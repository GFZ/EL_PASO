# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import el_paso as ep
import pytest
from el_paso.dataset import DataSet
from el_paso.recipes.dmsp import process_dmsp_ssj_electrons


@pytest.mark.basic
def test_dmsp_ssj(
    tmpdir: Path,
    *,
    renew_solution: bool,
) -> None:

    start_time = datetime(2013, 9, 8, tzinfo=timezone.utc)
    end_time = start_time + timedelta(hours=4)

    processed_data_path = tmpdir

    process_dmsp_ssj_electrons(
        start_time=start_time,
        end_time=end_time,
        satellite="f17",
        raw_data_path=Path(__file__).parent / "data" / "raw",
        processed_data_path=processed_data_path,
        num_cores=32,
    )

    out_path = processed_data_path / "DMSP" / "f17" / f"f17_ssj_{start_time:%Y%m%d}_T89.nc"
    assert out_path.exists()

    dmsp_proc = DataSet(
        start_time=start_time,
        end_time=end_time,
        saving_strategy=ep.saving_strategies.DailyLEORBStrategy(
            tmpdir,
            "DMSP",
            "f17",
            "ssj",
            "T89",
            data_standard=ep.data_standards.GFZStandard(),
            file_format="nc",
        ),
    )

    if renew_solution:
        # Copy every output file this strategy produces (e.g. "full" and "solar_wind_indices"),
        # not just the primary one, so a strategy that later gains more output groups stays covered.
        # get_file_path() needs the strategy's own rounded interval, not the raw start_time/end_time,
        # to compute the correct file name.
        dest_dir = Path(__file__).parent / "data" / "processed" / "DMSP" / "f17"
        interval_start, interval_end = dmsp_proc.saving_strategy.get_time_intervals_to_save(start_time, end_time)[0]
        for output_file in dmsp_proc.saving_strategy.output_files:
            src_path = dmsp_proc.saving_strategy.get_file_path(interval_start, interval_end, output_file)
            if src_path.exists():
                shutil.copy(src_path, dest_dir)

    dmsp_true = DataSet(
        start_time=start_time,
        end_time=end_time,
        saving_strategy=ep.saving_strategies.DailyLEORBStrategy(
            Path(__file__).parent / "data" / "processed",
            "DMSP",
            "f17",
            "ssj",
            "T89",
            data_standard=ep.data_standards.GFZStandard(),
            file_format="nc",
        ),
    )

    dmsp_proc.assert_equal(dmsp_true)
