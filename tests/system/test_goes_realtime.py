# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from el_paso.data_standards import GFZStandard
from el_paso.dataset import GFZDataSet
from el_paso.recipes.goes import process_goes_real_time
from el_paso.saving_strategies import MonthlyRBStrategy


@pytest.mark.basic
def test_goes_realtime_snapshot(
    tmpdir: Path,
    *,
    renew_solution: bool,
) -> None:

    start_time = datetime(2025, 12, 17, tzinfo=timezone.utc)
    end_time = start_time + timedelta(days=0.1)

    processed_data_path = Path(tmpdir)

    process_goes_real_time(
        start_time=start_time,
        end_time=end_time,
        satellite="primary",
        raw_data_path=Path(__file__).parent / "data" / "raw",
        processed_data_path=processed_data_path,
        num_cores=32,
        save_strategy="netcdf",
        skip_existing=True,
    )

    out_path = processed_data_path / "GOES" / "goes_primary" / "goes_primary_mps-high_20251201to20251231_T89.nc"
    assert out_path.exists(), "Output path does not exist!"

    goes_proc = GFZDataSet(
        start_time=start_time,
        end_time=end_time,
        saving_strategy=MonthlyRBStrategy(
            processed_data_path, "GOES", "goes_primary", "mps-high", "T89", GFZStandard(), "nc"
        ),
    )

    if renew_solution:
        # Copy every output file this strategy produces (e.g. "full" and "solar_wind_indices"),
        # not just the primary one, so a strategy that later gains more output groups stays covered.
        # get_file_path() needs the strategy's own rounded interval (e.g. full-month bounds for
        # MonthlyRBStrategy), not the raw start_time/end_time, to compute the correct file name.
        dest_dir = Path(__file__).parent / "data" / "processed" / "GOES" / "goes_primary"
        interval_start, interval_end = goes_proc.saving_strategy.get_time_intervals_to_save(start_time, end_time)[0]
        for output_file in goes_proc.saving_strategy.output_files:
            src_path = goes_proc.saving_strategy.get_file_path(interval_start, interval_end, output_file)
            if src_path.exists():
                shutil.copy(src_path, dest_dir)

    goes_true = GFZDataSet(
        start_time=start_time,
        end_time=end_time,
        saving_strategy=MonthlyRBStrategy(
            Path(__file__).parent / "data" / "processed", "GOES", "goes_primary", "mps-high", "T89", GFZStandard(), "nc"
        ),
    )

    goes_true.assert_equal(goes_proc)
