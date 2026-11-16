# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import pytest
from el_paso.dataset import GFZDataSet
from el_paso.recipes.rbsp import process_rbsp_rept_electrons, rbsp_rept_electron_netcdf_strategy


@pytest.mark.basic
@pytest.mark.parametrize("mag_field", ["T89"])
def test_rbsp_rept_electrons_snapshot(
    mag_field: Literal["T89", "TS04", "OP77", "T96"],
    tmpdir: Path,
    *,
    renew_solution: bool,
) -> None:
    start_time = datetime(2017, 9, 8, tzinfo=timezone.utc)
    end_time = start_time + timedelta(days=0.4, seconds=-1)

    processed_data_path = tmpdir

    process_rbsp_rept_electrons(
        start_time=start_time,
        end_time=end_time,
        satellite="a",
        mag_field=mag_field,
        raw_data_path=Path(__file__).parent / "data" / "raw",
        processed_data_path=processed_data_path,
        num_cores=32,
        bin_cadence=timedelta(hours=1),
        save_strategy="netcdf",
    )

    start_date = start_time.replace(day=1)
    end_date = end_time.replace(day=30)

    out_path = (
        processed_data_path
        / "RBSP"
        / "rbspa"
        / "rept"
        / f"rbspa_rept_{start_date:%Y%m%d}to{end_date:%Y%m%d}_{mag_field}.nc"
    )
    assert out_path.exists()

    rbsp_proc = GFZDataSet(
        start_time=start_time,
        end_time=end_time,
        saving_strategy=rbsp_rept_electron_netcdf_strategy(tmpdir, mag_field, "a"),
    )

    if renew_solution:
        # Copy every output file this strategy produces, so a strategy that later gains
        # more output groups stays covered.
        # get_file_path() needs the strategy's own rounded interval (full-month bounds for
        # MonthlyRBStrategy), not the raw start_time/end_time, to compute the correct name.
        dest_dir = Path(__file__).parent / "data" / "processed" / "RBSP" / "rbspa" / "rept"
        dest_dir.mkdir(parents=True, exist_ok=True)
        interval_start, interval_end = rbsp_proc.saving_strategy.get_time_intervals_to_save(start_time, end_time)[0]
        for output_file in rbsp_proc.saving_strategy.output_files:
            src_path = rbsp_proc.saving_strategy.get_file_path(interval_start, interval_end, output_file)
            if src_path.exists():
                shutil.copy(src_path, dest_dir)

    rbsp_true = GFZDataSet(
        start_time=start_time,
        end_time=end_time,
        saving_strategy=rbsp_rept_electron_netcdf_strategy(
            Path(__file__).parent / "data" / "processed", mag_field, "a"
        ),
    )

    rbsp_proc.assert_equal(rbsp_true)
