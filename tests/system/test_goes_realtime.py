# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from el_paso.dataset import GFZDataSet
from el_paso.saving_strategies import MonthlyFileStrategy
from examples.GOES.process_goes_realtime import process_goes_real_time


def test_goes_realtime_snapshot(
    tmpdir: Path,
    *,
    renew_solution: bool,
) -> None:
    start_time = datetime(2025, 12, 17, tzinfo=timezone.utc)
    end_time = start_time + timedelta(days=0.1)

    irbem_lib_path = Path(__file__).parent / "../../libirbem.so"

    processed_data_path = Path(tmpdir)

    process_goes_real_time(
        start_time=start_time,
        end_time=end_time,
        sat_str="primary",
        irbem_lib_path=irbem_lib_path,
        raw_data_path=Path(__file__).parent / "data" / "raw",
        processed_data_path=processed_data_path,
        num_cores=32,
        save_strategy="netcdf",
    )

    out_path = processed_data_path / "GOES" / "primary" / "goes_primary_mps_high_20251201to20251231_T89.nc"
    assert out_path.exists(), "Output path does not exist!"

    if renew_solution:
        shutil.copy(out_path, Path(__file__).parent / "data" / "processed" / "GOES" / "primary")

    goes_proc = GFZDataSet(
        start_time=start_time,
        end_time=end_time,
        saving_strategy=MonthlyFileStrategy(processed_data_path, "GOES", "goes_primary", "mps_high", "T89", "nc"),
    )

    goes_true = GFZDataSet(
        start_time=start_time,
        end_time=end_time,
        saving_strategy=MonthlyFileStrategy(
            Path(__file__).parent / "data" / "processed", "GOES", "goes_primary", "mps_high", "T89", "nc"
        ),
    )

    assert goes_proc == goes_true, f"Different variables: {goes_proc.get_different_variables(goes_true)}"
