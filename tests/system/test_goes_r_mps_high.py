# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from el_paso.dataset import GFZDataSet
from el_paso.recipes.goes import goes_r_mps_high_netcdf_strategy, process_goes_r_mps_high


@pytest.mark.basic
def test_goes_r_mps_high_snapshot(
    tmpdir: Path,
    *,
    renew_solution: bool,
) -> None:
    start_time = datetime(2024, 5, 27, tzinfo=timezone.utc)
    end_time = start_time + timedelta(hours=2, seconds=-1)

    processed_data_path = Path(tmpdir)

    process_goes_r_mps_high(
        start_time=start_time,
        end_time=end_time,
        satellite="goes18",
        mag_field="T96",
        raw_data_path=Path(__file__).parent / "data" / "raw" / "GOES",
        processed_data_path=processed_data_path,
        num_cores=32,
        save_strategy="netcdf",
    )

    out_path = processed_data_path / "GOES" / "goes18" / "goes18_mps_high_20240501to20240531_T96.nc"
    assert out_path.exists(), "File did not get written!"

    if renew_solution:
        solution_path = Path(__file__).parent / "data" / "processed" / "GOES" / "goes18"
        solution_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(out_path, solution_path)

    goes_proc = GFZDataSet(
        goes_r_mps_high_netcdf_strategy(processed_data_path, "T96", "goes18"),
        start_time=start_time,
        end_time=end_time,
    )
    goes_true = GFZDataSet(
        goes_r_mps_high_netcdf_strategy(Path(__file__).parent / "data" / "processed", "T96", "goes18"),
        start_time=start_time,
        end_time=end_time,
    )

    goes_true.assert_equal(goes_proc)
