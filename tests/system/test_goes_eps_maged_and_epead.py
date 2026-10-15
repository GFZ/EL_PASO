# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from el_paso.dataset import GFZDataSet
from el_paso.recipes.goes import goes_eps_maged_and_epead_netcdf_strategy, process_goes_eps_maged_and_epead


@pytest.mark.basic
def test_goes_eps_maged_and_epead_snapshot(
    tmpdir: Path,
    *,
    renew_solution: bool,
) -> None:
    start_time = datetime(2017, 9, 20, tzinfo=timezone.utc)
    end_time = start_time + timedelta(hours=2, seconds=-1)

    processed_data_path = Path(tmpdir)

    process_goes_eps_maged_and_epead(
        start_time=start_time,
        end_time=end_time,
        satellite="goes15",
        mag_field="T01s",
        raw_data_path=Path(__file__).parent / "data" / "raw" / "GOES",
        processed_data_path=processed_data_path,
        num_cores=32,
        save_strategy="netcdf",
    )

    out_path = processed_data_path / "GOES" / "goes15" / "goes15_maged_epead_20170901to20170930_T01s.nc"
    assert out_path.exists(), "File did not get written!"

    if renew_solution:
        solution_path = Path(__file__).parent / "data" / "processed" / "GOES" / "goes15"
        solution_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(out_path, solution_path)

    goes_proc = GFZDataSet(
        goes_eps_maged_and_epead_netcdf_strategy(processed_data_path, "T01s", "goes15"),
        start_time=start_time,
        end_time=end_time,
    )
    goes_true = GFZDataSet(
        goes_eps_maged_and_epead_netcdf_strategy(Path(__file__).parent / "data" / "processed", "T01s", "goes15"),
        start_time=start_time,
        end_time=end_time,
    )

    # the invariant K (and V) is sensitive to floating-point differences of IRBEM on different hardware
    goes_true.assert_equal(goes_proc, rtol=1e-3)
