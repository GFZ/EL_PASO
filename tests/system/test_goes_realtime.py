# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import shutil
from collections.abc import Callable
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
    skip_if_unreachable: Callable[..., None],
    *,
    renew_solution: bool,
) -> None:
    skip_if_unreachable("https://spdf.gsfc.nasa.gov")

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

    if renew_solution:
        shutil.copy(out_path, Path(__file__).parent / "data" / "processed" / "GOES" / "goes_primary")

    goes_proc = GFZDataSet(
        start_time=start_time,
        end_time=end_time,
        saving_strategy=MonthlyRBStrategy(
            processed_data_path, "GOES", "goes_primary", "mps-high", "T89", GFZStandard(), "nc"
        ),
    )

    goes_true = GFZDataSet(
        start_time=start_time,
        end_time=end_time,
        saving_strategy=MonthlyRBStrategy(
            Path(__file__).parent / "data" / "processed", "GOES", "goes_primary", "mps-high", "T89", GFZStandard(), "nc"
        ),
    )

    goes_true.assert_equal(goes_proc)
