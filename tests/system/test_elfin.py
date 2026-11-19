# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Jaskirat Singh
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from el_paso.recipes.elfin import process_elfin_epd_electrons


@pytest.mark.network
@pytest.mark.usefixtures("enable_downloads")
def test_elfin_epd_data(
    tmpdir: Path,
    *,
    renew_solution: bool,  # noqa: ARG001
) -> None:

    processed_data_path = tmpdir

    dt_start = datetime(2021, 4, 1, tzinfo=timezone.utc)
    dt_end = dt_start + timedelta(hours=4)

    process_elfin_epd_electrons(
        start_time=dt_start,
        end_time=dt_end,
        satellite="ela",
        raw_data_path=processed_data_path,
        processed_data_path=processed_data_path,
        num_cores=32,
        bin_cadence=timedelta(minutes=4),
    )
    start_date = dt_start.replace(day=1)

    out_path = (
        processed_data_path / "ELFIN" / "ela" / "epd" / f"{start_date:%Y/%m}" / f"ela_epd_{start_date:%Y%m%d}_Dip.nc"
    )
    print(out_path)
    assert out_path.exists()
