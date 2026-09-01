# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Parvathy Santhini
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from el_paso.recipes.gps import process_gps_data


@pytest.mark.basic
def test_lanl_gps(
    tmpdir: Path,
    skip_if_unreachable: Callable[..., None],
    *,
    renew_solution: bool,  # noqa: ARG001
) -> None:

    skip_if_unreachable(
        "https://www.ngdc.noaa.gov/stp/space-weather/satellite-data/satellite-systems/lanl_gps/version_v1.10r2"
    )

    processed_data_path = tmpdir

    dt_start = datetime(2017, 4, 1, tzinfo=timezone.utc)
    dt_end = dt_start + timedelta(hours=4)

    process_gps_data(
        start_time=dt_start,
        end_time=dt_end,
        satellite_str="ns41",
        raw_data_path=processed_data_path,
        processed_data_path=processed_data_path,
        num_cores=64,
        bin_cadence=timedelta(minutes=4),
    )
    start_date = dt_start.replace(day=1)
    end_date = dt_end.replace(day=30)

    out_path = processed_data_path / "GPS" / "ns41" / f"ns41_cxd_{start_date:%Y%m%d}to{end_date:%Y%m%d}_T89.nc"
    assert out_path.exists()
