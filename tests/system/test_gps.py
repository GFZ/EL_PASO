# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Parvathy Santhini
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from el_paso.recipes.gps import process_gps_data


@pytest.mark.network
@pytest.mark.usefixtures("enable_downloads")
def test_lanl_gps(
    tmpdir: Path,
    *,
    renew_solution: bool,  # noqa: ARG001
) -> None:

    processed_data_path = tmpdir

    dt_start = datetime(2017, 4, 1, tzinfo=timezone.utc)
    dt_end = dt_start + timedelta(hours=4)

    process_gps_data(
        start_time=dt_start,
        end_time=dt_end,
        satellite="ns41",
        raw_data_path=processed_data_path,
        processed_data_path=processed_data_path,
        num_cores=32,
        bin_cadence=timedelta(minutes=4),
    )
    start_date = dt_start.replace(day=1)
    end_date = dt_end.replace(day=30)

    out_path = processed_data_path / "GPS" / "ns41" / "cxd" / f"ns41_cxd_{start_date:%Y%m%d}to{end_date:%Y%m%d}_T89.nc"
    assert out_path.exists()


@pytest.mark.visual
@pytest.mark.network
@pytest.mark.usefixtures("enable_downloads")
def test_lanl_gps_plot_default(tmp_path: Path, plot_default: Callable[..., None]) -> None:
    """Process GPS data and show the default figure of `el-paso plot gps cxd`. Check it by eye."""
    dt_start = datetime(2017, 4, 1, tzinfo=timezone.utc)
    dt_end = dt_start + timedelta(hours=4)

    process_gps_data(
        start_time=dt_start,
        end_time=dt_end,
        satellite="ns41",
        raw_data_path=tmp_path,
        processed_data_path=tmp_path,
        num_cores=32,
        bin_cadence=timedelta(minutes=4),
    )

    plot_default("gps", "cxd", tmp_path, dt_start, dt_end, "--satellite", "ns41")
