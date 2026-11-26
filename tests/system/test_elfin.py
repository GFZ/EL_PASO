# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Jaskirat Singh
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
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

    dt_start = datetime(2020, 9, 29, tzinfo=timezone.utc)
    dt_end = dt_start + timedelta(hours=8)

    process_elfin_epd_electrons(
        start_time=dt_start,
        end_time=dt_end,
        satellite="elb",
        raw_data_path=processed_data_path,
        processed_data_path=processed_data_path,
        num_cores=32,
        bin_cadence=timedelta(minutes=1.4),
    )

    out_path = processed_data_path / "ELFIN" / "elb" / "epd" / f"{dt_start:%Y/%m}" / f"elb_epd_{dt_start:%Y%m%d}_Dip.nc"
    print(f"Checking if {out_path} exists...")

    assert out_path.exists()


@pytest.mark.visual
@pytest.mark.network
@pytest.mark.usefixtures("enable_downloads")
def test_elfin_epd_plot_default(tmp_path: Path, plot_default: Callable[..., None]) -> None:
    """Process ELFIN EPD data and show the default figure of `el-paso plot elfin epd`. Check it by eye."""
    dt_start = datetime(2020, 9, 29, 6, 56, tzinfo=timezone.utc)
    dt_end = datetime(2020, 9, 29, 7, 2, tzinfo=timezone.utc)

    process_elfin_epd_electrons(
        start_time=dt_start,
        end_time=dt_end,
        satellite="elb",
        raw_data_path=tmp_path,
        processed_data_path=tmp_path,
        num_cores=32,
        bin_cadence=timedelta(seconds=4),
    )

    plot_default(
        "elfin",
        "epd",
        tmp_path,
        dt_start,
        dt_end,
        "--satellite",
        "elb",
        "--colormap",
        "jet",
        "--vmin",
        "1e3",
        "--vmax",
        "1e10",
    )
