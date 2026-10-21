# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

import calendar
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from el_paso.recipes.esa import process_gssc_emu_electrons, process_gssc_emu_protons


@pytest.mark.basic
@pytest.mark.usefixtures("enable_downloads")
def test_esa_gssc_emu_electrons(
    tmpdir: Path,
    skip_if_unreachable: Callable[..., None],
    *,
    renew_solution: bool,  # noqa: ARG001
) -> None:
    skip_if_unreachable("ftp://gssc.esa.int", "https://www.space-track.org")

    start_time = datetime(2019, 1, 4, tzinfo=timezone.utc)
    end_time = start_time + timedelta(hours=4)

    raw_data_path = Path(__file__).parent / "data" / "raw"
    processed_data_path = tmpdir

    process_gssc_emu_electrons(
        start_time=start_time,
        end_time=end_time,
        satellite="gsat0215",
        raw_data_path=raw_data_path,
        processed_data_path=processed_data_path,
        num_cores=32,
        calculate_Lstar=False,
    )

    start_date = start_time.replace(day=1)
    end_date = start_time.replace(day=calendar.monthrange(start_time.year, start_time.month)[1])

    out_path = (
        processed_data_path
        / "ESA"
        / "gsat0215"
        / f"gsat0215_emu-electron_{start_date:%Y%m%d}to{end_date:%Y%m%d}_T89.nc"
    )
    assert out_path.exists()


@pytest.mark.basic
@pytest.mark.usefixtures("enable_downloads")
def test_esa_gssc_emu_protons(
    tmpdir: Path,
    skip_if_unreachable: Callable[..., None],
    *,
    renew_solution: bool,  # noqa: ARG001
) -> None:
    skip_if_unreachable("ftp://gssc.esa.int", "https://www.space-track.org")

    start_time = datetime(2019, 1, 4, tzinfo=timezone.utc)
    end_time = start_time + timedelta(hours=4)

    raw_data_path = Path(__file__).parent / "data" / "raw"
    processed_data_path = tmpdir

    process_gssc_emu_protons(
        start_time=start_time,
        end_time=end_time,
        satellite="gsat0215",
        raw_data_path=raw_data_path,
        processed_data_path=processed_data_path,
        num_cores=32,
        calculate_Lstar=False,
    )

    start_date = start_time.replace(day=1)
    end_date = start_time.replace(day=calendar.monthrange(start_time.year, start_time.month)[1])

    out_path = (
        processed_data_path / "ESA" / "gsat0215" / f"gsat0215_emu-proton_{start_date:%Y%m%d}to{end_date:%Y%m%d}_T89.nc"
    )
    assert out_path.exists()
