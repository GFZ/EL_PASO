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

_GSAT0215_OMM_CSV = (
    "OBJECT_NAME,OBJECT_ID,EPOCH,MEAN_MOTION,ECCENTRICITY,INCLINATION,RA_OF_ASC_NODE,"
    "ARG_OF_PERICENTER,MEAN_ANOMALY,EPHEMERIS_TYPE,CLASSIFICATION_TYPE,NORAD_CAT_ID,"
    "ELEMENT_SET_NO,REV_AT_EPOCH,BSTAR,MEAN_MOTION_DOT,MEAN_MOTION_DDOT\n"
    "GSAT0215 (GALILEO 19),2017-079A,2026-09-07T17:21:47.336256,1.70473847,.00034237,"
    "55.0552,219.2362,279.8195,80.1042,0,U,43055,999,5439,0,.15E-6,0\n"
)


def _seed_omm_cache(raw_data_path: Path) -> None:
    """Pre-seed today's OMM cache file so _download_gssc_emu_omm never queries Celestrak live."""
    omm_dir = raw_data_path / "OMM" / "gsat0215"
    omm_dir.mkdir(parents=True, exist_ok=True)
    (omm_dir / f"gsat0215_omm_{datetime.now(timezone.utc):%Y%m%d}.csv").write_text(_GSAT0215_OMM_CSV)


@pytest.mark.basic
def test_esa_gssc_emu_electrons(
    tmpdir: Path,
    skip_if_unreachable: Callable[..., None],
    *,
    renew_solution: bool,  # noqa: ARG001
) -> None:
    skip_if_unreachable("ftp://gssc.esa.int")

    start_time = datetime(2019, 1, 4, tzinfo=timezone.utc)
    end_time = start_time + timedelta(hours=4)

    raw_data_path = Path(__file__).parent / "data" / "raw"
    processed_data_path = tmpdir

    _seed_omm_cache(raw_data_path)

    process_gssc_emu_electrons(
        start_time=start_time,
        end_time=end_time,
        satellite="gsat0215",
        raw_data_path=raw_data_path,
        processed_data_path=processed_data_path,
        num_cores=32,
        calculate_Lstar=True,
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
def test_esa_gssc_emu_protons(
    tmpdir: Path,
    skip_if_unreachable: Callable[..., None],
    *,
    renew_solution: bool,  # noqa: ARG001
) -> None:
    skip_if_unreachable("ftp://gssc.esa.int")

    start_time = datetime(2019, 1, 4, tzinfo=timezone.utc)
    end_time = start_time + timedelta(hours=4)

    raw_data_path = Path(__file__).parent / "data" / "raw"
    processed_data_path = tmpdir

    _seed_omm_cache(raw_data_path)

    process_gssc_emu_protons(
        start_time=start_time,
        end_time=end_time,
        satellite="gsat0215",
        raw_data_path=raw_data_path,
        processed_data_path=processed_data_path,
        num_cores=32,
        calculate_Lstar=True,
    )

    start_date = start_time.replace(day=1)
    end_date = start_time.replace(day=calendar.monthrange(start_time.year, start_time.month)[1])

    out_path = (
        processed_data_path / "ESA" / "gsat0215" / f"gsat0215_emu-proton_{start_date:%Y%m%d}to{end_date:%Y%m%d}_T89.nc"
    )
    assert out_path.exists()
