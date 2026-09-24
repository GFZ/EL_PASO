# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from el_paso.recipes.esa import process_ngrm_electron_fluxes


@pytest.mark.basic
def test_esa_ngrm(
    tmpdir: Path,
    *,
    renew_solution: bool,  # noqa: ARG001
) -> None:

    client_id = os.environ.get("ESA_CLIENT_ID")
    client_secret = os.environ.get("ESA_CLIENT_SECRET")

    start_time = datetime(2025, 9, 8, tzinfo=timezone.utc)
    end_time = start_time + timedelta(hours=4)

    processed_data_path = tmpdir

    process_ngrm_electron_fluxes(
        start_time=start_time,
        end_time=end_time,
        satellite="EDRS-C",
        raw_data_path=Path(__file__).parent / "data" / "raw",
        processed_data_path=processed_data_path,
        num_cores=32,
        client_id=client_id,
        client_secret=client_secret,
        bin_cadence=timedelta(minutes=5),
    )

    start_date = start_time.replace(day=1)
    end_date = end_time.replace(day=30)

    out_path = processed_data_path / "ESA" / "edrs-c" / f"edrs-c_ngrm_{start_date:%Y%m%d}to{end_date:%Y%m%d}_T89.nc"
    assert out_path.exists()
