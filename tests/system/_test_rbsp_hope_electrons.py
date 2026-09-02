# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import shutil
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import el_paso as ep
from el_paso.dataset import GFZDataSet
from el_paso.recipes.rbsp import process_rbsp_hope_electrons


@pytest.mark.basic
def test_rbsp_hope_electrons(
    tmpdir: Path,
    skip_if_unreachable: Callable[..., None],
    *,
    renew_solution: bool,
) -> None:
    skip_if_unreachable("https://spdf.gsfc.nasa.gov")

    start_time = datetime(2017, 9, 8, tzinfo=timezone.utc)
    end_time = start_time + timedelta(days=0.4, seconds=-1)

    processed_data_path = tmpdir

    process_rbsp_hope_electrons(
        start_time=start_time,
        end_time=end_time,
        satellite="a",
        mag_field="T89",
        raw_data_path=Path(__file__).parent / "data" / "raw",
        processed_data_path=processed_data_path,
        num_cores=32,
        save_strategy="netcdf",
    )

    start_date = start_time.replace(day=1)
    end_date = end_time.replace(day=30)

    out_path = (
        processed_data_path
        / "RBSP"
        / "rbspa"
        / f"rbspa_hope_{start_date:%Y%m%d}to{end_date:%Y%m%d}_T89.nc"
    )
    assert out_path.exists()

    if renew_solution:
        shutil.copy(out_path, Path(__file__).parent / "data" / "processed" / "RBSP" / "rbspa")

    rbsp_proc = GFZDataSet(
        start_time=start_time,
        end_time=end_time,
        saving_strategy=ep.saving_strategies.MonthlyRBStrategy(
            tmpdir,
            "RBSP",
            "rbspa",
            "hope",
            "T89",
            data_standard=ep.data_standards.GFZStandard(),
            file_format="nc",
        ),
    )

    rbsp_true = GFZDataSet(
        start_time=start_time,
        end_time=end_time,
        saving_strategy=ep.saving_strategies.MonthlyRBStrategy(
            Path(__file__).parent / "data" / "processed",
            "RBSP",
            "rbspa",
            "hope",
            "T89",
            data_standard=ep.data_standards.GFZStandard(),
            file_format="nc",
        ),
    )

    rbsp_proc.assert_equal(rbsp_true)
