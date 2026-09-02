# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import shutil
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import pytest

import el_paso as ep
from el_paso.dataset import DataSet
from el_paso.recipes.dmsp import process_dmsp_ssj_electrons


@pytest.mark.basic
def test_dmsp_ssj(
    tmpdir: Path,
    skip_if_unreachable: Callable[..., None],
    *,
    renew_solution: bool,
) -> None:
    skip_if_unreachable("https://spdf.gsfc.nasa.gov")

    start_time = datetime(2013, 9, 8, tzinfo=timezone.utc)
    end_time = start_time + timedelta(hours=4)

    processed_data_path = tmpdir

    process_dmsp_ssj_electrons(
        start_time=start_time,
        end_time=end_time,
        satellite="f17",
        raw_data_path=Path(__file__).parent / "data" / "raw",
        processed_data_path=processed_data_path,
        num_cores=32,
    )

    out_path = (
        processed_data_path
        / "DMSP"
        / "f17"
        / f"f17_ssj_{start_time:%Y%m%d}_T89.nc"
    )
    assert out_path.exists()

    if renew_solution:
        shutil.copy(out_path, Path(__file__).parent / "data" / "processed" / "DMSP" / "f17")

    dmsp_proc = DataSet(
        start_time=start_time,
        end_time=end_time,
        saving_strategy=ep.saving_strategies.DailyLEORBStrategy(
            tmpdir,
            "DMSP",
            "f17",
            "ssj",
            "T89",
            data_standard=ep.data_standards.GFZStandard(),
            file_format="nc",
        ),
    )

    dmsp_true = DataSet(
        start_time=start_time,
        end_time=end_time,
        saving_strategy=ep.saving_strategies.DailyLEORBStrategy(
            Path(__file__).parent / "data" / "processed",
            "DMSP",
            "f17",
            "ssj",
            "T89",
            data_standard=ep.data_standards.GFZStandard(),
            file_format="nc",
        ),
    )

    dmsp_proc.assert_equal(dmsp_true)
