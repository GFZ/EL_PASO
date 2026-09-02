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
from el_paso.recipes.poes import process_poes_meped_electron, process_poes_ted_electron


@pytest.mark.basic
def test_poes_ted_electron(
    tmpdir: Path,
    skip_if_unreachable: Callable[..., None],
    *,
    renew_solution: bool,
) -> None:
    skip_if_unreachable("https://spdf.gsfc.nasa.gov")

    start_time = datetime(2013, 9, 8, tzinfo=timezone.utc)
    end_time = start_time + timedelta(hours=4)

    processed_data_path = tmpdir

    process_poes_ted_electron(
        start_time=start_time,
        end_time=end_time,
        satellite="metop1",
        raw_data_path=Path(__file__).parent / "data" / "raw",
        processed_data_path=processed_data_path,
        num_cores=32,
        calculate_Lm_Lstar=True,
    )

    out_path = (
        processed_data_path
        / "POES"
        / "metop1"
        / f"metop1_ted_{start_time:%Y%m%d}_T89.nc"
    )
    assert out_path.exists()

    if renew_solution:
        shutil.copy(out_path, Path(__file__).parent / "data" / "processed" / "POES" / "metop1")

    poes_proc = DataSet(
        start_time=start_time,
        end_time=end_time,
        saving_strategy=ep.saving_strategies.DailyLEORBStrategy(
            tmpdir,
            "POES",
            "metop1",
            "ted",
            "T89",
            data_standard=ep.data_standards.GFZStandard(),
            file_format="nc",
        ),
    )

    poes_true = DataSet(
        start_time=start_time,
        end_time=end_time,
        saving_strategy=ep.saving_strategies.DailyLEORBStrategy(
            Path(__file__).parent / "data" / "processed",
            "POES",
            "metop1",
            "ted",
            "T89",
            data_standard=ep.data_standards.GFZStandard(),
            file_format="nc",
        ),
    )

    poes_proc.assert_equal(poes_true)

@pytest.mark.basic
def test_poes_meped_electron(
    tmpdir: Path,
    skip_if_unreachable: Callable[..., None],
    *,
    renew_solution: bool,
) -> None:
    skip_if_unreachable("https://spdf.gsfc.nasa.gov")

    start_time = datetime(2013, 9, 8, tzinfo=timezone.utc)
    end_time = start_time + timedelta(hours=4)

    processed_data_path = tmpdir

    process_poes_meped_electron(
        start_time=start_time,
        end_time=end_time,
        satellite="noaa18",
        raw_data_path=Path(__file__).parent / "data" / "raw",
        processed_data_path=processed_data_path,
        num_cores=32,
    )

    out_path = (
        processed_data_path
        / "POES"
        / "noaa18"
        / f"noaa18_meped_{start_time:%Y%m%d}_T89.nc"
    )
    assert out_path.exists()

    if renew_solution:
        shutil.copy(out_path, Path(__file__).parent / "data" / "processed" / "POES" / "noaa18")

    poes_proc = DataSet(
        start_time=start_time,
        end_time=end_time,
        saving_strategy=ep.saving_strategies.DailyLEORBStrategy(
            tmpdir,
            "POES",
            "noaa18",
            "meped",
            "T89",
            data_standard=ep.data_standards.GFZStandard(),
            file_format="nc",
        ),
    )

    poes_true = DataSet(
        start_time=start_time,
        end_time=end_time,
        saving_strategy=ep.saving_strategies.DailyLEORBStrategy(
            Path(__file__).parent / "data" / "processed",
            "POES",
            "noaa18",
            "meped",
            "T89",
            data_standard=ep.data_standards.GFZStandard(),
            file_format="nc",
        ),
    )

    poes_proc.assert_equal(poes_true)
