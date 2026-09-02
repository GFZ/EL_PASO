# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import shutil
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import el_paso as ep
from el_paso.dataset import DataSet
from el_paso.recipes.rbsp.process_rbsp_emfisis_waves import process_rbsp_emfisis_waves


@pytest.mark.basic
def test_rbsp_emfisis_waves(
    tmpdir: Path,
    skip_if_unreachable: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
    *,
    renew_solution: bool,
) -> None:
    skip_if_unreachable("https://cdaweb.gsfc.nasa.gov")

    monkeypatch.setattr(plt, "show", lambda: None)

    start_time = datetime(2017, 9, 8, tzinfo=timezone.utc)
    end_time = start_time + timedelta(hours=4)

    processed_data_path = tmpdir

    process_rbsp_emfisis_waves(
        start_time=start_time,
        end_time=end_time,
        satellite="a",
        raw_data_path=Path(__file__).parent / "data" / "raw",
        processed_data_path=processed_data_path,
    )

    out_path = (
        processed_data_path
        / "RBSP"
        / "rbspa"
        / f"rbspa_emfisis_{start_time:%Y%m%d}.nc"
    )
    assert out_path.exists()

    if renew_solution:
        shutil.copy(out_path, Path(__file__).parent / "data" / "processed" / "RBSP" / "rbspa")

    rbsp_proc = DataSet(
        start_time=start_time,
        end_time=end_time,
        saving_strategy=ep.saving_strategies.DailyWaveStrategy(
            tmpdir,
            "RBSP",
            "rbspa",
            "EMFISIS",
            ep.data_standards.GFZStandard(),
        ),
    )

    rbsp_true = DataSet(
        start_time=start_time,
        end_time=end_time,
        saving_strategy=ep.saving_strategies.DailyWaveStrategy(
            Path(__file__).parent / "data" / "processed",
            "RBSP",
            "rbspa",
            "EMFISIS",
            ep.data_standards.GFZStandard(),
        ),
    )

    rbsp_proc.assert_equal(rbsp_true)
