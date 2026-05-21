# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import pytest

import el_paso as ep
from el_paso.dataset import GFZDataSet
from el_paso.recipes.arase import process_arase_mepe


@pytest.mark.parametrize("mag_field", ["T89", "TS04"])
@pytest.mark.basic
def test_arase_mepe_snapshot(
    mag_field: Literal["T89", "TS04"],
    tmpdir: Path,
    *,
    renew_solution: bool,
) -> None:
    start_time = datetime(2017, 9, 8, tzinfo=timezone.utc)
    end_time = start_time + timedelta(hours=4, seconds=-1)

    processed_data_path = tmpdir

    process_arase_mepe(
        start_time,
        end_time,
        mag_field,
        raw_data_path=Path(__file__).parent / "data" / "raw",
        processed_data_path=processed_data_path,
        num_cores=32,
        cadence=timedelta(hours=1),
        save_strategy="netcdf",
        use_level_3_orbit_data=False,
    )

    start_date = start_time.replace(day=1)
    end_date = end_time.replace(day=30)

    out_path = (
        processed_data_path / "ARASE" / "arase" / f"arase_mepe_{start_date:%Y%m%d}to{end_date:%Y%m%d}_{mag_field}.nc"
    )
    assert out_path.exists(), "File did not get written!"

    if renew_solution:
        shutil.copy(out_path, Path(__file__).parent / "data" / "processed" / "ARASE" / "arase")

    arase_proc = GFZDataSet(
        ep.saving_strategies.MonthlyRBStrategy(
            tmpdir, "ARASE", "arase", "mepe", mag_field, data_standard=ep.data_standards.GFZStandard(), file_format="nc"
        ),
        start_time=start_time,
        end_time=end_time,
    )
    arase_true = GFZDataSet(
        ep.saving_strategies.MonthlyRBStrategy(
            Path(__file__).parent / "data" / "processed",
            "Arase",
            "arase",
            "mepe",
            mag_field,
            ep.data_standards.GFZStandard(),
            "nc",
        ),
        start_time=start_time,
        end_time=end_time,
    )

    arase_true.assert_equal(arase_proc)
