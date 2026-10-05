# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

import calendar
from datetime import datetime, timedelta, timezone
from pathlib import Path

import el_paso as ep
import numpy as np
import pytest
from el_paso.dataset import DataSet
from el_paso.recipes.arase import arase_xep_strategy, process_arase_xep

_START_TIME = datetime(2017, 9, 8, tzinfo=timezone.utc)
_END_TIME = _START_TIME + timedelta(hours=3)
_MAG_FIELD = "T89"

_XEP_VARIABLES: list[ep.typing.InternalName] = [
    "Epoch",
    "FEDU",
    "FEDO",
    "Energy_FEDU",
    "Energy_FEDO",
    "Alpha",
    "Alpha_range",
    "Alpha_Eq",
    "R_Eq",
    "MLT",
    "L_m",
    "L_star",
    "PSD",
    "InvK",
    "InvMu",
]


def _assert_variables_load(dataset: DataSet, internal_names: list[ep.typing.InternalName]) -> None:
    """Assert every internal name reads back as a non-empty array sharing the time axis."""
    n_records = len(dataset.get_by_internal_name("Epoch"))
    assert n_records > 0, "no records were written"

    for internal_name in internal_names:
        data = dataset.get_by_internal_name(internal_name)

        assert data is not None, f"{internal_name} could not be loaded"
        assert data.size > 0, f"{internal_name} loaded as an empty array"
        assert data.shape[0] == n_records, f"{internal_name} has {data.shape[0]} records, expected {n_records}"
        assert not np.all(np.isnan(data)), f"{internal_name} is entirely NaN"


@pytest.mark.basic
def test_arase_xep(tmpdir: Path) -> None:

    processed_data_path = Path(tmpdir)

    process_arase_xep(
        start_time=_START_TIME,
        end_time=_END_TIME,
        mag_field=_MAG_FIELD,
        raw_data_path=Path(__file__).parent / "data" / "raw",
        processed_data_path=processed_data_path,
        num_cores=4,
        use_level_3_orbit_data=False,
    )

    month_start = _START_TIME.replace(day=1)
    month_end = month_start.replace(day=calendar.monthrange(month_start.year, month_start.month)[1])
    out_path = (
        processed_data_path / "ARASE" / "arase" / f"arase_xep_{month_start:%Y%m%d}to{month_end:%Y%m%d}_{_MAG_FIELD}.nc"
    )
    assert out_path.exists(), f"expected output file was not written: {out_path}"

    dataset = DataSet(
        saving_strategy=arase_xep_strategy(processed_data_path, _MAG_FIELD),
        start_time=_START_TIME,
        end_time=_END_TIME,
    )

    _assert_variables_load(dataset, _XEP_VARIABLES)
