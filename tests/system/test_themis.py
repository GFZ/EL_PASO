# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""System tests for the THEMIS recipes.

These check that each recipe runs end to end against the live archive, writes the file its
saving strategy promises, and that every variable it claims to save is readable back through
`el_paso.dataset`. They deliberately do not compare against a reference solution, so they need
no Zenodo data; they are a smoke test of the download, processing and saving path.
"""

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

import el_paso as ep
from el_paso.dataset import DataSet
from el_paso.recipes.themis import (
    process_themis_fft_waves,
    process_themis_scpot_density,
    themis_fft_waves_strategy,
    themis_scpot_density_strategy,
)

_START_TIME = datetime(2024, 5, 10, 20, 0, tzinfo=timezone.utc)
_END_TIME = _START_TIME + timedelta(hours=3)
_SATELLITE = "a"
_MAG_FIELD = "T89"

_WAVE_VARIABLES: list[ep.typing.InternalName] = [
    "Epoch",
    "Wave_frequency",
    "Magnetic_Power_Spectral_Density",
    "Number_density",
    "B_total_obs",
    "MLat",
    "MLT",
    "R_Eq",
    "f_ce",
    "f_ce_Eq",
]

_DENSITY_VARIABLES: list[ep.typing.InternalName] = [
    "Epoch",
    "Number_density",
    "Number_density_Eq",
    "Position",
    "xGEO_Eq",
    "MLT",
    "R_Eq",
]


def _assert_variables_load(dataset: DataSet, internal_names: list[ep.typing.InternalName]) -> None:
    """Assert every internal name reads back as a non-empty array sharing the time axis."""
    n_records = len(dataset.get_by_internal_name("Epoch"))
    assert n_records > 0, "no records were written"

    for internal_name in internal_names:
        data = dataset.get_by_internal_name(internal_name)

        assert data is not None, f"{internal_name} could not be loaded"
        assert data.size > 0, f"{internal_name} loaded as an empty array"

        # Everything except the frequency axis is time dependent.
        if internal_name != "Wave_frequency":
            assert data.shape[0] == n_records, f"{internal_name} has {data.shape[0]} records, expected {n_records}"

        assert not np.all(np.isnan(data)), f"{internal_name} is entirely NaN"


@pytest.mark.basic
def test_themis_fft_waves(tmpdir: Path, skip_if_unreachable: Callable[..., None]) -> None:
    skip_if_unreachable("https://themis.ssl.berkeley.edu")

    processed_data_path = Path(tmpdir)

    process_themis_fft_waves(
        start_time=_START_TIME,
        end_time=_END_TIME,
        satellite=_SATELLITE,
        mag_field=_MAG_FIELD,
        raw_data_path=Path(__file__).parent / "data" / "raw" / "themis",
        processed_data_path=processed_data_path,
        num_cores=4,
    )

    out_path = processed_data_path / "THEMIS" / "tha" / f"tha_fft_{_START_TIME:%Y%m%d}.nc"
    assert out_path.exists(), f"expected output file was not written: {out_path}"

    dataset = DataSet(
        saving_strategy=themis_fft_waves_strategy(processed_data_path, _SATELLITE),
        start_time=_START_TIME,
        end_time=_END_TIME,
    )

    _assert_variables_load(dataset, _WAVE_VARIABLES)


@pytest.mark.basic
def test_themis_scpot_density(tmpdir: Path, skip_if_unreachable: Callable[..., None]) -> None:
    skip_if_unreachable("https://themis.ssl.berkeley.edu")

    processed_data_path = Path(tmpdir)

    process_themis_scpot_density(
        start_time=_START_TIME,
        end_time=_END_TIME,
        satellite=_SATELLITE,
        mag_field=_MAG_FIELD,
        raw_data_path=Path(__file__).parent / "data" / "raw" / "themis",
        processed_data_path=processed_data_path,
        num_cores=4,
    )

    # MonthlyDensityStrategy writes one file per calendar month.
    month_start = _START_TIME.replace(day=1)
    month_end = month_start + timedelta(days=30)
    out_path = (
        processed_data_path / "THEMIS" / "tha" / f"tha_scpot_{month_start:%Y%m%d}to{month_end:%Y%m%d}_{_MAG_FIELD}.nc"
    )
    assert out_path.exists(), f"expected output file was not written: {out_path}"

    dataset = DataSet(
        saving_strategy=themis_scpot_density_strategy(processed_data_path, _SATELLITE, _MAG_FIELD),
        start_time=_START_TIME,
        end_time=_END_TIME,
    )

    _assert_variables_load(dataset, _DENSITY_VARIABLES)
