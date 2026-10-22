# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""System tests for the combined RBSP EFW/EMFISIS density recipe.

These check that the recipe runs end to end against the live archive, writes the file its
saving strategy promises, and that every variable it claims to save is readable back through
`el_paso.dataset`. They deliberately do not compare against a reference solution, so they need
no reference test data; they are a smoke test of the download, processing and saving path.
"""

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import el_paso as ep
import numpy as np
import pytest
from el_paso.dataset import DataSet
from el_paso.recipes.rbsp import (
    process_rbsp_efw_emfisis_density_combined,
    rbsp_efw_emfisis_density_combined_strategy,
)

# Crosses midnight, so both daily EFW and EMFISIS files have to be downloaded and extracted.
_START_TIME = datetime(2017, 9, 8, 22, 0, tzinfo=timezone.utc)
_END_TIME = _START_TIME + timedelta(hours=3)
_SATELLITE = "a"
_MAG_FIELD = "OP77"

_DENSITY_VARIABLES: list[ep.typing.InternalName] = [
    "Epoch",
    "Number_density_scpot",
    "Number_density_scpot_Eq",
    "Number_density_spectrogram",
    "Number_density_spectrogram_Eq",
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
        assert data.shape[0] == n_records, f"{internal_name} has {data.shape[0]} records, expected {n_records}"
        assert not np.all(np.isnan(data)), f"{internal_name} is entirely NaN"


@pytest.mark.basic
def test_rbsp_density_without_hiss(
    tmpdir: Path,
    skip_if_unreachable: Callable[..., None],
) -> None:
    skip_if_unreachable("https://spdf.gsfc.nasa.gov")

    processed_data_path = Path(tmpdir)

    process_rbsp_efw_emfisis_density_combined(
        start_time=_START_TIME,
        end_time=_END_TIME,
        satellite=_SATELLITE,
        mag_field=_MAG_FIELD,
        raw_data_path=Path(__file__).parent / "data" / "raw" / "rbsp",
        processed_data_path=processed_data_path,
        num_cores=4,
        save_strategy="mat",
        add_hiss_derived_densitites=False,
    )

    file_format = "mat"

    # RBSPDensityStrategy writes one file per calendar month.
    month_start = _START_TIME.replace(day=1)
    month_end = month_start.replace(day=30)
    out_path = (
        processed_data_path
        / "RBSP"
        / "rbspa"
        / f"rbspa_densities_combined_{month_start:%Y%m%d}to{month_end:%Y%m%d}_{_MAG_FIELD}.{file_format}"
    )
    assert out_path.exists(), f"expected output file was not written: {out_path}"

    dataset = DataSet(
        saving_strategy=rbsp_efw_emfisis_density_combined_strategy(
            processed_data_path, _SATELLITE, _MAG_FIELD, file_format=file_format
        ),
        start_time=_START_TIME,
        end_time=_END_TIME,
    )

    _assert_variables_load(dataset, _DENSITY_VARIABLES)
