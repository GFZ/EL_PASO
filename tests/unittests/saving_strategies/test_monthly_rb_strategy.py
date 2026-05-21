# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Literal  # noqa: UP035

import numpy as np
import pytest
from astropy import units as u  # type: ignore[reportMissingTypeStubs]
from icecream import ic

import el_paso as ep
from el_paso.dataset.utils import python2matlab

if TYPE_CHECKING:
    from pathlib import Path

    from el_paso.typing import DataStandard, InternalName


def _mock_monthly_variables() -> dict[InternalName, ep.Variable]:
    """Create mocked monthly product variables without running processing code."""
    time_size = 145
    energy_size = 3
    alpha_size = 4

    start_time = datetime(2013, 1, 1, tzinfo=timezone.utc)
    datetimes = [start_time + i * np.timedelta64(3600, "s") for i in range(time_size)]
    epoch = np.array([python2matlab(i) for i in datetimes])

    variables: dict[InternalName, ep.Variable] = {
        "Epoch": ep.Variable(original_unit=ep.units.datenum, data=epoch),
        "FEDU": ep.Variable(
            original_unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
            data=np.arange(time_size * energy_size * alpha_size, dtype=float).reshape(
                time_size,
                energy_size,
                alpha_size,
            ),
        ),
        "Alpha_Eq": ep.Variable(original_unit=u.deg, data=np.full((time_size, alpha_size), 45.0)),
        "Energy_FEDU": ep.Variable(
            original_unit=u.MeV,
            data=np.tile(np.asarray([0.5, 1.0, 2.0]), (time_size, 1)),
        ),
        "Alpha": ep.Variable(
            original_unit=u.deg,
            data=np.tile(np.asarray([10.0, 30.0, 60.0, 90.0]), (time_size, 1)),
        ),
        "B_Calc": ep.Variable(original_unit=u.nT, data=np.full(time_size, 75.0)),
        "B_Eq": ep.Variable(original_unit=u.nT, data=np.full(time_size, 50.0)),
        "InvK": ep.Variable(
            original_unit=ep.units.RE * u.G**0.5,
            data=np.full((time_size, alpha_size), 1.5),
        ),
        "InvMu": ep.Variable(
            original_unit=u.MeV / u.G,
            data=np.full((time_size, energy_size, alpha_size), 2.5),
        ),
        "Position": ep.Variable(
            original_unit=ep.units.RE,
            data=np.arange(time_size * 3, dtype=float).reshape(time_size, 3),
        ),
        "PSD": ep.Variable(
            original_unit=(u.m * u.kg * u.m / u.s) ** (-3),
            data=np.full((time_size, energy_size, alpha_size), 3.5),
        ),
        "R_Eq": ep.Variable(original_unit=ep.units.RE, data=np.full(time_size, 6.0)),
        "MLT": ep.Variable(original_unit=u.hour, data=np.full(time_size, 12.0)),
        "L_m": ep.Variable(
            original_unit=u.dimensionless_unscaled,
            data=np.full((time_size, alpha_size), 4.5),
        ),
        "L_star": ep.Variable(
            original_unit=u.dimensionless_unscaled,
            data=np.full((time_size, alpha_size), 5.5),
        ),
    }

    for variable in variables.values():
        variable.metadata.source_files = ["mocked_input.cdf"]

    return variables


_STANDARD_META_KEYS = {"unit", "original_cadence_seconds", "source_files", "processing_notes", "description"}


def check_metadata(keys: set[str]):
    return lambda actual: actual >= keys


_FORMAT_PARAMS = [
    ("nc", check_metadata(_STANDARD_META_KEYS)),
    ("h5", check_metadata(_STANDARD_META_KEYS)),
    ("cdf", check_metadata(_STANDARD_META_KEYS)),  # this also contains "Compress" hence ">=" check
    ("mat", check_metadata(_STANDARD_META_KEYS)),
]


@pytest.mark.basic
@pytest.mark.parametrize("data_standard", [ep.data_standards.GFZStandard, ep.data_standards.PRBEMStandard])
@pytest.mark.parametrize(("output_format", "meta_keys_check"), _FORMAT_PARAMS)
def test_monthly_strategy_saves_mocked_variables_to_netcdf_with_data_standards(
    tmp_path: Path,
    data_standard: type[DataStandard],
    output_format: Literal["nc", "h5", "cdf", "mat"],
    meta_keys_check: Callable[[set[str]], bool],
) -> None:
    variables = _mock_monthly_variables()
    start_time = datetime(2013, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2013, 1, 7, tzinfo=timezone.utc)
    MISSION = "GOES"
    SATELLITE = "primary"
    INSTRUMENT = "MAGED"
    MAG_FIELD = "T89"
    strategy = ep.saving_strategies.MonthlyRBStrategy(
        base_data_path=tmp_path,
        mission=MISSION,
        satellite=SATELLITE,
        instrument=INSTRUMENT,
        mag_field=MAG_FIELD,
        file_format=output_format,
        data_standard=data_standard(),
    )

    ep.save(
        variables,
        strategy,
        start_time=start_time,
        end_time=end_time,
        time_var=variables["Epoch"],
    )

    output_path = (
        tmp_path
        / MISSION
        / SATELLITE
        / f"{SATELLITE}_{INSTRUMENT.lower()}_20130101to20130131_{MAG_FIELD}.{output_format}"
    )
    assert output_path.exists()

    loader = {
        "nc": ep.utils.load_netcdf_data,
        "h5": ep.utils.load_h5_data,
        "cdf": ep.utils.load_cdf_data,
        "mat": ep.utils.load_mat_data,
    }[output_format]

    loaded_data = loader(output_path)
    metadata = loaded_data.get("metadata", {})
    for internal_name in strategy.output_files[0].names_to_save:
        var_key = data_standard().get_standard_name(internal_name)
        saved_variable = loaded_data[var_key]
        assert saved_variable.shape == variables[internal_name].get_data().shape
        var_attrs = metadata.get(var_key, {})
        assert meta_keys_check(var_attrs.keys())
        assert var_attrs.get("unit", "unknown") != "unknown"


def _mock_monthly_variables_for_append() -> dict[InternalName, ep.Variable]:
    """Create mocked monthly product variables without running processing code."""
    time_size = 121
    energy_size = 3
    alpha_size = 4

    start_time = datetime(2013, 1, 5, tzinfo=timezone.utc)
    datetimes = [start_time + i * np.timedelta64(3600, "s") for i in range(time_size)]
    epoch = np.array([python2matlab(i) for i in datetimes])

    variables: dict[InternalName, ep.Variable] = {
        "Epoch": ep.Variable(original_unit=ep.units.datenum, data=epoch),
        "FEDU": ep.Variable(
            original_unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
            data=np.arange(time_size * energy_size * alpha_size, dtype=float).reshape(
                time_size,
                energy_size,
                alpha_size,
            )
            * 2.0,  # different from the initial mock to ensure we can detect appending data correctly
        ),
        "Alpha_Eq": ep.Variable(original_unit=u.deg, data=np.full((time_size, alpha_size), 45.0)),
        "Energy_FEDU": ep.Variable(
            original_unit=u.MeV,
            data=np.tile(np.asarray([0.5, 1.0, 2.0]), (time_size, 1)),
        ),
        "Alpha": ep.Variable(
            original_unit=u.deg,
            data=np.tile(np.asarray([10.0, 30.0, 60.0, 90.0]), (time_size, 1)),
        ),
        "B_Calc": ep.Variable(original_unit=u.nT, data=np.full(time_size, 75.0)),
        "B_Eq": ep.Variable(original_unit=u.nT, data=np.full(time_size, 50.0)),
        "InvK": ep.Variable(
            original_unit=ep.units.RE * u.G**0.5,
            data=np.full((time_size, alpha_size), 1.5),
        ),
        "InvMu": ep.Variable(
            original_unit=u.MeV / u.G,
            data=np.full((time_size, energy_size, alpha_size), 2.5),
        ),
        "Position": ep.Variable(
            original_unit=ep.units.RE,
            data=np.arange(time_size * 3, dtype=float).reshape(time_size, 3),
        ),
        "PSD": ep.Variable(
            original_unit=(u.m * u.kg * u.m / u.s) ** (-3),
            data=np.full((time_size, energy_size, alpha_size), 8.5),
        ),
        "R_Eq": ep.Variable(original_unit=ep.units.RE, data=np.full(time_size, 5.0)),
        "MLT": ep.Variable(original_unit=u.hour, data=np.full(time_size, 5.0)),
        "L_m": ep.Variable(
            original_unit=u.dimensionless_unscaled,
            data=np.full((time_size, alpha_size), 4.5),
        ),
        "L_star": ep.Variable(
            original_unit=u.dimensionless_unscaled,
            data=np.full((time_size, alpha_size), 8.5),
        ),
    }
    return variables


@pytest.mark.basic
@pytest.mark.parametrize("data_standard", [ep.data_standards.GFZStandard, ep.data_standards.PRBEMStandard])
@pytest.mark.parametrize(("output_format", "meta_keys_check"), _FORMAT_PARAMS)
def test_append_data(
    tmp_path: Path,
    data_standard: type[DataStandard],
    output_format: Literal["nc", "h5", "cdf", "mat"],
    meta_keys_check: Callable[[set[str]], bool],
) -> None:

    ic(data_standard().get_standard_name("Epoch"))
    variables = _mock_monthly_variables()
    start_time = datetime(2013, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2013, 1, 7, tzinfo=timezone.utc)
    MISSION = "GOES"
    SATELLITE = "primary"
    INSTRUMENT = "MAGED"
    MAG_FIELD = "T89"
    strategy = ep.saving_strategies.MonthlyRBStrategy(
        base_data_path=tmp_path,
        mission=MISSION,
        satellite=SATELLITE,
        instrument=INSTRUMENT,
        mag_field=MAG_FIELD,
        file_format=output_format,
        data_standard=data_standard(),
    )

    ep.save(
        variables,
        strategy,
        start_time=start_time,
        end_time=end_time,
        time_var=variables["Epoch"],
    )

    variables = _mock_monthly_variables_for_append()
    start_time = datetime(2013, 1, 5, tzinfo=timezone.utc)
    end_time = datetime(2013, 1, 10, tzinfo=timezone.utc)
    MISSION = "GOES"
    SATELLITE = "primary"
    INSTRUMENT = "MAGED"
    MAG_FIELD = "T89"
    strategy = ep.saving_strategies.MonthlyRBStrategy(
        base_data_path=tmp_path,
        mission=MISSION,
        satellite=SATELLITE,
        instrument=INSTRUMENT,
        mag_field=MAG_FIELD,
        file_format=output_format,
        data_standard=data_standard(),
    )

    ep.save(
        variables,
        strategy,
        start_time=start_time,
        end_time=end_time,
        time_var=variables["Epoch"],
        append=True,
    )

    output_path = (
        tmp_path
        / MISSION
        / SATELLITE
        / f"{SATELLITE}_{INSTRUMENT.lower()}_20130101to20130131_{MAG_FIELD}.{output_format}"
    )
    assert output_path.exists()

    loader = {
        "nc": ep.utils.load_netcdf_data,
        "h5": ep.utils.load_h5_data,
        "cdf": ep.utils.load_cdf_data,
        "mat": ep.utils.load_mat_data,
    }[output_format]

    loaded_data = loader(output_path)
    metadata = loaded_data.get("metadata", {})
    for internal_name in strategy.output_files[0].names_to_save:
        var_key = data_standard().get_standard_name(internal_name)
        saved_variable = loaded_data[var_key]
        assert saved_variable.shape[1:] == variables[internal_name].get_data().shape[1:]
        assert saved_variable.shape[0] == 217
        var_attrs = metadata.get(var_key, {})
        assert meta_keys_check(var_attrs.keys())
        assert var_attrs.get("unit", "unknown") != "unknown"
