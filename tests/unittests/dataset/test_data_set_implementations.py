# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import shutil
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import pytest
from astropy import units as u  # type: ignore[reportMissingTypeStubs]

import el_paso as ep
from el_paso.dataset import DataOrgDataSet
from el_paso.dataset.utils import matlab2python, python2matlab

if TYPE_CHECKING:
    from pathlib import Path

    from el_paso.typing import InternalName, MFSFormats


def _mock_monthly_variables() -> dict[InternalName, ep.Variable]:
    """Create mocked monthly product variables without running processing code."""
    time_size = 144
    energy_size = 3
    alpha_size = 4

    start_time = datetime(2013, 1, 1, tzinfo=timezone.utc)
    datetimes = [start_time + i * np.timedelta64(6000, "s") for i in range(time_size)]
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


@pytest.mark.basic
@pytest.mark.parametrize("formats", ["nc", "h5", "cdf", "mat"])
def test_data_org_dataset_loads_saved_monthly_nc_and_rejects_invalid_variable(
    tmp_path: Path, formats: MFSFormats
) -> None:
    variables = _mock_monthly_variables()
    start_time = datetime(2013, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2013, 1, 2, tzinfo=timezone.utc)

    strategy = ep.saving_strategies.MonthlyFileStrategy(
        base_data_path=tmp_path,
        mission="GOES",
        satellite="primary",
        instrument="MAGED",
        mag_field="T89",
        file_format=formats,
        data_standard=ep.data_standards.DataOrgStandard(),
    )

    ep.save(
        variables,
        strategy,
        start_time=start_time,
        end_time=end_time,
        time_var=variables["Epoch"],
    )

    dataset = DataOrgDataSet(
        saving_strategy=strategy,
        start_time=start_time,
        end_time=end_time,
        preferred_extension=formats,
        verbose=False,
    )

    inv_K_repeated = np.repeat(dataset.InvK[:, np.newaxis, :], dataset.InvMu.shape[1], axis=1)
    expected_InvV = dataset.InvMu * (inv_K_repeated + 0.5) ** 2
    expected_P = ((dataset.MLT + 12) / 12 * np.pi) % (2 * np.pi)

    epoch = np.asarray(variables["Epoch"].get_data())
    time_mask = (epoch >= python2matlab(start_time)) & (epoch <= python2matlab(end_time))  # ty:ignore[unsupported-operator]

    expected_datetime = matlab2python(epoch[time_mask])
    expected_time = epoch[time_mask]
    expected_flux = np.asarray(variables["FEDU"].get_data())[time_mask, ...]
    expected_energy_channels = np.asarray(variables["Energy_FEDU"].get_data())[time_mask, ...]
    expected_alpha_local = np.deg2rad(np.asarray(variables["Alpha"].get_data())[time_mask, ...])
    expected_alpha_eq_model = np.deg2rad(np.asarray(variables["Alpha_Eq"].get_data())[time_mask, ...])
    expected_b_local = np.asarray(variables["B_Calc"].get_data())[time_mask, ...]
    expected_b_eq = np.asarray(variables["B_Eq"].get_data())[time_mask, ...]
    expected_invk = np.asarray(variables["InvK"].get_data())[time_mask, ...]
    expected_invmu = np.asarray(variables["InvMu"].get_data())[time_mask, ...]
    expected_xgeo = np.asarray(variables["Position"].get_data())[time_mask, ...]
    expected_psd = np.asarray(variables["PSD"].get_data())[time_mask, ...]
    expected_r0 = np.asarray(variables["R_Eq"].get_data())[time_mask, ...]
    expected_mlt = np.asarray(variables["MLT"].get_data())[time_mask, ...]
    expected_lm = np.asarray(variables["L_m"].get_data())[time_mask, ...]
    expected_lstar = np.asarray(variables["L_star"].get_data())[time_mask, ...]

    assert dataset.datetime == expected_datetime

    np.testing.assert_allclose(dataset.time, expected_time)
    np.testing.assert_equal(dataset.Flux, expected_flux)
    np.testing.assert_equal(dataset.get_var_by_internal_name("FEDU"), expected_flux)
    np.testing.assert_equal(dataset.energy_channels, expected_energy_channels)
    np.testing.assert_equal(dataset.alpha_local, expected_alpha_local)
    np.testing.assert_equal(dataset.alpha_eq_model, expected_alpha_eq_model)
    np.testing.assert_equal(dataset.B_sat, expected_b_local)
    np.testing.assert_equal(dataset.B_eq, expected_b_eq)
    np.testing.assert_equal(dataset.InvK, expected_invk)
    np.testing.assert_equal(dataset.InvMu, expected_invmu)
    np.testing.assert_equal(dataset.xGEO, expected_xgeo)
    np.testing.assert_equal(dataset.PSD, expected_psd)
    np.testing.assert_equal(dataset.R0, expected_r0)
    np.testing.assert_equal(dataset.MLT, expected_mlt)
    np.testing.assert_equal(dataset.Lm, expected_lm)
    np.testing.assert_equal(dataset.Lstar, expected_lstar)

    np.testing.assert_equal(dataset.InvV, expected_InvV)
    np.testing.assert_equal(dataset.P, expected_P)

    with pytest.raises(AttributeError, match="Maybe you meant "):
        dataset.lstar  # Levenstein variable check  # noqa: B018

    with pytest.raises(AttributeError, match="DataOrgDataSet object has no attribute somethingrandom"):
        dataset.somethingrandom  # this does not exist as a variable  # noqa: B018

    # shutil.rmtree(tmp_path)
