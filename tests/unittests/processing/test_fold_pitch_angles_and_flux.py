# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

import el_paso as ep
import numpy as np
import pytest
from astropy import units as u


def _make_vars(pitch_angles: np.ndarray) -> tuple[ep.Variable, ep.Variable]:
    n_time, n_pa = pitch_angles.shape
    flux = np.ones((n_time, 2, n_pa))
    return ep.Variable(original_unit=u.dimensionless_unscaled, data=flux), ep.Variable(
        original_unit=u.deg, data=pitch_angles
    )


@pytest.mark.basic
def test_fold_constant_pitch_angles() -> None:
    pa = np.tile(np.array([[30.0, 60.0, 120.0, 150.0]]), (3, 1))
    flux_var, pa_var = _make_vars(pa)

    ep.processing.fold_pitch_angles_and_flux(flux_var, pa_var)

    assert np.allclose(pa_var.get_data(u.deg), np.tile([30.0, 60.0], (3, 1)))
    assert flux_var.get_data().shape == (3, 2, 2)


@pytest.mark.basic
def test_fold_constant_pitch_angles_with_nan() -> None:
    pa = np.tile(np.array([[30.0, np.nan, 150.0]]), (3, 1))
    flux_var, pa_var = _make_vars(pa)

    ep.processing.fold_pitch_angles_and_flux(flux_var, pa_var)

    assert flux_var.get_data().shape == (3, 2, 1)


@pytest.mark.basic
def test_fold_raises_for_time_varying_pitch_angles() -> None:
    pa = np.array([[30.0, 60.0, 120.0], [31.0, 60.0, 120.0], [30.0, 60.0, 120.0]])
    flux_var, pa_var = _make_vars(pa)

    with pytest.raises(ValueError, match="do not change in time"):
        ep.processing.fold_pitch_angles_and_flux(flux_var, pa_var)
