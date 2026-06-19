# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

import el_paso as ep
from el_paso.processing.magnetic_field_utils import IrbemOptions, MagneticField
from el_paso.processing.magnetic_field_utils.dipole_field import (
    B0_DEFAULT,
    _dipole_B_at_latitude,
    _footpoint_latitude,
    _geo_to_dipole_params,
    dipole_get_footpoint_atmosphere,
    dipole_get_local_B_field,
    dipole_get_Lstar,
    dipole_get_magequator,
    dipole_get_mirror_point,
    dipole_get_MLT,
)
from el_paso.processing.magnetic_field_utils.irbem import InternalFieldModel

MAG = MagneticField.DIPOLE


def _make_time_var(n: int = 1) -> ep.Variable:
    return ep.Variable(original_unit=ep.units.posixtime, data=np.arange(n, dtype=np.float64))


def _make_equatorial_xgeo(L: float, n: int = 1) -> ep.Variable:
    data = np.tile([L, 0.0, 0.0], (n, 1))
    return ep.Variable(original_unit=ep.units.RE, data=data)


@pytest.mark.basic
def test_dipole_B_at_equator() -> None:
    """At the equator (λ=0), B = B0_DEFAULT / L^3."""
    L = np.array([6.0])
    lam = np.array([0.0])
    result = _dipole_B_at_latitude(L, lam)
    expected = B0_DEFAULT / 6.0**3
    np.testing.assert_allclose(result, expected, rtol=1e-10)


@pytest.mark.basic
def test_dipole_B_increases_with_latitude() -> None:
    """B must increase with magnetic latitude along the same L-shell."""
    L = np.array([6.0, 6.0, 6.0])
    lam = np.radians([0.0, 30.0, 60.0])
    result = _dipole_B_at_latitude(L, lam)
    assert np.all(np.diff(result) > 0)


@pytest.mark.basic
def test_geo_to_dipole_params_equator() -> None:
    """A point at (6, 0, 0) RE is at L=6, λ=0."""
    xgeo = np.array([[6.0, 0.0, 0.0]])
    lam, L, r = _geo_to_dipole_params(xgeo)
    np.testing.assert_allclose(r, [6.0])
    np.testing.assert_allclose(lam, [0.0], atol=1e-15)
    np.testing.assert_allclose(L, [6.0])


@pytest.mark.basic
def test_footpoint_latitude_reasonable() -> None:
    """Footpoint latitude at L=6 should be close to ~64 degrees."""
    L = np.array([6.0])
    lam_foot = _footpoint_latitude(L)
    deg = np.degrees(lam_foot)
    assert 60 < deg[0] < 70


@pytest.mark.basic
def test_dipole_get_local_B_field_at_equator() -> None:
    """B_Calc at equator matches analytic B0_DEFAULT/L^3."""
    xgeo = _make_equatorial_xgeo(6.0)
    result = dipole_get_local_B_field(xgeo, _make_time_var(), MAG)
    b = result["B_Calc_dipole"].get_data(u.nT)
    np.testing.assert_allclose(b, B0_DEFAULT / 6.0**3, rtol=1e-10)


@pytest.mark.basic
def test_footpoint_B_greater_than_equatorial_B() -> None:
    """B_fofl must be much larger than B at the equator."""
    xgeo = _make_equatorial_xgeo(6.0)
    t = _make_time_var()
    b_local = dipole_get_local_B_field(xgeo, t, MAG)["B_Calc_dipole"].get_data(u.nT)
    b_fofl = dipole_get_footpoint_atmosphere(xgeo, t, MAG)["B_fofl_dipole"].get_data(u.nT)
    assert b_fofl[0] > 10 * b_local[0]


@pytest.mark.basic
def test_Lstar_equals_L_in_dipole() -> None:
    """L_star and L_m must both equal L in a pure dipole."""
    xgeo = _make_equatorial_xgeo(5.0, n=3)
    pa = ep.Variable(original_unit=u.deg, data=np.full((3, 2), 45.0))
    result = dipole_get_Lstar(xgeo, _make_time_var(3), pa, MAG)
    np.testing.assert_allclose(result["L_m_dipole"].get_data(), 5.0)
    np.testing.assert_allclose(result["L_star_dipole"].get_data(), 5.0)


@pytest.mark.basic
def test_mirror_B_increases_with_smaller_pitch_angle() -> None:
    """Smaller pitch angles mirror at higher B (closer to Earth)."""
    xgeo = _make_equatorial_xgeo(6.0, n=1)
    pa = ep.Variable(original_unit=u.deg, data=np.array([[80.0, 45.0, 10.0]]))
    result = dipole_get_mirror_point(xgeo, _make_time_var(), pa, MAG)
    b_mirr = result["B_mirr_dipole"].get_data(u.nT)
    # ignore NaNs (particles in loss cone)
    finite = np.isfinite(b_mirr[0])
    finite_vals = b_mirr[0, finite]
    if len(finite_vals) > 1:
        assert np.all(np.diff(finite_vals) > 0)


@pytest.mark.basic
def test_mirror_B_nan_for_loss_cone() -> None:
    """A very small pitch angle at high L should be in the loss cone → NaN."""
    xgeo = _make_equatorial_xgeo(6.0, n=1)
    pa = ep.Variable(original_unit=u.deg, data=np.array([[1.0]]))
    result = dipole_get_mirror_point(xgeo, _make_time_var(), pa, MAG)
    b_mirr = result["B_mirr_dipole"].get_data(u.nT)
    assert np.isnan(b_mirr[0, 0])


# ── Comparison with _calculate_orbital_vars from RBSP recipe ──────────────────


@pytest.mark.basic
def test_dipole_matches_recipe_orbital_vars() -> None:
    """Dipole functions must produce the same L, MLT as the RBSP recipe's inline dipole code."""
    coords_km = np.array([[20000.0, 10000.0, 5000.0], [15000.0, -8000.0, 3000.0]])
    coords_re = coords_km / 6371.0

    # --- recipe logic (from _calculate_orbital_vars) ---
    x, y, z = coords_km[:, 0], coords_km[:, 1], coords_km[:, 2]
    r_xy = np.hypot(x, y)
    r = np.sqrt(x**2 + y**2 + z**2)
    mlat_rad = np.arctan2(z, r_xy)
    l_recipe = r / np.cos(mlat_rad) ** 2
    mlt_recipe = np.mod(np.degrees(np.arctan2(y, x)) / 15.0 + 12.0, 24.0)

    # --- our dipole functions ---
    xgeo_var = ep.Variable(original_unit=ep.units.RE, data=coords_re)
    time_var = _make_time_var(n=2)

    b_local = dipole_get_local_B_field(xgeo_var, time_var, MAG)
    mlt_result = dipole_get_MLT(xgeo_var, time_var, MAG)

    # L from our params
    lam, L_ours, _ = _geo_to_dipole_params(coords_re)

    # L values must match (recipe uses km, ours uses RE — same formula)
    l_recipe_re = l_recipe / 6371.0
    np.testing.assert_allclose(L_ours, l_recipe_re, rtol=1e-10)

    # MLT must match
    np.testing.assert_allclose(
        mlt_result["MLT_dipole"].get_data(u.hour), mlt_recipe, rtol=1e-10,
    )


# ── Comparison with IRBEM centered dipole ────────────────────────────────────


def _irbem_dipole_input(n: int) -> tuple:
    """Create IRBEM IrbemInput configured for centered dipole + no external field."""
    from el_paso.processing.magnetic_field_utils.magnetic_field_functions import IrbemInput

    irbem_lib = Path(ep.__file__).parent / "libirbem.so"
    if not irbem_lib.exists():
        pytest.skip("IRBEM library not available")

    irbem_options = IrbemOptions(internal_field_model=InternalFieldModel.CENTERED_DIPOLE)
    maginput: dict[str, np.ndarray] = {
        k: np.zeros(n) for k in [
            "Kp", "Dst", "dens", "velo", "Pdyn", "ByIMF", "BzIMF",
            "G1", "G2", "G3", "W1", "W2", "W3", "W4", "W5", "W6", "AL",
        ]
    }
    irbem_input = IrbemInput(
        magnetic_field=MagneticField.DIPOLE,
        maginput=maginput,
        irbem_options=irbem_options,
        num_cores=1,
        irbem_lib_path=irbem_lib,
    )
    return irbem_input, irbem_lib


def _calibrate_b0_from_irbem(irbem_input, irbem_lib: Path) -> float:  # noqa: ANN001
    """Derive IRBEM's effective B0 by measuring B_Eq at L=1 (equatorial surface)."""
    from el_paso.processing.magnetic_field_utils.magnetic_field_functions import get_magequator

    xgeo = ep.Variable(original_unit=ep.units.RE, data=np.array([[1.0, 0.0, 0.0]]))
    start = datetime(2017, 9, 8, tzinfo=timezone.utc)
    t = ep.Variable(original_unit=ep.units.posixtime, data=np.array([start.timestamp()]))

    from el_paso.processing.magnetic_field_utils.magnetic_field_functions import IrbemInput
    inp1 = IrbemInput(
        magnetic_field=MagneticField.DIPOLE,
        maginput={k: np.zeros(1) for k in irbem_input.maginput},
        irbem_options=irbem_input.irbem_options,
        num_cores=1,
        irbem_lib_path=irbem_lib,
    )
    result = get_magequator(xgeo, t, inp1)
    return float(result["B_Eq_dipole"].get_data(u.nT)[0])


@pytest.mark.basic
def test_dipole_all_variables_agree_with_irbem() -> None:
    """All dipole variables should closely match IRBEM (CENTERED_DIPOLE, kext=0)
    when using the same B0 value."""
    n = 2
    irbem_input, irbem_lib = _irbem_dipole_input(n)

    # Calibrate: extract IRBEM's effective B0
    irbem_b0 = _calibrate_b0_from_irbem(irbem_input, irbem_lib)

    xgeo_data = np.array([[5.0, 1.0, 0.5], [4.0, -0.5, 0.3]])
    xgeo_var = ep.Variable(original_unit=ep.units.RE, data=xgeo_data)

    start = datetime(2017, 9, 8, tzinfo=timezone.utc)
    timestamps = np.array([start.timestamp(), start.timestamp() + 60])
    time_var = ep.Variable(original_unit=ep.units.posixtime, data=timestamps)

    pa_data = np.array([[60.0, 30.0], [45.0, 20.0]])
    pa_var = ep.Variable(original_unit=u.deg, data=pa_data)

    from el_paso.processing.magnetic_field_utils.magnetic_field_functions import (
        get_footpoint_atmosphere,
        get_local_B_field,
        get_magequator,
        get_mirror_point,
    )

    # --- B_Calc ---
    our_b_calc = dipole_get_local_B_field(xgeo_var, time_var, MAG, b0=irbem_b0)
    irbem_b_calc = get_local_B_field(xgeo_var, time_var, irbem_input)
    np.testing.assert_allclose(
        our_b_calc["B_Calc_dipole"].get_data(u.nT),
        irbem_b_calc["B_Calc_dipole"].get_data(u.nT),
        rtol=0.02, err_msg="B_Calc mismatch",
    )

    # --- B_Eq, R_Eq, MLT_Eq ---
    our_eq = dipole_get_magequator(xgeo_var, time_var, MAG, irbem_lib, b0=irbem_b0)
    irbem_eq = get_magequator(xgeo_var, time_var, irbem_input)
    np.testing.assert_allclose(
        our_eq["B_Eq_dipole"].get_data(u.nT),
        irbem_eq["B_Eq_dipole"].get_data(u.nT),
        rtol=0.02, err_msg="B_Eq mismatch",
    )
    np.testing.assert_allclose(
        our_eq["R_Eq_dipole"].get_data(ep.units.RE),
        irbem_eq["R_Eq_dipole"].get_data(ep.units.RE),
        rtol=0.02, err_msg="R_Eq mismatch",
    )
    np.testing.assert_allclose(
        our_eq["MLT_Eq_dipole"].get_data(u.hour),
        irbem_eq["MLT_Eq_dipole"].get_data(u.hour),
        atol=0.5, err_msg="MLT_Eq mismatch",
    )

    # --- B_fofl ---
    our_fofl = dipole_get_footpoint_atmosphere(xgeo_var, time_var, MAG, b0=irbem_b0)
    irbem_fofl = get_footpoint_atmosphere(xgeo_var, time_var, irbem_input)
    np.testing.assert_allclose(
        our_fofl["B_fofl_dipole"].get_data(u.nT),
        irbem_fofl["B_fofl_dipole"].get_data(u.nT),
        rtol=0.05, err_msg="B_fofl mismatch",
    )

    # --- B_mirr ---
    our_mirr = dipole_get_mirror_point(xgeo_var, time_var, pa_var, MAG, b0=irbem_b0)
    irbem_mirr = get_mirror_point(xgeo_var, time_var, pa_var, irbem_input)
    our_bm = our_mirr["B_mirr_dipole"].get_data(u.nT)
    irbem_bm = irbem_mirr["B_mirr_dipole"].get_data(u.nT)
    both_finite = np.isfinite(our_bm) & np.isfinite(irbem_bm)
    if np.any(both_finite):
        np.testing.assert_allclose(
            our_bm[both_finite], irbem_bm[both_finite],
            rtol=0.05, err_msg="B_mirr mismatch",
        )
