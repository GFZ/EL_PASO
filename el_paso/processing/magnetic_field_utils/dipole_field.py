# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u

import el_paso as ep
from el_paso.processing.magnetic_field_utils.irbem import Coords
from el_paso.processing.magnetic_field_utils.mag_field_enum import MagneticField
from el_paso.processing.magnetic_field_utils.magnetic_field_functions import create_var_name
from el_paso.utils import timed_function

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

B0 = 31200.0  # nT — Earth's dipole moment at surface equator
R_EARTH_KM = 6371.0
FOOTPOINT_ALT_KM = 100.0


def _geo_to_dipole_params(
    xgeo: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Extract r (RE), magnetic latitude λ (rad), and L from GEO coordinates."""
    x, y, z = xgeo[:, 0], xgeo[:, 1], xgeo[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    r_xy = np.sqrt(x**2 + y**2)
    lam = np.arctan2(z, r_xy)
    cos_lam = np.cos(lam)
    L = r / np.where(cos_lam == 0, np.nan, cos_lam**2)
    return lam, L, r


def _dipole_B_at_latitude(L: NDArray[np.float64], lam: NDArray[np.float64]) -> NDArray[np.float64]:
    """Dipole field magnitude at (L, λ)."""
    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)
    return B0 / L**3 * np.sqrt(1.0 + 3.0 * sin_lam**2) / cos_lam**6


def _footpoint_latitude(L: NDArray[np.float64]) -> NDArray[np.float64]:
    """Magnetic latitude where the field line at L intersects r_foot = 1 + 100km/R_E."""
    r_foot = 1.0 + FOOTPOINT_ALT_KM / R_EARTH_KM
    cos2_lam = np.clip(r_foot / L, 0.0, 1.0)
    return np.arccos(np.sqrt(cos2_lam))


@timed_function()
def dipole_get_local_B_field(
    xgeo_var: ep.Variable, time_var: ep.Variable, mag_field: MagneticField
) -> dict[str, ep.Variable]:
    """Local magnetic field magnitude from dipole formula."""
    logger.info("\tCalculating local magnetic field (dipole) ...")
    xgeo = xgeo_var.get_data(ep.units.RE).astype(np.float64)
    lam, L, _ = _geo_to_dipole_params(xgeo)
    b_local = _dipole_B_at_latitude(L, lam)

    var = ep.Variable(data=b_local.astype(np.float64), original_unit=u.nT)
    var.metadata.add_processing_note("Calculated local B field using dipole model.")
    return {create_var_name("B_Calc", mag_field): var}


@timed_function()
def dipole_get_magequator(
    xgeo_var: ep.Variable,
    time_var: ep.Variable,
    mag_field: MagneticField,
    irbem_lib_path: str | Path,
) -> dict[str, ep.Variable]:
    """Equatorial B, R_Eq, MLT_Eq, xGEO_Eq from dipole formula."""
    logger.info("\tCalculating equatorial quantities (dipole) ...")
    xgeo = xgeo_var.get_data(ep.units.RE).astype(np.float64)
    timestamps = time_var.get_data(ep.units.posixtime)
    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]

    lam, L, _ = _geo_to_dipole_params(xgeo)

    b_eq = B0 / L**3
    b_eq_var = ep.Variable(data=b_eq.astype(np.float64), original_unit=u.nT)
    b_eq_var.metadata.add_processing_note("Equatorial B field from dipole model: B0/L^3.")

    # xGEO_Eq: project to equatorial plane along the field line
    x, y = xgeo[:, 0], xgeo[:, 1]
    r_xy = np.sqrt(x**2 + y**2)
    r_xy_safe = np.where(r_xy == 0, 1.0, r_xy)
    xgeo_eq = np.column_stack([L * x / r_xy_safe, L * y / r_xy_safe, np.zeros_like(L)])
    xgeo_eq_var = ep.Variable(data=xgeo_eq.astype(np.float64), original_unit=ep.units.RE)
    xgeo_eq_var.metadata.add_processing_note("Equatorial position projected along dipole field line.")

    # R_Eq and MLT_Eq from GSM coordinates
    x_gsm = Coords(lib_path=irbem_lib_path).transform(
        datetimes, xgeo_eq, ep.IRBEM_SYSAXIS_GEO, ep.IRBEM_SYSAXIS_GSM,
    )

    r_eq = np.linalg.norm(x_gsm, ord=2, axis=1).astype(np.float64)
    r_eq_var = ep.Variable(data=r_eq, original_unit=ep.units.RE)
    r_eq_var.metadata.add_processing_note("Equatorial radial distance in GSM from dipole model.")

    p_gsm = np.arctan2(x_gsm[:, 1], x_gsm[:, 0])
    mlt_gsm = ((p_gsm * 12 / np.pi) + 12) % 24
    mlt_eq_var = ep.Variable(data=mlt_gsm.astype(np.float64), original_unit=u.hour)
    mlt_eq_var.metadata.add_processing_note("Equatorial MLT in GSM from dipole model.")

    return {
        create_var_name("B_Eq", mag_field): b_eq_var,
        create_var_name("R_Eq", mag_field): r_eq_var,
        create_var_name("MLT_Eq", mag_field): mlt_eq_var,
        create_var_name("xGEO_Eq", mag_field): xgeo_eq_var,
    }


@timed_function()
def dipole_get_MLT(
    xgeo_var: ep.Variable, time_var: ep.Variable, mag_field: MagneticField
) -> dict[str, ep.Variable]:
    """Magnetic local time from GEO coordinates."""
    logger.info("\tCalculating MLT (dipole) ...")
    xgeo = xgeo_var.get_data(ep.units.RE).astype(np.float64)
    mlt = (np.arctan2(xgeo[:, 1], xgeo[:, 0]) * 12 / np.pi + 12) % 24

    var = ep.Variable(data=mlt.astype(np.float64), original_unit=u.hour)
    var.metadata.add_processing_note("Calculated MLT from GEO coordinates using dipole model.")
    return {create_var_name("MLT", mag_field): var}


@timed_function()
def dipole_get_Lstar(
    xgeo_var: ep.Variable,
    time_var: ep.Variable,
    pa_local_var: ep.Variable,
    mag_field: MagneticField,
) -> dict[str, ep.Variable]:
    """L_m, L_star (both = L in dipole), and I (third adiabatic invariant)."""
    logger.info("\tCalculating L-shell (dipole) ...")
    xgeo = xgeo_var.get_data(ep.units.RE).astype(np.float64)
    pa_local = pa_local_var.get_data(u.deg).astype(np.float64)

    _, L_scalar, _ = _geo_to_dipole_params(xgeo)

    # Broadcast L to match pitch angle dimensions: (n_time,) → (n_time, n_pa)
    L = np.broadcast_to(L_scalar[:, np.newaxis], pa_local.shape).copy()

    lm_var = ep.Variable(data=L.astype(np.float64), original_unit=u.dimensionless_unscaled)
    lm_var.metadata.add_processing_note("Dipole L-shell: r/cos^2(lambda).")

    lstar_var = ep.Variable(data=L.astype(np.float64), original_unit=u.dimensionless_unscaled)
    lstar_var.metadata.add_processing_note("L* = L in a pure dipole (no drift-shell splitting).")

    # I (related to second invariant) — set to zero placeholder in dipole
    xj_var = ep.Variable(data=np.zeros_like(L, dtype=np.float64), original_unit=ep.units.RE)
    xj_var.metadata.add_processing_note("I set to zero in dipole approximation.")

    return {
        create_var_name("L_m", mag_field): lm_var,
        create_var_name("L_star", mag_field): lstar_var,
        create_var_name("I", mag_field): xj_var,
    }


@timed_function()
def dipole_get_footpoint_atmosphere(
    xgeo_var: ep.Variable, time_var: ep.Variable, mag_field: MagneticField
) -> dict[str, ep.Variable]:
    """B at the atmospheric footpoint (100 km altitude) from dipole formula."""
    logger.info("\tCalculating footpoint B field (dipole) ...")
    xgeo = xgeo_var.get_data(ep.units.RE).astype(np.float64)
    _, L, _ = _geo_to_dipole_params(xgeo)

    lam_foot = _footpoint_latitude(L)
    b_foot = _dipole_B_at_latitude(L, lam_foot)

    var = ep.Variable(data=b_foot.astype(np.float64), original_unit=u.nT)
    var.metadata.add_processing_note(
        f"Footpoint B at {FOOTPOINT_ALT_KM:.0f} km altitude from dipole model."
    )
    return {create_var_name("B_fofl", mag_field): var}


@timed_function()
def dipole_get_mirror_point(
    xgeo_var: ep.Variable,
    time_var: ep.Variable,
    pa_local_var: ep.Variable,
    mag_field: MagneticField,
) -> dict[str, ep.Variable]:
    """B at mirror point for each pitch angle using dipole field line."""
    logger.info("\tCalculating mirror point B (dipole) ...")
    xgeo = xgeo_var.get_data(ep.units.RE).astype(np.float64)
    pa_local = pa_local_var.get_data(u.deg).astype(np.float64)

    lam, L_scalar, _ = _geo_to_dipole_params(xgeo)
    b_local = _dipole_B_at_latitude(L_scalar, lam)

    # B_mirr = B_local / sin²(alpha)
    sin2_alpha = np.sin(np.radians(pa_local)) ** 2
    sin2_alpha = np.where(sin2_alpha == 0, np.nan, sin2_alpha)

    # Broadcast b_local to match pa dimensions
    if pa_local.ndim == 2:
        b_local_broadcast = b_local[:, np.newaxis]
    else:
        b_local_broadcast = b_local

    b_mirr = b_local_broadcast / sin2_alpha

    # Particles outside loss cone: B_mirr > B_fofl means not trapped → NaN
    lam_foot = _footpoint_latitude(L_scalar)
    b_fofl = _dipole_B_at_latitude(L_scalar, lam_foot)
    if pa_local.ndim == 2:
        b_fofl_broadcast = b_fofl[:, np.newaxis]
    else:
        b_fofl_broadcast = b_fofl
    b_mirr = np.where(b_mirr > b_fofl_broadcast, np.nan, b_mirr)

    var = ep.Variable(data=b_mirr.astype(np.float64), original_unit=u.nT)
    var.metadata.add_processing_note("Mirror point B from dipole: B_local/sin^2(alpha).")
    return {create_var_name("B_mirr", mag_field): var}
