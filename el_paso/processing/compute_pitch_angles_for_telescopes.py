# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
from astropy import units as u  # type: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray

import el_paso as ep


def _compute_pitch_angles_for_telescopes(
    b_tele_aligned: NDArray[np.floating],
    tele_alpha_angles: NDArray[np.floating],
    tele_beta_angles: NDArray[np.floating],
) -> NDArray[np.floating]:
    if b_tele_aligned.shape[1] != 3:  # noqa: PLR2004
        msg = "Magnetic field input must be a vector with 3 components!"
        raise ValueError(msg)

    btot = np.linalg.norm(b_tele_aligned, axis=1, ord=2, keepdims=True)

    b_unit_vectors = b_tele_aligned / btot

    # convert to standard shperical coordinates
    theta_zone = np.pi / 2 - tele_alpha_angles
    phi_zone = tele_beta_angles

    # velocity directions of particles: reverse telescope look direction
    theta_v = np.pi - theta_zone
    phi_v = np.pi + phi_zone

    dircosx_v = np.multiply(np.sin(theta_v), np.cos(phi_v))
    dircosy_v = np.multiply(np.sin(theta_v), np.sin(phi_v))
    dircosz_v = np.cos(theta_v)

    v_unit_vectors = np.stack([dircosx_v, dircosy_v, dircosz_v], axis=1)

    cos_pitch = np.dot(b_unit_vectors, v_unit_vectors.T)
    pitch_angles = np.arccos(np.clip(cos_pitch, -1, 1))

    return pitch_angles


def compute_pitch_angles_for_telescopes(
    b_tele_aligned: ep.Variable,
    tele_alpha_angles: ep.Variable,
    tele_beta_angles: ep.Variable,
) -> ep.Variable:
    """Calculates the particle pitch angles for specific telescope orientations.

    This function computes the angle between the local magnetic field vector and the
    velocity vectors of particles entering the telescopes.

    Args:
        b_tele_aligned (ep.Variable): The magnetic field vector already rotated
            into the sensor-aligned reference frame. Expected to be a
            time-series array of shape (n_times, 3).
        tele_alpha_angles (ep.Variable): The telescope alpha angles (elevation-like).
            These are converted to radians internally before processing.
        tele_beta_angles (ep.Variable): The telescope beta angles (azimuth-like).
            These are converted to radians internally before processing.

    Returns:
        ep.Variable: A variable containing the computed pitch angles in radians.
            The data shape will be (n_times, n_telescopes).

    Notes:
        We thank Juan V. Rodriguez (University of Colorado CIRES) for his help with the
        pitch-angle calculations.

    """
    pitch_angle_data = _compute_pitch_angles_for_telescopes(
        b_tele_aligned.get_data().astype(np.float32),
        tele_alpha_angles.get_data(u.rad).astype(np.float32),
        tele_beta_angles.get_data(u.rad).astype(np.float32),
    )

    return ep.Variable(data=pitch_angle_data, original_unit=u.rad)
