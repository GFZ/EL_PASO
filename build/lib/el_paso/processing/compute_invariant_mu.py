# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from astropy import units as u  # type: ignore[reportMissingTypeStubs]

import el_paso as ep
from el_paso.physics import ParticleLiteral, en2pc, rest_energy


def compute_invariant_mu(
    energy_var: ep.Variable,
    alpha_local_var: ep.Variable,
    B_local_var: ep.Variable,  # noqa: N803
    particle_species: ParticleLiteral,
) -> ep.Variable:
    r"""Computes the first adiabatic invariant (mu) for given particle species.

    The first adiabatic invariant ($\mu$) is calculated using the formula:
    $\mu = \frac{p_{\perp}^2}{2mB}$, where $p_{\perp}$ is the perpendicular
    momentum, $m$ is the particle's rest mass, and $B$ is the local magnetic
    field strength.

    The momentum is derived from the total energy and local pitch angle.
    The result is in units of $MeV/G$.

    Args:
        energy_var (ep.Variable): A Variable object containing the total energy
            of the particles in MeV. Expected to be a 2D array (time, energy_bins).
        alpha_local_var (ep.Variable): A Variable object containing the local
            pitch angles in radians. Expected to be a 2D array (time, angle_bins).
        B_local_var (ep.Variable): A Variable object containing the local magnetic
            field strength in nT. Expected to be a 1D array (time).
        particle_species (ParticleLiteral): The species of the particles
            (e.g., "electron", "proton").

    Returns:
        ep.Variable: A new Variable object containing the computed invariant mu
            data, with dimensions (time, energy_bins, angle_bins) and unit $MeV/G$.

    Raises:
        ValueError: If input variables do not have the correct dimensions
            (energy, alpha_local must be 2D; B_local must be 1D)
            or if their time dimensions do not match.

    Notes:
        Values of invariant mu that are less than or equal to zero are replaced with `NaN`.
    """
    energy = energy_var.get_data(u.MeV)
    alpha_local = alpha_local_var.get_data(u.radian)
    magnetic_field = B_local_var.get_data(u.G)

    if energy.ndim != 2 or alpha_local.ndim != 2 or magnetic_field.ndim != 1:  # noqa: PLR2004
        msg = (
            "Input variables must have the correct dimensions: energy (2D), alpha_local (2D), and magnetic_field (1D)."
        )
        raise ValueError(msg)

    if energy.shape[0] != alpha_local.shape[0] or energy.shape[0] != magnetic_field.shape[0]:
        msg = (
            "Input variables must have matching time dimensions: "
            f"energy ({energy.shape[0]}), alpha_local ({alpha_local.shape[0]}), "
            f"and magnetic_field ({magnetic_field.shape[0]})."
        )
        raise ValueError(msg)

    mc2 = rest_energy(particle_species)
    pct = en2pc(energy, particle_species)  # Relativistic energy for the particles

    # Calculate InvMu using broadcasting
    sin_alpha_eq = np.sin(alpha_local)

    inv_mu = (pct[:, :, np.newaxis] * sin_alpha_eq[:, np.newaxis, :]) ** 2 / (
        magnetic_field[:, np.newaxis, np.newaxis] * 2 * mc2
    )  # MeV/G

    inv_mu[inv_mu <= 0] = np.nan

    inv_mu_var = ep.Variable(data=inv_mu, original_unit=u.MeV / u.G)  # type: ignore[reportUnknownArgumentType]

    inv_mu_var.metadata.add_processing_note(f"Created with compute_invariant_mu for {particle_species} particles")

    return inv_mu_var
