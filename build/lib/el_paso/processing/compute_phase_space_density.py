# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import logging
import typing
from typing import Literal

import numpy as np
from astropy import units as u  # type: ignore[reportMissingTypeStubs]

import el_paso as ep


def compute_phase_space_density(
    flux_var: ep.Variable, energy_var: ep.Variable, particle_species: Literal["electron", "proton"]
) -> ep.Variable:
    r"""Computes the Phase Space Density (PSD) from differential flux.

    This function calculates the Phase Space Density (PSD) for a given
    differential flux and energy spectrum, based on the particle species.
    The PSD is a fundamental quantity in space plasma physics.
    The units of the resulting PSD are typically $(m \cdot kg \cdot m/s)^{-3}$.

    The formula used is:
    $PSD = \frac{j}{p^2}$
    where $j$ is the differential flux, and $p$ is the relativistic momentum.

    Args:
        flux_var (ep.Variable): A Variable object containing the differential
            flux data. Expected units are inverse of (cm^2 s keV sr).
        energy_var (ep.Variable): A Variable object containing the energy
            spectrum data in MeV.
        particle_species (Literal["electron", "proton"]): The species of the
            particles (e.g., "electron", "proton") for which the PSD is computed.

    Returns:
        ep.Variable: A new Variable object containing the computed Phase Space
            Density (PSD) data, with unit $(m \cdot kg \cdot m/s)^{-3}$.

    Notes:
        - The constant `(1e3 / 2.997e10)` converts units appropriately (e.g., cm to m, keV to J, etc.)
          and accounts for the speed of light.
    """
    logger = logging.getLogger(__name__)
    logger.info("Computing PSD...")

    flux_data = flux_var.get_data(typing.cast("u.UnitBase", (u.cm**2 * u.s * u.keV * u.sr) ** (-1)))
    energies = energy_var.get_data(u.MeV)

    # Calculate pct for each energy value
    pct = ep.physics.en2pc(energies, particle_species)  # Relativistic energy for electrons
    psd_data = (1e3 / 2.997e10) * flux_data / (pct[:, :, np.newaxis] ** 2)

    var = ep.Variable(data=psd_data, original_unit=typing.cast("u.UnitBase", (u.m * u.kg * u.m / u.s) ** (-3)))

    var.metadata.add_processing_note("Created with compute_PSD")

    return var
