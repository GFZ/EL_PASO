# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
from astropy import units as u  # type: ignore[reportMissingTypeStubs]

import el_paso as ep


def construct_pitch_angle_distribution(
    omni_flux_var: ep.Variable,
    pa_local_var: ep.Variable,
    pa_eq_var: ep.Variable,
    method: str = "sin",
) -> ep.Variable:
    """Construct a pitch angle distribution from omni-directional flux.

    This function calculates a pitch angle distribution based on an omni-directional flux
    variable and local pitch angles. Currently, it supports a 'sin' distribution method.

    Args:
        omni_flux_var (ep.Variable): Omni-directional flux as an `el_paso` variable.
                                     The data is expected to be an array with dimensions
                                     (time, energy_channels).
        pa_local_var (ep.Variable): Local pitch angles as an `el_paso` variable.
                                    The data is expected to be an array with dimensions
                                    (time, pitch_angles).
        pa_eq_var (ep.Variable): Equatorial pitch angles as an `el_paso` variable.
                                 The data is expected to be an array with dimensions
                                 (time, pitch_angles).
        method (str, optional): The method to use for constructing the distribution.
                                Currently, only "sin" is supported. Defaults to "sin".

    Raises:
        ValueError: If an unsupported method is provided.

    Returns:
        ep.Variable: A new `el_paso` variable containing the differential flux. The returned data
                     is a 3D array with dimensions (time, energy_channels, pitch_angles),
                     and the units are updated accordingly (e.g., flux per steradian).
    """
    omni_flux = omni_flux_var.get_data()
    pa_local = pa_local_var.get_data()

    pa_eq_max = np.max(pa_eq_var.get_data(), axis=1)

    match method:
        case "sin":
            # Create a sin distribution
            # Calculate the factor outside the loops
            sin_pa = np.sin(np.deg2rad(pa_local))
            mean_sin = np.mean(np.sin(np.linspace(0, np.pi, 36)))

            # Calculate the factor matrix
            fact_matrix = sin_pa / (mean_sin * np.reciprocal(np.sin(np.deg2rad(pa_eq_max[:, np.newaxis]))))

            differential_flux = omni_flux[:, :, np.newaxis] * fact_matrix[:, np.newaxis, :]

        case _:
            msg = f"Encountered invalid method to constrouct pitch angle distribution: {method}!"
            raise ValueError(msg)

    return ep.Variable(data=differential_flux, original_unit=(omni_flux_var.metadata.unit / u.sr))
