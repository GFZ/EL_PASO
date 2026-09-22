# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u

import el_paso as ep

if TYPE_CHECKING:
    from el_paso import Variable

logger = logging.getLogger(__name__)


def map_to_dipole_equator(variable: Variable, mlat_var: Variable) -> Variable:
    """Maps a quantity proportional to the field strength down to the magnetic equator.

    Along a dipole field line the field strength varies as
    ``B(lambda) = B_eq * sqrt(1 + 3 sin^2(lambda)) / cos^6(lambda)``, so a local value is
    brought to the equator by the inverse factor. Any quantity proportional to ``|B|`` scales
    the same way, so this works on the field strength itself and on the electron
    gyrofrequency alike.

    This uses dipole geometry only; it needs no magnetic field model. Where a traced model
    field is available, taking the ratio ``B_Eq/B_Calc`` from
    `compute_magnetic_field_variables` is the more accurate mapping, at the cost of depending
    on that model.

    Args:
        variable (Variable): The locally measured quantity, proportional to the field strength.
        mlat_var (Variable): Magnetic latitude at the same time steps.

    Returns:
        Variable: The quantity mapped to the magnetic equator, in the input's unit.
    """
    mlat_rad = np.radians(np.asarray(mlat_var.get_data(u.deg)).astype(np.float64))

    scale_factor = np.cos(mlat_rad) ** 6 / np.sqrt(1 + 3 * np.sin(mlat_rad) ** 2)

    local_unit = variable.metadata.unit
    mapped = np.asarray(variable.get_data(local_unit)).astype(np.float64) * scale_factor

    mapped_var = ep.Variable(data=mapped, original_unit=local_unit)
    mapped_var.metadata.description = variable.metadata.description
    mapped_var.metadata.add_processing_note(
        "Mapped to the magnetic equator assuming dipole geometry, by the factor cos^6(MLat) / sqrt(1 + 3 sin^2(MLat))."
    )

    return mapped_var
