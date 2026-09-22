# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from astropy.constants import e as elementary_charge  # ty:ignore[unresolved-import]
from astropy.constants import m_e as electron_mass  # ty:ignore[unresolved-import]

import el_paso as ep

if TYPE_CHECKING:
    from el_paso import Variable

logger = logging.getLogger(__name__)


def compute_electron_gyrofrequency(b_var: Variable) -> Variable:
    """Computes the electron gyrofrequency from a magnetic field strength.

    Takes any magnetic field magnitude, so it serves both a magnetometer observation and a
    modelled field. Prefer the observation where one exists: it carries no field-model error.
    `compute_magnetic_field_variables` requests ``"f_ce"``/``"f_ce_Eq"`` route through this
    same function using the modelled field, for recipes without a magnetometer.

    Args:
        b_var (Variable): Variable containing the magnetic field magnitude.

    Returns:
        Variable: The electron gyrofrequency, in Hz.
    """
    b_field = np.asarray(b_var.get_data(u.T)).astype(np.float64)

    f_ce = (elementary_charge.si.value * b_field) / (2 * np.pi * electron_mass.si.value)

    f_ce_var = ep.Variable(data=f_ce, original_unit=u.Hz)
    f_ce_var.metadata.description = "Electron gyrofrequency."
    f_ce_var.metadata.add_processing_note("Computed as e*B/(2*pi*m_e) from the magnetic field strength.")

    return f_ce_var
