# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u

import el_paso as ep
from el_paso.processing.magnetic_field_utils.irbem import Coords
from el_paso.utils import timed_function

if TYPE_CHECKING:
    from el_paso import Variable

logger = logging.getLogger(__name__)


@timed_function("Magnetic latitude calculation")
def compute_magnetic_latitude(time_var: Variable, xgeo_var: Variable) -> Variable:
    """Computes the magnetic latitude of a satellite from its geographic position.

    The position is rotated into SM coordinates, whose z-axis is the geomagnetic dipole axis,
    and the latitude is taken in that frame.

    Unlike most quantities in `el_paso.processing`, this does not depend on a magnetic field
    model: magnetic latitude is a coordinate, so every model would return the same number. It
    does depend on time, because the dipole axis moves with respect to the geographic frame as
    the Earth rotates.

    Take care not to compute this as the latitude of a GEO position directly. That returns the
    *geographic* latitude, which differs from the magnetic latitude by the roughly 11 degree
    dipole tilt and its longitude dependence.

    Args:
        time_var (Variable): Variable containing time data, used to orient the dipole axis.
        xgeo_var (Variable): Variable containing geocentric (XGEO) coordinates, shaped
            (n_time, 3).

    Returns:
        Variable: The magnetic latitude, in degrees.
    """
    logger.info("\tCalculating magnetic latitude ...")

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in time_var.get_data(ep.units.posixtime)]

    pos_sm = Coords().transform(
        time=datetimes,
        pos=np.asarray(xgeo_var.get_data(ep.units.RE)).astype(np.float64),
        sysaxes_in=ep.IRBEM_SYSAXIS_GEO,
        sysaxes_out=ep.IRBEM_SYSAXIS_SM,
    )

    mlat = np.degrees(np.arctan2(pos_sm[:, 2], np.hypot(pos_sm[:, 0], pos_sm[:, 1])))

    mlat_var = ep.Variable(data=mlat, original_unit=u.deg)
    mlat_var.metadata.description = "Magnetic latitude of the satellite location."
    mlat_var.metadata.add_processing_note(
        "Computed as the latitude of the position in SM coordinates, i.e. measured from the geomagnetic dipole equator."
    )

    return mlat_var
