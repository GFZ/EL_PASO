# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Alwin Roy
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from astropy import units as u

import el_paso as ep
from el_paso.variable import Variable


def add_derived_params(variables: dict, re: float = ep.units.RE) -> dict:
    """Compute derived magnetospheric parameters and add them to EL_PASO variables.

    The parameters are based on the following formulas:
    - L-shell (L) is the distance from the Earth's center in units of Earth radii, derived from the coordinates.
    - Magnetic latitude (MLAT) is computed from the position vector.
    - Magnetic local time (MLT) is derived based on the coordinates, wrapping to the 0-24 hour range.
    - Electron gyrofrequency (fce) and equatorial electron gyrofrequency (fce_eq) are derived using the magnetic field strength.

    Args:
        variables (dict): Dictionary of EL_PASO Variable objects. Must contain:
            - "Bt" (np.ndarray): Magnetic field strength (nT).
            - "Coordinates" (np.ndarray): Cartesian coordinates [x, y, z] in km.
        re (float, optional): Earth radius in km. Default is `ep.units.RE`.

    Returns:
        dict: Updated dictionary with new Variable entries:
            - "L" (dimensionless): L-shell parameter.
            - "mlat" (degrees): Magnetic latitude.
            - "mlt" (hours): Magnetic local time.
            - "fce" (Hz): Electron gyrofrequency.
            - "fce_eq" (Hz): Equatorial electron gyrofrequency.
    """
    # Extract data arrays
    bt = np.asarray(variables["Bt"].get_data())
    coords = np.asarray(variables["Coordinates"].get_data())

    # Normalize coordinates by Earth radius
    x, y, z = coords[:, 0] / re, coords[:, 1] / re, coords[:, 2] / re

    # Compute derived spatial parameters
    r_xy = np.hypot(x, y)
    r = np.sqrt(x**2 + y**2 + z**2)
    mlat_rads = np.arctan2(z, r_xy)
    mlat = np.degrees(mlat_rads)

    # Dipole geometry formulas
    l_shell = r / np.cos(mlat_rads) ** 2
    mlt = np.degrees(np.arctan2(y, x)) / 15.0 + 12.0  # convert deg→hours, center at noon
    mlt = np.mod(mlt, 24.0)  # wrap to 0-24 h range

    # Physical constants
    q = 1.602e-19  # electron charge (C)
    me = 9.109e-31  # electron mass (kg)

    # Electron gyrofrequency and equatorial version
    fce = (q * bt * 1e-9) / (2 * np.pi * me)
    fce_eq = fce * (np.cos(mlat_rads) ** 6) / np.sqrt(1 + 3 * np.sin(mlat_rads) ** 2)

    # Create EL_PASO Variable objects with proper units
    variables["L"] = Variable(u.dimensionless_unscaled, data=l_shell)
    variables["mlat"] = Variable(u.deg, data=mlat)
    variables["mlt"] = Variable(u.hourangle, data=mlt)
    variables["fce"] = Variable(u.Hz, data=fce)
    variables["fce_eq"] = Variable(u.Hz, data=fce_eq)

    variables.add_processing_note(
        "L, mlt, mlat, fce and fce_eq are calculated using compute_L_mlt_mlat_fce_eq with dipole assumption"
    )

    return variables
