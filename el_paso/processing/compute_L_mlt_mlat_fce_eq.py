import numpy as np
from astropy import units as u
from el_paso.variable import Variable


def add_derived_params(variables, Re=6371.0):
    """
    Compute derived magnetospheric parameters (L, MLT, MLAT, fce, fce_eq)
    and add them to a dictionary of EL_PASO Variable objects.

    Parameters
    ----------
    variables : dict
        Dictionary of EL_PASO Variable objects. Must contain:
        - "Bt" : magnetic field strength (nT)
        - "Coordinates" : Cartesian coordinates [x, y, z] in km
    Re : float, optional
        Earth radius in km (default: 6371).

    Returns
    -------
    dict
        Updated dictionary with new Variable entries:
        - "L" : L-shell parameter (dimensionless)
        - "mlat" : magnetic latitude (degrees)
        - "mlt" : magnetic local time (hours)
        - "fce" : electron gyrofrequency (Hz)
        - "fce_eq" : equatorial electron gyrofrequency (Hz)
    """

    # Extract data arrays
    Bt = np.asarray(variables["Bt"].get_data())
    coords = np.asarray(variables["Coordinates"].get_data())

    # Normalize coordinates by Earth radius
    x, y, z = coords[:, 0] / Re, coords[:, 1] / Re, coords[:, 2] / Re

    # Compute derived spatial parameters
    r_xy = np.hypot(x, y)
    r = np.sqrt(x**2 + y**2 + z**2)
    mlat_rads = np.arctan2(z, r_xy)
    mlat = np.degrees(mlat_rads)

    # Dipole geometry formulas
    L = r / np.cos(mlat_rads) ** 2
    mlt = np.degrees(np.arctan2(y, x)) / 15.0 + 12.0  # convert deg→hours, center at noon
    mlt = np.mod(mlt, 24.0)  # wrap to 0–24 h range

    # Physical constants
    q = 1.602e-19  # electron charge (C)
    me = 9.109e-31  # electron mass (kg)

    # Electron gyrofrequency and equatorial version
    fce = (q * Bt * 1e-9) / (2 * np.pi * me)
    fce_eq = fce * (np.cos(mlat_rads) ** 6) / np.sqrt(1 + 3 * np.sin(mlat_rads) ** 2)

    # Create EL_PASO Variable objects with proper units
    variables["L"] = Variable(u.dimensionless_unscaled, data=L)
    variables["mlat"] = Variable(u.deg, data=mlat)
    variables["mlt"] = Variable(u.hourangle, data=mlt)
    variables["fce"] = Variable(u.Hz, data=fce)
    variables["fce_eq"] = Variable(u.Hz, data=fce_eq)

    return variables
