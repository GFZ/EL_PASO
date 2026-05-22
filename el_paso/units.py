# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import cdflib
import numpy as np
from astropy import units as u
from astropy.constants import R_earth  # ty:ignore[unresolved-import]

if TYPE_CHECKING:
    from numpy.typing import NDArray

# -----------------------------------------------------------------------------
# 1. Custom Unit Definitions
# -----------------------------------------------------------------------------

# Time units for scientific data formats
cdf_epoch = u.def_unit("cdf_epoch")
tt2000 = u.def_unit("tt2000")
posixtime = u.def_unit("posixtime")
datenum = u.def_unit("datenum")
j2k = u.def_unit("j2k")
datetime = u.def_unit("datetime")

J2000_EPOCH = dt.datetime(2000, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)

# Position units
RE = u.def_unit("RE", R_earth)

# -----------------------------------------------------------------------------
# 2. Time Conversion Functions
# -----------------------------------------------------------------------------


def posixtime_to_datenum(posixtime_array: NDArray[np.floating]) -> NDArray[np.floating]:
    """Converts an array of POSIX timestamps to MATLAB datenums.

    The MATLAB datenum format is the number of days since 0000-01-01 plus 1.
    This function handles the time zone and reference date differences.

    Parameters:
        posixtime_array (NDArray[np.floating]): An array of POSIX timestamps (seconds since 1970-01-01 UTC).

    Returns:
        NDArray[np.floating]: An array of MATLAB datenums.
    """
    posixtime_array = np.atleast_1d(posixtime_array)
    dt_array = [dt.datetime.fromtimestamp(posixtime, tz=dt.timezone.utc) for posixtime in posixtime_array]
    matlab_datenum_offset = 366  # Difference between MATLAB's and Python's reference dates
    return np.array(
        [
            dt.toordinal()
            + dt.hour / 24
            + dt.minute / 1440
            + dt.second / 86400
            + dt.microsecond / 86400000000
            + matlab_datenum_offset
            for dt in dt_array
        ]
    )


def datenum_to_posixtime(datenum_array: NDArray[np.floating]) -> NDArray[np.floating]:
    """Converts an array of MATLAB datenums to POSIX timestamps.

    Parameters:
        datenum_array (NDArray[np.floating]): An array of MATLAB datenums.

    Returns:
        NDArray[np.floating]: An array of POSIX timestamps (seconds since 1970-01-01 UTC).
    """
    return (datenum_array - 719529) * 24 * 60 * 60  # 719529 is the datenum for 1970-01-01


def tt2000_to_datenum(tt2000_val: NDArray[np.floating]) -> NDArray[np.floating]:
    """Converts tt2000 nanoseconds to MATLAB datenum days via POSIX time."""
    posix_val = cdflib.cdfepoch.unixtime(tt2000_val.astype(np.int64))
    return posixtime_to_datenum(posix_val)  # ty:ignore[invalid-argument-type]


def datenum_to_tt2000(datenum_val: NDArray[np.floating]) -> NDArray[np.floating]:
    """Converts MATLAB datenum days to tt2000 nanoseconds via POSIX time."""
    posix_val = datenum_to_posixtime(datenum_val)
    return cdflib.cdfepoch.timestamp_to_tt2000(posix_val)


def datenum_to_cdf_epoch(datenum_array: NDArray[np.floating]) -> NDArray[np.floating]:
    """Converts MATLAB datenum days to CDF_EPOCH milliseconds via POSIX time."""
    posix_val = datenum_to_posixtime(datenum_array)
    return cdflib.cdfepoch.timestamp_to_cdfepoch(posix_val)


def cdf_epoch_to_datenum(cdf_epoch_array: NDArray[np.floating]) -> NDArray[np.floating]:
    """Converts CDF_EPOCH milliseconds to MATLAB datenum days via POSIX time."""
    posix_val = np.atleast_1d(cdflib.cdfepoch.unixtime(cdf_epoch_array))
    return posixtime_to_datenum(posix_val)


def j2k_to_datetime(j2k_seconds: NDArray[np.floating]) -> NDArray[dt.datetime]:  # ty:ignore[invalid-type-arguments]
    """Convert J2000 seconds to UTC dt.datetime."""
    return np.array([J2000_EPOCH + dt.timedelta(seconds=float(sec)) for sec in np.atleast_1d(j2k_seconds)])


def j2k_to_posix(j2k_seconds: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert J2000 seconds to POSIX timestamp."""
    return np.array([dt.timestamp() for dt in j2k_to_datetime(j2k_seconds)], dtype=np.float64)


def posix_to_datetime(posix_seconds: NDArray[np.floating]) -> NDArray[dt.datetime]:  # ty:ignore[invalid-type-arguments]
    """Convert POSIX seconds to UTC dt.datetime."""
    return np.array([dt.datetime.fromtimestamp(float(sec), tz=dt.timezone.utc) for sec in np.atleast_1d(posix_seconds)])


# -----------------------------------------------------------------------------
# 3. Astropy Equivalencies
# -----------------------------------------------------------------------------

# Equivalency: TT2000 <-> POSIX
tt2000_posixtime_equiv = [
    (
        tt2000,
        posixtime,
        lambda x: cdflib.cdfepoch.unixtime(x.astype(np.int64)),
        cdflib.cdfepoch.timestamp_to_tt2000,
    )
]

# Equivalency: POSIX <-> DATENUM
posixtime_datenum_equiv = [
    (
        posixtime,
        datenum,
        posixtime_to_datenum,
        datenum_to_posixtime,
    )
]

# Equivalency: TT2000 <-> DATENUM
tt2000_datenum_equiv = [
    (
        tt2000,
        datenum,
        tt2000_to_datenum,
        datenum_to_tt2000,
    )
]

# Equivalency: CDF_EPOCH <-> POSIX
cdf_epoch_posixtime_equiv = [
    (
        cdf_epoch,
        posixtime,
        cdflib.cdfepoch.unixtime,
        cdflib.cdfepoch.timestamp_to_cdfepoch,
    )
]

# Equivalency: DATENUM <-> CDF_EPOCH
datenum_cdf_epoch_equiv = [
    (
        datenum,
        cdf_epoch,
        datenum_to_cdf_epoch,
        cdf_epoch_to_datenum,
    )
]

# Equivalency: CDF_EPOCH <-> CDF_TT2000
cdf_epoch_cdf_tt2000_equiv = [
    (
        cdf_epoch,
        tt2000,
        lambda x: cdflib.cdfepoch.compute_tt2000(cdflib.cdfepoch.breakdown_epoch(x)),
        lambda x: cdflib.cdfepoch.compute_epoch(cdflib.cdfepoch.breakdown_tt2000(x)),
    )
]

j2k_posixtime_equiv = [
    (
        j2k,
        posixtime,
        j2k_to_posix,
        lambda x: (j2k_to_datetime(x) - J2000_EPOCH).total_seconds(),
    )
]

# -----------------------------------------------------------------------------
# 4. Enable Custom Units and Conversions
# -----------------------------------------------------------------------------

# Add custom units to the astropy.units namespace for direct access (e.g., u.RE)
u.add_enabled_units(RE)
u.add_enabled_units(tt2000)
u.add_enabled_units(posixtime)
u.add_enabled_units(datenum)
u.add_enabled_units(cdf_epoch)
u.add_enabled_units(datetime)
u.add_enabled_units(j2k)

# Add custom equivalencies for seamless unit conversions
u.add_enabled_equivalencies(u.dimensionless_angles())
u.add_enabled_equivalencies(tt2000_posixtime_equiv)
u.add_enabled_equivalencies(posixtime_datenum_equiv)
u.add_enabled_equivalencies(tt2000_datenum_equiv)
u.add_enabled_equivalencies(cdf_epoch_posixtime_equiv)
u.add_enabled_equivalencies(datenum_cdf_epoch_equiv)
u.add_enabled_equivalencies(cdf_epoch_cdf_tt2000_equiv)
u.add_enabled_equivalencies(j2k_posixtime_equiv)
