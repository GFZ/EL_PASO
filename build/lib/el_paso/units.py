# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import cdflib
import numpy as np
from astropy import units as u
from astropy.constants import R_earth  # type:ignore [reportAttributeAccessIssue]

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
    dt_array = [datetime.fromtimestamp(posixtime, tz=timezone.utc) for posixtime in posixtime_array]
    matlab_datenum_offset = 366  # Difference between MATLAB's and Python's reference dates
    return np.array(
        [
            dt.toordinal() + dt.hour / 24 + dt.minute / 1440 + dt.second / 86400 + matlab_datenum_offset
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
    return posixtime_to_datenum(posix_val)  # type: ignore[reportArgumentType]


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


# -----------------------------------------------------------------------------
# 3. Astropy Equivalencies
# -----------------------------------------------------------------------------

# Equivalency: TT2000 <-> POSIX
tt2000_posixtime_equiv = [
    (
        tt2000,
        posixtime,
        lambda x: cdflib.cdfepoch.unixtime(x.astype(np.int64)),
        lambda x: cdflib.cdfepoch.timestamp_to_tt2000(x),
    )
]

# Equivalency: POSIX <-> DATENUM
posixtime_datenum_equiv = [
    (
        posixtime,
        datenum,
        lambda x: posixtime_to_datenum(x),
        lambda x: datenum_to_posixtime(x),
    )
]

# Equivalency: TT2000 <-> DATENUM
tt2000_datenum_equiv = [
    (
        tt2000,
        datenum,
        lambda x: tt2000_to_datenum(x),
        lambda x: datenum_to_tt2000(x),
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
        lambda x: datenum_to_cdf_epoch(x),
        lambda x: cdf_epoch_to_datenum(x),
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


# -----------------------------------------------------------------------------
# 4. Enable Custom Units and Conversions
# -----------------------------------------------------------------------------

# Add custom units to the astropy.units namespace for direct access (e.g., u.RE)
u.add_enabled_units(RE)
u.add_enabled_units(tt2000)
u.add_enabled_units(posixtime)
u.add_enabled_units(datenum)
u.add_enabled_units(cdf_epoch)

# Add custom equivalencies for seamless unit conversions
u.add_enabled_equivalencies(u.dimensionless_angles())
u.add_enabled_equivalencies(tt2000_posixtime_equiv)
u.add_enabled_equivalencies(posixtime_datenum_equiv)
u.add_enabled_equivalencies(tt2000_datenum_equiv)
u.add_enabled_equivalencies(cdf_epoch_posixtime_equiv)
u.add_enabled_equivalencies(datenum_cdf_epoch_equiv)
u.add_enabled_equivalencies(cdf_epoch_cdf_tt2000_equiv)
