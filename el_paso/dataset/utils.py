# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
from swvo.io.utils import enforce_utc_timezone

if TYPE_CHECKING:
    from numpy.typing import NDArray


def join_var(
    var1: NDArray[np.generic] | list[xr.Variable] | xr.Variable, var2: NDArray[np.generic] | xr.Variable
) -> NDArray[np.generic] | list[xr.Variable]:
    """Join two variables along the first axis.

    Args:
        var1 (NDArray | list[xr.Variable] | xr.Variable): The existing data.
            If an ``xr.Variable``, it is wrapped in a single-element list before
            appending, since lazy xarray variables can't be concatenated with
            :func:`numpy.concatenate`.
        var2 (NDArray | xr.Variable): The new data to append.

    Returns:
        NDArray | list[xr.Variable]: The concatenated array if both inputs are
        NumPy arrays, otherwise a list of ``xr.Variable`` with ``var2`` appended.
    """
    if isinstance(var1, np.ndarray) and isinstance(var2, np.ndarray):
        return np.concatenate((var1, var2), axis=0)

    if isinstance(var1, xr.Variable):
        var1 = [var1]

    var1.append(var2)  # ty: ignore[invalid-argument-type, unresolved-attribute]

    return var1


def round_seconds(obj: datetime) -> datetime:
    """Round datetime object to the nearest second.

    Args:
        obj (datetime): The datetime to round.

    Returns:
        datetime: A copy of ``obj`` rounded to the nearest second (microseconds
        of 500,000 or more round up).
    """
    if obj.microsecond >= 500_000:
        obj += timedelta(seconds=1)
    return obj.replace(microsecond=0)


def python2matlab(datenum: datetime) -> float:
    """Convert Python datetime to MATLAB datenum.

    Args:
        datenum (datetime): The datetime to convert. Must be timezone-aware
            (UTC), since the time-of-day fraction is computed against a UTC
            midnight reference.

    Returns:
        float: The corresponding MATLAB datenum (days since year 0), including
        a fractional component for the time of day.
    """
    mdn = datenum + timedelta(days=366)
    frac = (datenum - datetime(datenum.year, datenum.month, datenum.day, 0, 0, 0, tzinfo=timezone.utc)).seconds / (
        24.0 * 60.0 * 60.0
    )
    return mdn.toordinal() + round(frac, 6)


def matlab2python(datenum: float | Iterable[float]) -> Iterable[datetime] | datetime:
    """Convert MATLAB datenum to Python datetime.

    Args:
        datenum (float | Iterable[float]): One or more MATLAB datenum values
            (days since year 0) to convert.

    Returns:
        Iterable[datetime] | datetime: The converted, UTC-aware, second-rounded
        datetime(s), matching the scalar-vs-iterable shape of the input.
    """
    warnings.filterwarnings("ignore", message="Discarding nonzero nanoseconds in conversion")

    datenum = np.asarray(datenum, dtype=float)
    datenum = pd.to_datetime(datenum - 719529, unit="D", origin=pd.Timestamp("1970-01-01")).to_pydatetime()  # ty:ignore[unresolved-attribute]

    if isinstance(datenum, Iterable):
        datenum = enforce_utc_timezone(list(datenum))  # ty:ignore[invalid-assignment]
        datenum: Iterable[datetime] = [round_seconds(x) for x in datenum]  # ty: ignore[invalid-argument-type, not-iterable]
    else:
        datenum: datetime = round_seconds(enforce_utc_timezone(datenum))

    return datenum


def pol2cart(
    theta: NDArray[np.float64], radius: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Transforms polar coordinates theta (in rad) and radius to cartesian coordinates x, y.

    Args:
        theta (NDArray[np.float64]): Angle(s) in radians.
        radius (NDArray[np.float64]): Radius/radii.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: The ``(x, y)`` cartesian coordinates.
    """
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return (x, y)


def cart2pol(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Transforms cartesian coordinates x, y to polar coordinates theta (in rad) and radius.

    Args:
        x (NDArray[np.float64]): X coordinate(s).
        y (NDArray[np.float64]): Y coordinate(s).

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: The ``(theta, radius)`` polar
        coordinates, with ``theta`` in radians.
    """
    z = x + 1j * y
    return np.angle(z), np.abs(z)
