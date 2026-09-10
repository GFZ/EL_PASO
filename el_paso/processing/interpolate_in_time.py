# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import logging
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
from scipy.interpolate import interp1d

import el_paso as ep
from el_paso.utils import datenum_to_datetime, timed_function

InterpolationMethod = Literal["linear", "nearest", "zero", "slinear", "quadratic", "cubic", "previous", "next"]


@timed_function()
def interpolate_in_time(
    time_variable: ep.Variable,
    variables: dict[str, ep.Variable],
    interpolation_method_dict: dict[str, InterpolationMethod],
    target_cadence: timedelta | None = None,
    target_time_variable: ep.Variable | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    fill_value: Literal["extrapolate"] | float = np.nan,
    max_gap_seconds: float | None = None,
) -> ep.Variable:
    """Interpolates one or more variables by time according to specified methods and target axis.

    This function takes an original time variable and a dictionary of other variables, then
    interpolates these variables over a new time basis. The new basis can either be
    generated via a target cadence (with optional start/end times) or explicitly provided
    via a target time variable.

    Args:
        time_variable (ep.Variable): The master time variable that defines the original
            time basis for all other variables. Its data should be in a time
            unit (e.g., `ep.units.posixtime` or `ep.units.datenum`).
        variables (dict[str, ep.Variable]): A dictionary where keys are variable names (str) and values
            are the `ep.Variable` objects to be interpolated.
        interpolation_method_dict (dict[str, InterpolationMethod]): A dictionary mapping variable names (str) to
            interpolation method strings (e.g., "linear", "nearest"), specifying how each variable should be
            interpolated. If a variable is not present in this dictionary, it will be skipped.
        target_cadence (timedelta | None): Optional. A `datetime.timedelta` object specifying the
            duration of each time step for generating a regular target time axis.
        target_time_variable (ep.Variable | None): Optional. An explicit target time variable
            to interpolate onto. If provided, `target_cadence`, `start_time`, and `end_time` are ignored.
        start_time (datetime | None): Optional. A `datetime.datetime` object specifying the
            start time for generating the target axis. If None, the start time of `time_variable`
            is used.
        end_time (datetime | None): Optional. A `datetime.datetime` object specifying the end
            time for generating the target axis. If None, the end time of `time_variable` is used.
        fill_value (Literal["extrapolate"] | float): Optional. The value used to fill data points outside the bounds
            of the original time variable. Defaults to `np.nan`. Can also be set to `"extrapolate"`.
        max_gap_seconds (float | None): Optional. The maximum allowable time gap (in seconds) between two
            consecutive original timestamps. Target timestamps falling within a gap larger than this
            value will not be interpolated and will be masked with `np.nan`.

    Returns:
        ep.Variable: An `ep.Variable` object representing the new interpolated time axis. The
        `variables` dictionary passed as an argument is modified in place, with
        each variable's data updated to its interpolated values.

    Raises:
        ValueError: If neither `target_cadence` nor `target_time_variable` is provided,
            or if the first dimension size of any variable's data does not match the
            length of the `time_variable` data.
        TypeError: If an input data array is not a numeric type.
    """
    logger = logging.getLogger(__name__)
    logger.info("Interpolating by time...")

    if target_cadence is None and target_time_variable is None:
        msg = "Either target_cadence or target_time_variable must be provided!"
        raise ValueError(msg)

    # Determine target time coordinates
    if target_time_variable is not None:
        target_timestamps = target_time_variable.get_data(ep.units.posixtime).astype(np.float64)
        new_time_var = target_time_variable
    else:
        start_time = start_time or datenum_to_datetime(time_variable.get_data(ep.units.datenum)[0])
        end_time = end_time or datenum_to_datetime(time_variable.get_data(ep.units.datenum)[-1])

        if target_cadence is None:
            msg = "target_cadence must be specified if target_time_variable is None!"
            raise ValueError(msg)

        target_timestamps = np.arange(start_time.timestamp(), end_time.timestamp(), target_cadence.total_seconds())
        new_time_var = ep.Variable(data=target_timestamps, original_unit=ep.units.posixtime)
        new_time_var.metadata.add_processing_note("Created while time interpolating.")

    original_cadence = float(np.nanmedian(np.diff(time_variable.get_data(ep.units.posixtime))))
    timestamps = time_variable.get_data(ep.units.posixtime).astype(np.float64)

    # Pre-calculate reusable search indices and exact match conditions for gaps
    idx = np.searchsorted(timestamps, target_timestamps)
    in_bounds = (idx > 0) & (idx < len(timestamps))
    exact_match_right = (idx < len(timestamps)) & (
        target_timestamps == timestamps[np.minimum(idx, len(timestamps) - 1)]
    )
    exact_match_left = (idx > 0) & (target_timestamps == timestamps[idx - 1])

    # Pre-calculate the max_gap_seconds mask
    max_gap_mask = None
    if max_gap_seconds is not None:
        gaps = np.zeros_like(target_timestamps, dtype=float)
        gaps[in_bounds] = timestamps[idx[in_bounds]] - timestamps[idx[in_bounds] - 1]
        is_large_gap = in_bounds & (gaps > max_gap_seconds)
        max_gap_mask = is_large_gap & ~exact_match_right & ~exact_match_left

    for key, var in variables.items():
        if key not in interpolation_method_dict:
            continue

        # Check if time variable and data content sizes match
        if var.get_data().shape[0] != len(timestamps):
            msg = f"Variable {key}: size of dimension 0 does not match length of time variable!"
            raise ValueError(msg)

        old_data = var.get_data()
        if not np.issubdtype(old_data.dtype, np.number):
            msg = f"Interpolation (method: {interpolation_method_dict[key]}) is only supported for numeric types!"
            raise TypeError(msg)

        # Identify NaN rows along the time axis (axis=0)
        if old_data.ndim > 1:
            is_nan_time = np.any(np.isnan(old_data), axis=tuple(range(1, old_data.ndim)))
        else:
            is_nan_time = np.isnan(old_data)

        # Generate a per-variable mask to isolate target timestamps falling inside an original NaN gap
        is_nan_gap = None
        if np.any(is_nan_time):
            left_is_nan = np.zeros_like(target_timestamps, dtype=bool)
            right_is_nan = np.zeros_like(target_timestamps, dtype=bool)

            left_is_nan[idx > 0] = is_nan_time[idx[idx > 0] - 1]
            right_is_nan[idx < len(timestamps)] = is_nan_time[
                np.minimum(idx[idx < len(timestamps)], len(timestamps) - 1)
            ]

            # Target timestamps are inside a NaN gap if bounded by an original NaN point
            is_nan_gap = in_bounds & (left_is_nan | right_is_nan)
            # Protect target items that land perfectly on an original valid measurement
            is_nan_gap = is_nan_gap & ~(exact_match_right & ~right_is_nan) & ~(exact_match_left & ~left_is_nan)

        # Interpolate using only valid data pairs to prevent spline corruption or matrix failures
        if np.any(is_nan_time):
            valid_mask = ~is_nan_time
            if not np.any(valid_mask):
                interpolated_data = np.full((len(target_timestamps), *old_data.shape[1:]), np.nan)
                f = None
            else:
                f = interp1d(
                    timestamps[valid_mask],
                    old_data[valid_mask, ...],
                    kind=interpolation_method_dict[key],
                    axis=0,
                    bounds_error=False,
                    fill_value=fill_value,
                )
        else:
            f = interp1d(
                timestamps,
                old_data,
                kind=interpolation_method_dict[key],
                axis=0,
                bounds_error=False,
                fill_value=fill_value,
            )

        if f is not None:
            interpolated_data = f(target_timestamps)

        # Enforce maximum permissible data gaps
        if max_gap_mask is not None:
            interpolated_data[max_gap_mask, ...] = np.nan

        # Enforce NaN gap rules (no bridging across missing measurements)
        if is_nan_gap is not None:
            interpolated_data[is_nan_gap, ...] = np.nan

        if interpolated_data.shape[0] != len(target_timestamps):
            msg = "Encountered shape missmatch after time interpolation!"
            raise ValueError(msg)

        # Update data and metadata in-place
        var.set_data(np.array(interpolated_data), "same")
        var.metadata.original_cadence_seconds = original_cadence

        cadence_desc = (
            f"cadence of {target_cadence.total_seconds() / 60} minutes" if target_cadence else "custom target time axis"
        )
        var.metadata.add_processing_note(
            f"Time interpolated with method '{interpolation_method_dict[key]}' to {cadence_desc}"
        )

    return new_time_var
