# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import logging
import typing
from datetime import datetime, timedelta
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

import el_paso as ep
from el_paso.utils import datenum_to_datetime, timed_function


class TimeBinMethod(Enum):
    """Enum for time binning methods.

    Attributes:
        Mean (str): Calculates the mean of the data.
        NanMean (str): Calculates the mean of the data, ignoring NaNs.
        Median (str): Calculates the median of the data.
        NanMedian (str): Calculates the median of the data, ignoring NaNs.
        Merge (str): Concatenates the data.
        NanMax (str): Calculates the maximum of the data, ignoring NaNs.
        NanMin (str): Calculates the minimum of the data, ignoring NaNs.
        NoBinning (str): Applies no binning.
        Repeat (str): Repeats the data.
        Unique (str): Returns unique values from the data.
    """

    Mean = "Mean"
    NanMean = "NanMean"
    Median = "Median"
    NanMedian = "NanMedian"
    Merge = "Merge"
    NanMax = "NanMax"
    NanMin = "NanMin"
    NoBinning = "NoBinning"
    Repeat = "Repeat"
    Unique = "Unique"

    def __call__(self, data: NDArray[np.generic], drop_percent: float = 0) -> NDArray[np.generic]:  # noqa: C901, PLR0912
        """Applies the binning method to the provided data.

        Args:
            data (NDArray[np.generic]): The input data array to be binned or aggregated.
            drop_percent (float, optional): The percentage of the lowest and highest
                values to drop before performing a statistical aggregation.
                Defaults to 0.

        Returns:
            NDArray[np.generic]: The resulting array after applying the selected
                binning or aggregation method.

        Raises:
            TypeError: If the selected binning method requires numeric types and the
                input data is not numeric.
        """
        binned_array: NDArray[np.generic]

        if self.value in ["Mean", "NanMean", "Median", "NanMedian", "NanMax", "NanMin"] and not np.issubdtype(
            data.dtype, np.number
        ):
            msg = f"{self.value} time bin method is only supported for numeric types!"
            raise TypeError(msg)

        num_to_remove = int(len(data) * drop_percent / 100)
        if num_to_remove > 0 and np.issubdtype(data.dtype, np.number):
            data = np.sort(data, axis=0)
            data = data[num_to_remove:-num_to_remove]

        match self.value:
            case "Mean":
                data = typing.cast("NDArray[np.floating]", data)
                binned_array = np.mean(data, axis=0)
            case "NanMean":
                data = typing.cast("NDArray[np.floating]", data)
                binned_array = np.nanmean(data, axis=0)
            case "Median":
                data = typing.cast("NDArray[np.floating]", data)
                binned_array = np.nanmedian(data, axis=0)
            case "NanMedian":
                data = typing.cast("NDArray[np.floating]", data)
                binned_array = np.nanmedian(data, axis=0)
            case "Merge":
                binned_array = np.concatenate(data, axis=0)
            case "NanMax":
                binned_array = np.nanmax(data, axis=0)
            case "NanMin":
                binned_array = np.nanmin(data, axis=0)
            case "NoBinning":
                binned_array = data
            case "Repeat":
                binned_array = data
            case "Unique":
                binned_array = np.unique(data, axis=0)

                if data.dtype.kind in {"U", "S"}:
                    binned_array = np.asarray(["".join(binned_array)])

        return binned_array


@timed_function()
def bin_by_time(  # noqa: C901
    time_variable: ep.Variable,
    variables: dict[str, ep.Variable],
    time_bin_method_dict: dict[str, TimeBinMethod],
    time_binning_cadence: timedelta,
    window_alignement: Literal["center", "left", "right"] = "center",
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    drop_percent: float = 0,
) -> ep.Variable:
    """Bins one or more variables by time according to specified methods and cadence.

    This function takes a time variable and a dictionary of other variables, then
    bins these variables over time. Each variable can have a specific binning
    method applied (e.g., mean, median, sum). The binning is performed over
    defined time intervals (cadence) with a specified alignment.

    Args:
        time_variable (ep.Variable): The master time variable that defines the
            time basis for all other variables. Its data should be in a time
            unit (e.g., `ep.units.posixtime` or `ep.units.datenum`).
        variables (dict[str, ep.Variable]): A dictionary where keys are variable names (str) and values
            are the `ep.Variable` objects to be binned.
        time_bin_method_dict (dict[str, ep.TimeBinMethod]): A dictionary mapping variable names (str) to
            `ep.TimeBinMethod` enums, specifying how each variable should be
            binned within each time window. If a variable is not present in
            this dictionary, it will be skipped.
        time_binning_cadence (timedelta): A `datetime.timedelta` object specifying the
            duration of each time bin.
        window_alignement (Literal["center", "left", "right"]): Determines how the time windows are aligned.
            Defaults to "center".
            * "center": The time bin represents the center of the window.
            * "left": The time bin represents the left (start) of the window.
            * "right": The time bin represents the right (end) of the window.
        start_time (datetime | None): Optional. A `datetime.datetime` object specifying the
            start time for binning. If None, the start time of `time_variable`
            is used.
        end_time (datetime | None): Optional. A `datetime.datetime` object specifying the end
            time for binning. If None, the end time of `time_variable` is used.
        drop_percent (float): Optional. The percentage of the lowest and highest values to
            drop from each time bin before calculating statistical aggregates
            like mean or median. Defaults to 0.

    Returns:
        ep.Variable: An `ep.Variable` object representing the new binned time axis. The
        `variables` dictionary passed as an argument is modified in place, with
        each variables's data updated to its binned values.

    Raises:
        ValueError: If the first dimension size of any variable's data does not
            match the length of the `time_variable` data.
    """
    logger = logging.getLogger(__name__)
    logger.info("Binning by time...")

    start_time = start_time or datenum_to_datetime(time_variable.get_data(ep.units.datenum)[0])
    end_time = end_time or datenum_to_datetime(time_variable.get_data(ep.units.datenum)[-1])

    original_cadence = float(np.nanmedian(np.diff(time_variable.get_data(ep.units.posixtime))))

    binned_time, time_bins = _create_binned_time_and_bins(start_time, end_time, time_binning_cadence, window_alignement)

    # Cache digitized indices for every time variable
    index_iterables = None

    for key, var in variables.items():
        if key not in time_bin_method_dict:
            continue

        # Just repeat in case of no time dependency
        if time_bin_method_dict[key] == ep.TimeBinMethod.Repeat:
            var.set_data(np.repeat(var.get_data()[np.newaxis, ...], len(binned_time), axis=0), "same")
            var.metadata.original_cadence_seconds = 0
            continue

        # check if time variable and data content sizes match
        if var.get_data().shape[0] != len(time_variable.get_data()):
            msg = f"Variable {key}: size of dimension 0 does not match length of time variable!"
            raise ValueError(msg)

        # calculate bin indices for given time array if it has not been calculated before
        if not index_iterables:
            timestamps = typing.cast("NDArray[np.floating]", time_variable.get_data(ep.units.posixtime))
            index_iterables = _calculate_index_iterables(timestamps, time_bins)

        unique_indices, indices_separation = index_iterables

        # Initialize binned_data as an array of np.nans with the same shape as self._data,
        # but with the length of the first dimension matching the length of time_array
        if var.get_data().dtype.kind in {"U", "S", "O"}:  # Check if the data is string or object type
            binned_data = np.full((len(binned_time),), "", dtype=var.get_data().dtype)
        else:
            binned_data_shape = (len(binned_time), *var.get_data().shape[1:])
            binned_data = np.full(binned_data_shape, np.nan)

        # Iterate over unique indices
        for i, unique_index in enumerate(unique_indices):
            bin_data = var.get_data()[indices_separation[i] : indices_separation[i + 1]]
            if len(bin_data) == 0:
                continue  # no data found
            if bin_data.dtype.kind in {"i", "f"} and not np.any(np.isfinite(bin_data)):
                continue  # no finite data found
            binned_value = time_bin_method_dict[key](bin_data, drop_percent=drop_percent)

            # Update the relevant slice of binned_data
            binned_data[unique_index, ...] = binned_value

        # Update relevant metadata fields
        # Ensure binned_data works for both numeric and string data
        if isinstance(binned_data[0], str):
            var.set_data(np.array(binned_data, dtype=object), "same")
        else:
            var.set_data(np.array(binned_data), "same")

        # update metadata
        var.metadata.original_cadence_seconds = original_cadence
        var.metadata.add_processing_note(
            f"Time binned with method {time_bin_method_dict[key].value}"
            f" and cadence of {time_binning_cadence.total_seconds() / 60} minutes"
        )

    new_time_var = ep.Variable(data=binned_time, original_unit=ep.units.posixtime)
    new_time_var.metadata.add_processing_note("Created while time binning.")

    return new_time_var


def _create_binned_time_and_bins(
    start_time: datetime,
    end_time: datetime,
    time_binning_cadence: timedelta,
    window_alignement: Literal["left", "center", "right"],
) -> tuple[NDArray[np.floating], list[float]]:
    binned_time = np.arange(start_time.timestamp(), end_time.timestamp(), time_binning_cadence.total_seconds())

    # built bins
    total_duration = binned_time[-1] - binned_time[0]

    # Generate timestamps by stepping through the time range
    num_steps = (
        int(total_duration // time_binning_cadence.total_seconds()) + 1
    )  # Number of steps including start and end

    if window_alignement == "center":
        time_bins = [binned_time[0] - time_binning_cadence.total_seconds() / 2]
        for _ in range(num_steps):
            current_time = time_bins[-1] + time_binning_cadence.total_seconds()
            time_bins.append(current_time)
    elif window_alignement == "left":
        time_bins = [binned_time[0] - time_binning_cadence.total_seconds()]
        for _ in range(num_steps):
            current_time = time_bins[-1] + time_binning_cadence.total_seconds()
            time_bins.append(current_time)
    elif window_alignement == "right":
        time_bins = [binned_time[0]]
        for _ in range(num_steps):
            current_time = time_bins[-1] + time_binning_cadence.total_seconds()
            time_bins.append(current_time)
    else:
        msg = f"Encountered invalid window_alignment argument in time binning: {window_alignement}!"
        raise ValueError(msg)

    return binned_time, time_bins


def _calculate_index_iterables(
    timestamps: NDArray[np.floating], time_bins: list[float]
) -> tuple[NDArray[np.intp], list[int]]:
    index_set = np.digitize(timestamps, time_bins)
    index_set = index_set - 1  # shift indices by one to match time array

    unique_indices = np.unique(index_set)

    indices_separation: list[int] = []
    cursor = 0

    for i in range(len(unique_indices)):
        # ignore values before the desired interval
        if unique_indices[i] == -1:
            continue

        while cursor < len(index_set) and index_set[cursor] != unique_indices[i]:
            cursor += 1

        indices_separation.append(cursor)

    indices_separation.append(len(index_set))
    unique_indices = np.delete(
        unique_indices,
        np.argwhere((unique_indices == -1) | (unique_indices == len(time_bins) - 1)),
    )

    return unique_indices, indices_separation
