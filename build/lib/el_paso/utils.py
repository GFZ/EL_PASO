# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import re
import time
import timeit
import typing
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import pandas as pd
import tqdm
from packaging import version as version_pkg

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from multiprocessing.pool import MapResult

    import el_paso as ep

logger = logging.getLogger(__name__)


def fill_str_template_with_time(input_str: str, time: datetime) -> str:
    """Fills a string template with time-based placeholders.

    This function replaces common time-based placeholders in a string with
    the corresponding values from a `datetime` object. The placeholders
    are case-sensitive.

    Parameters:
        input_str (str): The input string containing placeholders like 'yyyymmdd', 'YYYYMMDD',
                         'YYYY', 'MM', and 'DD'.
        time (datetime): The datetime object to use for filling the template.

    Returns:
        str: The string with all placeholders replaced by their time values.
    """
    yyyymmdd_str = time.strftime("%Y%m%d")
    yyyy_str = time.strftime("%Y")
    mm_str = time.strftime("%m")
    dd_str = time.strftime("%d")

    return (
        input_str.replace("yyyymmdd", yyyymmdd_str)
        .replace("YYYYMMDD", yyyymmdd_str)
        .replace("YYYY", yyyy_str)
        .replace("MM", mm_str)
        .replace("DD", dd_str)
    )


def extract_version(file_name: str | Path) -> tuple[str, version_pkg.Version]:
    """Extracts the version string from a file name.

    The function looks for a version string pattern `_v*` (e.g., '_v1.2.3' or '_v1_2-3')
    located just before the file extension. It returns the base file name and a
    parsed version object. If no version is found, it returns the original file name
    and a default version '0'.

    Parameters:
        file_name (str | Path): The name or path of the file.

    Returns:
        tuple[str, version_pkg.Version]: A tuple containing:
            - The base file name without the version string.
            - The parsed version object (`packaging.version.Version`).
    """
    # convert to str in case of Path object
    file_name = str(file_name)

    # Regular expression to find the version part (_v* or _v*.*-*.*) before the file extension
    match = re.search(r"_(v[\d._-]+)(?=\.\w+$)", file_name)
    if match:
        base_name = file_name[: match.start()]
        ver_str = match.group(1)
        # Normalize the version string by replacing separators with dots
        normalized_ver_str = re.sub(r"[_-]", ".", ver_str.replace("v", ""))
        return base_name, version_pkg.parse(normalized_ver_str)
    return file_name, version_pkg.parse("0")


T = TypeVar("T", bound=Path | str)


def get_file_by_version(file_paths: Iterable[T], version: str) -> T | None:
    """Filters a list of file paths to find a specific version or the latest one.

    If a specific version string (e.g., 'v1.2.3') is provided, the function returns
    the file that matches exactly. If the `version` parameter is 'latest', it
    returns the file with the highest version number among all provided file paths.

    Parameters:
        file_paths (Iterable[T]): An iterable of file paths (as strings or `Path` objects).
        version (str): The specific version string to match (e.g., 'v1.2.3') or 'latest'
                       to retrieve the most recent version.

    Returns:
        T | None: The file path that matches the criteria, or `None` if no matching
                  file is found.
    """
    latest_file = None

    if version != "latest":
        normalized_version = re.sub(r"[_-]", ".", version.replace("v", ""))
        target_version = version_pkg.parse(normalized_version)
    else:
        target_version = None

    for file in file_paths:
        _, ver_obj = extract_version(file)

        # Check if the current file matches the target version if specified
        if target_version and ver_obj == target_version:
            return file

        # If no specific version is targeted, find the highest version
        if latest_file is None or ver_obj > extract_version(latest_file)[1]:
            latest_file = file

    # Extract the file names from the dictionary
    return latest_file


P = ParamSpec("P")
R = TypeVar("R")


def timed_function(func_name: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """A decorator that logs the execution time of a function.

    This decorator measures the time it takes for a decorated function to execute
    and logs the result to a logger at the INFO level. The log message can be
    prefixed with an optional function name.

    Parameters:
        func_name (str | None): An optional name to use in the log message. If `None`,
                                a generic message is used.

    Returns:
        Callable: A decorator that wraps the target function with timing logic.
    """

    def timed_function_(f: Callable[P, R]) -> Callable[P, R]:
        @wraps(f)
        def wrap(*args: P.args, **kwargs: P.kwargs) -> R:
            tic = timeit.default_timer()
            result = f(*args, **kwargs)
            toc = timeit.default_timer()
            if func_name:
                logger.info(f"\t\t{func_name} finished in {toc - tic:0.3f} seconds")
            else:
                logger.info(f"\t\tFinished in {toc - tic:0.3f} seconds")

            return result

        return wrap

    return timed_function_


def enforce_utc_timezone(time: datetime) -> datetime:
    """Ensures a datetime object has UTC timezone information.

    If the provided datetime object is naive (lacks timezone info), it is assigned
    the UTC timezone. If it already has a timezone, it is returned unchanged.

    Parameters:
        time (datetime): The datetime object to process.

    Returns:
        datetime: The datetime object with `timezone.utc` assigned.
    """
    if time.tzinfo is None:
        time = time.replace(tzinfo=timezone.utc)
    return time


def datenum_to_datetime(datenum_val: float) -> datetime:
    """Converts a MATLAB datenum value to a timezone-aware datetime object.

    This function leverages pandas to convert the datenum (days since year 0)
    into a UTC-aware datetime object.

    Parameters:
        datenum_val (float): The MATLAB datenum value.

    Returns:
        datetime: The converted datetime object with UTC timezone.
    """
    return (
        pd.to_datetime(datenum_val - 719529, unit="D", origin=pd.Timestamp("1970-01-01"))
        .to_pydatetime()
        .replace(tzinfo=timezone.utc)
    )


def datetime_to_datenum(datetime_val: datetime) -> float:
    """Converts a datetime object to a MATLAB datenum value.

    This function calculates the datenum value, which represents the number of days
    since year 0, including a fractional component for the time of day.

    Parameters:
        datetime_val (datetime): The datetime object to convert.

    Returns:
        float: The corresponding MATLAB datenum value.
    """
    mdn = datetime_val + timedelta(days=366)
    dt = datetime(datetime_val.year, datetime_val.month, datetime_val.day, 0, 0, 0, tzinfo=timezone.utc)
    frac = (datetime_val - dt).seconds / (24.0 * 60.0 * 60.0)

    return mdn.toordinal() + round(frac, 6)


def assert_n_dim(var: ep.Variable, n_dims: int, name_in_file: str) -> None:
    """Asserts that a variable's data has a specific number of dimensions.

    Raises a `ValueError` if the provided variable's data does not match the
    expected number of dimensions.

    Parameters:
        var (ep.Variable): The variable instance to check.
        n_dims (int): The expected number of dimensions.
        name_in_file (str): The name of the variable, used in the error message.
    """
    provided = var.get_data().ndim

    if provided != n_dims:
        msg = (
            f"Encountered dimension missmatch for variable with name {name_in_file}:"
            f"should be {n_dims}, got: {provided}!"
        )
        raise ValueError(msg)


def show_process_bar_for_map_async(map_result: MapResult[Any], chunksize: int) -> None:
    """Displays a progress bar for a `multiprocessing.pool.MapResult` object.

    This function creates a `tqdm` progress bar that tracks the completion of
    a parallel map operation. It polls the `MapResult`'s internal state to
    update the progress bar until the operation is complete.

    Parameters:
        map_result (MapResult): The result object from `Pool.map_async()`.
        chunksize (int): The chunk size used in the `map_async` call.
    """
    init = typing.cast("int", map_result._number_left) * chunksize  # type: ignore[reportUnknownMemberType] # noqa: SLF001
    with tqdm.tqdm(total=init) as t:
        while True:
            if map_result.ready():
                break
            t.n = init - map_result._number_left * chunksize  # type: ignore[reportUnknownMemberType] # noqa: SLF001
            t.refresh()
            time.sleep(1)


class Hashabledict(dict[Any, Any]):
    """A dictionary subclass that is hashable.

    This class enables a dictionary to be used in sets or as keys in other dictionaries
    by providing a custom hash implementation based on its contents.
    """

    def __hash__(self) -> int:  # type: ignore[reportIncompatibleVariableOverride]
        """Computes a hash value for the dictionary.

        The hash is computed based on the frozensets of the dictionary's keys
        and values. This ensures that two `Hashabledict` instances with the same
        key-value pairs will have the same hash, regardless of the order of
        insertion.

        Returns:
            int: The hash value of the dictionary.
        """
        return hash((frozenset(self), frozenset(self.itervalues())))  # type: ignore[reportAttributeAccessIssue]


def make_dict_hashable(dict_input: dict[Any, Any] | None) -> Hashabledict | None:
    """Converts a standard dictionary into a hashable one.

    If the input is `None`, it is returned as is. Otherwise, a new `Hashabledict`
    instance is created and returned.

    Parameters:
        dict_input (dict | None): The dictionary to convert.

    Returns:
        Hashabledict | None: The new hashable dictionary, or `None` if the input was `None`.
    """
    if dict_input is None:
        return dict_input

    return Hashabledict(dict_input)
