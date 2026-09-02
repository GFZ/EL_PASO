# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import calendar
import logging
import os
import re
import timeit
import typing
import warnings
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

import cdflib
import h5py
import netCDF4 as nC
import numpy as np
import pandas as pd
import xarray as xr
from packaging import version as version_pkg
from scipy.io.matlab import loadmat, savemat

import el_paso as ep

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from el_paso.typing import DataStandard, SavedDataDict, StandardName, TimeInterval

    DataDict = SavedDataDict

logger = logging.getLogger(__name__)


def get_el_paso_model_data_path() -> Path:
    """Return the directory used to store downloaded model coefficient data.

    Resolved from the `EL_PASO_MODEL_DATA_PATH` environment variable if set,
    otherwise defaults to `~/.elpaso/model_data`. The directory is created if
    it does not already exist.

    Returns:
        Path: Absolute path to the model data directory.
    """
    data_path = Path(os.environ.get("EL_PASO_MODEL_DATA_PATH", Path.home() / ".elpaso")).expanduser()
    data_path.mkdir(parents=True, exist_ok=True)

    return data_path


def get_el_paso_indices_solar_wind_param_path() -> Path:
    """Return the directory used to store downloaded solar wind index/parameter data.

    Resolved from the `EL_PASO_INDICES_SW_PARAM_DATA_PATH` environment variable
    if set, otherwise defaults to `~/.elpaso/`. The directory
    is created if it does not already exist.

    Returns:
        Path: Absolute path to the solar wind indices/parameters directory.
    """
    data_path = Path(os.environ.get("EL_PASO_INDICES_SW_PARAM_DATA_PATH", Path.home() / ".elpaso")).expanduser()
    data_path.mkdir(parents=True, exist_ok=True)

    return data_path


def fill_str_template_with_time(input_str: str, time: datetime) -> str:
    """Fills a string template with time-based placeholders.

    This function replaces common time-based placeholders in a string with
    the corresponding values from a `datetime` object. The placeholders
    are case-sensitive.

    Args:
        input_str (str): The input string containing placeholders like 'yyyymmdd', 'YYYYMMDD',
                         'YYYY', 'YY', 'MM', and 'DD'.
        time (datetime): The datetime object to use for filling the template.

    Returns:
        str: The string with all placeholders replaced by their time values.
    """
    yyyymmdd_str = time.strftime("%Y%m%d")
    yyyy_str = time.strftime("%Y")
    yy_str = time.strftime("%y")
    mm_str = time.strftime("%m")
    dd_str = time.strftime("%d")

    return (
        input_str.replace("yyyymmdd", yyyymmdd_str)
        .replace("YYYYMMDD", yyyymmdd_str)
        .replace("YYYY", yyyy_str)
        .replace("MM", mm_str)
        .replace("DD", dd_str)
        .replace("YY", yy_str)
    )


def extract_version(file_name: str | Path) -> tuple[str, version_pkg.Version]:
    """Extracts the version string from a file name.

    The function looks for a version string pattern `_v*` (e.g., '_v1.2.3' or '_v1_2-3')
    located just before the file extension. It returns the base file name and a
    parsed version object. If no version is found, it returns the original file name
    and a default version '0'.

    Args:
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

    Args:
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

    Args:
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
            name = func_name or f"{f.__name__}"  # ty:ignore[unresolved-attribute]
            log = logging.getLogger(f.__module__)
            log.info(f"{name} finished in {toc - tic:0.3f} seconds", stacklevel=2)

            return result

        return wrap

    return timed_function_


def enforce_utc_timezone(time: datetime) -> datetime:
    """Ensures a datetime object has UTC timezone information.

    If the provided datetime object is naive (lacks timezone info), it is assigned
    the UTC timezone. If it already has a timezone, it is returned unchanged.

    Args:
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

    Args:
        datenum_val (float): The MATLAB datenum value.

    Returns:
        datetime: The converted datetime object with UTC timezone.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Discarding nonzero nanoseconds", category=UserWarning)

        return (
            pd.to_datetime(datenum_val - 719529, unit="D", origin=pd.Timestamp("1970-01-01"))
            .to_pydatetime()
            .replace(tzinfo=timezone.utc)
        )


def datetime_to_datenum(datetime_val: datetime) -> float:
    """Converts a datetime object to a MATLAB datenum value.

    This function calculates the datenum value, which represents the number of days
    since year 0, including a fractional component for the time of day.

    Args:
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

    Args:
        var (ep.Variable): The variable instance to check.
        n_dims (int): The expected number of dimensions.
        name_in_file (str): The name of the variable, used in the error message.
    """
    provided = var.get_data().ndim

    if provided != n_dims:
        msg = (
            f"Encountered dimension missmatch for variable with name {name_in_file}: "
            f"should be {n_dims}, got: {provided}!"
        )
        raise ValueError(msg)


class Hashabledict(dict[Any, Any]):
    """A dictionary subclass that is hashable.

    This class enables a dictionary to be used in sets or as keys in other dictionaries
    by providing a custom hash implementation based on its contents.
    """

    def __hash__(self) -> int:
        """Computes a hash value for the dictionary.

        The hash is computed based on the frozensets of the dictionary's keys
        and values. This ensures that two `Hashabledict` instances with the same
        key-value pairs will have the same hash, regardless of the order of
        insertion.

        Returns:
            int: The hash value of the dictionary.
        """
        return hash((frozenset(self), frozenset(self.itervalues())))  # ty:ignore[unresolved-attribute]


def make_dict_hashable(dict_input: dict[Any, Any] | None) -> Hashabledict | None:
    """Converts a standard dictionary into a hashable one.

    If the input is `None`, it is returned as is. Otherwise, a new `Hashabledict`
    instance is created and returned.

    Args:
        dict_input (dict | None): The dictionary to convert.

    Returns:
        Hashabledict | None: The new hashable dictionary, or `None` if the input was `None`.
    """
    if dict_input is None:
        return dict_input

    return Hashabledict(dict_input)


def load_h5_data(file_path: Path) -> dict[StandardName, Any]:
    """Load all datasets and dataset attributes from an HDF5 file.

    Groups are flattened into slash-delimited paths (e.g. ``"group/dataset"``),
    mirroring the hierarchy written by :func:`write_h5_file`.

    Args:
        file_path (Path): Path to the HDF5 file to load.

    Returns:
        dict[StandardName, Any]: Mapping from flattened dataset path to its data
        as a NumPy array, plus a ``"metadata"`` entry mapping each path to its
        HDF5 attributes.
    """
    loaded_data: dict[StandardName, Any] = {"metadata": {}}

    def _recursively_load_datasets(group: h5py.Group | h5py.File, prefix: str = "") -> None:
        for key, item in group.items():
            full_path = f"{prefix}{key}" if prefix else key
            if isinstance(item, h5py.Dataset):
                loaded_data[full_path] = np.array(item)  # ty:ignore[invalid-assignment]
                loaded_data["metadata"][full_path] = dict(item.attrs.items())
            elif isinstance(item, h5py.Group):
                _recursively_load_datasets(item, f"{full_path}/")

    with h5py.File(file_path, "r") as file:
        _recursively_load_datasets(file)

    return loaded_data


def load_netcdf_data(file_path: Path, target_var_names: list[str] | None = None) -> dict[StandardName, Any]:
    """Load all variables and variable metadata from a NetCDF file.

    Groups are flattened into slash-delimited paths (e.g. ``"group/variable"``),
    mirroring the hierarchy written by :func:`write_netcdf_file`. Loading is
    eager: all matched variables are read fully into memory as NumPy arrays.

    Args:
        file_path (Path): Path to the NetCDF file to load.
        target_var_names (list[str] | None): If provided, only variables whose
            (unprefixed) name is in this list are loaded. If ``None``, all
            variables are loaded.

    Returns:
        dict[StandardName, Any]: Mapping from flattened variable path to its data
        as a NumPy array, plus a ``"metadata"`` entry mapping each path to a dict
        of known attributes (unit, source_files, processing_notes, description,
        original_cadence_seconds, standard_name). Returns an empty dict if
        ``file_path`` does not exist.
    """
    loaded_data: dict[StandardName, Any] = {"metadata": {}}

    def _recursively_load(group: nC.Group | nC.Dataset, prefix: str = "") -> None:
        for var_name, variable in group.variables.items():
            if not target_var_names or var_name in target_var_names:
                full_path = f"{prefix}{var_name}" if prefix else var_name
                loaded_data[full_path] = np.array(variable[:])  # ty:ignore[invalid-assignment]
                loaded_data["metadata"][full_path] = {
                    "unit": getattr(variable, "units", "unknown"),
                    "source_files": getattr(variable, "source", "unknown"),
                    "processing_notes": getattr(variable, "history", "unknown"),
                    "description": getattr(variable, "description", "unknown"),
                    "original_cadence_seconds": getattr(variable, "original_cadence_seconds", "unknown"),
                    "standard_name": getattr(variable, "standard_name", "unknown"),
                }

        for group_name, subgroup in group.groups.items():
            _recursively_load(subgroup, f"{prefix}{group_name}/")

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return {}

    with nC.Dataset(file_path, "r", format="NETCDF4") as file:
        _recursively_load(file)

    return loaded_data


def load_netcdf_data_lazy(file_path: Path) -> dict[StandardName, Any]:
    """Load all variables and variable metadata from a NetCDF file lazily using xarray.

    Unlike :func:`load_netcdf_data`, variable data is returned as lazily-loaded
    xarray ``DataArray`` objects rather than eagerly read into NumPy arrays,
    which avoids loading large datasets fully into memory.

    Args:
        file_path (Path): Path to the NetCDF file to load.

    Returns:
        dict[StandardName, Any]: Mapping from flattened variable path (group
        prefix included, e.g. ``"group/variable"``) to its data as an xarray
        ``DataArray``, plus a ``"metadata"`` entry mapping each path to a dict
        of known attributes (unit, source_files, processing_notes, description,
        original_cadence_seconds, standard_name). Returns an empty dict if
        ``file_path`` does not exist.
    """
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return {}

    loaded_data: dict[StandardName, Any] = {"metadata": {}}
    grouped_datasets = xr.open_groups(file_path)

    for group_path, ds in grouped_datasets.items():
        prefix = f"{group_path}/" if group_path != "/" else ""

        for var_name, data_array in ds.variables.items():
            full_path = f"{prefix}{var_name}"
            loaded_data[full_path] = data_array  # ty:ignore[invalid-assignment]

            attrs = data_array.attrs
            loaded_data["metadata"][full_path] = {
                "unit": attrs.get("units", "unknown"),
                "source_files": attrs.get("source", "unknown"),
                "processing_notes": attrs.get("history", "unknown"),
                "description": attrs.get("description", "unknown"),
                "original_cadence_seconds": attrs.get("original_cadence_seconds", "unknown"),
                "standard_name": attrs.get("standard_name", "unknown"),
            }

    return loaded_data


def load_cdf_data(file_path: Path) -> dict[StandardName, Any]:
    """Load all zVariables from an existing CDF file.

    zVariables with no records are skipped with a warning rather than raised,
    since :func:`write_cdf_file` never writes empty variables.

    Args:
        file_path (Path): Path to the CDF file to load.

    Returns:
        dict[StandardName, Any]: Mapping from zVariable name to its data as a
        NumPy array, plus a ``"metadata"`` entry mapping each variable name to
        its CDF variable attributes (empty dict if attributes could not be read).
    """
    loaded_data: dict[StandardName, Any] = {"metadata": {}}
    cdf_file = cdflib.CDF(str(file_path))
    try:
        info = cdf_file.cdf_info()
        z_variables = getattr(info, "zVariables", None)
        if z_variables is None and isinstance(info, dict):
            z_variables = info.get("zVariables", [])

        for variable_name in z_variables or []:
            try:
                loaded_data[variable_name] = np.asarray(cdf_file.varget(variable_name))
            except ValueError as exc:
                if "No records found" not in str(exc):
                    raise
                logger.warning(f"Skipping empty CDF variable {variable_name} in {file_path.name}")
                continue

            try:
                loaded_data["metadata"][variable_name] = cdf_file.varattsget(variable_name)
            except Exception:  # noqa: BLE001
                loaded_data["metadata"][variable_name] = {}
    finally:
        close = getattr(cdf_file, "close", None)
        if close is not None:
            close()

    return loaded_data


def load_mat_data(file_path: Path) -> dict[StandardName, Any]:
    """Load an existing MATLAB file.

    Args:
        file_path (Path): Path to the .mat file to load.

    Returns:
        dict[StandardName, Any]: Mapping from variable name to its data, with
        MATLAB's internal ``__*__`` entries stripped. If present, the
        ``"metadata"`` entry's per-variable attribute dicts have their NumPy
        array values converted to plain Python scalars/lists (or ``""`` for
        empty arrays) for JSON/MATLAB-struct compatibility.
    """
    loaded = loadmat(str(file_path), simplify_cells=True)
    data: dict[StandardName, Any] = {key: value for key, value in loaded.items() if not key.startswith("__")}

    if "metadata" in data and isinstance(data["metadata"], dict):
        for var_key, attrs in data["metadata"].items():
            if not isinstance(attrs, dict):
                continue
            data["metadata"][var_key] = {
                k: v.item()
                if isinstance(v, np.ndarray) and v.ndim == 0
                else v.tolist()
                if isinstance(v, np.ndarray) and v.size != 0
                else ""
                if isinstance(v, np.ndarray) and v.size == 0
                else v
                for k, v in attrs.items()
            }

    return data


def normalize_file_format(file_format: str) -> str:
    """Return a normalized file extension for the requested monthly format.

    Args:
        file_format (str): A file extension or format name, with or without a
            leading dot (e.g. ``"nc"``, ``".NC"``).

    Returns:
        str: The lowercased extension with a leading dot (e.g. ``".nc"``).

    Raises:
        ValueError: If the normalized extension is not one of ``.nc``, ``.cdf``,
            ``.h5``, or ``.mat``.
    """
    normalized = file_format.lower()
    if not normalized.startswith("."):
        normalized = f".{normalized}"

    if normalized not in {".nc", ".cdf", ".h5", ".mat"}:
        msg = "MonthlyRBStrategy supports only 'nc', 'cdf', 'h5', and 'mat' formats."
        raise ValueError(msg)

    return normalized


def write_mat_file(file_path: Path, data_dict: DataDict, data_standard: DataStandard) -> None:
    """Write a MATLAB file, resolving standard variable paths and flattening hierarchy.

    Data variables are stored under their flattened canonical names (``/`` → ``__``).
    Per-variable metadata is stored in a parallel ``metadata`` struct whose field
    names mirror the data variable names, matching how HDF5 stores attrs per dataset.

    Args:
        file_path (Path): Destination path for the .mat file.
        data_dict (DataDict): Data to save, keyed by internal name. The
            ``"metadata"`` key, if present, maps internal names to per-variable
            attribute dicts.
        data_standard (DataStandard): Used to resolve each internal name to its
            standard (canonical) name for the on-disk variable name.
    """
    mat_dict: dict[str, Any] = {}
    mat_metadata: dict[str, Any] = {}

    for internal_name, value in data_dict.items():
        if internal_name == "metadata":
            continue

        path = data_standard.get_standard_name(internal_name)
        mat_var_name = path.replace("/", "__")

        value_to_write = value
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == 1:
            value_to_write = value.reshape(-1)

        mat_dict[mat_var_name] = value_to_write

        # Attach per-variable metadata under a matching key in the metadata struct,
        # mirroring how _write_h5_file stores attrs on each dataset.
        variable_meta = data_dict.get("metadata", {}).get(internal_name, {})
        if isinstance(variable_meta, dict) and variable_meta:
            mat_metadata[mat_var_name] = {
                "unit": variable_meta.get("unit", "unknown"),
                "source_files": variable_meta.get("source_files", "unknown"),
                "processing_notes": variable_meta.get("processing_notes", "unknown"),
                "description": variable_meta.get("description", "unknown"),
                "original_cadence_seconds": variable_meta.get("original_cadence_seconds", "unknown"),
                "standard_name": variable_meta.get("standard_name", "unknown"),
            }

    if mat_metadata:
        mat_dict["metadata"] = mat_metadata

    savemat(str(file_path), mat_dict)


def write_h5_file(file_path: Path, data_dict: SavedDataDict, data_standard: DataStandard) -> None:
    """Write an HDF5 file with hierarchical groups from slash-delimited paths.

    Each variable's standard name is split on ``/`` to build a nested group
    structure, with the final path component used as the dataset name.

    Args:
        file_path (Path): Destination path for the .h5 file.
        data_dict (SavedDataDict): Data to save, keyed by internal name. The
            ``"metadata"`` key, if present, maps internal names to per-variable
            attribute dicts written as HDF5 dataset attributes.
        data_standard (DataStandard): Used to resolve each internal name to its
            standard (canonical) name, which determines the group/dataset path.
    """
    with h5py.File(file_path, "w") as file:
        for internal_name, value in data_dict.items():
            if internal_name == "metadata":
                continue
            path = data_standard.get_standard_name(internal_name)

            path_parts = path.split("/")
            groups = path_parts[:-1]
            dataset_name = path_parts[-1]

            curr_hierarchy = file
            for group in groups:
                if group not in curr_hierarchy:
                    curr_hierarchy = curr_hierarchy.create_group(group)
                else:
                    curr_hierarchy = cast("h5py.Group", curr_hierarchy[group])

            # Normalize 2D arrays with shape (n, 1) back to 1D for consistency with other formats
            value_to_write = value
            if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == 1:
                value_to_write = value.reshape(-1)

            data_set = curr_hierarchy.create_dataset(
                dataset_name, data=value_to_write, compression="gzip", shuffle=True
            )

            metadata_dict = data_dict.get("metadata", {}).get(internal_name, {})
            if not isinstance(metadata_dict, dict):
                continue

            for key, metadata in metadata_dict.items():
                if getattr(metadata, "size", None) == 0:
                    continue
                data_set.attrs[key] = metadata


def _write_data_to_netcdf_file(file: nC.Dataset | nC.Group, data_dict: DataDict, data_standard: DataStandard) -> None:
    """Write variables to a NetCDF file or group.

    Args:
        file (nC.Dataset | nC.Group): The NetCDF dataset or group to write into.
            Dimensions referenced by variables must already be created.
        data_dict (DataDict): Data to write, keyed by internal name. Variables
            with zero size are skipped. The ``"metadata"`` key, if present, maps
            internal names to per-variable attribute dicts.
        data_standard (DataStandard): Used to resolve each internal name to its
            standard (canonical) name and dimension names.
    """
    for internal_name, value in data_dict.items():
        if internal_name == "metadata":
            continue

        value_array = np.asarray(value)
        if value_array.size == 0:
            continue

        standard_name = data_standard.get_standard_name(internal_name)

        path_parts = standard_name.split("/")
        groups = path_parts[:-1]
        dataset_name = path_parts[-1]

        curr_hierarchy: nC.Group | nC.Dataset = file
        for group in groups:
            if group not in curr_hierarchy.groups:
                curr_hierarchy = curr_hierarchy.createGroup(group)
            else:
                curr_hierarchy = curr_hierarchy.groups[group]

        dimensions = data_standard.get_dependencies(internal_name)
        data_set = cast(
            "nC.Variable[Any]",
            curr_hierarchy.createVariable(
                dataset_name,
                "float64",
                dimensions,
                zlib=True,
                complevel=5,
                shuffle=True,
            ),
        )

        value_to_write = value_array
        if len(dimensions) == 1 and value_array.ndim == 2 and value_array.shape[1] == 1:
            value_to_write = value_array.reshape(-1)

        data_set[:] = value_to_write

        metadata_dict = data_dict.get("metadata", {})
        metadata = {}
        if isinstance(metadata_dict, dict):
            metadata = metadata_dict.get(internal_name, metadata_dict.get(internal_name, {}))

        if not isinstance(metadata, dict):
            continue

        valid_internal_names = {
            arg for names in typing.get_args(ep.typing.InternalName) for arg in typing.get_args(names)
        }

        coordinates = [
            data_standard.get_standard_name(int_name)  # ty:ignore[invalid-argument-type]
            for int_name in data_standard.get_dependencies(internal_name)
            if int_name in valid_internal_names
        ]

        data_set.coordinates = " ".join(coordinates)
        data_set.units = metadata.get("unit", "unknown")
        data_set.source = metadata.get("source_files", "unknown")
        data_set.history = metadata.get("processing_notes", "unknown")
        data_set.description = metadata.get("description", "unknown")
        data_set.original_cadence_seconds = metadata.get("original_cadence_seconds", "unknown")
        data_set.standard_name = metadata.get("standard_name", "unknown")


def write_netcdf_file(file_path: Path, data_dict: DataDict, data_standard: DataStandard) -> None:
    """Create and write a NetCDF file from a data dictionary.

    The "Epoch" dimension is created as unlimited so the file can later be
    appended to via :meth:`el_paso.saving_strategy.SavingStrategy.append_data`.

    Args:
        file_path (Path): Destination path for the .nc file.
        data_dict (DataDict): Data to save, keyed by internal name, including an
            "Epoch" entry. If "Epoch" has length 0, the write is skipped entirely.
        data_standard (DataStandard): Used to resolve each internal name to its
            standard (canonical) name, dimensions, and dimension sizes.
    """
    with nC.Dataset(file_path, "w", format="NETCDF4") as file:
        size_time = np.asarray(data_dict["Epoch"]).shape[0]
        if size_time == 0:
            logger.info(f"Skipping write for {file_path.name} (time has length 0).")
            return

        dimensions = _calculate_dimensions(data_dict, data_standard)
        for dim_name, dim_size in dimensions.items():
            if dim_name == "Epoch":
                # we create the time dimension as unilimited to allow for append later on
                file.createDimension(dim_name, size=None)
            else:
                file.createDimension(dim_name, dim_size)

        _write_data_to_netcdf_file(file, data_dict, data_standard)


def _calculate_dimensions(data_dict: DataDict, data_standard: DataStandard) -> dict[str, int]:
    """Calculate NetCDF dimension sizes from the data dictionary.

    Args:
        data_dict (DataDict): Data to inspect, keyed by internal name.
        data_standard (DataStandard): Used to resolve each internal name's
            dependent dimension names.

    Returns:
        dict[str, int]: Mapping from dimension name to its size, with
        "min_max" and "Position_components" special-cased to 2 and 3
        respectively.
    """
    unique_dims = {}

    for internal_name in data_dict:
        if internal_name == "metadata":
            continue
        dim_names = data_standard.get_dependencies(internal_name)

        for dim_name in dim_names:
            if dim_name not in unique_dims:
                # handle special cases
                if dim_name == "min_max":
                    unique_dims[dim_name] = 2
                elif dim_name == "Position_components":
                    unique_dims[dim_name] = 3
                elif dim_name in data_dict:
                    dims_of_dim = data_standard.get_dependencies(dim_name)

                    target_idx = np.where(dim_name == np.asarray(dims_of_dim))[0][0]  # ty:ignore[no-matching-overload]

                    if data_dict[dim_name].ndim <= target_idx:
                        unique_dims[dim_name] = 1  # dimesion of size 1 can be collapsed
                    else:
                        unique_dims[dim_name] = data_dict[dim_name].shape[target_idx]

    return unique_dims


def _get_cdf_variable_attrs(var_name: str, data_dict: DataDict) -> DataDict:
    """Return non-empty CDF variable attributes for a saved variable."""
    metadata = data_dict.get("metadata", {}).get(var_name, {})
    var_attrs: SavedDataDict = {}

    if isinstance(metadata, dict):
        for attr_name, attr_value in metadata.items():
            if _is_empty_cdf_attribute(attr_value):
                logger.debug(f"Skipping empty CDF attribute {var_name}:{attr_name}")
                continue

            var_attrs[str(attr_name)] = attr_value  # ty:ignore[invalid-assignment]

    var_attrs["Compress"] = 6  # ty:ignore[invalid-assignment]
    return var_attrs


def _is_empty_cdf_attribute(value: Any) -> bool:  # noqa: ANN401
    """Return True if cdflib cannot infer a datatype from the attribute value."""
    if value is None:
        return True

    if isinstance(value, (list, tuple, dict, str, bytes)):
        return len(value) == 0

    return getattr(value, "size", None) == 0


def write_cdf_file(file_path: Path, data_dict: DataDict, data_standard: DataStandard) -> None:
    """Write a CDF file, resolving standard variable paths and embedding metadata.

    Args:
        file_path (Path): Destination path for the .cdf file. An existing file
            at this path is deleted first.
        data_dict (DataDict): Data to save, keyed by internal name. Variables
            with zero size are skipped. The ``"metadata"`` key, if present, maps
            internal/standard names to per-variable attribute dicts.
        data_standard (DataStandard): Used to resolve each internal name to its
            standard (canonical) name for the on-disk zVariable name.

    Raises:
        RuntimeError: If writing the CDF file fails for any reason.
    """
    try:
        cdf_file = cdflib.cdfwrite.CDF(str(file_path), delete=True)
        try:
            for internal_name, var_data in data_dict.items():
                if internal_name == "metadata":
                    continue

                if getattr(var_data, "size", 0) == 0:
                    logger.warning(f"Skipping empty variable {internal_name}")
                    continue

                # Resolve the canonical name via the data standard, matching H5/NC behaviour.
                # CDF does not support '/' in variable names, so we replace path separators
                # with '__' to preserve hierarchy information without violating the spec.
                path = data_standard.get_standard_name(internal_name)
                cdf_var_name = path
                value_to_write = var_data
                if isinstance(var_data, np.ndarray) and var_data.ndim == 2 and var_data.shape[1] == 1:
                    value_to_write = var_data.reshape(-1)

                var_data_array = np.asarray(value_to_write)
                if np.issubdtype(var_data_array.dtype, np.integer):
                    if var_data_array.dtype == np.int8:
                        cdf_dtype = cdflib.cdfwrite.CDF.CDF_INT1
                    elif var_data_array.dtype == np.int16:
                        cdf_dtype = cdflib.cdfwrite.CDF.CDF_INT2
                    elif var_data_array.dtype == np.int32:
                        cdf_dtype = cdflib.cdfwrite.CDF.CDF_INT4
                    else:
                        cdf_dtype = cdflib.cdfwrite.CDF.CDF_INT8
                elif np.issubdtype(var_data_array.dtype, np.floating):
                    cdf_dtype = (
                        cdflib.cdfwrite.CDF.CDF_FLOAT
                        if var_data_array.dtype == np.float32
                        else cdflib.cdfwrite.CDF.CDF_DOUBLE
                    )
                else:
                    var_data_array = var_data_array.astype(np.float64)
                    cdf_dtype = cdflib.cdfwrite.CDF.CDF_DOUBLE

                var_spec: dict[str, Any] = {
                    "Variable": cdf_var_name,
                    "Data_Type": cdf_dtype,
                    "Num_Elements": 1,
                    "Rec_Vary": True,
                    "Dim_Sizes": (list(var_data_array.shape[1:]) if var_data_array.ndim > 1 else []),
                }

                metadata_dict = data_dict.get("metadata", {})
                metadata: dict[str, Any] = {}
                if isinstance(metadata_dict, dict):
                    metadata = metadata_dict.get(path, metadata_dict.get(internal_name, {}))

                var_attrs = {}
                if isinstance(metadata, dict):
                    for attr_name, attr_value in metadata.items():
                        if _is_empty_cdf_attribute(attr_value):
                            logger.debug(f"Skipping empty CDF attribute {cdf_var_name}:{attr_name}")
                            continue
                        var_attrs[str(attr_name)] = attr_value
                if isinstance(metadata, dict):
                    for field, nc_key in {
                        "unit": "unit",
                        "source_files": "source_files",
                        "processing_notes": "processing_notes",
                        "description": "description",
                        "original_cadence_seconds": "original_cadence_seconds",
                        "standard_name": "standard_name",
                    }.items():
                        value = metadata.get(nc_key)
                        if value is None or _is_empty_cdf_attribute(value):
                            var_attrs.setdefault(nc_key, "empty")
                            continue
                        if value and not _is_empty_cdf_attribute(value):
                            var_attrs.setdefault(field, value)

                var_attrs["Compress"] = 6

                cdf_file.write_var(var_spec, var_attrs=var_attrs, var_data=var_data_array)
        finally:
            cdf_file.close()
    except Exception as e:
        msg = f"Failed to write CDF file {file_path}: {e}"
        logger.exception(msg)
        raise RuntimeError(msg) from e


def get_monthly_datetime_intervals(start_time: datetime | None, end_time: datetime | None) -> list[TimeInterval]:
    """Splits a time range into a list of full calendar-month intervals.

    Each interval spans from the first second of a month to the last second
    of that same month, in UTC. The first and last intervals are not clipped
    to ``start_time``/``end_time``; they always cover the full calendar month.

    Args:
        start_time (datetime | None): The start of the time range. Must not be None.
        end_time (datetime | None): The end of the time range. Must not be None.

    Returns:
        list[TimeInterval]: One ``(month_start, month_end)`` tuple per calendar
        month overlapping ``[start_time, end_time]``.

    Raises:
        ValueError: If ``start_time`` or ``end_time`` is None.
    """
    time_intervals: list[TimeInterval] = []

    if start_time is None or end_time is None:
        msg = "start_time and end_time must be provided!"
        logger.error(msg)
        raise ValueError(msg)

    current_time = start_time.replace(day=1)
    while current_time <= end_time:
        year = current_time.year
        month = current_time.month
        eom_day = calendar.monthrange(year, month)[1]

        month_start = datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc)
        month_end = datetime(year, month, eom_day, 23, 59, 59, tzinfo=timezone.utc)
        time_intervals.append((month_start, month_end))
        current_time = (
            datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            if month == 12
            else datetime(year, month + 1, 1, tzinfo=timezone.utc)
        )
    return time_intervals
