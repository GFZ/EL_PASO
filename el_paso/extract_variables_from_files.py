# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, cast  # noqa: UP035

import cdflib
import h5py
import netCDF4
import numpy as np
import pandas as pd

from el_paso import Variable
from el_paso.utils import enforce_utc_timezone, fill_str_template_with_time, get_file_by_version

if TYPE_CHECKING:
    from collections.abc import Iterable

    from astropy import units as u
    from numpy.typing import DTypeLike, NDArray

    from el_paso.typing import TimeInterval

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, eq=False)
class ExtractionInfo:
    """Store metadata required to extract a variable from a source file.

    Attributes:
        name_or_column:
            Name of the variable or column to extract from the source file.

        unit:
            Physical unit associated with the extracted variable.

        is_time_dependent:
            Whether the variable is time-dependent.

            If ``True``, data from multiple files will be concatenated
            along the time axis.

            If ``False``, data from multiple files will be used to fill
            missing (`np.nan`) values instead of being concatenated.

        result_key:
            Key to use for the extracted variable in the resulting
            variables dictionary.

            If ``None``, ``name_or_column`` is used as the key.

        dependent_variables:
            Names of variables that the extracted variable depends on.

            This is mainly used for JSON extraction to determine how
            extracted data should be reshaped.

        np_dtype:
            Optional NumPy dtype used to cast the extracted data.

            If ``None``, the dtype is inferred from the source data.
    """

    name_or_column: str | int
    unit: u.UnitBase
    is_time_dependent: bool = True
    result_key: str | None = None
    dependent_variables: list[str] | None = None
    np_dtype: DTypeLike | None = None


def extract_variables_from_files(
    start_time: datetime,
    end_time: datetime,
    file_cadence: Literal["daily", "monthly", "single_file"],
    data_path: Path | str,
    file_name_stem: str,
    extraction_infos: Iterable[ExtractionInfo],
    pd_read_csv_kwargs: dict[str, Any] | None = None,
    custom_extractors: dict[str, Callable] | None = None,
) -> dict[str, Variable]:
    """Extract variable data from files with any file format.

    Args:
        start_time (datetime): The start time for data extraction.
        end_time (datetime): The end time for data extraction.
        file_cadence (Literal["daily", "monthly", "single_file"]): The cadence at which files are organized.
        data_path (Path or str): The directory path where data files are stored.
        file_name_stem (str): The stem of the file name to match files.
        extraction_infos (Iterable[ExtractionInfo]): Information about which variables to extract and how.
        pd_read_csv_kwargs (dict[str, Any], optional): Additional keyword arguments to pass to pandas.read_csv.
        custom_extractors (dict[str, Callable], optional): A dictionary mapping file suffixes to custom extractor functions.

    Returns:
        dict[str, Variable]: A dictionary mapping result keys to extracted Variable objects.

    Raises:
        ValueError: If no files are found for extraction.

    """  # noqa: E501
    logger.info("Extracting variables ...")

    if pd_read_csv_kwargs is None:
        pd_read_csv_kwargs = {}

    start_time = enforce_utc_timezone(start_time)
    end_time = enforce_utc_timezone(end_time)

    if start_time > end_time:
        msg = "start_time must be before end_time!"
        logger.error(msg)
        raise ValueError(msg)

    data_path = Path(data_path)

    files_list, _ = _construct_file_list(start_time, end_time, file_cadence, data_path / file_name_stem)

    if len(files_list) == 0:
        msg = f"No file found to extract variables! Search at: {data_path / file_name_stem}"
        logger.error(msg)
        raise ValueError(msg)

    variable_data = _extract_data_from_files(files_list, extraction_infos, pd_read_csv_kwargs, custom_extractors)

    # create variables based on the extraction_infos
    variables: dict[str, Variable] = {}

    for info in extraction_infos:
        if info.result_key is None:
            if isinstance(info.name_or_column, str):
                dict_key = info.name_or_column
            else:
                msg = "Result key cannot be inferred from a integer column! Please provide a result_key!"
                logger.error(msg)
                raise ValueError(msg)
        else:
            dict_key = info.result_key
        variables[dict_key] = Variable(original_unit=info.unit, data=variable_data[info.name_or_column])
        variables[dict_key].metadata.source_files = [path.name for path in files_list]

    return variables


def _construct_file_list(
    start_time: datetime, end_time: datetime, file_cadence: Literal["daily", "monthly", "single_file"], file_path: Path
) -> tuple[list[Path], list[TimeInterval]]:
    file_paths: list[Path] = []
    time_intervals: list[TimeInterval] = []

    match file_cadence:
        case "daily":
            current_time = start_time
            while current_time <= end_time:
                file_path_current = _fill_file_name_and_check_version(current_time, file_path)
                if file_path_current is None:
                    logger.warning(
                        (
                            f"No file found for {current_time.strftime('%Y-%m-%d')} under path: {file_path}."
                            "Skipping this day."
                        ),
                        stacklevel=2,
                    )
                else:
                    file_paths.append(file_path_current)

                    day_start = datetime(
                        current_time.year, current_time.month, current_time.day, 0, 0, 0, tzinfo=timezone.utc
                    )
                    day_end = datetime(
                        current_time.year, current_time.month, current_time.day, 23, 59, 59, tzinfo=timezone.utc
                    )
                    time_intervals.append((day_start, day_end))

                current_time += timedelta(days=1)

        case "monthly":
            msg = "Monthly file cadence is not implemented yet!"
            logger.error(msg)
            raise NotImplementedError(msg)

        case "single_file":
            file_path_current = _fill_file_name_and_check_version(start_time, file_path)
            if file_path_current is None:
                msg = f"No file found under the specified path: {file_path}. Please check your data_path."
                logger.error(msg)
                raise FileNotFoundError(msg)

            file_paths.append(file_path_current)
            time_intervals.append((start_time, end_time))

    return file_paths, time_intervals


def _extract_data_from_files(
    files_list: list[Path],
    extraction_infos: Iterable[ExtractionInfo],
    pd_read_csv_kwargs: dict[str, Any] | None,
    custom_extractors: dict[str, Callable] | None,
) -> dict[str | int, NDArray[np.generic]]:
    extraction_infos = tuple(extraction_infos)

    variable_data = {info.name_or_column: np.array([]) for info in extraction_infos}

    custom_extractor_msg = f" If you have a custom extractor, pass it via the `custom_extractors` argument with the key '{files_list[0].suffix}'."  # noqa: E501

    EXTRACTOR_MAP: dict[str, Callable[..., dict[str | int, NDArray[np.generic]]]] = {
        ".cdf": lambda path: _extract_data_from_cdf(path, extraction_infos),
        ".txt": lambda path: _extract_data_from_ascii(path, extraction_infos, pd_read_csv_kwargs),
        ".asc": lambda path: _extract_data_from_ascii(path, extraction_infos, pd_read_csv_kwargs),
        ".csv": lambda path: _extract_data_from_ascii(path, extraction_infos, pd_read_csv_kwargs),
        ".tab": lambda path: _extract_data_from_ascii(path, extraction_infos, pd_read_csv_kwargs),
        ".dat": lambda path: _extract_data_from_ascii(path, extraction_infos, pd_read_csv_kwargs),
        ".h5": lambda path: _extract_data_from_h5(path, extraction_infos),
        ".nc": lambda path: _extract_data_from_netcdf(path, extraction_infos),
        ".json": lambda path: _extract_data_from_json(path, extraction_infos),
    }

    unsupported_formats = {
        ".mat": "MATLAB reading is not supported yet!",
    }

    for file_path in files_list:
        suffix = file_path.suffix

        if custom_extractors is not None and suffix in custom_extractors:
            extractor = custom_extractors[suffix]
            new_data = extractor(str(file_path), extraction_infos)

        elif suffix in EXTRACTOR_MAP:
            new_data = EXTRACTOR_MAP[suffix](str(file_path))

        elif suffix in unsupported_formats:
            msg = unsupported_formats[suffix] + custom_extractor_msg.format(suffix=suffix)
            logger.error(msg)
            raise NotImplementedError(msg)

        else:
            msg = f"Unsupported file extension: {suffix}"
            logger.error(msg)
            raise ValueError(msg)

        # Update the data content of variables
        for info in extraction_infos:
            key = info.name_or_column

            if info.np_dtype is not None:
                variable_data[key] = variable_data[key].astype(info.np_dtype)

            if variable_data[key].size == 0:
                variable_data[key] = new_data[key]

            elif info.is_time_dependent:
                logger.debug(f"Concatenating data for {key} ...")
                variable_data[key] = np.concatenate((variable_data[key], new_data[key]), axis=0)

            elif np.any(pd.isna(variable_data[key])):
                logger.debug(f"Found NaNs in non-time-dependent variable {key}. Trying to fill with next file ...")
                nan_idx = np.isnan(variable_data[key])
                variable_data[key] = new_data[key][nan_idx]

    return variable_data


def _fill_file_name_and_check_version(time: datetime, file_path: Path) -> Path | None:
    file_path = Path(fill_str_template_with_time(str(file_path), time))

    file_names_all_versions = _find_files_with_regex(file_path.parent, file_path.name)

    file_path_latest = get_file_by_version(file_names_all_versions, version="latest")

    if file_path_latest is None:
        warnings.warn(
            (
                f"No file found under path: {file_path.parent}! Have you called download()?"
                "Check your download_path argument."
            ),
            stacklevel=2,
        )

    return file_path_latest


def _extract_data_from_ascii(
    file_path: str, extraction_infos: tuple[ExtractionInfo, ...], pd_read_csv_kwargs: dict[str, Any] | None
) -> dict[str | int, NDArray[np.generic]]:
    pd_read_csv_kwargs = pd_read_csv_kwargs if pd_read_csv_kwargs is not None else {}

    file_df = pd.read_csv(file_path, **pd_read_csv_kwargs)

    data: dict[str | int, NDArray[np.generic]] = {}

    for info in extraction_infos:
        if isinstance(info.name_or_column, int):
            data[info.name_or_column] = file_df.iloc[:, info.name_or_column].to_numpy()
        elif isinstance(info.name_or_column, str):
            data[info.name_or_column] = file_df.loc[:, info.name_or_column].to_numpy()
        else:
            msg = f"Encountered invalid name_or_column value: {info.name_or_column}! Must be int or str!"
            raise TypeError(msg)

    return data


def _extract_data_from_json(
    file_path: str, extraction_infos: tuple[ExtractionInfo, ...]
) -> dict[str | int, NDArray[np.generic]]:
    logger.info("Enountered JSON file. Please specify your dependent variables!")

    with Path(file_path).open("r") as f:
        data = json.load(f)

    json_df = pd.DataFrame(data)

    data: dict[str | int, NDArray[np.generic]] = {}

    for info in extraction_infos:
        if info.name_or_column in json_df:
            if info.dependent_variables is None:
                data[info.name_or_column] = np.array(pd.unique(json_df[info.name_or_column]))
            else:
                dependent_data = [json_df[dep_var_name] for dep_var_name in info.dependent_variables]

                unique_values: list[NDArray[np.generic]] = [pd.unique(dep) for dep in dependent_data]
                shape = tuple(len(uq) for uq in unique_values)

                # Determine the correct dtype for the data_array based on the column data type
                dtype = object if json_df[info.name_or_column].dtype == object else np.float64

                data_array = np.full(shape, np.nan, dtype=dtype)

                for indices in np.ndindex(*shape):
                    mask = np.ones(len(json_df), dtype=bool)
                    for i, idx in enumerate(indices):
                        mask &= dependent_data[i] == unique_values[i][idx]
                    if mask.any():
                        data_array[indices] = json_df[info.name_or_column][mask].to_numpy()[0]

                data[info.name_or_column] = data_array
        else:
            logger.warning(
                f"Variable with name_or_column_in_file: {info.name_or_column} was not found in file {file_path}!",
            )

    return data


def _extract_data_from_cdf(
    file_path: str, extraction_infos: tuple[ExtractionInfo, ...]
) -> dict[str | int, NDArray[np.generic]]:
    # Open the CDF file
    cdf_file = cdflib.CDF(file_path)
    cdfinfo = cdf_file.cdf_info()
    variable_data: dict[str | int, NDArray[np.generic]] = {}

    # Extract data for each variable in self.variables
    for info in extraction_infos:
        if info.name_or_column in cdfinfo.zVariables:
            # Retrieve data corresponding to the variable name from the CDF file
            var_content = cdf_file.varget(info.name_or_column)  # ty:ignore[invalid-argument-type]
            var_content = cast("NDArray[Any]", var_content)

            if isinstance(var_content, str):
                var_content = np.array([var_content])

            if np.issubdtype(var_content.dtype, np.floating) and "FILLVAL" in cdf_file.varattsget(info.name_or_column):
                fill_value = cdf_file.attget("FILLVAL", info.name_or_column).Data
                var_content[var_content == fill_value] = np.nan

            variable_data[info.name_or_column] = var_content
        else:
            logger.warning(f"Data with name {info.name_or_column} was not found in file {file_path}!", stacklevel=2)

    return variable_data


def _extract_data_from_h5(
    file_path: str, extraction_infos: tuple[ExtractionInfo, ...]
) -> dict[str | int, NDArray[np.generic]]:
    # Open the h5 file
    variable_data: dict[str | int, NDArray[np.generic]] = {}

    with h5py.File(file_path) as file:
        entries: list[str] = list(file.keys())

        # Extract data for each variable in self.variables
        for info in extraction_infos:
            if info.name_or_column in entries:
                entry_data = file[info.name_or_column]
                if isinstance(entry_data, h5py.Datatype):
                    msg = f"Encountered invalid DataType while extracting variables for entry: {info.name_or_column}"
                    logger.error(msg)
                    raise TypeError(msg)

                if isinstance(entry_data, h5py.Group):
                    msg = "Groups for h5 files has not been implemented! You can implement it as a custom extractor and"
                    " pass it via the `custom_extractors` argument."
                    logger.error(msg)
                    raise NotImplementedError(msg)

                var_content = cast("NDArray[np.generic]|str", entry_data[...])
                if isinstance(var_content, str):
                    # If the content is a string, convert it to a numpy array
                    var_content = np.array([var_content])
                variable_data[info.name_or_column] = var_content
            else:
                logger.warning(
                    (
                        f"Data with name {info.name_or_column} was not found in file {file_path}!"
                        "Available entries: {entries}"
                    ),
                    stacklevel=2,
                )

        return variable_data


def _extract_data_from_netcdf(
    file_path: str, extraction_infos: tuple[ExtractionInfo, ...]
) -> dict[str | int, NDArray[np.generic]]:
    variable_data: dict[str | int, NDArray[np.generic]] = {}

    with netCDF4.Dataset(file_path) as file:
        if file.groups:
            msg = "Groups for nc files has not been implemented! You can implement it as a custom extractor and"
            " pass it via the `custom_extractors` argument."
            logger.error(msg)
            raise NotImplementedError(msg)

        entries: list[str] = list(file.variables.keys())

        for info in extraction_infos:
            if info.name_or_column in entries:
                entry_data = file[info.name_or_column]  # ty: ignore[invalid-argument-type]
                var_content = np.asarray([entry_data]).squeeze()
                variable_data[info.name_or_column] = var_content
            else:
                logger.warning(
                    (
                        f"Data with name {info.name_or_column} was not found in file {file_path}!"
                        "Available entries: {entries}"
                    ),
                    stacklevel=2,
                )

    return variable_data


def _find_files_with_regex(directory: Path, regex_pattern: str) -> list[Path]:
    """Find files in a directory that match a given regex pattern."""
    compiled_regex = re.compile(regex_pattern)  # Compile for efficiency

    return [file for file in directory.glob("*") if compiled_regex.fullmatch(str(file.name))]
