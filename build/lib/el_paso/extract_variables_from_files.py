# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import re
import typing
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import cdflib  # type: ignore[reportMissingTypeStubs]
import h5py  # type: ignore[reportMissingTypeStubs]
import numpy as np
import pandas as pd

from el_paso import Variable
from el_paso.utils import enforce_utc_timezone, fill_str_template_with_time, get_file_by_version

if TYPE_CHECKING:
    from collections.abc import Iterable

    from astropy import units as u  # type: ignore[reportMissingTypeStubs]
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, eq=False)
class ExtractionInfo:
    """Class to store information about the extraction of variables from a file."""

    name_or_column: str | int
    unit: u.UnitBase
    is_time_dependent: bool = True
    result_key: str | None = None
    dependent_variables: list[str] | None = None


def extract_variables_from_files(
    start_time: datetime,
    end_time: datetime,
    file_cadence: Literal["daily", "monthly", "single_file"],
    data_path: Path | str,
    file_name_stem: str,
    extraction_infos: Iterable[ExtractionInfo],
    pd_read_csv_kwargs: dict[str, Any] | None = None,
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

    Returns:
        dict[str, Variable]: A dictionary mapping result keys to extracted Variable objects.

    Raises:
        ValueError: If no files are found for extraction.

    """
    logger.info("Extracting variables ...")

    if pd_read_csv_kwargs is None:
        pd_read_csv_kwargs = {}

    start_time = enforce_utc_timezone(start_time)
    end_time = enforce_utc_timezone(end_time)

    data_path = Path(data_path)

    files_list, _ = _construct_file_list(start_time, end_time, file_cadence, data_path / file_name_stem)

    if len(files_list) == 0:
        msg = "No file found to extract variables!"
        raise ValueError(msg)

    variable_data = _extract_data_from_files(files_list, extraction_infos, pd_read_csv_kwargs)

    # create variables based on the extraction_infos
    variables: dict[str, Variable] = {}

    for info in extraction_infos:
        if info.result_key is None:
            if isinstance(info.name_or_column, str):
                dict_key = info.name_or_column
            else:
                msg = "Result key cannot be inferred from a integer column! Please provide a result_key!"
                raise ValueError(msg)
        else:
            dict_key = info.result_key
        variables[dict_key] = Variable(original_unit=info.unit, data=variable_data[info.name_or_column])
        variables[dict_key].metadata.source_files = [path.name for path in files_list]

    return variables


def _construct_file_list(
    start_time: datetime, end_time: datetime, file_cadence: Literal["daily", "monthly", "single_file"], file_path: Path
) -> tuple[list[Path], list[tuple[datetime, datetime]]]:
    file_paths: list[Path] = []
    time_intervals: list[tuple[datetime, datetime]] = []

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
            raise NotImplementedError(msg)

        case "single_file":
            file_path_current = _fill_file_name_and_check_version(start_time, file_path)
            if file_path_current is None:
                msg = f"No file found under the specified path: {file_path}. Please check your data_path."
                raise FileNotFoundError(msg)

            file_paths.append(file_path_current)
            time_intervals.append((start_time, end_time))

    return file_paths, time_intervals


def _extract_data_from_files(  # noqa: C901
    files_list: list[Path],
    extraction_infos: Iterable[ExtractionInfo],
    pd_read_csv_kwargs: dict[str, Any] | None,
) -> dict[str | int, NDArray[np.generic]]:
    variable_data = {info.name_or_column: np.array([]) for info in extraction_infos}

    for file_path in files_list:
        match file_path.suffix:
            case ".cdf":
                new_data = _extract_data_from_cdf(str(file_path), tuple(extraction_infos))
            case ".txt" | ".asc" | ".csv" | ".tab":
                new_data = _extract_data_from_ascii(str(file_path), tuple(extraction_infos), pd_read_csv_kwargs)
            case ".nc":
                msg = "NetCDF reading is not supported yet!"
                raise NotImplementedError(msg)
            case ".h5":
                new_data = _extract_data_from_h5(str(file_path), tuple(extraction_infos))
            case ".json":
                new_data = _extract_data_from_json(str(file_path), tuple(extraction_infos))
            case ".mat":
                msg = "MATLAB .mat file reading is not supported yet!"
                raise NotImplementedError(msg)
            case _:
                msg = f"Unsupported file extension: {file_path.suffix}"
                raise ValueError(msg)

        # Update the data content of variables
        for info in extraction_infos:
            key = info.name_or_column
            if variable_data[key].size == 0:
                variable_data[key] = new_data[key]

            elif info.is_time_dependent:
                logger.debug(f"Concatenating data for {key} ...")
                variable_data[key] = np.concatenate((variable_data[key], new_data[key]), axis=0)

            elif np.any(np.isnan(variable_data[key])):
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

    file_df = pd.read_csv(file_path, **pd_read_csv_kwargs)  # type: ignore[reportUnknownMemberType]

    data: dict[str | int, NDArray[np.generic]] = {}

    for info in extraction_infos:
        if isinstance(info.name_or_column, int):
            data[info.name_or_column] = file_df.iloc[:, info.name_or_column].to_numpy()  # type: ignore[reportUnknownMemberType]
        elif isinstance(info.name_or_column, str):  # type: ignore[reportUnnecessaryIsInstance]
            data[info.name_or_column] = file_df.loc[:, info.name_or_column].to_numpy()  # type: ignore[reportUnknownMemberType]
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
                data[info.name_or_column] = np.array(pd.unique(json_df[info.name_or_column]))  # type: ignore[reportUnknownMemberType]
            else:
                dependent_data = [json_df[dep_var_name] for dep_var_name in info.dependent_variables]  # type: ignore[reportUnknownMemberType]

                unique_values: list[NDArray[np.generic]] = [pd.unique(dep) for dep in dependent_data]  # type: ignore[reportUnknownMemberType]
                shape = tuple(len(uq) for uq in unique_values)

                # Determine the correct dtype for the data_array based on the column data type
                dtype = object if json_df[info.name_or_column].dtype == object else np.float64

                data_array = np.full(shape, np.nan, dtype=dtype)

                for indices in np.ndindex(*shape):
                    mask = np.ones(len(json_df), dtype=bool)
                    for i, idx in enumerate(indices):
                        mask &= dependent_data[i] == unique_values[i][idx]
                    if mask.any():
                        data_array[indices] = json_df[info.name_or_column][mask].to_numpy()[0]  # type: ignore[reportUnknownMemberType]

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
            var_content = cdf_file.varget(info.name_or_column)  # type: ignore[reportUnknownMemberType]
            var_content = typing.cast("NDArray[Any]", var_content)

            if isinstance(var_content, str):
                var_content = np.array([var_content])

            if np.issubdtype(var_content.dtype, np.floating) and "FILLVAL" in cdf_file.varattsget(info.name_or_column):
                fill_value = cdf_file.attget("FILLVAL", info.name_or_column).Data  # type: ignore[reportUnknownMemberType]
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
                    raise TypeError(msg)

                if isinstance(entry_data, h5py.Group):
                    msg = "Groups for h5 files has not been implemented!"
                    raise NotImplementedError(msg)

                var_content = typing.cast("NDArray[np.generic]|str", entry_data[...])
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


def _find_files_with_regex(directory: Path, regex_pattern: str) -> list[Path]:
    """Find files in a directory that match a given regex pattern."""
    compiled_regex = re.compile(regex_pattern)  # Compile for efficiency

    return [file for file in directory.glob("*") if compiled_regex.fullmatch(str(file.name))]
