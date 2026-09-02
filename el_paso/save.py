# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import logging
import stat
from typing import TYPE_CHECKING, Any, get_args

import numpy as np

from el_paso.typing import FixedDimensionName, InternalName, Variable
from el_paso.utils import enforce_utc_timezone, timed_function

if TYPE_CHECKING:
    from datetime import datetime

    from el_paso.data_standard import DataStandard
    from el_paso.saving_strategy import SavingStrategy
    from el_paso.typing import SavedDataDict

logger = logging.getLogger(__name__)


@timed_function()
def save(
    variables_dict: dict[InternalName, Variable],
    saving_strategy: SavingStrategy,
    start_time: datetime,
    end_time: datetime,
    time_var: Variable | None = None,
    *,
    append: bool = False,
    ignore_validation: bool = False,
) -> None:
    """Saves variables to files based on the specified saving strategy and time intervals.

    This function iterates through defined time intervals, calling the saving strategy
    to save a set of variables to corresponding files. It handles variable filtering
    and includes a check for missing data.

    Args:
        variables_dict (dict[str, Variable]): A dictionary mapping variable names to their
            `Variable` objects to be saved.
        saving_strategy (SavingStrategy): The strategy object that defines how to
            organize, standardize, and save the data (e.g., file paths, formats).
        start_time (datetime): The start of the overall time range for which data
            should be saved.
        end_time (datetime): The end of the overall time range.
        time_var (Variable, optional): The variable representing time. If not provided,
            the saving strategy must handle time internally. Defaults to None.
        append (bool, optional): If `True`, data will be appended to existing files
            rather than overwriting them. Defaults to `False`.
        ignore_validation (bool, optional): If `True`, validation of the input against the `ep.typing.InternalName`
            variables will be skipped. Defaults to `False`.

    Raises:
        TypeError: If `variables_dict` is not a dictionary of `Variable` objects.
        KeyError: If `variables_dict` contains invalid internal variable names.

    Note:
        If an output file is missing one or more of its required variables, a
        warning is logged and that output file is skipped (no exception is raised).
    """
    if not ignore_validation:
        data_standard = saving_strategy.data_standard if hasattr(saving_strategy, "data_standard") else None
        _validate_variables_dict(variables_dict, data_standard)

    start_time = enforce_utc_timezone(start_time)
    end_time = enforce_utc_timezone(end_time)

    time_intervals_to_save = saving_strategy.get_time_intervals_to_save(start_time, end_time)

    for interval_start, interval_end in time_intervals_to_save:
        for output_file in saving_strategy.output_files:
            file_path = saving_strategy.get_file_path(interval_start, interval_end, output_file)

            target_variables = saving_strategy.get_target_variables(
                output_file, variables_dict, time_var, interval_start, interval_end
            )

            if target_variables is None:
                logger.warning(
                    f"Saving attempted, but product is missing some required variables for output {output_file.name}!",
                    stacklevel=2,
                )
            else:
                data_dict = _get_data_dict_to_save(target_variables)

                if len(data_dict.keys()) == 1:
                    logger.info("No data to save! Skipping...")
                    continue

                saving_strategy.save_single_file(file_path, data_dict, append=append)
                file_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP)


def _validate_variables_dict(variables_dict: dict[InternalName, Variable], data_standard: DataStandard | None) -> None:
    """Validates runtime types for data passed to ``save``.

    This guard complements static type checking by rejecting invalid keys and
    values even when annotations are ignored.
    """
    valid_internal_names = {arg for names in get_args(InternalName) for arg in get_args(names)}

    invalid_keys = [key for key in variables_dict if not isinstance(key, str) or key not in valid_internal_names]
    if invalid_keys:
        msg = (
            "variables_dict contains invalid internal name keys: "
            f"{invalid_keys}. Allowed keys are: {valid_internal_names}"
        )
        raise KeyError(msg)

    invalid_values = {
        key: type(value).__name__ for key, value in variables_dict.items() if not isinstance(value, Variable)
    }
    if invalid_values:
        msg = f"variables_dict must map each internal name to an ep.Variable. Invalid entries: {invalid_values}"
        raise TypeError(msg)

    # check if necessary dimensions are saved
    if data_standard:
        all_dependencies = [data_standard.get_dependencies(name) for name in variables_dict]
        unique_dimensions = np.unique(list(itertools.chain.from_iterable(all_dependencies)))
        missing = {
            dim: [name for name in variables_dict if dim in data_standard.get_dependencies(name)]
            for dim in unique_dimensions
            if dim not in get_args(FixedDimensionName) and dim not in variables_dict
        }
        if len(missing) > 0:
            missing_details = "; ".join(
                f"'{dim}' (required by: {', '.join(required_by)})" for dim, required_by in missing.items()
            )
            msg = f"Data for the following dimensions is not saved: {missing_details}"
            raise ValueError(msg)


def _get_data_dict_to_save(
    target_variables: dict[InternalName, Variable],
) -> SavedDataDict:
    """Generates a dictionary of data and metadata for saving.

    This internal function iterates through a dictionary of variables, extracts their
    data and metadata, and formats them into a single dictionary suitable for
    persistence. It also sanitizes the metadata to handle `None` values.

    Args:
        target_variables (dict[str, Variable]): A dictionary of variables to be prepared
            for saving.

    Returns:
        dict[str, Any]: The formatted dictionary containing all variable data and
            associated metadata.
    """
    data_dict: SavedDataDict = {}
    metadata_dict: dict[Any, Any] = {}

    for save_name, variable in target_variables.items():
        data_content = variable.get_data()

        if data_content.size == 0:
            logger.debug(f"Variable {save_name} does not hold any content! Skipping ...", stacklevel=2)
            continue

        if np.all(np.isnan(data_content)):
            logger.warning(f"Variable {save_name} only holds NaN values!")

        # Save the data_content into a field named by save_name
        data_dict[save_name] = data_content

        # Create metadata for each variable
        metadata_dict[save_name] = {
            "unit": str(variable.metadata.unit),
            "original_cadence_seconds": variable.metadata.original_cadence_seconds,
            "source_files": variable.metadata.source_files,
            "description": variable.metadata.description,
            "processing_notes": variable.metadata.processing_notes,
            "standard_name": variable.metadata.standard_name or save_name,
        }

    # Add metadata to the dictionary
    data_dict["metadata"] = _sanitize_metadata_dict(metadata_dict)

    return data_dict


def _sanitize_metadata_dict(metadata_dict: dict[Any, Any]) -> dict[Any, Any]:
    """Recursively sanitizes a metadata dictionary by replacing `None` with empty NumPy arrays.

    This function ensures that the metadata dictionary is free of `None` values,
    which can cause issues with certain serialization formats (e.g., MAT files).
    It traverses the dictionary and its nested dictionaries to replace any `None`
    instance with an empty `numpy.ndarray`.

    Args:
        metadata_dict (dict): The dictionary to be sanitized. It can contain nested
            dictionaries.

    Returns:
        dict: The sanitized dictionary with `None` values replaced.
    """
    sanitized_dict: dict[Any, Any] = {}

    for key, value in metadata_dict.items():
        if isinstance(value, dict):
            # Recursively sanitize nested dictionaries
            sanitized_dict[key] = _sanitize_metadata_dict(value)
        elif value is None:
            # Replace None with an empty numpy array
            sanitized_dict[key] = np.array([])
        else:
            # Retain other values as they are
            sanitized_dict[key] = value

    return sanitized_dict
