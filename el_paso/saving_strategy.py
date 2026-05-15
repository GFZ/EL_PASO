# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from astropy import units as u  # type: ignore[reportMissingTypeStubs]

from el_paso import Variable

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from el_paso.processing.magnetic_field_utils import MagneticFieldLiteral
    from el_paso.typing import InternalName, SavedDataDict, DataStandardClass


class OutputFile(NamedTuple):
    """Represents an output file with its name and a list of variable names to save.

    Attributes:
        name (str): The name of the output file.
        names_to_save (list[str]): List of variable names to be saved in the output file.
        save_incomplete (bool): If True, allows saving even if some variables are missing.
    """

    name: str
    names_to_save: list[InternalName]
    save_incomplete: bool = False


class SavingStrategy(ABC):
    """Abstract base class for defining strategies to save output files with specific time intervals and variables.

    Attributes:
        output_files (list[OutputFile]): List of output files to be managed by the saving strategy.

    Methods:
        get_time_intervals_to_save(start_time: datetime | None, end_time: datetime | None)
            -> list[tuple[datetime, datetime]]:
            Abstract method to determine the time intervals for saving data between start_time and end_time.

        get_file_path(interval_start: datetime, interval_end: datetime, output_file: OutputFile) -> Path:
            Abstract method to generate the file path for a given time interval and output file.

        standardize_variable(variable: Variable, name_in_file: str) -> Variable:
            Abstract method to standardize a variable before saving, possibly renaming or formatting it.

        get_target_variables(output_file: OutputFile, variables_dict: dict[str, Variable], time_var: Variable | None,
                             start_time: datetime | None, end_time: datetime | None) -> dict[str, Variable] | None:
            Selects and prepares variables to be saved in the output file, optionally truncating them to a time range.

        save_single_file(file_path: Path, dict_to_save: dict[str, Any], *, append: bool = False):
            Saves the provided dictionary to a file in the specified format (.mat, .pickle, .h5, .nc),
            optionally appending data.

        append_data(file_path: Path, dict_to_save: dict[str, Any]) -> dict[str, Any]:
            Abstract method to append data to an existing file; must be implemented by subclasses.
    """

    output_files: list[OutputFile]
    data_standard: DataStandardClass
    base_data_path: Path
    satellite: str
    mission: str
    instrument: str
    mag_field: MagneticFieldLiteral

    @abstractmethod
    def get_time_intervals_to_save(self, start_time: datetime, end_time: datetime) -> list[tuple[datetime, datetime]]:
        """Generates a list of time intervals to save between the specified start and end times.

        Args:
            start_time (datetime | None): The starting datetime for the intervals.
                                          If None, intervals may start from the earliest available time.
            end_time (datetime | None): The ending datetime for the intervals.
                                        If None, intervals may end at the latest available time.

        Returns:
            list[tuple[datetime, datetime]]: A list of tuples, each representing a time interval (start, end)
                                             to be saved.
        """

    @abstractmethod
    def get_file_path(self, interval_start: datetime, interval_end: datetime, output_file: OutputFile) -> Path:
        """Generates a file path for saving variables based on the provided interval and output file information.

        Args:
            interval_start (datetime): The start of the interval for which the file is being generated.
            interval_end (datetime): The end of the interval for which the file is being generated.
            output_file (OutputFile): An OutputFile containing the name of the output file,
                                      and which variables should be saved in this file.

        Returns:
            Path: The generated file path where the output data should be saved.
        """

    @abstractmethod
    def standardize_variable(
        self, variable: Variable, internal_name: InternalName, *, first_call_of_interval: bool
    ) -> Variable:
        """Standardizes the given variable according to the specified name in the file.

        Standardization may include checking of units, dimensions, and size consistency.

        Args:
            variable (Variable): The variable instance to be standardized.
            name_in_file (str): The name of the variable as it appears in the file.
            first_call_of_interval (bool): Flag to indicate if it is the first call of a time interval

        Returns:
            Variable: The standardized variable instance.
        """

    @abstractmethod
    def save_single_file(self, file_path: Path, dict_to_save: SavedDataDict, *, append: bool = False) -> None:
        """Saves the provided dictionary to a single file in one of the supported formats (.mat, .pickle, .h5, .nc).

        Parameters:
            file_path (Path): The path where the file should be saved.
            dict_to_save (dict[str, Any]): The dictionary containing variable data and metadata to be saved.
            append (bool, optional): If True, data will be appended to existing files rather than overwriting them.
                    Defaults to False.
        """

    @abstractmethod
    def get_file_path_stem(self) -> Path:
        pass

    @abstractmethod
    def get_file_name_stem(self) -> str:
        pass

    def get_target_variables(
        self,
        output_file: OutputFile,
        variables_dict: dict[InternalName, Variable],
        time_var: Variable | None,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> dict[InternalName, Variable] | None:
        """Retrieves and processes target variables for saving based on the specified output file.

        Parameters:
            output_file (OutputFile): The output file configuration containing variable names to save.
            variables_dict (dict[str, Variable]): Dictionary mapping variable names to Variable objects.
            time_var (Variable | None): The time variable used for truncation, if applicable.
            start_time (datetime | None): The start time for truncating variables, if specified.
            end_time (datetime | None): The end time for truncating variables, if specified.

        Returns:
            dict[str, Variable] | None:
                - A dictionary of processed Variable objects keyed by their names,
                    or None if any specified variable name is not found in variables_dict.

        Notes:
            - If no variable names are specified in output_file, all variables in variables_dict are processed.
            - Variables are deep-copied before processing.
            - Each variable is standardized using the `standardize_variable` method.
            - If a requested variable name is not found, a warning is issued and None is returned.
        """
        target_variables: dict[InternalName, Variable] = {}
        first_call_of_interval = True

        # if no variables have been specified, we save all of them
        if len(output_file.names_to_save) == 0:
            for key, var in variables_dict.items():
                var_to_save = deepcopy(var)

                if start_time is not None and end_time is not None and time_var is not None:
                    var_to_save.truncate(time_var, start_time.timestamp(), end_time.timestamp())
                var_to_save = self.standardize_variable(var_to_save, key, first_call_of_interval=first_call_of_interval)
                first_call_of_interval = False

                target_variables[key] = var_to_save

            return target_variables

        for name_to_save in output_file.names_to_save:
            if name_to_save in variables_dict:
                var_to_save = deepcopy(variables_dict[name_to_save])

                if start_time is not None and end_time is not None and time_var is not None:
                    var_to_save.truncate(time_var, start_time.timestamp(), end_time.timestamp())

                var_to_save = self.standardize_variable(
                    var_to_save, name_to_save, first_call_of_interval=first_call_of_interval
                )
                first_call_of_interval = False

                target_variables[name_to_save] = var_to_save
            else:
                msg = f"Could not find target variable {name_to_save}!"
                logger.warning(msg, stacklevel=2)
                if output_file.save_incomplete:
                    target_variables[name_to_save] = Variable(original_unit=u.dimensionless_unscaled, data=np.array([]))
                else:
                    return None

        return target_variables

    def get_output_file(
        self, *, standard_name: str | None = None, internal_name: InternalName | None = None
    ) -> OutputFile | None:

        if internal_name is None:
            if standard_name is None:
                msg = "Either standard_name or internal_name must be provided!"
                raise ValueError(msg)
            internal_name = self.data_standard.get_internal_name(standard_name)

        if internal_name is None:
            return None

        for output_file in self.output_files:
            if internal_name in output_file.names_to_save:
                return output_file

        return None
