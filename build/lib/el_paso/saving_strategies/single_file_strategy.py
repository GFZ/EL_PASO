# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from el_paso.saving_strategy import OutputFile, SavingStrategy

if TYPE_CHECKING:
    from datetime import datetime

    from el_paso import Variable


class SingleFileStrategy(SavingStrategy):
    """A concrete saving strategy that saves all data to a single file.

    This strategy implements the `SavingStrategy` abstract methods to manage saving all variables
    for the entire time range into a single output file. It is a simple, non-partitioning approach.

    Attributes:
        file_path (Path): The path to the single output file where all data will be saved.

    Methods:
        __init__(file_path): Initializes the strategy with the file path.
        get_time_intervals_to_save: Returns the entire time range as a single interval.
        get_file_path: Always returns the pre-defined single file path.
        standardize_variable: Passes the variable through without any standardization.
    """

    map_standard_name: dict[str, str]
    output_files: list[OutputFile]

    file_path: Path

    def __init__(self, file_path: str | Path) -> None:
        """Initializes the SingleFileStrategy with the specified file path.

        Parameters:
            file_path (str | Path): The full path to the output file.
        """
        self.file_path = Path(file_path)
        self.output_files = [OutputFile(self.file_path.name, [])]

        self.map_standard_name = {}

    def get_time_intervals_to_save(self, start_time: datetime, end_time: datetime) -> list[tuple[datetime, datetime]]:
        """Returns the entire time range as a single interval.

        This strategy does not split data by time; it saves everything in one go.

        Parameters:
            start_time (datetime): The start time of the data range.
            end_time (datetime): The end time of the data range.

        Returns:
            list[tuple[datetime, datetime]]: A list containing a single tuple with the start and end times.
        """
        return [(start_time, end_time)]

    def get_file_path(
        self,
        interval_start: datetime,  # noqa: ARG002
        interval_end: datetime,  # noqa: ARG002
        output_file: OutputFile,  # noqa: ARG002
    ) -> Path:
        """Returns the pre-defined single file path, ignoring the interval.

        This method ensures all data is saved to the same file, regardless of the time interval.

        Parameters:
            interval_start (datetime): The start of the time interval (ignored).
            interval_end (datetime): The end of the time interval (ignored).
            output_file (OutputFile): The output file configuration (ignored).

        Returns:
            Path: The `file_path` of this strategy instance.
        """
        return self.file_path

    def standardize_variable(self, variable: Variable, name_in_file: str, *, first_call_of_interval: bool) -> Variable:  # noqa: ARG002
        """Does not modify the variable.

        This strategy does not perform any specific standardization on the variables before saving.

        Parameters:
            variable (Variable): The variable instance to be standardized.
            name_in_file (str): The name of the variable as it appears in the file (ignored).
            first_call_of_interval (bool): Flag to indicate if it is the first call of a time interval

        Returns:
            Variable: The original variable instance, unchanged.
        """
        return variable
