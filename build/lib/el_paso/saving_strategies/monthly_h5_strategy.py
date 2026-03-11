# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import calendar
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import el_paso as ep
from el_paso.saving_strategy import OutputFile, SavingStrategy

if TYPE_CHECKING:
    from el_paso.data_standard import DataStandard


class MonthlyH5Strategy(SavingStrategy):
    """A saving strategy that organizes and saves data into a series of monthly HDF5 files.

    This strategy partitions data by month, with each month's data being saved to
    a separate HDF5 file. It standardizes variables to a consistent set of units
    and dimensions before saving and performs consistency checks to ensure data
    integrity. The file name is constructed from a user-defined stem, a date range,
    and a magnetic field model identifier.

    Attributes:
        output_files (list[OutputFile]): Pre-defined list of files to be saved,
            each containing a comprehensive list of variables to be included.
        base_data_path (Path): The root directory for all saved `.h5` files.
        file_name_stem (str): The base name for the output files.
        mag_field (ep.processing.magnetic_field_utils.MagneticFieldLiteral):
            A string specifying the magnetic field model used.
        data_standard (DataStandard): An instance of a data standard class
            that handles the standardization of variables.

    Methods:
        __init__: Initializes the strategy with file paths, names, and a magnetic field model.
        get_time_intervals_to_save: Splits a given time range into a list of monthly intervals.
        get_file_path: Generates the file path for a monthly HDF5 file.
        standardize_variable: Standardizes a variable's units, dimensions, and shape.
    """

    output_files: list[OutputFile]

    file_path: Path

    def __init__(
        self,
        base_data_path: str | Path,
        file_name_stem: str,
        mag_field: ep.processing.magnetic_field_utils.MagneticFieldLiteral,
        data_standard: DataStandard | None = None,
    ) -> None:
        """Initializes the MonthlyH5Strategy.

        Parameters:
            base_data_path (str | Path): The base directory for saving all data.
            file_name_stem (str): The base name for the output files.
            mag_field (ep.processing.magnetic_field_utils.MagneticFieldLiteral):
                The magnetic field model used, e.g., 'TS04'.
            data_standard (DataStandard | None): An optional data standard instance.
                If `None`, `ep.data_standards.PRBEMStandard` is used by default.
        """
        self.base_data_path = Path(base_data_path)
        self.file_name_stem = file_name_stem
        self.mag_field = mag_field

        if data_standard is None:
            data_standard = ep.data_standards.PRBEMStandard()
        self.data_standard = data_standard

        self.output_files = [
            OutputFile(
                "full",
                [
                    "time",
                    "flux/FEDU",
                    "flux/FEDO",
                    "flux/alpha_eq",
                    "flux/energy",
                    "flux/alpha_local",
                    "position/xGEO",
                    f"position/{mag_field}/MLT",
                    f"position/{mag_field}/R0",
                    f"position/{mag_field}/Lstar",
                    f"position/{mag_field}/Lm",
                    f"mag_field/{mag_field}/B_eq",
                    f"mag_field/{mag_field}/B_local",
                    "psd/PSD",
                    f"psd/{mag_field}/inv_mu",
                    f"psd/{mag_field}/inv_K",
                    "density/density_local",
                    f"density/{mag_field}/density_eq",
                ],
                save_incomplete=True,
            ),
        ]

    def get_time_intervals_to_save(
        self, start_time: datetime | None, end_time: datetime | None
    ) -> list[tuple[datetime, datetime]]:
        """Splits the provided time range into a list of full-month intervals.

        This method generates a list of (start_datetime, end_datetime) tuples, where each tuple
        represents a single calendar month.

        Parameters:
            start_time (datetime | None): The start time of the data range.
            end_time (datetime | None): The end time of the data range.

        Returns:
            list[tuple[datetime, datetime]]: A list of tuples, each defining a monthly interval.

        Raises:
            ValueError: If either `start_time` or `end_time` is not provided.
        """
        time_intervals: list[tuple[datetime, datetime]] = []

        if start_time is None or end_time is None:
            msg = "start_time and end_time must be provided for MonthlyH5Strategy!"
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
                if month == 12  # noqa: PLR2004
                else datetime(year, month + 1, 1, tzinfo=timezone.utc)
            )

        return time_intervals

    def get_file_path(self, interval_start: datetime, interval_end: datetime, output_file: OutputFile) -> Path:  # noqa: ARG002
        """Generates a structured file path for the HDF5 file.

        The file name is constructed from a predefined stem, the date range, and the magnetic
        field model, with a `.h5` extension.

        Parameters:
            interval_start (datetime): The start of the time interval.
            interval_end (datetime): The end of the time interval.
            output_file (OutputFile): The configuration for the output file. (ignored)

        Returns:
            Path: The full file path for the HDF5 file.
        """
        start_year_month_day = interval_start.strftime("%Y%m%d")
        end_year_month_day = interval_end.strftime("%Y%m%d")

        file_name = f"{self.file_name_stem}_{start_year_month_day}to{end_year_month_day}_{self.mag_field}.h5"

        return self.base_data_path / file_name

    def standardize_variable(
        self, variable: ep.Variable, name_in_file: str, *, first_call_of_interval: bool
    ) -> ep.Variable:
        """Standardizes a variable's units and dimensions by delegating to a DataStandard instance.

        This method acts as a wrapper, passing the variable and its file name to the
        `standardize_variable` method of the `data_standard` attribute.
        Parameters:
            variable (ep.Variable): The variable instance to be standardized.
            name_in_file (str): The name of the variable as it appears in the file.
            first_call_of_interval (bool): Flag to indicate if it is the first call of a time interval

        Returns:
            ep.Variable: The standardized variable.
        """
        return self.data_standard.standardize_variable(
            name_in_file, variable, reset_consistency_check=first_call_of_interval
        )
