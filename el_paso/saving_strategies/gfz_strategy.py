# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, cast

import numpy as np
from scipy.io import savemat

import el_paso as ep
from el_paso.data_standards import GFZStandard
from el_paso.saving_strategy import OutputFile, SavingStrategy

if TYPE_CHECKING:
    from datetime import datetime

    from numpy.typing import NDArray

    from el_paso import Variable
    from el_paso.typing import DataStandard, InternalName, SavedDataDict, StandardName, TimeInterval

logger = logging.getLogger(__name__)


class GFZStrategy(SavingStrategy):
    """A concrete saving strategy for saving data based on the satellite mission into separate monthly files.

    This strategy implements the data standard used at GFZ in the past.
    It organizes the output files into a specific directory structure
    (e.g., `base_path/MISSION/SATELLITE/Processed_Mat_Files/`) and standardizes
    variables to specific units and dimensions before saving. The data is saved
    in `.mat` format.

    Attributes:
        output_files (list[OutputFile]): Pre-defined list of files to be saved,
            each with a specific set of variables.
        base_data_path (Path): The root directory for all saved data.
        mission (str): The name of the space mission (e.g., "MMS").
        satellite (str): The name of the satellite (e.g., "MMS1").
        instrument (str): The name of the instrument.
        kext (str): A model-related identifier, with "TS04" being mapped to "T04s"
            for backward compatibility.

    Methods:
        __init__: Initializes the strategy with file paths and metadata.
        standardize_variable: Standardizes variables to specific units and dimensions based on their name.
        get_time_intervals_to_save: Splits the given time range into a list of monthly intervals.
        get_file_path: Generates a complete file path based on the mission, satellite, and date.
        append_data: Appends new data to an existing file by concatenating NumPy arrays based on time.
    """

    output_files: list[OutputFile]

    file_path: Path

    def __init__(
        self,
        base_data_path: str | Path,
        mission: str,
        satellite: str,
        instrument: str,
        mag_field: ep.typing.MagneticFieldLiteral,
        data_standard: Optional[DataStandard[StandardName]] = None,
    ) -> None:
        """Initializes the data organization strategy.

        Args:
            base_data_path (str | Path): The base directory for saving all data.
            mission (str): The mission name.
            satellite (str): The satellite name.
            instrument (str): The instrument name.
            mag_field (str): The model extension type. "TS04" is remapped to "T04s".
            data_standard (DataStandard | None, optional): An optional `DataStandard` instance to use for
                standardizing variables. If `None`, `ep.data_standards.GFZStandard` is used by default.
        """
        self.base_data_path = Path(base_data_path)
        self.mission = mission
        self.satellite = satellite
        self.instrument = instrument
        self.data_standard = data_standard or GFZStandard()

        # for backwards compatibility
        if mag_field == "TS04":
            mag_field = "T04s"
        self.mag_field = mag_field

        self.output_files = [
            OutputFile("flux", ["Epoch", ("FEDU", "FPDU")]),
            OutputFile("alpha_and_energy", ["Epoch", "Alpha", "Alpha_Eq", ("Energy_FEDU", "Energy_FPDU")]),
            OutputFile("mlt", ["Epoch", "MLT"]),
            OutputFile("lstar", ["Epoch", "L_star"]),
            OutputFile("lm", ["Epoch", "L_m"]),
            OutputFile("psd", ["Epoch", "PSD"]),
            OutputFile("xGEO", ["Epoch", "Position"]),
            OutputFile("invmu_and_invk", ["Epoch", "InvMu", "InvK"]),
            OutputFile("bfield", ["Epoch", "B_Eq", "B_Calc"]),
            OutputFile("R0", ["Epoch", "R_Eq"]),
        ]

        self._loader = ep.utils.load_mat_data

    def get_time_intervals_to_save(self, start_time: datetime | None, end_time: datetime | None) -> list[TimeInterval]:
        """Splits the time range into a list of full-month intervals.

        This method iterates from the start month to the end month, creating a new
        (start, end) tuple for each calendar month.

        Args:
            start_time (datetime | None): The start of the time range.
            end_time (datetime | None): The end of the time range.

        Returns:
            list[TimeInterval]: A list of tuples, where each tuple represents a
                monthly time interval.

        Raises:
            ValueError: If either `start_time` or `end_time` is not provided.
        """
        time_intervals: list[TimeInterval] = ep.utils.get_monthly_datetime_intervals(start_time, end_time)

        return time_intervals

    def get_file_path_stem(self) -> Path:
        return self.base_data_path / self.mission.upper() / self.satellite.lower() / "Processed_Mat_Files"

    def get_file_name_stem(self) -> str:
        return self.satellite.lower() + "_" + self.instrument.lower()

    def get_file_path(self, interval_start: datetime, interval_end: datetime, output_file: OutputFile) -> Path:
        """Generates a structured file path for the given time interval and output file.

        The path follows a specific format:
        `base_path/MISSION/SATELLITE/Processed_Mat_Files/satellite_instrument_YYYYMMDDtoYYYYMMDD_filename_ver4.mat`

        Args:
            interval_start (datetime): The start of the time interval.
            interval_end (datetime): The end of the time interval.
            output_file (OutputFile): The output file configuration.

        Returns:
            Path: The generated file path.
        """
        interval = ep.utils.get_monthly_datetime_intervals(interval_start, interval_end)[0]
        start_year_month_day = interval[0].strftime("%Y%m%d")
        end_year_month_day = interval[1].strftime("%Y%m%d")

        file_name = self.get_file_name_stem() + f"_{start_year_month_day}to{end_year_month_day}_{output_file.name}"

        if output_file.name in ["alpha_and_energy", "lstar", "lm", "invmu_and_invk", "mlt", "bfield", "R0"]:
            file_name += f"_n4_4_{self.mag_field}"

        file_name += "_ver4.mat"

        return self.get_file_path_stem() / file_name
