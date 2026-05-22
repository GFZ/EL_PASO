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
    from el_paso.typing import DataStandard, InternalName, SavedDataDict, StandardName

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

        Parameters:
            base_data_path (str | Path): The base directory for saving all data.
            mission (str): The mission name.
            satellite (str): The satellite name.
            instrument (str): The instrument name.
            mag_field (str): The model extension type. "TS04" is remapped to "T04s".
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
            OutputFile("flux", ["Epoch", "FEDU"]),
            OutputFile("alpha_and_energy", ["Epoch", "Alpha", "Alpha_Eq", "Energy_FEDU"]),
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

    def standardize_variable(
        self, variable: Variable, internal_name: InternalName, *, first_call_of_interval: bool
    ) -> Variable:
        """Standardizes a variable's units and dimensions based on its predefined name.

        This method acts as a proxy, delegating the actual standardization logic
        to the `GFZStandard` instance. It ensures that data conforms to the
        specified standard before it is saved.

        Parameters:
            variable (Variable): The variable instance to be standardized.
            name_in_file (str): The predefined name of the variable to use for
                determining the standardization rules.
            first_call_of_interval (bool): Flag to indicate if it is the first call of a time interval

        Returns:
            Variable: The standardized variable instance.

        Raises:
            ValueError: If an unknown `name_in_file` is encountered..
        """
        return self.data_standard.standardize_variable(
            internal_name, variable, reset_consistency_check=first_call_of_interval
        )

    def get_time_intervals_to_save(
        self, start_time: datetime | None, end_time: datetime | None
    ) -> list[tuple[datetime, datetime]]:
        """Splits the time range into a list of full-month intervals.

        This method iterates from the start month to the end month, creating a new
        (start, end) tuple for each calendar month.

        Parameters:
            start_time (datetime | None): The start of the time range.
            end_time (datetime | None): The end of the time range.

        Returns:
            list[tuple[datetime, datetime]]: A list of tuples, where each tuple represents a
                monthly time interval.

        Raises:
            ValueError: If either `start_time` or `end_time` is not provided.
        """
        time_intervals: list[tuple[datetime, datetime]] = ep.utils.get_monthly_datetime_intervals(start_time, end_time)

        return time_intervals

    def get_file_path_stem(self) -> Path:
        return self.base_data_path / self.mission.upper() / self.satellite.lower() / "Processed_Mat_Files"

    def get_file_name_stem(self) -> str:
        return self.satellite.lower() + "_" + self.instrument.lower()

    def get_file_path(self, interval_start: datetime, interval_end: datetime, output_file: OutputFile) -> Path:
        """Generates a structured file path for the given time interval and output file.

        The path follows a specific format:
        `base_path/MISSION/SATELLITE/Processed_Mat_Files/satellite_instrument_YYYYMMDDtoYYYYMMDD_filename_ver4.mat`

        Parameters:
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

    def _append_mat_data(self, file_path: Path, data_dict_to_save: SavedDataDict) -> SavedDataDict:
        """Load an existing MATLAB file and merge the new data into it."""
        data_dict_old = self._loader(file_path)
        time_key = self.data_standard.get_standard_name("Epoch")

        def _normalize_1d(arr: NDArray) -> NDArray:
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[1] == 1:
                return arr.reshape(-1)
            return arr

        time_1 = np.atleast_1d(np.squeeze(data_dict_old[time_key]))
        time_2 = np.atleast_1d(np.squeeze(data_dict_to_save["Epoch"]))

        idx_to_insert = int(np.searchsorted(time_1, time_2[0]))

        time_1_in_2 = np.isin(time_1, time_2)

        for key, value_1 in data_dict_old.items():
            if key.startswith("__"):
                continue

            if key == "metadata":
                value_2 = data_dict_to_save.get(key)
                if isinstance(value_1, dict) and isinstance(value_2, dict):
                    data_dict_to_save[key] = {**value_1, **value_2}
                elif key not in data_dict_to_save:
                    data_dict_to_save[key] = value_1
                continue

            internal_name = self.data_standard.get_internal_name(key)
            if internal_name is None:
                msg = "Encountered unexpected standard name!"
                raise ValueError(msg)

            if internal_name not in data_dict_to_save:
                msg = "Key mismatch when concatenating data dicts!"
                logger.error(msg)
                raise ValueError(msg)

            if isinstance(value_1, np.ndarray):
                value_1_truncated = cast("NDArray[np.floating]", value_1[~time_1_in_2])

                value_2 = data_dict_to_save[internal_name]

                value_1_truncated = _normalize_1d(value_1_truncated)
                value_2 = _normalize_1d(value_2)
                concatenated_value = (
                    value_2
                    if value_1_truncated.size == 0
                    else np.insert(value_1_truncated, idx_to_insert, value_2, axis=0)
                )

                if key == time_key and len(np.unique(concatenated_value)) != len(concatenated_value):
                    msg = "Time values were not unique when concatenating arrays!"
                    logger.error(msg)
                    raise ValueError(msg)
                data_dict_to_save[internal_name] = concatenated_value

        return data_dict_to_save

    def append_data(self, file_path: Path, data_dict_to_save: SavedDataDict) -> SavedDataDict:
        """Appends new data to an existing GFZ file.

        Existing data is loaded from the file, overlapping time stamps are replaced
        by the new block, and the merged dictionary is returned for the caller to
        write back to disk.
        Parameters:
            file_path (Path): The path to the existing file to append to.
            data_dict_to_save (dict[str, Any]): The dictionary with new data to be added.

        Returns:
            dict[str, Any]: A new dictionary containing the merged old and new data.

        Raises:
            ValueError: If there are mismatches in keys or if time values are not unique after concatenation or
                        if invalid standard names are encountered.
        """
        return self._append_mat_data(file_path, data_dict_to_save)

    def save_single_file(self, file_path: Path, dict_to_save: SavedDataDict, *, append: bool = False) -> None:
        """Saves variable data to a single file in ".mat" format.

        Parameters:
            file_path (Path): The path to the file where the dictionary will be saved.
                            The file extension determines the format.
            dict_to_save (dict[str, Any]): The dictionary containing variable data to save.
            append (bool, optional): If True and the file exists, appends data to the existing file (if supported).
                                    Defaults to False.

        Raises:
            NotImplementedError: If the file format specified by the file extension is not supported.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.exists() and append:
            dict_to_save = self.append_data(file_path, dict_to_save)

        standard_dict = {}
        for key, value in dict_to_save.items():
            if key == "metadata":
                standard_dict["metadata"] = value
            else:
                standard_dict[self.data_standard.get_standard_name(key)] = value

        logger.info(f"Saving file: {file_path}")
        savemat(str(file_path), standard_dict)
