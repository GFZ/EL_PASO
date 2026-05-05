# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import calendar
import logging
import pickle
import typing
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.io import loadmat

from el_paso.data_standards import DataOrgStandard
from el_paso.saving_strategy import OutputFile, SavingStrategy

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray

    from el_paso import Variable

logger = logging.getLogger(__name__)


class DataOrgStrategy(SavingStrategy):
    """A concrete saving strategy for saving data based on the satellite mission into separate monthly files.

    This strategy implements the data standard used at GFZ in the past.
    It organizes the output files into a specific directory structure
    (e.g., `base_path/MISSION/SATELLITE/Processed_Mat_Files/`) and standardizes
    variables to specific units and dimensions before saving. The data is saved
    in either `.mat` or `.pickle` format, depending on user preference.

    !!! warning "Deprecation"
        The ``".pickle"`` file format is deprecated and will be removed in a future
        release. Use ``".mat"`` instead.

    Attributes:
        output_files (list[OutputFile]): Pre-defined list of files to be saved,
            each with a specific set of variables.
        base_data_path (Path): The root directory for all saved data.
        mission (str): The name of the space mission (e.g., "MMS").
        satellite (str): The name of the satellite (e.g., "MMS1").
        instrument (str): The name of the instrument.
        kext (str): A model-related identifier, with "TS04" being mapped to "T04s"
            for backward compatibility.
        file_format (Literal[".mat", ".pickle"]): The file extension for the output files.

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
        kext: str,
        file_format: Literal[".mat", ".pickle"] = ".mat",
    ) -> None:
        """Initializes the data organization strategy.

        Parameters:
            base_data_path (str | Path): The base directory for saving all data.
            mission (str): The mission name.
            satellite (str): The satellite name.
            instrument (str): The instrument name.
            kext (str): The model extension type. "TS04" is remapped to "T04s".
            file_format (Literal[".mat", ".pickle"]): The desired format for the output files.
            .. deprecated:: 1.0.3rc0
                    Passing ``".pickle"`` is deprecated and will be removed in a future release.
                    Use ``".mat"`` instead.
        """
        if file_format == ".pickle":
            warnings.warn(
                "The '.pickle' file format for DataOrgStrategy is deprecated and will be removed "
                "in a future release. Use '.mat' instead.",
                FutureWarning,
                stacklevel=2,
            )

        self.base_data_path = Path(base_data_path)
        self.mission = mission
        self.satellite = satellite
        self.instrument = instrument

        # for backwards compatibility
        if kext == "TS04":
            kext = "T04s"
        self.kext = kext

        self.file_format = file_format

        self.output_files = [
            OutputFile("flux", ["time", "Flux"]),
            OutputFile("alpha_and_energy", ["time", "alpha_local", "alpha_eq_model", "energy_channels"]),
            OutputFile("mlt", ["time", "MLT"]),
            OutputFile("lstar", ["time", "Lstar"]),
            OutputFile("lm", ["time", "Lm"]),
            OutputFile("psd", ["time", "PSD"]),
            OutputFile("xGEO", ["time", "xGEO"]),
            OutputFile("invmu_and_invk", ["time", "InvMu", "InvK"]),
            OutputFile("bfield", ["time", "B_eq", "B_local"]),
            OutputFile("R0", ["time", "R0"]),
            OutputFile("density", ["time", "density"]),
        ]

        self.data_standard = DataOrgStandard()

    def standardize_variable(self, variable: Variable, name_in_file: str, *, first_call_of_interval: bool) -> Variable:
        """Standardizes a variable's units and dimensions based on its predefined name.

        This method acts as a proxy, delegating the actual standardization logic
        to the `DataOrgStandard` instance. It ensures that data conforms to the
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
            name_in_file, variable, reset_consistency_check=first_call_of_interval
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
        time_intervals: list[tuple[datetime, datetime]] = []

        if start_time is None or end_time is None:
            msg = "start_time and end_time must be provided for DataOrgStrategy!"
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
                if month == 12  # noqa: PLR2004
                else datetime(year, month + 1, 1, tzinfo=timezone.utc)
            )

        return time_intervals

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
        start_year_month_day = interval_start.strftime("%Y%m%d")
        end_year_month_day = interval_end.strftime("%Y%m%d")

        file_name = (
            f"{self.satellite.lower()}_{self.instrument.lower()}_"
            f"{start_year_month_day}to{end_year_month_day}_{output_file.name}"
        )

        if output_file.name in ["alpha_and_energy", "lstar", "lm", "invmu_and_invk", "mlt", "bfield", "R0"]:
            file_name += f"_n4_4_{self.kext}"

        file_name += "_ver4" + self.file_format

        return self.base_data_path / self.mission.upper() / self.satellite.lower() / "Processed_Mat_Files" / file_name

    def _merge_data_dicts_by_time(
        self, data_dict_old: dict[str, Any], data_dict_to_save: dict[str, Any]
    ) -> dict[str, Any]:
        """Appends new data to an existing file by combining the new and old data dictionaries.

        Parameters:
            data_dict_old (dict[str, Any]): The dictionary with existing data.
            data_dict_to_save (dict[str, Any]): The dictionary with new data to be added.

        Returns:
            dict[str, Any]: A new dictionary containing the merged old and new data.

        Raises:
            ValueError: If a key mismatch occurs between the dictionaries or if the concatenated
                time array contains non-unique values.
        """
        time_1 = np.atleast_1d(np.squeeze(data_dict_old["time"]))
        time_2 = np.atleast_1d(np.squeeze(data_dict_to_save["time"]))

        idx_to_insert = int(np.searchsorted(time_1, time_2[0]))

        time_1_in_2 = np.isin(time_1, time_2)

        for key, value_1 in data_dict_old.items():
            if key.startswith("__") or key == "metadata":
                continue

            if key not in data_dict_to_save:
                msg = "Key mismatch when concatenating data dicts!"
                logger.error(msg)
                raise ValueError(msg)

            if isinstance(value_1, np.ndarray):
                value_1_truncated = typing.cast("NDArray[np.floating]", value_1[~time_1_in_2])

                value_2 = data_dict_to_save[key]

                concatenated_value = (
                    value_2
                    if value_1_truncated.size == 0
                    else np.insert(value_1_truncated, idx_to_insert, value_2, axis=0)
                )

                if key == "time" and len(np.unique(concatenated_value)) != len(concatenated_value):
                    msg = "Time values were not unique when concatenating arrays!"
                    logger.error(msg)
                    raise ValueError(msg)
                data_dict_to_save[key] = concatenated_value

        return data_dict_to_save

    def _append_mat_data(self, file_path: Path, data_dict_to_save: dict[str, Any]) -> dict[str, Any]:
        """Load an existing MATLAB file and merge the new data into it."""
        data_dict_old = loadmat(str(file_path))
        return self._merge_data_dicts_by_time(data_dict_old, data_dict_to_save)

    def _append_pickle_data(self, file_path: Path, data_dict_to_save: dict[str, Any]) -> dict[str, Any]:
        """Load an existing pickle file and merge the new data into it."""
        warnings.warn(
            "Appending to '.pickle' files is deprecated alongside the '.pickle' format and will "
            "be removed in a future release. Switch to '.mat' to avoid this warning.",
            FutureWarning,
            stacklevel=2,
        )

        with file_path.open("rb") as file:
            data_dict_old = pickle.load(file)  # noqa: S301

        return self._merge_data_dicts_by_time(data_dict_old, data_dict_to_save)

    def append_data(self, file_path: Path, data_dict_to_save: dict[str, Any]) -> dict[str, Any]:
        """Appends new data to an existing DataOrg file.

        Existing data is loaded from the file, overlapping time stamps are replaced
        by the new block, and the merged dictionary is returned for the caller to
        write back to disk.

        .. deprecated:: 1.0.3rc0
            Support for appending to ``.pickle`` files is deprecated alongside the ``.pickle``
            format itself. This code path will be removed in a future release.

        Parameters:
            file_path (Path): The path to the existing file to append to.
            data_dict_to_save (dict[str, Any]): The dictionary with new data to be added.

        Returns:
            dict[str, Any]: A new dictionary containing the merged old and new data.

        Raises:
            ValueError: If a key mismatch occurs between the dictionaries or if the concatenated
                time array contains non-unique values.
            NotImplementedError: If ``append`` is requested for an unsupported file format.
        """
        match file_path.suffix:
            case ".mat":
                return self._append_mat_data(file_path, data_dict_to_save)
            case ".pickle":
                return self._append_pickle_data(file_path, data_dict_to_save)
            case _:
                msg = f"Appending to '{file_path.suffix}' files is not supported by DataOrgStrategy."
                logger.error(msg)
                raise NotImplementedError(msg)
