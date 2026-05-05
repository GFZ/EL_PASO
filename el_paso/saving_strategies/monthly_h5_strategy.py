# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import calendar
import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

import el_paso as ep
from el_paso.saving_strategies.single_file_strategy import SingleFileStrategy
from el_paso.saving_strategy import OutputFile

if TYPE_CHECKING:
    from el_paso.data_standard import DataStandard

logger = logging.getLogger(__name__)


class MonthlyH5Strategy(SingleFileStrategy):
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
        append_data: Appends new data to an existing HDF5 file while maintaining temporal order.
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

    def _load_h5_data(self, file_path: Path) -> dict[str, Any]:
        """Load all data from an existing HDF5 file.

        Parameters:
            file_path (Path): The path to the HDF5 file to load.

        Returns:
            dict[str, Any]: A dictionary containing all variables from the file.
        """
        loaded_data: dict[str, Any] = {}

        def _recursively_load_datasets(group: h5py.Group | h5py.File, prefix: str = "") -> None:
            """Recursively load datasets from groups and subgroups."""
            for key, item in group.items():
                full_path = f"{prefix}{key}" if prefix else key
                if isinstance(item, h5py.Dataset):
                    loaded_data[full_path] = np.array(item)
                elif isinstance(item, h5py.Group):
                    _recursively_load_datasets(item, f"{full_path}/")

        with h5py.File(file_path, "r") as file:
            _recursively_load_datasets(file)

        return loaded_data

    def _merge_and_sort_data_h5(  # noqa: C901
        self,
        existing_data: dict[str, Any],
        new_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge and sort HDF5 data by time, handling overlapping timestamps.

        Parameters:
            existing_data (dict[str, Any]): The dictionary with existing data.
            new_data (dict[str, Any]): The dictionary with new data to be added.

        Returns:
            dict[str, Any]: A new dictionary containing the merged and sorted data.

        Raises:
            ValueError: If a key mismatch occurs or if the concatenated time array contains non-unique values.
        """

        def _normalize_1d(arr: np.ndarray) -> np.ndarray:
            """Convert (N,1) -> (N,) but leave all other shapes untouched."""
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[1] == 1:  # noqa: PLR2004
                return arr.reshape(-1)
            return arr

        if "time" not in existing_data or existing_data["time"].size == 0:
            return new_data

        if "time" not in new_data or new_data["time"].size == 0:
            return existing_data

        existing_time = _normalize_1d(existing_data["time"])
        new_time = _normalize_1d(new_data["time"])

        # remove overlapping timestamps from existing data
        mask_keep_existing = ~np.isin(existing_time, new_time)

        # insertion index (assumes sorted time)
        insert_idx = int(np.searchsorted(existing_time, new_time[0]))

        merged: dict[str, Any] = {}

        all_keys = set(existing_data.keys()) | set(new_data.keys())

        for key in all_keys:
            if key not in existing_data:
                merged[key] = new_data[key]
                continue

            if key not in new_data:
                merged[key] = existing_data[key]
                continue

            v1 = np.asarray(existing_data[key])
            v2 = np.asarray(new_data[key])

            # normalize 1D inconsistencies
            v1 = _normalize_1d(v1)
            v2 = _normalize_1d(v2)

            if not isinstance(v1, np.ndarray) or not isinstance(v2, np.ndarray):
                merged[key] = v2
                continue

            # check dimensional compatibility (except time axis)
            if v1.ndim != v2.ndim:
                msg = f"{key}: ndim mismatch {v1.shape} vs {v2.shape}"
                logger.error(msg)
                raise ValueError(msg)

            if v1.ndim > 1 and v1.shape[1:] != v2.shape[1:]:
                msg = f"{key}: shape mismatch {v1.shape} vs {v2.shape}"
                logger.error(msg)
                raise ValueError(msg)

            # remove overlapping timestamps from existing
            v1_trunc = v1[mask_keep_existing]

            # insert new block
            merged_val = v2 if v1_trunc.size == 0 else np.insert(v1_trunc, insert_idx, v2, axis=0)

            # enforce time uniqueness
            if key == "time":
                t = np.asarray(merged_val)
                if len(np.unique(t)) != len(t):
                    msg = f"Time values are not unique after merge for key '{key}'"
                    logger.error(msg)
                    raise ValueError(msg)

            merged[key] = merged_val

        return merged

    def append_data(self, file_path: Path, data_dict_to_save: dict[str, Any]) -> dict[str, Any]:
        """Append new data to an existing HDF5 file, maintaining sorted order by timestamp.

        This method loads the existing data, merges it with new data, sorts by timestamp,
        and writes back to the file. If any step fails, the original file remains intact.

        The process is atomic:
        1. Load existing data from the file
        2. Merge new data with existing data
        3. Sort combined data by time
        4. Write to a temporary file
        5. Only if successful, replace the original file

        Parameters:
            file_path (Path): The path to the existing HDF5 file to which data will be inserted.
            data_dict_to_save (dict[str, Any]): The dictionary containing variable data to insert.
                Must include a "time" key.

        Returns:
            dict[str, Any]: The merged and sorted `data_dict_to_save`.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            KeyError: If the "time" key is missing from `data_dict_to_save`.
            ValueError: If dimensional mismatches occur or if time values are not unique.
        """
        if not file_path.exists():
            msg = f"Cannot append: file does not exist: {file_path}"
            raise FileNotFoundError(msg)

        if "time" not in data_dict_to_save:
            msg = "Cannot append: missing 'time' in data_dict_to_save."
            raise KeyError(msg)

        new_time = data_dict_to_save["time"]
        new_time_len = int(new_time.shape[0])
        if new_time_len == 0:
            logger.info(f"No new time data to insert for {file_path}")
            return data_dict_to_save

        try:
            logger.info(f"Loading existing data from {file_path}...")
            existing_data = self._load_h5_data(file_path)

            logger.info(f"Merging and sorting data for {file_path}...")
            merged_data = self._merge_and_sort_data_h5(existing_data, data_dict_to_save)

            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False, dir=file_path.parent) as tmp_file:
                tmp_path = Path(tmp_file.name)

            try:
                logger.info(f"Writing merged data to temporary file {tmp_path.name}...")
                self._write_h5_file(tmp_path, merged_data)

                logger.info(f"Replacing original file with merged data for {file_path}...")
                shutil.move(str(tmp_path), str(file_path))
                logger.info(f"Successfully inserted data into {file_path}")

                return merged_data  # noqa: TRY300

            except Exception:
                if tmp_path.exists():
                    tmp_path.unlink()
                logger.exception("Failed to write merged data to temporary file")
                raise

        except Exception:
            logger.exception(f"Failed to insert data into {file_path}")
            raise

    def save_single_file(self, file_path: Path, dict_to_save: dict[str, Any], *, append: bool = False) -> None:
        """Saves variable data to a monthly HDF5 file.

        Parameters:
            file_path (Path): The path to the file where the dictionary will be saved.
                              The file extension determines the format.
            dict_to_save (dict[str, Any]): The dictionary containing variable data to save.
        """
        logger.info(f"Saving file {file_path}...")

        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.exists() and append:
            self.append_data(file_path, dict_to_save)
            return

        super()._write_h5_file(file_path, dict_to_save)
