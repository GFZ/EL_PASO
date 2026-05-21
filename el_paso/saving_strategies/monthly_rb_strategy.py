# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import calendar
import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import netCDF4 as nC
import numpy as np

import el_paso as ep
from el_paso.data_standards import PRBEMStandard
from el_paso.saving_strategy import OutputFile, SavingStrategy

if TYPE_CHECKING:
    from el_paso.processing.magnetic_field_utils import MagneticFieldLiteral
    from el_paso.typing import (
        DataStandard,
        FileLoader,
        FileWriter,
        InternalName,
        MFSFormats,
        SavedDataDict,
        StandardName,
        Variable,
    )

logger = logging.getLogger(__name__)


class MonthlyRBStrategy(SavingStrategy):
    """Save PRBEM-standard data into one monthly file per interval.

    The strategy supports NetCDF, CDF, HDF5, and MATLAB output through a format
    dispatch table. Existing files can be appended by loading the current file,
    replacing overlapping timestamps with the new data block, and atomically
    rewriting the merged data.
    """

    output_files: list[OutputFile]
    dependency_dict: dict[InternalName, list[str]]

    def __init__(
        self,
        base_data_path: str | Path,
        mission: str,
        satellite: str,
        instrument: str,
        mag_field: MagneticFieldLiteral,
        data_standard: DataStandard[StandardName],
        file_format: MFSFormats = "nc",
    ) -> None:
        """Initialize a monthly file saving strategy.

        Parameters:
            base_data_path (str | Path): Directory where monthly files are written.
            mission (str): Mission name, used in file path and name generation.
            satellite (str): Satellite name, used in file path and name generation.
            instrument (str): Instrument name, used in file path and name generation.
            mag_field (MagneticFieldLiteral): Magnetic field model name. Monthly files use one model.
            file_format (MFSFormats): One of ``"nc"``, ``"cdf"``, ``"h5"``, or ``"mat"``.
                A leading dot is also accepted.
            data_standard (DataStandard): Instance of the data standard implementation. Deafults to
                `ep.data_standards.PRBEMStandard()`

        Attributes:
            output_files: List of output file configurations, with variable names
                defined by ``_get_output_file_entries()``.
            dependency_dict: Dictionary defining NetCDF dimension dependencies for
                all variables in ``output_files``.
        """
        self.base_data_path = Path(base_data_path)
        self.mission = mission
        self.satellite = satellite
        self.instrument = instrument
        self.mag_field = mag_field
        self.data_standard = data_standard
        self.file_format = ep.utils.normalize_file_format(file_format)

        self.output_files = [
            OutputFile("full", self._get_output_file_entries(), save_incomplete=True),
        ]

        self._writers: dict[str, FileWriter] = {
            ".mat": ep.utils.write_mat_file,
            ".h5": ep.utils.write_h5_file,
            ".nc": ep.utils.write_netcdf_file,
            ".cdf": ep.utils.write_cdf_file,
        }
        self._loaders: dict[str, FileLoader] = {
            ".mat": ep.utils.load_mat_data,
            ".h5": ep.utils.load_h5_data,
            ".nc": ep.utils.load_netcdf_data,
            ".cdf": ep.utils.load_cdf_data,
        }

    def _get_output_file_entries(self) -> list[InternalName]:
        """Return the standard variable list plus user-defined custom variables."""
        return [
            "FEDU",
            "Epoch",
            "Alpha_Eq",
            "Energy_FEDU",
            "Alpha",
            "B_Calc",
            "B_Eq",
            "InvK",
            "InvMu",
            "Position",
            "PSD",
            "R_Eq",
            "MLT",
            "L_m",
            "L_star",
        ]

    def _sanitize_dimension_name(self, variable_name: str) -> str:
        """Return a NetCDF-safe root dimension name derived from a variable path."""
        return "".join(char if char.isalnum() else "_" for char in variable_name).strip("_") or "custom"

    def _register_writer(self, extension: str, writer: FileWriter) -> None:
        """Register or replace the writer used for a file extension.

        TODO: We may want to support user defined formats in the future, so this method could be extended to check.
        """
        normalized = ep.utils.normalize_file_format(extension)
        self._writers[normalized] = writer

    def get_time_intervals_to_save(
        self, start_time: datetime | None, end_time: datetime | None
    ) -> list[tuple[datetime, datetime]]:
        """Split the requested time range into full monthly intervals."""
        time_intervals: list[tuple[datetime, datetime]] = []

        if start_time is None or end_time is None:
            msg = "start_time and end_time must be provided for MonthlyFileStrategy!"
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
                if month == 12
                else datetime(year, month + 1, 1, tzinfo=timezone.utc)
            )

        return time_intervals

    def get_file_path_stem(self) -> Path:
        return self.base_data_path / self.mission.upper() / self.satellite.lower()

    def get_file_name_stem(self) -> str:
        return self.satellite.lower() + "_" + self.instrument.lower()

    def get_file_path(self, interval_start: datetime, interval_end: datetime, output_file: OutputFile) -> Path:  # noqa: ARG002
        """Generate the monthly file path for the configured format."""
        start_year_month_day = interval_start.strftime("%Y%m%d")
        end_year_month_day = interval_end.strftime("%Y%m%d")
        file_name = (
            f"{self.get_file_name_stem()}_{start_year_month_day}to{end_year_month_day}_"
            f"{self.mag_field}{self.file_format}"
        )

        return self.get_file_path_stem() / file_name

    def standardize_variable(
        self,
        variable: Variable,
        internal_name: InternalName,
        *,
        first_call_of_interval: bool,
    ) -> Variable:
        """Standardize a variable through the configured data standard."""
        return self.data_standard.standardize_variable(
            internal_name, variable, reset_consistency_check=first_call_of_interval
        )

    def save_single_file(self, file_path: Path, dict_to_save: SavedDataDict, *, append: bool = False) -> None:
        """Save one monthly file, optionally appending to an existing file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        format_name = ep.utils.normalize_file_format(file_path.suffix)
        writer = self._writers.get(format_name)

        if writer is None:
            msg = f"The '{format_name}' format is not implemented."
            logger.error(msg)
            raise NotImplementedError(msg)

        if file_path.exists() and append:
            logger.info(f"Appending and saving to existing file: {file_path.resolve()}")
            self.append_data(file_path, dict_to_save)
            return

        logger.info(f"Saving file: {file_path.resolve()}")

        writer(file_path, dict_to_save, self.data_standard)

    def append_data(self, file_path: Path, data_dict_to_save: SavedDataDict) -> SavedDataDict:
        """Append data to any supported monthly file format.

        Existing data is loaded with the loader for ``file_path.suffix``, merged
        by timestamp with the new dictionary, and written to a temporary file
        before replacing the original file.
        """
        if not file_path.exists():
            msg = f"Cannot append: file does not exist: {file_path}"
            raise FileNotFoundError(msg)

        new_time = np.asarray(data_dict_to_save["Epoch"])
        if int(new_time.shape[0]) == 0:
            logger.info(f"No new time data to insert for {file_path.name}")
            return data_dict_to_save

        format_name = ep.utils.normalize_file_format(file_path.suffix)
        loader = self._loaders.get(format_name)
        writer = self._writers.get(format_name)
        if loader is None or writer is None:
            msg = f"Appending to '{format_name}' files is not supported by MonthlyFileStrategy."
            logger.error(msg)
            raise NotImplementedError(msg)

        if format_name == ".nc":
            self._validate_netcdf_appendable(file_path)

        logger.info(f"Loading existing data from {file_path.name}")
        existing_data = loader(file_path)

        logger.info(f"Merging and sorting data for {file_path.name}")
        merged_data = self._merge_and_sort_data(existing_data, data_dict_to_save)

        with tempfile.NamedTemporaryFile(suffix=format_name, delete=False, dir=file_path.parent) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            logger.info(f"Writing merged data to temporary file {tmp_path.name}")
            writer(tmp_path, merged_data, self.data_standard)

            logger.info(f"Replacing original file with merged data for {file_path.name}")
            shutil.move(str(tmp_path), str(file_path))
            logger.info(f"Successfully inserted data into {file_path.resolve()}")

            return merged_data  # noqa: TRY300
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            logger.exception("Failed to write merged data to temporary file")
            raise

    def _merge_and_sort_data(
        self,
        existing_data: dict[StandardName | Literal["metadata"], Any],
        new_data: SavedDataDict,
    ) -> SavedDataDict:
        """Merge two dictionaries along the time axis, replacing duplicate times."""

        def _normalize_1d(arr: np.ndarray) -> np.ndarray:
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[1] == 1:
                return arr.reshape(-1)
            return arr

        existing_data_internal: SavedDataDict = {}
        for name, value in existing_data.items():
            if name == "metadata":
                existing_data_internal["metadata"] = value
            else:
                internal_name = self.data_standard.get_internal_name(name)
                if internal_name is None:
                    msg = f"Could not find necessary internal name for variable: {name}"
                    raise ValueError(msg)
                existing_data_internal[internal_name] = value

        existing_time = _normalize_1d(existing_data_internal["Epoch"])
        new_time = _normalize_1d(new_data["Epoch"])
        mask_keep_existing = ~np.isin(existing_time, new_time)
        insert_idx = int(np.searchsorted(existing_time, new_time[0]))

        merged: SavedDataDict = {}
        existing_metadata = existing_data_internal.get("metadata", {})
        new_metadata = new_data.get("metadata", {})
        if isinstance(existing_metadata, dict) and isinstance(new_metadata, dict):
            merged["metadata"] = {**existing_metadata, **new_metadata}
        elif "metadata" in new_data:
            merged["metadata"] = new_metadata
        elif "metadata" in existing_data_internal:
            merged["metadata"] = existing_metadata

        all_keys = set(existing_data_internal.keys()) | set(new_data.keys())
        for key in all_keys:
            if key == "metadata" or key.startswith("__"):
                continue

            if key not in existing_data_internal:
                merged[key] = new_data[key]
                continue

            if key not in new_data:
                merged[key] = existing_data_internal[key]
                continue

            if key.startswith("custom/"):
                merged[key] = new_data[key]
                continue

            v1 = _normalize_1d(np.asarray(existing_data_internal[key]))
            v2 = _normalize_1d(np.asarray(new_data[key]))

            if v1.ndim != v2.ndim:
                msg = f"{key}: ndim mismatch {v1.shape} vs {v2.shape}"
                logger.error(msg)
                raise ValueError(msg)

            if v1.ndim > 1 and v1.shape[1:] != v2.shape[1:]:
                msg = f"{key}: shape mismatch {v1.shape} vs {v2.shape}"
                logger.error(msg)
                raise ValueError(msg)

            v1_trunc = v1[mask_keep_existing]
            merged_val = v2 if v1_trunc.size == 0 else np.insert(v1_trunc, insert_idx, v2, axis=0)

            if key == "Epoch":
                t = np.asarray(merged_val)
                if len(np.unique(t)) != len(t):
                    msg = "Time values are not unique after merge for key 'time'"
                    logger.error(msg)
                    raise ValueError(msg)

            merged[key] = merged_val

        return merged

    def _validate_netcdf_appendable(self, file_path: Path) -> None:
        """Validate that the existing NetCDF file has an unlimited time dimension."""
        with nC.Dataset(file_path, "r", format="NETCDF4") as file:
            time_dim = file.dimensions.get("time")
            if time_dim is None or not time_dim.isunlimited():
                msg = (
                    "Cannot append: the existing NetCDF file does not have an "
                    "unlimited 'time' dimension. Recreate the file with 'time' "
                    "created as unlimited (None)."
                )
                raise ValueError(msg)
