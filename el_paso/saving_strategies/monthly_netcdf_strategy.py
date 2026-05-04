# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import shutil
import tempfile
import typing
from pathlib import Path
from typing import Any

import netCDF4 as nC
import numpy as np
from swvo.io.RBMDataSet.utils import read_all_datasets_netcdf

import el_paso as ep
from el_paso.saving_strategies.monthly_h5_strategy import MonthlyH5Strategy
from el_paso.saving_strategy import OutputFile

if typing.TYPE_CHECKING:
    from datetime import datetime

    from el_paso.data_standard import DataStandard
    from el_paso.processing.magnetic_field_utils import MagneticFieldLiteral

logger = logging.getLogger(__name__)


class MonthlyNetCDFStrategy(MonthlyH5Strategy):
    """A saving strategy that saves data to monthly NetCDF files.

    This strategy organizes and saves processed scientific data into a series of
    NetCDF files, partitioned by month. It inherits from `MonthlyH5Strategy` but
    overrides the file saving logic to use the NetCDF format, which is widely used
    in climate and earth science for storing array-oriented scientific data.

    The strategy standardizes variables based on a provided `DataStandard` and
    structures the output files with a consistent naming convention that includes
    the file stem, date range, and magnetic field models used. It supports
    multiple magnetic field models and automatically configures the output files
    and their dependencies.
    """

    output_files: list[OutputFile]

    file_path: Path
    dependency_dict: dict[str, list[str]]

    def __init__(
        self,
        base_data_path: str | Path,
        file_name_stem: str,
        mag_field: MagneticFieldLiteral | list[MagneticFieldLiteral],
        data_standard: DataStandard | None = None,
        root_metadata: dict[str, str] | None = None,
    ) -> None:
        """Initializes the monthly NetCDF saving strategy.

        Parameters:
            base_data_path (str | Path): The base directory where the output NetCDF files will be saved.
            file_name_stem (str): The base name for the output files (e.g., "my_data").
            mag_field (MagneticFieldLiteral | list[MagneticFieldLiteral]):
                A string or list of strings specifying the magnetic field models used.
            data_standard (DataStandard | None):
                An optional `DataStandard` instance to use for standardizing variables.
                If `None`, `ep.data_standards.PRBEMStandard` is used by default.
        """
        if isinstance(mag_field, str):
            mag_field = [mag_field]

        if data_standard is None:
            data_standard = ep.data_standards.PRBEMStandard()

        self.base_data_path = Path(base_data_path)
        self.file_name_stem = file_name_stem
        self.mag_field_list = mag_field
        self.standard = data_standard
        self.root_metadata = root_metadata

        output_file_entries = [
            "time",
            "flux/FEDU",
            "flux/FEDO",
            "flux/FEIU",
            "flux/alpha_eq",
            "flux/energy",
            "flux/alpha_local",
            "position/xGEO",
            "psd/PSD",
            "density/density_local",
        ]

        for single_mag_field in self.mag_field_list:
            output_file_entries.extend(
                [
                    f"position/{single_mag_field}/MLT",
                    f"position/{single_mag_field}/R0",
                    f"position/{single_mag_field}/Lstar",
                    f"position/{single_mag_field}/Lm",
                    f"mag_field/{single_mag_field}/B_eq",
                    f"mag_field/{single_mag_field}/B_local",
                    f"psd/{single_mag_field}/inv_mu",
                    f"psd/{single_mag_field}/inv_K",
                    f"density/{single_mag_field}/density_eq",
                ]
            )
        self.output_files = [
            OutputFile("full", output_file_entries, save_incomplete=True),
        ]

        self.dependency_dict = {
            "time": ["time"],
            "flux/FEDU": ["time", "energy", "alpha"],
            "flux/FEDO": ["time", "energy"],
            "flux/FEIU": ["time", "energy", "alpha"],
            "flux/alpha_eq": ["time", "alpha"],
            "flux/energy": ["time", "energy"],
            "flux/alpha_local": ["time", "alpha"],
            "position/xGEO": ["time", "xGEO_components"],
            "psd/PSD": ["time", "energy", "alpha"],
            "density/density_local": ["time"],
        }

        for single_mag_field in mag_field:
            self.dependency_dict |= {
                f"position/{single_mag_field}/MLT": ["time"],
                f"position/{single_mag_field}/R0": ["time"],
                f"position/{single_mag_field}/Lstar": ["time", "alpha"],
                f"position/{single_mag_field}/Lm": ["time", "alpha"],
                f"mag_field/{single_mag_field}/B_eq": ["time"],
                f"mag_field/{single_mag_field}/B_local": ["time"],
                f"psd/{single_mag_field}/inv_mu": ["time", "energy", "alpha"],
                f"psd/{single_mag_field}/inv_K": ["time", "alpha"],
                f"density/{single_mag_field}/density_eq": ["time"],
            }

    def get_file_path(self, interval_start: datetime, interval_end: datetime, output_file: OutputFile) -> Path:  # noqa: ARG002
        """Generates the file path for a monthly NetCDF file.

        The file name is constructed from the `file_name_stem`, the date range of the interval,
        and the specified magnetic field models, with a `.nc` extension.

        Parameters:
            interval_start (datetime): The start of the time interval.
            interval_end (datetime): The end of the time interval.
            output_file (OutputFile): The configuration for the output file.

        Returns:
            Path: The full file path for the NetCDF file.
        """
        start_year_month_day = interval_start.strftime("%Y%m%d")
        end_year_month_day = interval_end.strftime("%Y%m%d")

        file_name = f"{self.file_name_stem}_{start_year_month_day}to{end_year_month_day}"

        for mag_field in self.mag_field_list:
            file_name += f"_{mag_field}"

        file_name += ".nc"

        return self.base_data_path / file_name

    def standardize_variable(
        self, variable: ep.Variable, name_in_file: str, *, first_call_of_interval: bool
    ) -> ep.Variable:
        """Standardizes a variable based on the configured `DataStandard`.

        This method delegates the standardization process to a `DataStandard` instance,
        ensuring that the variable's units and dimensions are consistent with the
        defined standard.

        Parameters:
            variable (ep.Variable): The variable instance to be standardized.
            name_in_file (str): The name of the variable as it will appear in the file.
            first_call_of_interval (bool): Flag to indicate if it is the first call of a time interval

        Returns:
            ep.Variable: The standardized variable.
        """
        return self.standard.standardize_variable(
            name_in_file, variable, reset_consistency_check=first_call_of_interval
        )

    def _load_netcdf_data(self, file_path: Path) -> dict[str, Any]:
        """Load all data from an existing NetCDF file, including metadata.

        Uses read_all_datasets_netcdf from swvo if available for efficient recursive loading,
        otherwise falls back to custom implementation.

        Parameters:
            file_path (Path): The path to the NetCDF file to load.

        Returns:
            dict[str, Any]: A dictionary containing all variables and metadata from the file.
        """
        loaded_data: dict[str, Any] = {"metadata": {}}

        datasets = read_all_datasets_netcdf(file_path)
        loaded_data.update(datasets)

        # load metadata

        with nC.Dataset(file_path, "r", format="NETCDF4") as file:

            def _recursively_load_metadata(group: nC.Group | nC.Dataset, prefix: str = "") -> None:
                """Recursively load metadata from groups and subgroups."""
                for var_name, var in group.variables.items():
                    full_path = f"{prefix}{var_name}" if prefix else var_name
                    if full_path not in loaded_data:
                        continue
                    loaded_data["metadata"][full_path] = {
                        "unit": getattr(var, "units", "unknown"),
                        "source_files": getattr(var, "source", "unknown"),
                        "processing_notes": getattr(var, "history", "unknown"),
                        "description": getattr(var, "description", "unknown"),
                    }

                for group_name, subgroup in group.groups.items():
                    _recursively_load_metadata(subgroup, f"{prefix}{group_name}/")

            _recursively_load_metadata(file)

        return loaded_data

    def _merge_and_sort_data(  # noqa: C901
        self,
        existing_data: dict[str, Any],
        new_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge using pickle-style logic with robust shape handling."""

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

        # insertion index (assumes sorted time, same as pickle version)
        insert_idx = int(np.searchsorted(existing_time, new_time[0]))

        merged: dict[str, Any] = {"metadata": existing_data.get("metadata", {}).copy()}

        if "metadata" in new_data:
            merged["metadata"].update(new_data["metadata"])

        all_keys = set(existing_data.keys()) | set(new_data.keys())

        for key in all_keys:
            if key == "metadata":
                continue

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

    def _calculate_dimensions(self, data_dict: dict[str, Any]) -> dict[str, int]:
        """Calculate dimension sizes from the data dictionary.

        Parameters:
            data_dict (dict[str, Any]): The data dictionary containing variables.

        Returns:
            dict[str, int]: Dictionary with dimension names as keys and sizes as values.
        """
        dimensions = {
            "time": data_dict["time"].shape[0],
            "alpha": 0,
            "energy": 0,
        }

        if "flux/alpha_eq" in data_dict and data_dict["flux/alpha_eq"].size > 0:
            dimensions["alpha"] = data_dict["flux/alpha_eq"].shape[1]
        elif "flux/alpha_local" in data_dict and data_dict["flux/alpha_local"].size > 0:
            dimensions["alpha"] = data_dict["flux/alpha_local"].shape[1]

        if "flux/energy" in data_dict and data_dict["flux/energy"].size > 0:
            dimensions["energy"] = data_dict["flux/energy"].shape[1]

        if "position/xGEO" in data_dict and data_dict["position/xGEO"].size > 0:
            dimensions["xGEO_components"] = 3

        return dimensions

    def _write_data_to_netcdf_file(self, file: nC.Dataset | nC.Group, data_dict: dict[str, Any]) -> None:
        """Write variables to a NetCDF file or group.

        Parameters:
            file (nC.Dataset | nC.Group): The NetCDF dataset or group to write to.
            data_dict (dict[str, Any]): The data dictionary containing variables to write.
        """
        for path, value in data_dict.items():
            if path == "metadata":
                continue

            if getattr(value, "size", 0) == 0:
                continue

            path_parts = path.split("/")
            groups = path_parts[:-1]
            dataset_name = path_parts[-1]

            curr_hierarchy: nC.Group | nC.Dataset = file
            for group in groups:
                if group not in curr_hierarchy.groups:
                    curr_hierarchy = curr_hierarchy.createGroup(group)
                else:
                    curr_hierarchy = typing.cast("nC.Group", curr_hierarchy[group])

            data_set = typing.cast(
                "nC.Variable[Any]",
                curr_hierarchy.createVariable(
                    dataset_name, "float64", self.dependency_dict[path], zlib=True, complevel=5, shuffle=True
                ),
            )

            data_set[:, ...] = value

            if path in data_dict.get("metadata", {}):
                metadata = data_dict["metadata"][path]
                data_set.units = metadata["unit"]
                data_set.source = metadata["source_files"]
                data_set.history = metadata["processing_notes"]
                data_set.description = metadata["description"]

    def _write_netcdf_file(self, file_path: Path, data_dict: dict[str, Any]) -> None:
        """Create and write a NetCDF file from a data dictionary.

        Parameters:
            file_path (Path): The path to the file where the data will be saved.
            data_dict (dict[str, Any]): The data dictionary containing variables and metadata.
        """
        with nC.Dataset(file_path, "w", format="NETCDF4") as file:
            if self.root_metadata is not None:
                for key, value in self.root_metadata.items():
                    setattr(file, key, value)

            size_time = data_dict["time"].shape[0]
            if size_time == 0:
                logger.info(f"Skipping write for {file_path.name} (time has length 0).")
                return

            # Calculate and create dimensions
            dimensions = self._calculate_dimensions(data_dict)
            file.createDimension("time", None)  # Unlimited dimension
            for dim_name, dim_size in dimensions.items():
                if dim_name != "time":
                    file.createDimension(dim_name, dim_size)

            # Write variables
            self._write_data_to_netcdf_file(file, data_dict)

    def append_data(self, file_path: Path, data_dict_to_save: dict[str, Any]) -> dict[str, Any]:
        """Insert new data into an existing NetCDF file, maintaining sorted order by timestamp.

        This method loads the existing data, merges it with new data, sorts by timestamp,
        and writes back to the file. If any step fails, the original file remains intact.

        The process is atomic:
        1. Load existing data from the file
        2. Merge new data with existing data
        3. Sort combined data by time
        4. Write to a temporary file
        5. Only if successful, replace the original file

        Parameters:
            file_path (Path): The path to the existing NetCDF file to which data will be inserted.
            data_dict_to_save (dict[str, Any]): The dictionary containing variable data to insert.
                Must include a "time" key.

        Returns:
            dict[str, Any]: The merged and sorted `data_dict_to_save`.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            KeyError: If the "time" key is missing from `data_dict_to_save`.
            ValueError: If the existing NetCDF file does not have an unlimited "time" dimension.
        """
        if not file_path.exists():
            msg = f"Cannot append: file does not exist: {file_path}"
            raise FileNotFoundError(msg)

        if "time" not in data_dict_to_save:
            msg = "Cannot append: missing 'time' in data_dict_to_save."
            raise KeyError(msg)

        # Validate that the time dimension is unlimited
        with nC.Dataset(file_path, "r", format="NETCDF4") as file:
            time_dim = file.dimensions.get("time")
            if time_dim is None or not time_dim.isunlimited():
                msg = (
                    "Cannot append: the existing NetCDF file does not have an "
                    "unlimited 'time' dimension. Recreate the file with 'time' "
                    "created as unlimited (None)."
                )
                raise ValueError(msg)

        new_time = data_dict_to_save["time"]
        new_time_len = int(new_time.shape[0])
        if new_time_len == 0:
            logger.info(f"No new time data to insert for {file_path.name}")
            return data_dict_to_save

        try:
            logger.info(f"Loading existing data from {file_path.name}...")
            existing_data = self._load_netcdf_data(file_path)

            logger.info(f"Merging and sorting data for {file_path.name}...")
            merged_data = self._merge_and_sort_data(existing_data, data_dict_to_save)

            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False, dir=file_path.parent) as tmp_file:
                tmp_path = Path(tmp_file.name)

            try:
                logger.info(f"Writing merged data to temporary file {tmp_path.name}...")
                self._write_netcdf_file(tmp_path, merged_data)

                logger.info(f"Replacing original file with merged data for {file_path.name}...")
                shutil.move(str(tmp_path), str(file_path))
                logger.info(f"Successfully inserted data into {file_path.name}")

                return merged_data  # noqa: TRY300

            except Exception:
                if tmp_path.exists():
                    tmp_path.unlink()
                logger.exception("Failed to write merged data to temporary file")
                raise

        except Exception:
            logger.exception(f"Failed to insert data into {file_path.name}")
            raise

    def save_single_file(self, file_path: Path, dict_to_save: dict[str, Any], *, append: bool = False) -> None:
        """Saves a dictionary of variables to a single NetCDF file.

        This method creates a new NetCDF4 file, defines dimensions based on the data,
        and writes each variable as a dataset. It also attaches metadata as attributes
        to the datasets.

        Parameters:
            file_path (Path): The path to the file where the data will be saved.
            dict_to_save (dict[str, Any]): The dictionary containing variable data.
            append (bool, optional): If `True`, attempts to append data to an existing file.
                If the existing file has an unlimited `time` dimension, the data will be
                inserted in sorted order by timestamp, replacing any duplicate entries.

        Note:
            For appending to work, the original file must have been created with an
            unlimited `time` dimension.
        """
        logger.info(f"Saving file {file_path.name}...")

        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.exists() and append:
            self.append_data(file_path, dict_to_save)
            return

        self._write_netcdf_file(file_path, dict_to_save)
