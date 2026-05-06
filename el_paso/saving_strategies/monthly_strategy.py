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
import typing
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import cdflib
import h5py
import netCDF4 as nC
import numpy as np
from scipy.io import loadmat, savemat

import el_paso as ep
from el_paso.saving_strategy import OutputFile, SavingStrategy
from el_paso.variable import Variable

if TYPE_CHECKING:
    from el_paso.data_standard import DataStandard
    from el_paso.processing.magnetic_field_utils import MagneticFieldLiteral

logger = logging.getLogger(__name__)

MFSFormats = Literal["nc", "cdf", "h5", "mat", ".nc", ".cdf", ".h5", ".mat"]
FormatWriter = Callable[[Path, dict[str, Any]], None]
FormatLoader = Callable[[Path], dict[str, Any]]


class MonthlyFileStrategy(SavingStrategy):
    """Save PRBEM-standard data into one monthly file per interval.

    The strategy supports NetCDF, CDF, HDF5, and MATLAB output through a format
    dispatch table. Existing files can be appended by loading the current file,
    replacing overlapping timestamps with the new data block, and atomically
    rewriting the merged data.
    """

    output_files: list[OutputFile]
    dependency_dict: dict[str, list[str]]

    def __init__(
        self,
        base_data_path: str | Path,
        file_name_stem: str,
        mag_field: MagneticFieldLiteral,
        file_format: MFSFormats = "h5",
        data_standard: DataStandard | None = None,
        root_metadata: dict[str, str] | None = None,
        *,
        custom_variables: dict[str, Variable] | None = None,
    ) -> None:
        """Initialize a monthly file saving strategy.

        Parameters:
            base_data_path (str | Path): Directory where monthly files are written.
            file_name_stem (str): Prefix used in generated monthly file names.
            mag_field (MagneticFieldLiteral): Magnetic field model name. Monthly files use one model.
            file_format (MFSFormats): One of ``"nc"``, ``"cdf"``, ``"h5"``, or ``"mat"``.
                A leading dot is also accepted.
            data_standard (DataStandard | None): Standardization implementation. Defaults to
                [`el_paso.data_standards.PRBEMStandard`][]
            root_metadata (dict[str, str] | None): Optional global NetCDF attributes.
            custom_variables (dict[str, Variable] | None): Custom variables to include in the output.
                Each entry is saved below ``custom/`` using its dictionary key as the variable path.

        Attributes:
            output_files: List of output file configurations, with variable names
                defined by ``_get_output_file_entries()``.
            dependency_dict: Dictionary defining NetCDF dimension dependencies for
                all variables in ``output_files``.
        """
        self.base_data_path = Path(base_data_path)
        self.file_name_stem = file_name_stem
        self.mag_field = mag_field
        self.file_format = self._normalize_file_format(file_format)
        self.root_metadata = root_metadata
        self.custom_variables = self._validate_custom_variables(custom_variables)

        if data_standard is None:
            data_standard = ep.data_standards.PRBEMStandard()
        self.standard = data_standard
        self.data_standard = data_standard

        self.output_files = [
            OutputFile("full", self._get_output_file_entries(), save_incomplete=True),
        ]
        self.dependency_dict = self._get_dependency_dict()

        self._writers: dict[str, FormatWriter] = {
            ".mat": self._write_mat_file,
            ".h5": self._write_h5_file,
            ".nc": self._write_netcdf_file,
            ".cdf": self._write_cdf_file,
        }
        self._loaders: dict[str, FormatLoader] = {
            ".mat": self._load_mat_data,
            ".h5": self._load_h5_data,
            ".nc": self._load_netcdf_data,
            ".cdf": self._load_cdf_data,
        }

    def _normalize_file_format(self, file_format: str) -> str:
        """Return a normalized file extension for the requested monthly format."""
        normalized = file_format.lower()
        if not normalized.startswith("."):
            normalized = f".{normalized}"

        if normalized not in {".nc", ".cdf", ".h5", ".mat"}:
            msg = "MonthlyFileStrategy supports only 'nc', 'cdf', 'h5', and 'mat' formats."
            raise ValueError(msg)

        return normalized

    def _validate_custom_variables(self, custom_variables: dict[str, Variable] | None) -> dict[str, Variable]:
        """Validate and copy user-defined custom variables."""
        if custom_variables is None:
            return {}

        standard_entries = set(self._get_standard_output_file_entries())
        validated: dict[str, Variable] = {}
        custom_output_paths: set[str] = set()
        for name, variable in custom_variables.items():
            if not isinstance(name, str) or len(name.strip()) == 0:
                msg = "Custom variable names must be non-empty strings."
                raise ValueError(msg)

            output_path = self._get_custom_variable_path(name)
            if output_path == "custom/":
                msg = "Custom variable names must contain a non-empty name after the custom group."
                raise ValueError(msg)

            if name == "metadata" or output_path == "custom/metadata":
                msg = f"Custom variable name '{name}' is reserved."
                raise ValueError(msg)

            if output_path in standard_entries:
                msg = f"Custom variable '{name}' conflicts with a standard monthly variable."
                raise ValueError(msg)

            if output_path in custom_output_paths:
                msg = f"Custom variable '{name}' maps to a duplicate output path '{output_path}'."
                raise ValueError(msg)

            if not isinstance(variable, Variable):
                msg = f"Custom variable '{name}' must be an el_paso.Variable instance."
                raise TypeError(msg)

            custom_output_paths.add(output_path)
            validated[name] = variable

        return validated

    def _get_standard_output_file_entries(self) -> list[str]:
        """Return the standard PRBEM monthly variable list."""
        return [
            "time",
            "flux/FEDU",
            "flux/FEDO",
            "flux/FEIU",
            "flux/alpha_eq",
            "flux/energy",
            "flux/alpha_local",
            "position/xGEO",
            f"position/{self.mag_field}/MLT",
            f"position/{self.mag_field}/R0",
            f"position/{self.mag_field}/Lstar",
            f"position/{self.mag_field}/Lm",
            f"mag_field/{self.mag_field}/B_eq",
            f"mag_field/{self.mag_field}/B_local",
            "psd/PSD",
            f"psd/{self.mag_field}/inv_mu",
            f"psd/{self.mag_field}/inv_K",
            "density/density_local",
            f"density/{self.mag_field}/density_eq",
        ]

    def _get_output_file_entries(self) -> list[str]:
        """Return the standard variable list plus user-defined custom variables."""
        entries = self._get_standard_output_file_entries()
        entries.extend(self._get_custom_variable_path(name) for name in self.custom_variables)
        return entries

    def _get_dependency_dict(self) -> dict[str, list[str]]:
        """Return NetCDF dimension dependencies for all monthly variables."""
        dependency_dict = {
            "time": ["time"],
            "flux/FEDU": ["time", "energy", "alpha"],
            "flux/FEDO": ["time", "energy"],
            "flux/FEIU": ["time", "energy", "alpha"],
            "flux/alpha_eq": ["time", "alpha"],
            "flux/energy": ["time", "energy"],
            "flux/alpha_local": ["time", "alpha"],
            "position/xGEO": ["time", "xGEO_components"],
            f"position/{self.mag_field}/MLT": ["time"],
            f"position/{self.mag_field}/R0": ["time"],
            f"position/{self.mag_field}/Lstar": ["time", "alpha"],
            f"position/{self.mag_field}/Lm": ["time", "alpha"],
            f"mag_field/{self.mag_field}/B_eq": ["time"],
            f"mag_field/{self.mag_field}/B_local": ["time"],
            "psd/PSD": ["time", "energy", "alpha"],
            f"psd/{self.mag_field}/inv_mu": ["time", "energy", "alpha"],
            f"psd/{self.mag_field}/inv_K": ["time", "alpha"],
            "density/density_local": ["time"],
            f"density/{self.mag_field}/density_eq": ["time"],
        }
        dependency_dict.update(self._get_custom_dependency_dict())
        return dependency_dict

    def _get_custom_dependency_dict(self) -> dict[str, list[str]]:
        """Infer NetCDF dimension dependencies for custom variables.

        Custom monthly variables are independent payloads and do not need to share
        the monthly file's time dimension. Every custom axis gets a variable-specific
        dimension name.
        """
        return {
            self._get_custom_variable_path(name): self._infer_custom_variable_dimensions(
                self._get_custom_variable_path(name),
                variable,
            )
            for name, variable in self.custom_variables.items()
        }

    def _get_custom_variable_path(self, name: str) -> str:
        """Return the output path for a custom variable."""
        name_without_group = name.removeprefix("custom/").strip("/")
        return f"custom/{name_without_group}"

    def _infer_custom_variable_dimensions(self, name: str, variable: Variable) -> list[str]:
        """Infer dimensions for one custom variable from its data shape."""
        return self._infer_custom_array_dimensions(name, np.asarray(variable.get_data()))

    def _infer_custom_array_dimensions(self, name: str, data: np.ndarray) -> list[str]:
        """Infer custom variable dimensions from an array shape."""
        if data.ndim == 0:
            return []

        if data.ndim == 2 and data.shape[1] == 1:  # noqa: PLR2004
            return [f"{self._sanitize_dimension_name(name)}_dim_0"]

        return [f"{self._sanitize_dimension_name(name)}_dim_{axis}" for axis in range(data.ndim)]

    def _sanitize_dimension_name(self, variable_name: str) -> str:
        """Return a NetCDF-safe root dimension name derived from a variable path."""
        return "".join(char if char.isalnum() else "_" for char in variable_name).strip("_") or "custom"

    def _register_writer(self, extension: str, writer: FormatWriter) -> None:
        """Register or replace the writer used for a file extension.

        TODO: We may want to support user defined formats in the future, so this method could be extended to check.
        """
        normalized = self._normalize_file_format(extension)
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
                if month == 12  # noqa: PLR2004
                else datetime(year, month + 1, 1, tzinfo=timezone.utc)
            )

        return time_intervals

    def get_file_path(self, interval_start: datetime, interval_end: datetime, output_file: OutputFile) -> Path:  # noqa: ARG002
        """Generate the monthly file path for the configured format."""
        start_year_month_day = interval_start.strftime("%Y%m%d")
        end_year_month_day = interval_end.strftime("%Y%m%d")
        file_name = (
            f"{self.file_name_stem}_{start_year_month_day}to{end_year_month_day}_{self.mag_field}{self.file_format}"
        )

        return self.base_data_path / file_name

    def standardize_variable(
        self,
        variable: ep.Variable,
        name_in_file: str,
        *,
        first_call_of_interval: bool,
    ) -> ep.Variable:
        """Standardize a variable through the configured data standard."""
        if name_in_file.startswith("custom/"):
            return variable

        return self.standard.standardize_variable(
            name_in_file, variable, reset_consistency_check=first_call_of_interval
        )

    def get_target_variables(
        self,
        output_file: OutputFile,
        variables_dict: dict[str, Variable],
        time_var: Variable | None,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> dict[str, Variable] | None:
        """Return standard monthly variables plus configured custom variables."""
        standard_output_file = OutputFile(
            output_file.name,
            [name for name in output_file.names_to_save if not name.startswith("custom/")],
            output_file.save_incomplete,
        )
        target_variables = super().get_target_variables(
            standard_output_file,
            variables_dict,
            time_var,
            start_time,
            end_time,
        )

        if target_variables is None:
            return None

        for name, variable in self.custom_variables.items():
            output_path = self._get_custom_variable_path(name)
            var_to_save = variables_dict.get(output_path, variables_dict.get(name, variable))
            target_variables[output_path] = deepcopy(var_to_save)

        return target_variables

    def save_single_file(self, file_path: Path, dict_to_save: dict[str, Any], *, append: bool = False) -> None:
        """Save one monthly file, optionally appending to an existing file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        format_name = self._normalize_file_format(file_path.suffix)
        writer = self._writers.get(format_name)

        if writer is None:
            msg = f"The '{format_name}' format is not implemented."
            logger.error(msg)
            raise NotImplementedError(msg)

        if file_path.exists() and append:
            self.append_data(file_path, dict_to_save)
            logger.info(f"Saving file {file_path.resolve()}")
            return

        writer(file_path, dict_to_save)

    def append_data(self, file_path: Path, data_dict_to_save: dict[str, Any]) -> dict[str, Any]:
        """Append data to any supported monthly file format.

        Existing data is loaded with the loader for ``file_path.suffix``, merged
        by timestamp with the new dictionary, and written to a temporary file
        before replacing the original file.
        """
        if not file_path.exists():
            msg = f"Cannot append: file does not exist: {file_path}"
            raise FileNotFoundError(msg)

        if "time" not in data_dict_to_save:
            msg = "Cannot append: missing 'time' in data_dict_to_save."
            raise KeyError(msg)

        new_time = np.asarray(data_dict_to_save["time"])
        if int(new_time.shape[0]) == 0:
            logger.info(f"No new time data to insert for {file_path.name}")
            return data_dict_to_save

        format_name = self._normalize_file_format(file_path.suffix)
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
            writer(tmp_path, merged_data)

            logger.info(f"Replacing original file with merged data for {file_path.name}")
            shutil.move(str(tmp_path), str(file_path))
            logger.info(f"Successfully inserted data into {file_path.resolve()}")

            return merged_data  # noqa: TRY300
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            logger.exception("Failed to write merged data to temporary file")
            raise

    def _merge_and_sort_data(self, existing_data: dict[str, Any], new_data: dict[str, Any]) -> dict[str, Any]:  # noqa: C901, PLR0912
        """Merge two dictionaries along the time axis, replacing duplicate times."""

        def _normalize_1d(arr: np.ndarray) -> np.ndarray:
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[1] == 1:  # noqa: PLR2004
                return arr.reshape(-1)
            return arr

        if "time" not in existing_data or np.asarray(existing_data["time"]).size == 0:
            return new_data

        if "time" not in new_data or np.asarray(new_data["time"]).size == 0:
            return existing_data

        existing_time = _normalize_1d(existing_data["time"])
        new_time = _normalize_1d(new_data["time"])
        mask_keep_existing = ~np.isin(existing_time, new_time)
        insert_idx = int(np.searchsorted(existing_time, new_time[0]))

        merged: dict[str, Any] = {}
        existing_metadata = existing_data.get("metadata", {})
        new_metadata = new_data.get("metadata", {})
        if isinstance(existing_metadata, dict) and isinstance(new_metadata, dict):
            merged["metadata"] = {**existing_metadata, **new_metadata}
        elif "metadata" in new_data:
            merged["metadata"] = new_metadata
        elif "metadata" in existing_data:
            merged["metadata"] = existing_metadata

        all_keys = set(existing_data.keys()) | set(new_data.keys())
        for key in all_keys:
            if key == "metadata" or key.startswith("__"):
                continue

            if key not in existing_data:
                merged[key] = new_data[key]
                continue

            if key not in new_data:
                merged[key] = existing_data[key]
                continue

            if key.startswith("custom/"):
                merged[key] = new_data[key]
                continue

            v1 = _normalize_1d(np.asarray(existing_data[key]))
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

            if key == "time":
                t = np.asarray(merged_val)
                if len(np.unique(t)) != len(t):
                    msg = "Time values are not unique after merge for key 'time'"
                    logger.error(msg)
                    raise ValueError(msg)

            merged[key] = merged_val

        return merged

    def _load_mat_data(self, file_path: Path) -> dict[str, Any]:
        """Load an existing MATLAB file."""
        loaded = loadmat(str(file_path), simplify_cells=True)
        return {key: value for key, value in loaded.items() if not key.startswith("__")}

    def _write_mat_file(self, file_path: Path, data_dict: dict[str, Any]) -> None:
        """Write a MATLAB file."""
        savemat(str(file_path), data_dict)

    def _load_h5_data(self, file_path: Path) -> dict[str, Any]:
        """Load all datasets and dataset attributes from an HDF5 file."""
        loaded_data: dict[str, Any] = {"metadata": {}}

        def _recursively_load_datasets(group: h5py.Group | h5py.File, prefix: str = "") -> None:
            for key, item in group.items():
                full_path = f"{prefix}{key}" if prefix else key
                if isinstance(item, h5py.Dataset):
                    loaded_data[full_path] = np.array(item)
                    loaded_data["metadata"][full_path] = dict(item.attrs.items())
                elif isinstance(item, h5py.Group):
                    _recursively_load_datasets(item, f"{full_path}/")

        with h5py.File(file_path, "r") as file:
            _recursively_load_datasets(file)

        return loaded_data

    def _write_h5_file(self, file_path: Path, data_dict: dict[str, Any]) -> None:
        """Write an HDF5 file with hierarchical groups from slash-delimited paths."""
        with h5py.File(file_path, "w") as file:
            for path, value in data_dict.items():
                if path == "metadata":
                    continue

                path_parts = path.split("/")
                groups = path_parts[:-1]
                dataset_name = path_parts[-1]

                curr_hierarchy = file
                for group in groups:
                    if group not in curr_hierarchy:
                        curr_hierarchy = curr_hierarchy.create_group(group)
                    else:
                        curr_hierarchy = typing.cast("h5py.Group", curr_hierarchy[group])

                data_set = curr_hierarchy.create_dataset(dataset_name, data=value, compression="gzip", shuffle=True)

                metadata_dict = data_dict.get("metadata", {}).get(path, {})
                if not isinstance(metadata_dict, dict):
                    continue

                for key, metadata in metadata_dict.items():
                    if getattr(metadata, "size", None) == 0:
                        continue
                    data_set.attrs[key] = metadata

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

    def _load_netcdf_data(self, file_path: Path) -> dict[str, Any]:
        """Load all variables and variable metadata from a NetCDF file."""
        loaded_data: dict[str, Any] = {"metadata": {}}

        def _recursively_load(group: nC.Group | nC.Dataset, prefix: str = "") -> None:
            for var_name, variable in group.variables.items():
                full_path = f"{prefix}{var_name}" if prefix else var_name
                loaded_data[full_path] = np.array(variable[:])
                loaded_data["metadata"][full_path] = {
                    "unit": getattr(variable, "units", "unknown"),
                    "source_files": getattr(variable, "source", "unknown"),
                    "processing_notes": getattr(variable, "history", "unknown"),
                    "description": getattr(variable, "description", "unknown"),
                }

            for group_name, subgroup in group.groups.items():
                _recursively_load(subgroup, f"{prefix}{group_name}/")

        with nC.Dataset(file_path, "r", format="NETCDF4") as file:
            _recursively_load(file)

        return loaded_data

    def _calculate_dimensions(self, data_dict: dict[str, Any]) -> dict[str, int]:
        """Calculate NetCDF dimension sizes from the data dictionary."""
        dimensions = {
            "time": np.asarray(data_dict["time"]).shape[0],
            "alpha": 0,
            "energy": 0,
        }

        if "flux/alpha_eq" in data_dict and np.asarray(data_dict["flux/alpha_eq"]).size > 0:
            dimensions["alpha"] = np.asarray(data_dict["flux/alpha_eq"]).shape[1]
        elif "flux/alpha_local" in data_dict and np.asarray(data_dict["flux/alpha_local"]).size > 0:
            dimensions["alpha"] = np.asarray(data_dict["flux/alpha_local"]).shape[1]

        if "flux/energy" in data_dict and np.asarray(data_dict["flux/energy"]).size > 0:
            dimensions["energy"] = np.asarray(data_dict["flux/energy"]).shape[1]

        if "position/xGEO" in data_dict and np.asarray(data_dict["position/xGEO"]).size > 0:
            dimensions["xGEO_components"] = 3

        for name in self.custom_variables:
            path = self._get_custom_variable_path(name)
            if path not in data_dict:
                continue

            value_array = np.asarray(data_dict[path])
            if value_array.size == 0:
                continue

            path_dimensions = self._infer_custom_array_dimensions(path, value_array)
            self.dependency_dict[path] = path_dimensions

            for axis, dimension_name in enumerate(path_dimensions):
                if dimension_name == "time":
                    continue

                dimensions[dimension_name] = int(value_array.shape[axis])

        return dimensions

    def _write_data_to_netcdf_file(self, file: nC.Dataset | nC.Group, data_dict: dict[str, Any]) -> None:
        """Write variables to a NetCDF file or group."""
        for path, value in data_dict.items():
            if path == "metadata":
                continue

            value_array = np.asarray(value)
            if value_array.size == 0:
                continue

            path_parts = path.split("/")
            groups = path_parts[:-1]
            dataset_name = path_parts[-1]

            curr_hierarchy: nC.Group | nC.Dataset = file
            for group in groups:
                if group not in curr_hierarchy.groups:
                    curr_hierarchy = curr_hierarchy.createGroup(group)
                else:
                    curr_hierarchy = typing.cast("nC.Group", curr_hierarchy.groups[group])

            dimensions = self.dependency_dict[path]
            data_set = typing.cast(
                "nC.Variable[Any]",
                curr_hierarchy.createVariable(
                    dataset_name,
                    "float64",
                    dimensions,
                    zlib=True,
                    complevel=5,
                    shuffle=True,
                ),
            )

            value_to_write = value_array
            if len(dimensions) == 1 and value_array.ndim == 2 and value_array.shape[1] == 1:  # noqa: PLR2004
                value_to_write = value_array.reshape(-1)

            if len(dimensions) == 0:
                data_set[...] = value_to_write
            else:
                data_set[:, ...] = value_to_write

            if path in data_dict.get("metadata", {}):
                metadata = data_dict["metadata"][path]
                if not isinstance(metadata, dict):
                    continue
                data_set.units = metadata.get("unit", "unknown")
                data_set.source = metadata.get("source_files", "unknown")
                data_set.history = metadata.get("processing_notes", "unknown")
                data_set.description = metadata.get("description", "unknown")

    def _write_netcdf_file(self, file_path: Path, data_dict: dict[str, Any]) -> None:
        """Create and write a NetCDF file from a data dictionary."""
        with nC.Dataset(file_path, "w", format="NETCDF4") as file:
            if self.root_metadata is not None:
                for key, value in self.root_metadata.items():
                    setattr(file, key, value)

            size_time = np.asarray(data_dict["time"]).shape[0]
            if size_time == 0:
                logger.info(f"Skipping write for {file_path.name} (time has length 0).")
                return

            dimensions = self._calculate_dimensions(data_dict)
            file.createDimension("time", None)
            for dim_name, dim_size in dimensions.items():
                if dim_name != "time":
                    file.createDimension(dim_name, dim_size)

            self._write_data_to_netcdf_file(file, data_dict)

    def _load_cdf_data(self, file_path: Path) -> dict[str, Any]:
        """Load all zVariables from an existing CDF file."""
        loaded_data: dict[str, Any] = {"metadata": {}}
        cdf_file = cdflib.CDF(str(file_path))
        try:
            info = cdf_file.cdf_info()
            z_variables = getattr(info, "zVariables", None)
            if z_variables is None and isinstance(info, dict):
                z_variables = info.get("zVariables", [])

            for variable_name in z_variables or []:
                try:
                    loaded_data[variable_name] = np.asarray(cdf_file.varget(variable_name))
                except ValueError as exc:
                    if "No records found" not in str(exc):
                        raise
                    logger.warning(f"Skipping empty CDF variable {variable_name} in {file_path.name}")
                    continue

                try:
                    loaded_data["metadata"][variable_name] = cdf_file.varattsget(variable_name)
                except Exception:  # noqa: BLE001
                    loaded_data["metadata"][variable_name] = {}
        finally:
            close = getattr(cdf_file, "close", None)
            if close is not None:
                close()

        return loaded_data

    def _get_cdf_variable_attrs(self, var_name: str, data_dict: dict[str, Any]) -> dict[str, Any]:
        """Return non-empty CDF variable attributes for a saved variable."""
        metadata = data_dict.get("metadata", {}).get(var_name, {})
        var_attrs: dict[str, Any] = {}

        if isinstance(metadata, dict):
            for attr_name, attr_value in metadata.items():
                if self._is_empty_cdf_attribute(attr_value):
                    logger.debug(f"Skipping empty CDF attribute {var_name}:{attr_name}")
                    continue

                var_attrs[str(attr_name)] = attr_value

        var_attrs["Compress"] = 6
        return var_attrs

    def _is_empty_cdf_attribute(self, value: Any) -> bool:  # noqa: ANN401
        """Return True if cdflib cannot infer a datatype from the attribute value."""
        if value is None:
            return True

        if isinstance(value, (list, tuple, dict, str, bytes)):
            return len(value) == 0

        return getattr(value, "size", None) == 0

    def _write_cdf_file(self, file_path: Path, data_dict: dict[str, Any]) -> None:
        """Write a CDF file."""
        try:
            cdf_file = cdflib.cdfwrite.CDF(str(file_path), delete=True)
            try:
                for var_name, var_data in data_dict.items():
                    if var_name == "metadata":
                        continue

                    if getattr(var_data, "size", 0) == 0:
                        logger.warning(f"Skipping empty variable {var_name}")
                        continue

                    var_data_array = np.asarray(var_data)
                    if np.issubdtype(var_data_array.dtype, np.integer):
                        if var_data_array.dtype == np.int8:
                            cdf_dtype = cdflib.cdfwrite.CDF.CDF_INT1
                        elif var_data_array.dtype == np.int16:
                            cdf_dtype = cdflib.cdfwrite.CDF.CDF_INT2
                        elif var_data_array.dtype == np.int32:
                            cdf_dtype = cdflib.cdfwrite.CDF.CDF_INT4
                        else:
                            cdf_dtype = cdflib.cdfwrite.CDF.CDF_INT8
                    elif np.issubdtype(var_data_array.dtype, np.floating):
                        cdf_dtype = (
                            cdflib.cdfwrite.CDF.CDF_FLOAT
                            if var_data_array.dtype == np.float32
                            else cdflib.cdfwrite.CDF.CDF_DOUBLE
                        )
                    else:
                        var_data_array = var_data_array.astype(np.float64)
                        cdf_dtype = cdflib.cdfwrite.CDF.CDF_DOUBLE

                    var_spec: dict[str, Any] = {
                        "Variable": var_name,
                        "Data_Type": cdf_dtype,
                        "Num_Elements": 1,
                        "Rec_Vary": True,
                        "Dim_Sizes": (list(var_data_array.shape[1:]) if var_data_array.ndim > 1 else []),
                    }
                    var_attrs = self._get_cdf_variable_attrs(var_name, data_dict)

                    cdf_file.write_var(var_spec, var_attrs=var_attrs, var_data=var_data_array)
            finally:
                cdf_file.close()
        except Exception as e:
            msg = f"Failed to write CDF file {file_path}: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e
