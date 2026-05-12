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
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import cdflib
import h5py
import netCDF4 as nC
import numpy as np
from scipy.io import loadmat, savemat

import el_paso as ep
from el_paso.data_standard import DataStandard, InternalName
from el_paso.saving_strategy import OutputFile, SavingStrategy

if TYPE_CHECKING:
    from el_paso.processing.magnetic_field_utils import MagneticFieldLiteral

logger = logging.getLogger(__name__)

MFSFormats = Literal["nc", "cdf", "h5", "mat", ".nc", ".cdf", ".h5", ".mat"]
DataDict = dict[InternalName | Literal["metadata"], Any]
FormatWriter = Callable[[Path, DataDict], None]
FormatLoader = Callable[[Path], DataDict]


class MonthlyFileStrategy(SavingStrategy):
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
        file_name_stem: str,
        mag_field: MagneticFieldLiteral,
        file_format: MFSFormats = "h5",
        data_standard: DataStandard[Any] | None = None,
        root_metadata: dict[str, str] | None = None,
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

        if data_standard is None:
            data_standard = ep.data_standards.PRBEMStandard()
        self.standard = data_standard
        self.data_standard = data_standard

        self.output_files = [
            OutputFile("full", self._get_output_file_entries(), save_incomplete=True),
        ]

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

    def _get_standard_output_file_entries(self) -> list[InternalName]:
        """Return the standard PRBEM monthly variable list."""
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

    def _get_output_file_entries(self) -> list[InternalName]:
        """Return the standard variable list plus user-defined custom variables."""
        entries = self._get_standard_output_file_entries()
        return entries

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
        internal_name: InternalName,
        *,
        first_call_of_interval: bool,
    ) -> ep.Variable:
        """Standardize a variable through the configured data standard."""
        return self.standard.standardize_variable(
            internal_name, variable, reset_consistency_check=first_call_of_interval
        )

    def save_single_file(
        self, file_path: Path, dict_to_save: dict[InternalName | Literal["metadata"], Any], *, append: bool = False
    ) -> None:
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

    def append_data(
        self, file_path: Path, data_dict_to_save: dict[InternalName | Literal["metadata"], Any]
    ) -> dict[InternalName | Literal["metadata"], Any]:
        """Append data to any supported monthly file format.

        Existing data is loaded with the loader for ``file_path.suffix``, merged
        by timestamp with the new dictionary, and written to a temporary file
        before replacing the original file.
        """
        if not file_path.exists():
            msg = f"Cannot append: file does not exist: {file_path}"
            raise FileNotFoundError(msg)

        time_key = self.standard.get_full_var_name("Epoch")

        if time_key not in data_dict_to_save:
            msg = f"Cannot append: missing {time_key} in data_dict_to_save."
            raise KeyError(msg)

        new_time = np.asarray(data_dict_to_save[time_key])
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

    def _merge_and_sort_data(  # noqa: C901, PLR0912, PLR0915
        self,
        existing_data: dict[InternalName | Literal["metadata"], Any],
        new_data: dict[InternalName | Literal["metadata"], Any],
    ) -> dict[InternalName | Literal["metadata"], Any]:
        """Merge two dictionaries along the time axis, replacing duplicate times."""

        def _normalize_1d(arr: np.ndarray) -> np.ndarray:
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[1] == 1:  # noqa: PLR2004
                return arr.reshape(-1)
            return arr

        time_key = self.standard.get_full_var_name("Epoch")

        if time_key not in existing_data or np.asarray(existing_data[time_key]).size == 0:
            return new_data

        if time_key not in new_data or np.asarray(new_data[time_key]).size == 0:
            return existing_data

        existing_time = _normalize_1d(existing_data[time_key])
        new_time = _normalize_1d(new_data[time_key])
        mask_keep_existing = ~np.isin(existing_time, new_time)
        insert_idx = int(np.searchsorted(existing_time, new_time[0]))

        merged: dict[InternalName | Literal["metadata"], Any] = {}
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

            if key == time_key:
                t = np.asarray(merged_val)
                if len(np.unique(t)) != len(t):
                    msg = "Time values are not unique after merge for key 'time'"
                    logger.error(msg)
                    raise ValueError(msg)

            merged[key] = merged_val

        return merged

    def _load_mat_data(self, file_path: Path) -> dict[InternalName | Literal["metadata"], Any]:
        """Load an existing MATLAB file."""
        loaded = loadmat(str(file_path), simplify_cells=True)
        data = {key: value for key, value in loaded.items() if not key.startswith("__")}

        if "metadata" in data and isinstance(data["metadata"], dict):
            for var_key, attrs in data["metadata"].items():
                if not isinstance(attrs, dict):
                    continue
                data["metadata"][var_key] = {
                    k: v.item() if isinstance(v, np.ndarray) and v.ndim == 0
                    else str(v) if isinstance(v, np.ndarray)
                    else v
                    for k, v in attrs.items()
                }

        return data

    def _write_mat_file(self, file_path: Path, data_dict: DataDict) -> None:
        """Write a MATLAB file, resolving standard variable paths and flattening hierarchy.

        Data variables are stored under their flattened canonical names (``/`` → ``__``).
        Per-variable metadata is stored in a parallel ``metadata`` struct whose field
        names mirror the data variable names, matching how HDF5 stores attrs per dataset.
        """
        mat_dict: dict[str, Any] = {}
        mat_metadata: dict[str, Any] = {}

        for internal_name, value in data_dict.items():
            if internal_name == "metadata":
                continue

            path = self.standard.get_full_var_name(internal_name)
            mat_var_name = path.replace("/", "__")

            value_to_write = value
            if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == 1:  # noqa: PLR2004
                value_to_write = value.reshape(-1)

            mat_dict[mat_var_name] = value_to_write

            # Attach per-variable metadata under a matching key in the metadata struct,
            # mirroring how _write_h5_file stores attrs on each dataset.
            variable_meta = data_dict.get("metadata", {}).get(internal_name, {})
            if isinstance(variable_meta, dict) and variable_meta:
                mat_metadata[mat_var_name] = {
                    "unit": variable_meta.get("unit", "unknown"),
                    "source_files": variable_meta.get("source_files", "unknown"),
                    "processing_notes": variable_meta.get("processing_notes", "unknown"),
                    "description": variable_meta.get("description", "unknown"),
                    "original_cadence_seconds": variable_meta.get("original_cadence_seconds", "unknown"),
                }

        if mat_metadata:
            mat_dict["metadata"] = mat_metadata

        savemat(str(file_path), mat_dict)

    def _load_h5_data(self, file_path: Path) -> DataDict:
        """Load all datasets and dataset attributes from an HDF5 file."""
        loaded_data: DataDict = {"metadata": {}}

        def _recursively_load_datasets(group: h5py.Group | h5py.File, prefix: str = "") -> None:
            for key, item in group.items():
                full_path = f"{prefix}{key}" if prefix else key
                if isinstance(item, h5py.Dataset):
                    loaded_data[full_path] = np.array(item)  # ty:ignore[invalid-assignment]
                    loaded_data["metadata"][full_path] = dict(item.attrs.items())
                elif isinstance(item, h5py.Group):
                    _recursively_load_datasets(item, f"{full_path}/")

        with h5py.File(file_path, "r") as file:
            _recursively_load_datasets(file)

        return loaded_data

    def _write_h5_file(self, file_path: Path, data_dict: dict[InternalName | Literal["metadata"], Any]) -> None:
        """Write an HDF5 file with hierarchical groups from slash-delimited paths."""
        with h5py.File(file_path, "w") as file:
            for internal_name, value in data_dict.items():
                if internal_name == "metadata":
                    continue
                path = self.standard.get_full_var_name(internal_name)

                path_parts = path.split("/")
                groups = path_parts[:-1]
                dataset_name = path_parts[-1]

                curr_hierarchy = file
                for group in groups:
                    if group not in curr_hierarchy:
                        curr_hierarchy = curr_hierarchy.create_group(group)
                    else:
                        curr_hierarchy = typing.cast("h5py.Group", curr_hierarchy[group])

                # Normalize 2D arrays with shape (n, 1) back to 1D for consistency with other formats
                value_to_write = value
                if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == 1:  # noqa: PLR2004
                    value_to_write = value.reshape(-1)

                data_set = curr_hierarchy.create_dataset(
                    dataset_name, data=value_to_write, compression="gzip", shuffle=True
                )

                metadata_dict = data_dict.get("metadata", {}).get(internal_name, {})
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

    def _load_netcdf_data(self, file_path: Path) -> DataDict:
        """Load all variables and variable metadata from a NetCDF file."""
        loaded_data: DataDict = {"metadata": {}}

        def _recursively_load(group: nC.Group | nC.Dataset, prefix: str = "") -> None:
            for var_name, variable in group.variables.items():
                full_path = f"{prefix}{var_name}" if prefix else var_name
                loaded_data[full_path] = np.array(variable[:])  # ty:ignore[invalid-assignment]
                loaded_data["metadata"][full_path] = {
                    "unit": getattr(variable, "units", "unknown"),
                    "source_files": getattr(variable, "source", "unknown"),
                    "processing_notes": getattr(variable, "history", "unknown"),
                    "description": getattr(variable, "description", "unknown"),
                    "original_cadence_seconds": getattr(variable, "original_cadence_seconds", "unknown"),
                }

            for group_name, subgroup in group.groups.items():
                _recursively_load(subgroup, f"{prefix}{group_name}/")

        with nC.Dataset(file_path, "r", format="NETCDF4") as file:
            _recursively_load(file)

        standard_internal_name_map: dict[str, InternalName] = {}

        for standard_name in loaded_data:
            standard_names: list[InternalName] = [
                internal_name
                for internal_name, var_info in self.standard.variable_infos.items()
                if var_info.standard_name == standard_name
            ]

            if len(standard_names) == 0:
                continue
            if len(standard_names) == 1:
                standard_internal_name_map[standard_name] = standard_names[0]
            else:
                msg = "More than one fitting internal name found!"
                raise ValueError(msg)

        return loaded_data

    def _calculate_dimensions(self, data_dict: DataDict) -> dict[str, int]:
        """Calculate NetCDF dimension sizes from the data dictionary."""
        dimensions = {
            "Epoch": np.asarray(data_dict["Epoch"]).shape[0],
            "Alpha": 0,
            "Energy_FEDU": 0,
        }

        if "Alpha_Eq" in data_dict and np.asarray(data_dict["Alpha_Eq"]).size > 0:
            dimensions["Alpha"] = np.asarray(data_dict["Alpha_Eq"]).shape[1]
        elif "Alpha" in data_dict and np.asarray(data_dict["Alpha"]).size > 0:
            dimensions["Alpha"] = np.asarray(data_dict["Alpha"]).shape[1]

        if "Energy_FEDU" in data_dict and np.asarray(data_dict["Energy_FEDU"]).size > 0:
            dimensions["Energy_FEDU"] = np.asarray(data_dict["Energy_FEDU"]).shape[1]

        if "Position" in data_dict and np.asarray(data_dict["Position"]).size > 0:
            dimensions["Position_components"] = 3

        return dimensions

    def _write_data_to_netcdf_file(
        self, file: nC.Dataset | nC.Group, data_dict: dict[InternalName | Literal["metadata"], Any]
    ) -> None:
        """Write variables to a NetCDF file or group."""
        for mfs_name, value in data_dict.items():
            if mfs_name == "metadata":
                continue

            value_array = np.asarray(value)
            if value_array.size == 0:
                continue

            path = self.standard.get_full_var_name(mfs_name)

            path_parts = path.split("/")
            groups = path_parts[:-1]
            dataset_name = path_parts[-1]

            curr_hierarchy: nC.Group | nC.Dataset = file
            for group in groups:
                if group not in curr_hierarchy.groups:
                    curr_hierarchy = curr_hierarchy.createGroup(group)
                else:
                    curr_hierarchy = curr_hierarchy.groups[group]

            dimensions = self.standard.get_dependencies(mfs_name)
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

            metadata_dict = data_dict.get("metadata", {})
            metadata = {}
            if isinstance(metadata_dict, dict):
                metadata = metadata_dict.get(path, metadata_dict.get(mfs_name, {}))

            if not isinstance(metadata, dict):
                continue

            data_set.units = metadata.get("unit", "unknown")
            data_set.source = metadata.get("source_files", "unknown")
            data_set.history = metadata.get("processing_notes", "unknown")
            data_set.description = metadata.get("description", "unknown")
            data_set.original_cadence_seconds = metadata.get("original_cadence_seconds", "unknown")

    def _write_netcdf_file(self, file_path: Path, data_dict: dict[InternalName | Literal["metadata"], Any]) -> None:
        """Create and write a NetCDF file from a data dictionary."""
        with nC.Dataset(file_path, "w", format="NETCDF4") as file:
            if self.root_metadata is not None:
                for key, value in self.root_metadata.items():
                    setattr(file, key, value)

            size_time = np.asarray(data_dict["Epoch"]).shape[0]
            if size_time == 0:
                logger.info(f"Skipping write for {file_path.name} (time has length 0).")
                return

            dimensions = self._calculate_dimensions(data_dict)
            file.createDimension("time", None)
            for dim_name, dim_size in dimensions.items():
                if dim_name != "time":
                    file.createDimension(dim_name, dim_size)

            self._write_data_to_netcdf_file(file, data_dict)

    def _load_cdf_data(self, file_path: Path) -> DataDict:
        """Load all zVariables from an existing CDF file."""
        loaded_data: DataDict = {"metadata": {}}
        cdf_file = cdflib.CDF(str(file_path))
        try:
            info = cdf_file.cdf_info()
            z_variables = getattr(info, "zVariables", None)
            if z_variables is None and isinstance(info, dict):
                z_variables = info.get("zVariables", [])  # ty:ignore[no-matching-overload]

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

    def _get_cdf_variable_attrs(self, var_name: str, data_dict: DataDict) -> DataDict:
        """Return non-empty CDF variable attributes for a saved variable."""
        metadata = data_dict.get("metadata", {}).get(var_name, {})
        var_attrs: DataDict = {}

        if isinstance(metadata, dict):
            for attr_name, attr_value in metadata.items():
                if self._is_empty_cdf_attribute(attr_value):
                    logger.debug(f"Skipping empty CDF attribute {var_name}:{attr_name}")
                    continue

                var_attrs[str(attr_name)] = attr_value  # ty:ignore[invalid-assignment]

        var_attrs["Compress"] = 6  # ty:ignore[invalid-assignment]
        return var_attrs

    def _is_empty_cdf_attribute(self, value: Any) -> bool:  # noqa: ANN401
        """Return True if cdflib cannot infer a datatype from the attribute value."""
        if value is None:
            return True

        if isinstance(value, (list, tuple, dict, str, bytes)):
            return len(value) == 0

        return getattr(value, "size", None) == 0

    def _write_cdf_file(self, file_path: Path, data_dict: DataDict) -> None:  # noqa: C901, PLR0912
        """Write a CDF file, resolving standard variable paths and embedding metadata."""
        try:
            cdf_file = cdflib.cdfwrite.CDF(str(file_path), delete=True)
            try:
                for internal_name, var_data in data_dict.items():
                    if internal_name == "metadata":
                        continue

                    if getattr(var_data, "size", 0) == 0:
                        logger.warning(f"Skipping empty variable {internal_name}")
                        continue

                    # Resolve the canonical name via the data standard, matching H5/NC behaviour.
                    # CDF does not support '/' in variable names, so we replace path separators
                    # with '__' to preserve hierarchy information without violating the spec.
                    path = self.standard.get_full_var_name(internal_name)
                    cdf_var_name = path
                    value_to_write = var_data
                    if isinstance(var_data, np.ndarray) and var_data.ndim == 2 and var_data.shape[1] == 1:
                        value_to_write = var_data.reshape(-1)

                    var_data_array = np.asarray(value_to_write)
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
                        "Variable": cdf_var_name,
                        "Data_Type": cdf_dtype,
                        "Num_Elements": 1,
                        "Rec_Vary": True,
                        "Dim_Sizes": (list(var_data_array.shape[1:]) if var_data_array.ndim > 1 else []),
                    }

                    metadata_dict = data_dict.get("metadata", {})
                    metadata: dict[str, Any] = {}
                    if isinstance(metadata_dict, dict):
                        metadata = metadata_dict.get(path, metadata_dict.get(internal_name, {}))

                    var_attrs = {}
                    if isinstance(metadata, dict):
                        for attr_name, attr_value in metadata.items():
                            if self._is_empty_cdf_attribute(attr_value):
                                logger.debug(f"Skipping empty CDF attribute {cdf_var_name}:{attr_name}")
                                continue
                            var_attrs[str(attr_name)] = attr_value
                    if isinstance(metadata, dict):
                        for field, nc_key in {
                            "unit": "unit",
                            "source_files": "source_files",
                            "processing_notes": "processing_notes",
                            "description": "description",
                            "original_cadence_seconds": "original_cadence_seconds",
                        }.items():
                            value = metadata.get(nc_key)
                            if value is None or self._is_empty_cdf_attribute(value):
                                var_attrs.setdefault(nc_key, "empty")
                                continue
                            if value and not self._is_empty_cdf_attribute(value):
                                var_attrs.setdefault(field, value)

                    var_attrs["Compress"] = 6

                    cdf_file.write_var(var_spec, var_attrs=var_attrs, var_data=var_data_array)
            finally:
                cdf_file.close()
        except Exception as e:
            msg = f"Failed to write CDF file {file_path}: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e
