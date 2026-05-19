# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import typing
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable  # noqa: UP035

import cdflib
import h5py
import netCDF4 as nC
import numpy as np
from scipy.io import savemat

from el_paso.saving_strategy import OutputFile, SavingStrategy

if TYPE_CHECKING:
    from datetime import datetime

    from el_paso.typing import InternalName, Variable

logger = logging.getLogger(__name__)


SingleFileFormatWriter = Callable[[Path, dict[str, Any]], None]


class SingleFileStrategy(SavingStrategy):
    """A concrete saving strategy that saves all data to a single file.

    This strategy implements the `SavingStrategy` abstract methods to manage saving all variables
    for the entire time range into a single output file. It is a simple, non-partitioning approach.
    Supports multiple file formats including MATLAB (.mat), HDF5 (.h5), NetCDF4 (.nc), and CDF (.cdf).
    Users can also register custom format writers for additional file formats.

    Attributes:
        file_path (Path): The path to the single output file where all data will be saved.
        output_files (list[OutputFile]): List of output files to be managed.

    Methods:
        __init__(file_path, format_writers): Initializes the strategy with file path and optional custom writers.
        get_time_intervals_to_save: Returns the entire time range as a single interval.
        get_file_path: Always returns the pre-defined single file path.
        standardize_variable: Passes the variable through without any standardization.
        save_single_file: Saves data to a file in the specified format using the dispatch table.
        register_writer: Registers a custom format writer for a file extension.

    Supported Formats:
        - .mat: MATLAB format using scipy.io.savemat
        - .h5: HDF5 format using h5py with optional gzip compression
        - .nc: NetCDF4 format using netCDF4 with optional compression
        - .cdf: CDF (Common Data Format) using cdflib with gzip compression
        - Custom: Any user-defined format via register_writer() or format_writers parameter

    Example:
        ```python
        def write_custom(file_path: Path, data_dict: dict[str, Any]) -> None:
            # Custom writer implementation
            pass
        strategy = SingleFileStrategy("output.myformat",format_writers={".myformat": write_custom})
        ep.save(variables, saving_strategy=strategy, ...)
        ```
    """

    output_files: list[OutputFile]
    file_path: Path
    _writers: dict[str, SingleFileFormatWriter]

    def __init__(
        self,
        file_path: str | Path,
        format_writers: dict[str, SingleFileFormatWriter] | None = None,
    ) -> None:
        """Initializes the SingleFileStrategy with the specified file path and optional custom format writers.

        Parameters:
            file_path (str | Path): The full path to the output file. The file extension determines
                the format unless a custom writer is registered.
            format_writers (dict[str, SingleFileFormatWriter] | None): Optional dictionary mapping file extensions
                (including the dot, e.g., ".myformat") to custom writer functions. Custom writers override
                built-in writers for the same extension. Defaults to None.

        Example:
            ```python
            def write_custom(file_path: Path, data_dict: dict[str, Any]) -> None:
                # Custom writer implementation
                pass
            strategy = SingleFileStrategy("output.myformat",format_writers={".myformat": write_custom})
            ep.save(variables, saving_strategy=strategy, ...)
            ```
        """
        self.file_path = Path(file_path)
        self.output_files = [OutputFile(self.file_path.name, [])]

        # Build the dispatch table with built-in writers
        self._writers: dict[str, SingleFileFormatWriter] = {
            ".mat": self._write_mat_file,
            ".h5": self._write_h5_file,
            ".nc": self._write_netcdf_file,
            ".cdf": self._write_cdf_file,
        }

        # Register custom writers (these override built-in writers if same extension)
        if format_writers:
            self._writers.update(format_writers)

    def get_file_path_stem(self):
        pass

    def get_file_name_stem(self):
        pass

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

    def standardize_variable(
        self,
        variable: Variable,
        internal_name: InternalName,  # noqa: ARG002
        *,
        first_call_of_interval: bool,  # noqa: ARG002
    ) -> Variable:
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

    def register_writer(self, extension: str, writer: SingleFileFormatWriter) -> None:
        """Register a custom format writer for a file extension.

        This method allows you to register custom writers for file formats not natively supported,
        or to override built-in writers. Custom writers are called when a file with the matching
        extension is saved.

        Parameters:
            extension (str): The file extension (including the dot), e.g., ".myformat" or ".bin".
            writer (SingleFileFormatWriter): A callable with signature `(Path, dict[str, Any]) -> None` that
                handles writing the data dictionary to the specified file path.

        Example:
            ```python
            def write_binary(path: Path, data: dict[str, Any]) -> None:
                import struct
                with open(path, 'wb') as f:
                    for key, value in data.items():
                        if key != "metadata":
                            f.write(value.tobytes())
            strategy = SingleFileStrategy("output.dat")
            strategy.register_writer(".dat", write_binary)
            ```
        """
        if not extension.startswith("."):
            extension = "." + extension
        self._writers[extension.lower()] = writer

    def _write_metadata_to_netcdf_variable(self, data_set: nC.Variable[Any], metadata: dict[str, Any]) -> None:
        """Attach metadata values that can be represented as NetCDF attributes."""
        for key, value in metadata.items():
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value)

            if getattr(value, "size", None) == 0:
                continue

            setattr(data_set, key, value)

    def _write_netcdf_file(self, file_path: Path, data_dict: dict[str, Any]) -> None:
        """Write data dictionary to NetCDF4 (.nc) format.

        Creates hierarchical groups based on paths (e.g., "group1/group2/dataset" becomes nested groups).
        Applies zlib compression, shuffle filter, and creates dimension variables automatically.
        Writes metadata as variable attributes.

        Parameters:
            file_path (Path): Path to save the .nc file.
            data_dict (dict[str, Any]): Dictionary with variable data and metadata.
                Keys are path strings (e.g., "var_name" or "group/subgroup/var_name").
                The "metadata" key is skipped; metadata is stored as variable attributes.
        """
        with nC.Dataset(file_path, "w", format="NETCDF4") as file:
            for path, value in data_dict.items():
                if path == "metadata":
                    continue

                if value.size == 0:
                    continue

                path_parts = path.split("/")
                groups = path_parts[:-1]
                dataset_name = path_parts[-1]

                curr_hierarchy: nC.Group | nC.Dataset = file
                for group in groups:
                    if group not in curr_hierarchy.groups:
                        curr_hierarchy = curr_hierarchy.createGroup(group)
                    else:
                        curr_hierarchy = curr_hierarchy.groups[group]

                dimensions = []
                for axis, size in enumerate(value.shape):
                    dimension_name = f"{dataset_name}_dim_{axis}"
                    if dimension_name not in curr_hierarchy.dimensions:
                        curr_hierarchy.createDimension(dimension_name, size)
                    dimensions.append(dimension_name)

                data_set = typing.cast(
                    "nC.Variable[Any]",
                    curr_hierarchy.createVariable(
                        dataset_name, value.dtype, dimensions, zlib=True, complevel=5, shuffle=True
                    ),
                )

                data_set[...] = value

                if path in data_dict.get("metadata", {}):
                    self._write_metadata_to_netcdf_variable(data_set, data_dict["metadata"][path])

    def save_single_file(self, file_path: Path, dict_to_save: dict[str, Any], *, append: bool = False) -> None:  # ty:ignore[invalid-method-override]
        """Saves variable data to a single file in one of the supported formats.

        The file format is determined by the file extension. Built-in formats include .mat, .h5, .nc, and .cdf.
        Custom format writers can be registered via the format_writers parameter during initialization or
        via the register_writer() method.

        It is primarily designed to be used with the `el_paso.save()` function, which handles the logic of determining
        what data to save and when.

        Parameters:
            file_path (Path): The path to the file where the dictionary will be saved.
                              The file extension determines the format.
            dict_to_save (dict[str, Any]): The dictionary containing variable data to save.
                Keys are variable names (strings), values are NumPy arrays or other serializable data.
                Should include a "metadata" key with metadata dictionary.
            append (bool, optional): If True, attempts to append to an existing file.
                Only supported for CDF format. For other formats, raises NotImplementedError.
                Defaults to False.

        Raises:
            NotImplementedError: If the file format is not registered or supported,
                or if append is requested for formats that don't support it.
            Any exception raised by the format writer function.

        Supported Built-in Formats:
            - .mat: MATLAB format using scipy.io.savemat
            - .h5: HDF5 format using h5py with gzip compression
            - .nc: NetCDF4 format using netCDF4 with compression
            - .cdf: CDF (Common Data Format) using cdflib with gzip compression
        """
        logger.info(f"Saving file {file_path.name}...")

        file_path.parent.mkdir(parents=True, exist_ok=True)
        format_name = file_path.suffix.lower()

        # Look up the writer in the dispatch table
        writer = self._writers.get(format_name)

        if writer is None:
            msg = f"The '{format_name}' format is not implemented. Registered formats: {list(self._writers.keys())}"
            logger.error(msg)
            raise NotImplementedError(msg)

        if append:
            msg = f"Appending to existing files is not supported for '{format_name}' format."
            logger.error(msg)
            raise NotImplementedError(msg)
        writer(file_path, dict_to_save)

    def _write_mat_file(self, file_path: Path, data_dict: dict[str, Any]) -> None:
        """Write data dictionary to MATLAB .mat format.

        Parameters:
            file_path (Path): Path to save the .mat file.
            data_dict (dict[str, Any]): Dictionary with variable data and metadata.
        """
        savemat(str(file_path), data_dict)

    def _write_h5_file(self, file_path: Path, data_dict: dict[str, Any]) -> None:
        """Write data dictionary to HDF5 (.h5) format.

        Creates hierarchical groups based on paths (e.g., "group1/group2/dataset" becomes nested groups).
        Applies gzip compression and shuffling to all datasets. Writes metadata as dataset attributes.

        Parameters:
            file_path (Path): Path to save the .h5 file.
            data_dict (dict[str, Any]): Dictionary with variable data and metadata.
                Keys are path strings (e.g., "var_name" or "group/subgroup/var_name").
                The "metadata" key is skipped; metadata is stored as dataset attributes.
        """
        with h5py.File(file_path, "w") as file:
            for path, value in data_dict.items():
                if path == "metadata":
                    continue

                path_parts = path.split("/")
                groups = path_parts[:-1]
                dataset_name = path_parts[-1]

                curr_hierachy = file
                for group in groups:
                    if group not in curr_hierachy:
                        curr_hierachy = curr_hierachy.create_group(group)
                    else:
                        curr_hierachy = typing.cast("h5py.Group", curr_hierachy[group])

                data_set = curr_hierachy.create_dataset(dataset_name, data=value, compression="gzip", shuffle=True)

                if path in data_dict["metadata"]:
                    for key, metadata in data_dict["metadata"][path].items():
                        data_set.attrs[key] = metadata

    def _write_cdf_file(self, file_path: Path, data_dict: dict[str, Any]) -> None:
        """Write data dictionary to CDF (Common Data Format) format.

        Converts NumPy arrays to appropriate CDF data types and writes them as zVariables.
        Supports global attributes and per-variable attributes from the metadata dictionary.
        Applies gzip compression (Compress=6) to all variables.

        Parameters:
            file_path (Path): Path to save the .cdf file.
            data_dict (dict[str, Any]): Dictionary with variable data and metadata.
                Keys are variable names. The "metadata" key contains global and variable attributes.
                Metadata should follow the format: {var_name: {attr_name: attr_value, ...}, ...}
        """
        try:
            cdf_file = cdflib.cdfwrite.CDF(str(file_path), delete=True)

            try:
                metadata = data_dict.get("metadata")

                if isinstance(metadata, dict):
                    global_attrs: dict[str, dict[int, Any]] = {}

                    for attr_name, attr_value in metadata.items():
                        attr_name_str = str(attr_name)

                        if isinstance(attr_value, dict):
                            keys = list(attr_value.keys())
                            if all(isinstance(k, (int, np.integer)) or str(k).isdigit() for k in keys):
                                global_attrs[attr_name_str] = {int(k): v for k, v in attr_value.items()}
                            else:
                                for sub_key, sub_val in attr_value.items():
                                    if isinstance(sub_val, (list, tuple)) and len(sub_val) == 0:
                                        logger.warning(f"Skipping empty global attribute {attr_name_str}_{sub_key}")
                                        continue
                                    flat_name = f"{attr_name_str}_{sub_key}"
                                    global_attrs[flat_name] = {0: sub_val}

                        elif isinstance(attr_value, (list, tuple)):
                            if len(attr_value) == 0:
                                logger.warning(f"Skipping empty global attribute {attr_name_str}")
                                continue
                            global_attrs[attr_name_str] = dict(enumerate(attr_value))

                        else:
                            global_attrs[attr_name_str] = {0: attr_value}

                    if global_attrs:
                        cdf_file.write_globalattrs(global_attrs)

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
                        if var_data_array.dtype == np.float32:
                            cdf_dtype = cdflib.cdfwrite.CDF.CDF_FLOAT
                        else:
                            cdf_dtype = cdflib.cdfwrite.CDF.CDF_DOUBLE

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

                    var_attrs: dict[str, Any] = {
                        "Compress": 6,
                    }

                    cdf_file.write_var(
                        var_spec,
                        var_attrs=var_attrs,
                        var_data=var_data_array,
                    )

            finally:
                cdf_file.close()

        except Exception as e:
            msg = f"Failed to write CDF file {file_path}: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e
