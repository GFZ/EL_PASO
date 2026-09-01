# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import logging
import shutil
import tempfile
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import netCDF4 as nC
import numpy as np
from astropy import units as u

import el_paso as ep

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from datetime import datetime

    from el_paso.typing import (
        DataStandard,
        InternalName,
        MagneticFieldLiteral,
        SavedDataDict,
        StandardName,
        TimeInterval,
    )


class OutputFile(NamedTuple):
    """Represents an output file with its name and a list of variable names to save.

    Attributes:
        name (str): The name of the output file.
        names_to_save (list[str]): List of variable names to be saved in the output file.
        save_incomplete (bool): If True, allows saving even if some variables are missing.
    """

    name: str
    names_to_save: list[InternalName | tuple[InternalName, ...]]
    save_incomplete: bool = False


_writers: dict[str, ep.typing.FileWriter] = {
    ".mat": ep.utils.write_mat_file,
    ".h5": ep.utils.write_h5_file,
    ".nc": ep.utils.write_netcdf_file,
    ".cdf": ep.utils.write_cdf_file,
}
_loaders: dict[str, ep.typing.FileLoader] = {
    ".mat": ep.utils.load_mat_data,
    ".h5": ep.utils.load_h5_data,
    ".nc": ep.utils.load_netcdf_data,
    ".cdf": ep.utils.load_cdf_data,
}


class SavingStrategy(ABC):
    """Abstract base class for defining strategies to save output files with specific time intervals and variables.

    Attributes:
        output_files (list[OutputFile]): List of output files to be managed by the saving strategy.
        data_standard (DataStandard[StandardName]): The data standard that defines the variable naming convention.
        base_data_path (Path): The base path where output files will be saved.
        satellite (str): The name of the satellite for which data is being saved.
        mission (str): The name of the mission for which data is being saved.
        instrument (str): The name of the instrument for which data is being saved.
        mag_field (MagneticFieldLiteral): The magnetic field model used for saving data, if applicable.
    """

    output_files: list[OutputFile]
    data_standard: DataStandard[StandardName]
    base_data_path: Path
    satellite: str
    mission: str
    instrument: str
    mag_field: MagneticFieldLiteral

    def __repr__(self) -> str:
        cls = type(self)

        constructor_params = inspect.signature(cls.__init__).parameters

        args = []

        for name in constructor_params:
            if name == "self":
                continue

            if hasattr(self, name):
                value = getattr(self, name)
                args.append(f"{name}={value!r}")

        return f"{cls.__name__}({', '.join(args)})"

    def __str__(self) -> str:
        return self.__repr__()

    @abstractmethod
    def get_time_intervals_to_save(self, start_time: datetime, end_time: datetime) -> list[TimeInterval]:
        """Generates a list of time intervals to save between the specified start and end times.

        Args:
            start_time (datetime): The starting datetime for the intervals.
            end_time (datetime): The ending datetime for the intervals.

        Returns:
            list[TimeInterval]: A list of tuples, each representing a time interval (start, end)
                                             to be saved.
        """

    @abstractmethod
    def get_file_path(self, interval_start: datetime, interval_end: datetime, output_file: OutputFile) -> Path:
        """Generates a file path for saving variables based on the provided interval and output file information.

        Args:
            interval_start (datetime): The start of the interval for which the file is being generated.
            interval_end (datetime): The end of the interval for which the file is being generated.
            output_file (OutputFile): An OutputFile containing the name of the output file,
                                      and which variables should be saved in this file.

        Returns:
            Path: The generated file path where the output data should be saved.
        """

    def standardize_variable(
        self,
        variable: ep.Variable,
        internal_name: InternalName,
        *,
        first_call_of_interval: bool,
        available_keys: set[InternalName] | None = None,
    ) -> ep.Variable:
        """Standardize a variable through the configured data standard.

        Args:
            variable (Variable): The variable to standardize.
            internal_name (InternalName): The internal name of the variable.
            first_call_of_interval (bool): If True, the data standard's consistency
                check is reset before standardizing, as this is the first variable
                processed for a new file/time interval.
            available_keys (set[InternalName] | None): The full set of internal names being
                saved together in this call.

        Returns:
            Variable: The standardized variable.
        """
        return self.data_standard.standardize_variable(
            internal_name, variable, reset_consistency_check=first_call_of_interval, available_keys=available_keys
        )

    def save_single_file(self, file_path: Path, dict_to_save: SavedDataDict, *, append: bool = False) -> None:
        """Save one monthly file, optionally appending to an existing file.

        Args:
            file_path (Path): Destination path for the file. Its suffix determines
                the output format (.mat, .h5, .nc, .cdf).
            dict_to_save (SavedDataDict): The data to save, keyed by standard name.
            append (bool): If True and ``file_path`` already exists, the data is
                merged into the existing file via :meth:`append_data` instead of
                overwriting it.

        Raises:
            NotImplementedError: If no writer is registered for the file's format.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        format_name = ep.utils.normalize_file_format(file_path.suffix)
        writer = _writers.get(format_name)

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

        Args:
            file_path (Path): Path to the existing file to append to.
            data_dict_to_save (SavedDataDict): The new data to merge in, keyed by
                internal name, including an "Epoch" entry with the new timestamps.

        Returns:
            SavedDataDict: The merged data that was written to ``file_path``. If
            ``data_dict_to_save`` contains no new timestamps, it is returned unchanged.

        Raises:
            FileNotFoundError: If ``file_path`` does not exist.
            NotImplementedError: If no loader/writer is registered for the file's format.
            ValueError: If the existing NetCDF file's time dimension is not unlimited,
                or if merging produces duplicate or mismatched-shape time values.
        """
        if not file_path.exists():
            msg = f"Cannot append: file does not exist: {file_path}"
            raise FileNotFoundError(msg)

        new_time = np.asarray(data_dict_to_save["Epoch"])
        if int(new_time.shape[0]) == 0:
            logger.info(f"No new time data to insert for {file_path.name}")
            return data_dict_to_save

        format_name = ep.utils.normalize_file_format(file_path.suffix)
        loader = _loaders.get(format_name)
        writer = _writers.get(format_name)
        if loader is None or writer is None:
            msg = f"Appending to '{format_name}' files is not supported by MonthlyRBStrategy."
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
            time_dim = file.dimensions.get("Epoch")
            if time_dim is None or not time_dim.isunlimited():
                msg = (
                    "Cannot append: the existing NetCDF file does not have an "
                    "unlimited 'Epoch' dimension. Recreate the file with 'Epoch' "
                    "created as unlimited (None)."
                )
                raise ValueError(msg)

    @abstractmethod
    def get_file_path_stem(self) -> Path:
        """Returns the directory under ``base_data_path`` where output files are stored.

        Returns:
            Path: The strategy-specific output directory, typically derived from
            ``mission``, ``satellite``, and ``instrument``.
        """

    @abstractmethod
    def get_file_name_stem(self) -> str:
        """Returns the base file name (without extension) used to build output file names.

        Returns:
            str: The strategy-specific file name stem, typically derived from
            ``mission``, ``satellite``, and ``instrument``.
        """

    def get_target_variables(
        self,
        output_file: OutputFile,
        variables_dict: dict[InternalName, ep.Variable],
        time_var: ep.Variable | None,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> dict[InternalName, ep.Variable] | None:
        """Retrieves and processes target variables for saving based on the specified output file.

        Args:
            output_file (OutputFile): The output file configuration containing variable names to save.
            variables_dict (dict[str, Variable]): Dictionary mapping variable names to Variable objects.
            time_var (Variable | None): The time variable used for truncation, if applicable.
            start_time (datetime | None): The start time for truncating variables, if specified.
            end_time (datetime | None): The end time for truncating variables, if specified.

        Returns:
            dict[str, Variable] | None:
                - A dictionary of processed Variable objects keyed by their names,
                    or None if any specified variable name is not found in variables_dict.

        Notes:
            - If no variable names are specified in output_file, all variables in variables_dict are processed.
            - Variables are deep-copied before processing.
            - Each variable is standardized using the `standardize_variable` method.
            - If a requested variable name is not found, a warning is issued and None is returned.
        """
        target_variables: dict[InternalName, ep.Variable] = {}
        first_call_of_interval = True
        available_keys = set(variables_dict.keys())

        # if no variables have been specified, we save all of them
        if len(output_file.names_to_save) == 0:
            for key, var in variables_dict.items():
                var_to_save = deepcopy(var)

                if start_time is not None and end_time is not None and time_var is not None:
                    var_to_save.truncate(time_var, start_time.timestamp(), end_time.timestamp())
                var_to_save = self.standardize_variable(
                    var_to_save, key, first_call_of_interval=first_call_of_interval, available_keys=available_keys
                )
                first_call_of_interval = False

                target_variables[key] = var_to_save

            return target_variables

        missing_names = []

        for name_to_save in output_file.names_to_save:
            # a tuple entry means "either of these species-specific names satisfies it"
            if isinstance(name_to_save, tuple):
                resolved_name = next((alt for alt in name_to_save if alt in variables_dict), None)
            else:
                resolved_name = name_to_save if name_to_save in variables_dict else None

            if resolved_name is not None:
                var_to_save = deepcopy(variables_dict[resolved_name])

                if start_time is not None and end_time is not None and time_var is not None:
                    var_to_save.truncate(time_var, start_time.timestamp(), end_time.timestamp())

                var_to_save = self.standardize_variable(
                    var_to_save,
                    resolved_name,
                    first_call_of_interval=first_call_of_interval,
                    available_keys=available_keys,
                )

                first_call_of_interval = False

                target_variables[resolved_name] = var_to_save
            else:
                label = " or ".join(name_to_save) if isinstance(name_to_save, tuple) else name_to_save
                missing_names.append(label)
                if output_file.save_incomplete:
                    fallback_name = name_to_save[0] if isinstance(name_to_save, tuple) else name_to_save
                    target_variables[fallback_name] = ep.Variable(
                        original_unit=u.dimensionless_unscaled, data=np.array([])
                    )
                else:
                    return None

        if len(missing_names) > 0:
            msg = f"Could not find target variable(s) {', '.join(sorted(missing_names))}!"
            logger.warning(msg, stacklevel=2)

        return target_variables

    def get_output_file(
        self, *, standard_name: StandardName | None = None, internal_name: InternalName | None = None
    ) -> OutputFile | None:
        """Finds the output file configured to save a given variable.

        Exactly one of ``standard_name`` or ``internal_name`` should be provided.

        Args:
            standard_name (StandardName | None): The standard name of the variable.
            internal_name (InternalName | None): The internal name of the variable.

        Returns:
            OutputFile | None: The output file whose ``names_to_save`` includes the
            variable, or None if no such output file is configured or the variable
            name could not be resolved.

        Raises:
            ValueError: If neither ``standard_name`` nor ``internal_name`` is provided.
        """
        if internal_name is None:
            if standard_name is None:
                msg = "Either standard_name or internal_name must be provided!"
                raise ValueError(msg)
            internal_name = self.data_standard.get_internal_name(standard_name)

        if internal_name is None:
            return None

        for output_file in self.output_files:
            for name_to_save in output_file.names_to_save:
                if name_to_save == internal_name or (isinstance(name_to_save, tuple) and internal_name in name_to_save):
                    return output_file

        return None

    def get_all_standard_names(self) -> list[StandardName]:
        """Returns the standard names of every variable managed by this strategy's output files.

        Returns:
            list[StandardName]: The unique standard names across all configured output files.
        """
        all_standard_names: list[StandardName] = []

        for output_file in self.output_files:
            for name_to_save in output_file.names_to_save:
                if isinstance(name_to_save, tuple):
                    all_standard_names.extend(self.data_standard.get_standard_name(alt) for alt in name_to_save)
                else:
                    all_standard_names.append(self.data_standard.get_standard_name(name_to_save))

        return list(set(all_standard_names))
