# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import typing
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import netCDF4 as nC
from scipy.io import savemat

from el_paso.saving_strategy import OutputFile, SavingStrategy

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from datetime import datetime

    from el_paso import Variable


class SingleFileStrategy(SavingStrategy):
    """A concrete saving strategy that saves all data to a single file.

    This strategy implements the `SavingStrategy` abstract methods to manage saving all variables
    for the entire time range into a single output file. It is a simple, non-partitioning approach.

    Attributes:
        file_path (Path): The path to the single output file where all data will be saved.

    Methods:
        __init__(file_path): Initializes the strategy with the file path.
        get_time_intervals_to_save: Returns the entire time range as a single interval.
        get_file_path: Always returns the pre-defined single file path.
        standardize_variable: Passes the variable through without any standardization.
    """

    map_standard_name: dict[str, str]
    output_files: list[OutputFile]

    file_path: Path

    def __init__(self, file_path: str | Path) -> None:
        """Initializes the SingleFileStrategy with the specified file path.

        Parameters:
            file_path (str | Path): The full path to the output file.
        """
        self.file_path = Path(file_path)
        self.output_files = [OutputFile(self.file_path.name, [])]

        self.map_standard_name = {}

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
        name_in_file: str,  # noqa: ARG002
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

    def _write_metadata_to_netcdf_variable(self, data_set: nC.Variable[Any], metadata: dict[str, Any]) -> None:
        """Attach metadata values that can be represented as NetCDF attributes."""
        for key, value in metadata.items():
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value)  # noqa: PLW2901

            if getattr(value, "size", None) == 0:
                continue

            setattr(data_set, key, value)

    def _write_netcdf_file(self, file_path: Path, data_dict: dict[str, Any]) -> None:
        """Write a generic output dictionary to NetCDF."""
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

    def save_single_file(self, file_path: Path, dict_to_save: dict[str, Any], *, append: bool = False) -> None:
        """Saves variable data to a single file in one of the supported formats (.mat, .h5, .nc).

        Parameters:
            file_path (Path): The path to the file where the dictionary will be saved.
                              The file extension determines the format.
            dict_to_save (dict[str, Any]): The dictionary containing variable data to save.

        Raises:
            NotImplementedError: If the file format specified by the file extension is not supported.
            RuntimeError: If the .pickle format is attempted to be used.
            NotImplementedError: If `append` is set to True, as appending is not supported by this strategy.

        Supported formats:
            - .mat: Saves using scipy.io.savemat.
            - .h5: Saves using h5py, with each key as a dataset (excluding "metadata").
            - .nc: Saves using netCDF4, with each key as a variable (excluding "metadata").
        """
        logger.info(f"Saving file {file_path.name}...")

        file_path.parent.mkdir(parents=True, exist_ok=True)
        format_name = file_path.suffix.lower()

        if append:
            msg = "Appending to existing files is not supported by `SingleFileStrategy`."
            logger.error(msg)
            raise NotImplementedError(msg)

        if format_name == ".pickle":
            msg = (
                "Pickle format has been removed from `SingleFileStrategy` and will be soon"
                "removed from `SavingStrategy` as well (already deprecated)."
            )
            logger.error(msg)
            raise RuntimeError(msg)

        if format_name == ".mat":
            savemat(str(file_path), dict_to_save)

        elif format_name == ".h5":
            self._write_h5_file(file_path, dict_to_save)

        elif format_name == ".nc":
            self._write_netcdf_file(file_path, dict_to_save)

        else:
            msg = f"The '{format_name}' format is not implemented."
            raise NotImplementedError(msg)

    def _write_h5_file(self, file_path: Path, data_dict: dict[str, Any]) -> None:
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
                        curr_hierachy = curr_hierachy.create_group(group)  # type: ignore[reportUnknownVariableType]
                    else:
                        curr_hierachy = typing.cast("h5py.Group", curr_hierachy[group])

                data_set = curr_hierachy.create_dataset(dataset_name, data=value, compression="gzip", shuffle=True)  # type: ignore[reportUnknownMemberType]

                if path in data_dict["metadata"]:
                    for key, metadata in data_dict["metadata"][path].items():
                        data_set.attrs[key] = metadata
