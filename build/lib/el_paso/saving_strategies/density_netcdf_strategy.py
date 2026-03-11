# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import typing
from pathlib import Path
from typing import Any, Literal

import netCDF4 as nC

import el_paso as ep
from el_paso.saving_strategies.monthly_h5_strategy import MonthlyH5Strategy
from el_paso.saving_strategy import OutputFile

if typing.TYPE_CHECKING:
    from datetime import datetime

    from el_paso.data_standard import DataStandard
    from el_paso.processing.magnetic_field_utils import MagneticFieldLiteral

logger = logging.getLogger(__name__)


class DensityNetCDFStrategy(MonthlyH5Strategy):
    """Saving strategy for writing plasma density and related data to monthly NetCDF files.

    This strategy extends `MonthlyH5Strategy` but implements saving to the NetCDF
    format (`.nc`), primarily targeting the time-series of density, position, and
    coordinate variables (e.g., L-star, MLT).

    The variables included and their dependencies are configured based on whether
    the data is associated with the **"RBSP"** satellites or **"Other"**.

    Attributes:
        output_files (list[OutputFile]): List of file configurations to be produced.
        file_path (Path): Base path for output files (inherited).
        dependency_dict (dict[str, list[str]]): Defines the NetCDF dimension names
            (e.g., 'time', 'xGEO_components') that each variable depends on.
    """

    output_files: list[OutputFile]

    file_path: Path
    dependency_dict: dict[str, list[str]]

    def __init__(
        self,
        base_data_path: str | Path,
        file_name_stem: str,
        mag_field: MagneticFieldLiteral | list[MagneticFieldLiteral],
        satellite: Literal["RBSP", "Other"] = "Other",
        data_standard: DataStandard | None = None,
    ) -> None:
        """Initializes the monthly NetCDF saving strategy.

        Parameters:
            base_data_path (str | Path): The base directory where the output NetCDF files will be saved.
            file_name_stem (str): The base name for the output files (e.g., "my_data").
            mag_field (MagneticFieldLiteral | list[MagneticFieldLiteral]):
                A string or list of strings specifying the magnetic field models used.
            satellite (Literal["RBSP", "Other"], optional):
                            Specifies the satellite associated with the data. This is often used to trigger
                            specific metadata or formatting conventions. Defaults to "Other".
            data_standard (DataStandard | None, optional):
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

        output_file_entries = [
            "time",
            "xGEO",
            "MLT",
            "R_eq",
            "Lstar",
            "xGEO_eq",
        ]

        self.dependency_dict = {
            "time": ["time"],
            "xGEO": ["time", "xGEO_components"],
            "MLT": ["time"],
            "R_eq": ["time"],
            "xGEO_eq": ["time", "xGEO_components"],
            "Lstar": ["time"],
        }

        if satellite == "Other":
            output_file_entries += ["density_local", "density_eq"]
            self.dependency_dict |= {"density_local": ["time"], "density_eq": ["time"]}

        elif satellite == "RBSP":
            output_file_entries += [
                "density_emfisis_local",
                "density_efw_local",
                "density_hiss_derived_local",
                "density_emfisis_eq",
                "density_efw_eq",
                "density_hiss_derived_eq",
            ]

            self.dependency_dict |= {
                "density_emfisis_local": ["time"],
                "density_efw_local": ["time"],
                "density_hiss_derived_local": ["time"],
                "density_emfisis_eq": ["time"],
                "density_efw_eq": ["time"],
                "density_hiss_derived_eq": ["time"],
            }

        else:
            msg = "Enountered invalid satellite! Valid names are: 'RBSP', 'Other'."
            raise ValueError(msg)

        self.output_files = [
            OutputFile("full", output_file_entries, save_incomplete=True),
        ]

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

    def save_single_file(self, file_path: Path, dict_to_save: dict[str, Any], *, append: bool = False) -> None:
        """Saves a dictionary of variables to a single NetCDF file.

        This method creates a new NetCDF4 file, defines dimensions based on the data,
        and writes each variable as a dataset. It also attaches metadata as attributes
        to the datasets.

        Parameters:
            file_path (Path): The path to the file where the data will be saved.
            dict_to_save (dict[str, Any]): The dictionary containing variable data.
            append (bool, optional): If `True`, attempts to append data to an existing file.
                Currently, this functionality is not fully implemented for NetCDF,
                so it defaults to creating a new file.

        Note:
            This method only supports creating new files (`append=False`) and does not
            handle appending to an existing NetCDF file.
        """
        logger.info(f"Saving file {file_path.name}...")

        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.exists() and append:
            dict_to_save = self.append_data(file_path, dict_to_save)

        # a NETCDF4_CLASSIC format allows for loading mulitple files via netCDF4.MFDataset
        with nC.Dataset(file_path, "w", format="NETCDF4_CLASSIC") as file:
            file.createDimension("time", size=None)  # time is unlimited

            if ("xGEO" in dict_to_save and dict_to_save["xGEO"].size) > 0 or (
                "xGEO_eq" in dict_to_save and dict_to_save["xGEO_eq"].size > 0
            ):
                file.createDimension("xGEO_components", 3)

            for dataset_name, value in dict_to_save.items():
                if dataset_name == "metadata":
                    continue

                if value.size == 0:
                    continue

                data_set = typing.cast(
                    "nC.Variable[Any]",
                    file.createVariable(  # type: ignore[reportUnknownMemberType]
                        dataset_name, "f4", self.dependency_dict[dataset_name], zlib=True, complevel=5, shuffle=True
                    ),
                )

                data_set[:, ...] = value

                if dataset_name in dict_to_save["metadata"]:
                    metadata = dict_to_save["metadata"][dataset_name]
                    data_set.units = metadata["unit"]
                    data_set.history = metadata["processing_notes"]
                    data_set.description = metadata["description"]
