# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing
from typing import Literal

from el_paso.saving_strategies.monthly_strategy import MonthlyFileStrategy
from el_paso.saving_strategy import OutputFile

if typing.TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    import el_paso as ep
    from el_paso.data_standard import DataStandard
    from el_paso.processing.magnetic_field_utils import MagneticFieldLiteral


class DensityNetCDFStrategy(MonthlyFileStrategy):
    """Saving strategy for writing plasma density and related data to monthly NetCDF files.

    This strategy extends `MonthlyFileStrategy` but implements saving to the NetCDF
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
        mag_field: MagneticFieldLiteral,
        satellite: Literal["RBSP", "Other"] = "Other",
        data_standard: DataStandard | None = None,
    ) -> None:
        """Initializes the monthly NetCDF saving strategy.

        Parameters:
            base_data_path (str | Path): The base directory where the output NetCDF files will be saved.
            file_name_stem (str): The base name for the output files (e.g., "my_data").
            mag_field (MagneticFieldLiteral):
                A string specifying the magnetic field model used.
            satellite (Literal["RBSP", "Other"], optional):
                            Specifies the satellite associated with the data. This is often used to trigger
                            specific metadata or formatting conventions. Defaults to "Other".
            data_standard (DataStandard | None, optional):
            data_standard (DataStandard | None):
                An optional `DataStandard` instance to use for standardizing variables.
                If `None`, `ep.data_standards.PRBEMStandard` is used by default.
        """
        self.mag_field = mag_field

        super().__init__(
            base_data_path=base_data_path,
            file_name_stem=file_name_stem,
            mag_field=self.mag_field,
            file_format="nc",
            data_standard=data_standard,
        )

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

    def _calculate_dimensions(self, data_dict: dict[str, np.ndarray]) -> dict[str, int]:
        """Calculate density NetCDF dimension sizes from the data dictionary."""
        dimensions = {"time": data_dict["time"].shape[0]}

        has_local_position = "xGEO" in data_dict and data_dict["xGEO"].size > 0
        has_equatorial_position = "xGEO_eq" in data_dict and data_dict["xGEO_eq"].size > 0
        if has_local_position or has_equatorial_position:
            dimensions["xGEO_components"] = 3

        return dimensions

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
