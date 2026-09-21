# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import el_paso as ep
from el_paso.saving_strategies.monthly_rb_strategy import MonthlyRBStrategy
from el_paso.saving_strategy import OutputFile

if TYPE_CHECKING:
    from pathlib import Path

    from el_paso.data_standard import DataStandard
    from el_paso.typing import InternalName, MagneticFieldLiteral, StandardName

logger = logging.getLogger(__name__)


class RBSPDensityStrategy(MonthlyRBStrategy):
    """Saving strategy for writing RBSP plasma density and related data to monthly NetCDF files.

    This strategy extends `MonthlyRBStrategy` but implements saving to the NetCDF
    format (`.nc`), primarily targeting the time-series of density, position, and
    coordinate variables.

    The variable set is specific to RBSP, which carries several independent density
    products (EMFISIS, EFW, and hiss-derived), each saved both at the satellite
    location and mapped to the magnetic equator. For every other mission use
    `MonthlyDensityStrategy`, which saves a single local/equatorial density pair.

    Attributes:
        output_files (list[OutputFile]): List of file configurations to be produced.
        file_path (Path): Base path for output files (inherited).
    """

    output_files: list[OutputFile]

    file_path: Path

    def __init__(
        self,
        base_data_path: str | Path,
        mission: str,
        instrument: str,
        mag_field: MagneticFieldLiteral,
        data_standard: Optional[DataStandard[StandardName]] = None,
    ) -> None:
        """Initializes the monthly RBSP density saving strategy.

        Args:
            base_data_path (str | Path): The base directory where the output NetCDF files will be saved.
            mission (str): The mission name, used in file path and name generation.
            instrument (str): The instrument name, used in file path and name generation.
            mag_field (MagneticFieldLiteral):
                A string specifying the magnetic field model used.
            data_standard (DataStandard | None, optional):
                An optional `DataStandard` instance to use for standardizing variables.
                If `None`, `ep.data_standards.PRBEMStandard` is used by default.
        """
        # Resolve the fallback before `super().__init__`, which assigns `self.data_standard`
        # from whatever it is handed.
        data_standard = data_standard or ep.data_standards.PRBEMStandard()

        super().__init__(
            base_data_path=base_data_path,
            satellite="RBSP",
            mission=mission,
            instrument=instrument,
            mag_field=mag_field,
            file_format="nc",
            data_standard=data_standard,
        )

        self.output_files = [
            OutputFile("full", self._get_output_file_entries(), save_incomplete=True),
        ]

    def _get_output_file_entries(self) -> list[InternalName | tuple[InternalName, ...]]:
        """Return the RBSP density variable list written by this strategy."""
        return [
            "Epoch",
            "Position",
            "xGEO_Eq",
            "MLT",
            "R_Eq",
            "Number_density_emfisis",
            "Number_density_efw",
            "Number_density_hiss_derived",
            "Number_density_emfisis_Eq",
            "Number_density_efw_Eq",
            "Number_density_hiss_derived_Eq",
        ]
