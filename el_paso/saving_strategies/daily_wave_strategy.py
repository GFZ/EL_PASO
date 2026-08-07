# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import el_paso as ep
from el_paso.saving_strategy import OutputFile, SavingStrategy
from el_paso.utils import write_netcdf_file

if TYPE_CHECKING:
    from el_paso.typing import (
        DataStandard,
        FileLoader,
        FileWriter,
        InternalName,
        MagneticFieldLiteral,
        MFSFormats,
        SavedDataDict,
        StandardName,
        TimeInterval,
        Variable,
    )

logger = logging.getLogger(__name__)


class DailyWaveStrategy(SavingStrategy):
    """Save wave and density data into one NetCDF (.nc) file per day.

    Appending to existing files is not yet implemented for this strategy.
    """

    output_files: list[OutputFile]
    dependency_dict: dict[InternalName, list[str]]

    def __init__(
        self,
        base_data_path: str | Path,
        mission: str,
        satellite: str,
        instrument: str,
        data_standard: DataStandard[StandardName],
    ) -> None:
        """Initialize a monthly file saving strategy.

        Args:
            base_data_path (str | Path): Directory where daily files are written.
            mission (str): Mission name, used in file path and name generation.
            satellite (str): Satellite name, used in file path and name generation.
            instrument (str): Instrument name, used in file path and name generation.
            data_standard (DataStandard): Instance of the data standard implementation.

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
        self.data_standard = data_standard

        self.output_files = [
            OutputFile("full", self._get_output_file_entries(), save_incomplete=True),
        ]

    def _get_output_file_entries(self) -> list[InternalName]:
        """Return the standard variable list plus user-defined custom variables."""
        return [
            "Epoch",
            "Wave_frequency",
            "Number_density",
            "Wave_ellipticity",
            "Wave_normal_angle",
            "Wave_planarity",
            "Magnetic_Power_Spectral_Density",
            "Wave_frequency_bandwidth",
            "B_total_obs",
            "MLat",
            "R_Eq",
            "MLT",
        ]

    def _sanitize_dimension_name(self, variable_name: str) -> str:
        """Return a NetCDF-safe root dimension name derived from a variable path."""
        return "".join(char if char.isalnum() else "_" for char in variable_name).strip("_") or "custom"

    def get_time_intervals_to_save(self, start_time: datetime | None, end_time: datetime | None) -> list[TimeInterval]:
        """Split the requested time range into full daily intervals."""
        time_intervals: list[TimeInterval] = []

        if start_time is None or end_time is None:
            msg = "start_time and end_time must be provided for DailyWaveStrategy!"
            raise ValueError(msg)

        current_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        while current_time <= end_time:
            interval_start = current_time
            interval_end = current_time + timedelta(days=1, microseconds=-1)

            time_intervals.append((interval_start, interval_end))
            current_time += timedelta(days=1)

        return time_intervals

    def get_file_path_stem(self) -> Path:
        return self.base_data_path / self.mission.upper() / self.satellite.lower()

    def get_file_name_stem(self) -> str:
        return self.satellite.lower() + "_" + self.instrument.lower()

    def get_file_path(self, interval_start: datetime, interval_end: datetime, output_file: OutputFile) -> Path:  # noqa: ARG002
        """Generate the daily file path for the configured format."""
        file_name = f"{self.get_file_name_stem()}_{interval_start.strftime('%Y%m%d')}.nc"

        return self.get_file_path_stem() / file_name

    def standardize_variable(
        self,
        variable: Variable,
        internal_name: InternalName,
        *,
        first_call_of_interval: bool,
        available_keys: set[InternalName] | None = None,
    ) -> Variable:
        """Standardize a variable through the configured data standard."""
        return self.data_standard.standardize_variable(
            internal_name, variable, reset_consistency_check=first_call_of_interval, available_keys=available_keys
        )

    def save_single_file(self, file_path: Path, dict_to_save: SavedDataDict, *, append: bool = False) -> None:
        """Save one daily file."""
        if append:
            msg = "Appending is not implemented yet for DailyWaveStrategy!"
            raise NotImplementedError(msg)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving file: {file_path.resolve()}")

        write_netcdf_file(file_path, dict_to_save, self.data_standard)
