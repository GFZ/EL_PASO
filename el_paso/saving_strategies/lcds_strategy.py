# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import calendar
import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import netCDF4 as nC
import numpy as np

import el_paso as ep
from el_paso.saving_strategy import OutputFile, SavingStrategy

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


class LCDSStrategy(SavingStrategy):
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
        mag_field: MagneticFieldLiteral,
        data_standard: DataStandard[StandardName],
        file_format: MFSFormats = "nc",
    ) -> None:
        """Initialize a monthly file saving strategy.

        Args:
            base_data_path (str | Path): Directory where monthly files are written.
            mag_field (MagneticFieldLiteral): Magnetic field model name. Monthly files use one model.
            file_format (MFSFormats): One of ``"nc"``, ``"cdf"``, ``"h5"``, or ``"mat"``.
                A leading dot is also accepted.
            data_standard (DataStandard): Instance of the data standard implementation.

        Attributes:
            output_files: List of output file configurations, with variable names
                defined by ``_get_output_file_entries()``.
            dependency_dict: Dictionary defining NetCDF dimension dependencies for
                all variables in ``output_files``.
        """
        self.base_data_path = Path(base_data_path)
        self.mag_field = mag_field
        self.data_standard = data_standard
        self.file_format = ep.utils.normalize_file_format(file_format)

        self.output_files = [
            OutputFile("full", self._get_output_file_entries(), save_incomplete=True),
        ]

    def _get_output_file_entries(self) -> list[InternalName]:
        """Return the standard variable list plus user-defined custom variables."""
        return [
            "Epoch",
            "Alpha",
            "LCDS",
            "InvK",
        ]

    def get_time_intervals_to_save(self, start_time: datetime | None, end_time: datetime | None) -> list[TimeInterval]:
        """Split the requested time range into full monthly intervals."""
        time_intervals: list[TimeInterval] = []

        if start_time is None or end_time is None:
            msg = "start_time and end_time must be provided for MonthlyRBStrategy!"
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
                if month == 12
                else datetime(year, month + 1, 1, tzinfo=timezone.utc)
            )

        return time_intervals

    def get_file_path_stem(self) -> Path:
        return self.base_data_path

    def get_file_name_stem(self) -> str:
        return "LCDS_"

    def get_file_path(self, interval_start: datetime, interval_end: datetime, output_file: OutputFile) -> Path:  # noqa: ARG002
        """Generate the monthly file path for the configured format."""
        start_year_month_day = interval_start.strftime("%Y%m%d")
        end_year_month_day = interval_end.strftime("%Y%m%d")
        file_name = (
            f"{self.get_file_name_stem()}_{start_year_month_day}to{end_year_month_day}_"
            f"{self.mag_field}{self.file_format}"
        )

        return self.get_file_path_stem() / file_name
