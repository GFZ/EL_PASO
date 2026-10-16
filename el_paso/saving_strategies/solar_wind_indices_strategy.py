# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import el_paso as ep
from el_paso.saving_strategy import OutputFile, SavingStrategy

if TYPE_CHECKING:
    from datetime import datetime

    from el_paso.load_indices_solar_wind_parameters import SW_Index
    from el_paso.typing import DataStandard, MFSFormats, StandardName, TimeInterval

ALL_SW_INDICES: list[SW_Index] = [
    "Kp",
    "Dst",
    "Pdyn",
    "ByIMF",
    "BzIMF",
    "Vsw",
    "Nsw",
    "G1",
    "G2",
    "G3",
    "W_params",
]


class SolarWindIndicesStrategy(SavingStrategy):
    """Save every solar wind/geomagnetic index EL-PASO can load, one file per month.

    Independent of any magnetic field model: the single output file holds the time axis plus
    every index in `ALL_SW_INDICES` that `data_standard` can also store.
    """

    output_files: list[OutputFile]

    def __init__(
        self,
        base_data_path: str | Path,
        data_standard: DataStandard[StandardName],
        file_format: MFSFormats = "nc",
    ) -> None:
        """Initialize a monthly solar wind indices saving strategy.

        Args:
            base_data_path (str | Path): Directory where monthly files are written.
            data_standard (DataStandard): Instance of the data standard implementation.
            file_format (MFSFormats): One of ``"nc"``, ``"cdf"``, ``"h5"``, or ``"mat"``.
                A leading dot is also accepted.
        """
        self.base_data_path = Path(base_data_path)
        self.data_standard = data_standard
        self.file_format = ep.utils.normalize_file_format(file_format)

        saveable_indices = [idx for idx in ALL_SW_INDICES if idx in data_standard.variable_infos]
        self.output_files = [
            OutputFile("full", ["Epoch", *saveable_indices], save_incomplete=True),
        ]

    def get_time_intervals_to_save(self, start_time: datetime | None, end_time: datetime | None) -> list[TimeInterval]:
        """Split the requested time range into full monthly intervals."""
        return ep.utils.get_monthly_datetime_intervals(start_time, end_time)

    def get_file_path_stem(self) -> Path:
        return self.base_data_path

    def get_file_name_stem(self) -> str:
        return "SolarWindIndices"

    def get_file_path(self, interval_start: datetime, interval_end: datetime, output_file: OutputFile) -> Path:  # noqa: ARG002
        """Generate the monthly file path for the configured format."""
        start_year_month_day = interval_start.strftime("%Y%m%d")
        end_year_month_day = interval_end.strftime("%Y%m%d")
        file_name = f"{self.get_file_name_stem()}_{start_year_month_day}to{end_year_month_day}{self.file_format}"

        return self.get_file_path_stem() / file_name
