# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime, timedelta
from pathlib import Path

import el_paso as ep


class DailyLEORBStrategy(ep.typing.MonthlyRBStrategy):
    """Save PRBEM-standard LEO radiation-belt data into one NetCDF file per day.

    This strategy extends `MonthlyRBStrategy` but splits the output into daily
    files instead of monthly ones, and fixes the output variable list and file
    format (NetCDF) for low-Earth-orbit radiation-belt missions.
    """

    def _get_output_file_entries(self) -> list[ep.typing.InternalName]:
        """Return the standard variable list plus user-defined custom variables."""
        return [
            "FEDU",
            "FEIU",
            "FEDO",
            "FPDU",
            "Epoch",
            "Alpha_Eq",
            "Alpha_Eq_range",
            "Energy_FEDU",
            "Energy_FEIU",
            "Energy_FEDO",
            "Energy_FPDU",
            "Alpha",
            "Alpha_range",
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
            "Alpha_LC",
            "Alpha_LC_Eq",
            "Position_geo_alt",
            "Position_geo_lat",
            "Position_geo_lon",
        ]

    def get_file_path(
        self,
        interval_start: datetime,
        interval_end: datetime,  # noqa: ARG002
        output_file: ep.typing.OutputFile,  # noqa: ARG002
    ) -> Path:
        """Generate the daily file path for the configured format."""
        file_name = f"{self.get_file_name_stem()}_{interval_start.strftime('%Y%m%d')}_{self.mag_field}.nc"

        return self.get_file_path_stem() / file_name

    def get_time_intervals_to_save(
        self, start_time: datetime | None, end_time: datetime | None
    ) -> list[ep.typing.TimeInterval]:
        """Split the requested time range into full daily intervals."""
        time_intervals: list[ep.typing.TimeInterval] = []

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
