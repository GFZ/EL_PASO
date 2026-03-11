# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from skyfield.api import EarthSatellite, load

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TLE:
    """A class to track satellite positions using TLE data.

    Args:
        tle_filename (str | Path): The file path containing the TLE data.

    Attributes:
        tle_filename (str | Path): The file path containing the TLE data.
        tle_data (list of tuple): A list containing pairs of TLE lines for each
            satellite.
        satellite_name (str): The name of the satellite based on the first line of
            the TLE data.
    """

    tle_filename: str | Path
    tle_data: list[tuple[str, str]] = field(init=False)
    satellite_name: str = field(init=False)
    tle_time_list: list[datetime] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize parsed and derived values from the input TLE file."""
        tle_path = Path(self.tle_filename)
        tle_data, satellite_name = self._read_tle_file(tle_path)
        tle_time_list = self.get_tle_time(tle_data)

        object.__setattr__(self, "tle_data", tle_data)
        object.__setattr__(self, "satellite_name", satellite_name)
        object.__setattr__(self, "tle_time_list", tle_time_list)

    def _read_tle_file(self, filename: Path) -> tuple[list[tuple[str, str]], str]:
        """Read TLE data from a file.

        Args:
            filename (Path): The file path containing the TLE data.

        Returns:
            tuple[list[tuple[str, str]], str]: A tuple containing:
                - tle_data: A list of satellite TLE line pairs.
                - satellite_name: The name of the satellite.
        """
        with filename.open() as file:
            lines = file.readlines()

        tle_data = [(lines[i].strip(), lines[i + 1].strip()) for i in range(0, len(lines), 2)]
        satellite_name = tle_data[0][0].split()[1]

        return tle_data, satellite_name

    def get_tle_time(self, tle_data: list[tuple[str, str]]) -> list[datetime]:
        """Generate a list of UTC datetime objects from the TLE data.

        Args:
            tle_data (list[tuple[str, str]]): The TLE data for the satellite.

        Returns:
            list[datetime]: A list of UTC datetime objects.
        """
        tle_times = []
        for tle in tle_data:
            year, doy = str(tle[0].split()[3])[:2], str(tle[0].split()[3])[2:]
            tle_times.append(
                datetime(2000 + int(year), 1, 1, 0, 0, 0, tzinfo=timezone.utc) + timedelta(days=float(doy))
            )

        return tle_times

    def calculate_geo_coords(self) -> NDArray[np.float64]:
        """Calculate GEO coordinates (x, y, z) in kilometers using Skyfield.

        Returns:
            NDArray[np.float64]: GEO coordinates for each TLE epoch in a shape
                `(n, 3)` array, where columns are `(x, y, z)` in kilometers.
        """
        timescale = load.timescale()
        geo_coordinates = []

        for tle_lines, tle_time in zip(self.tle_data, self.tle_time_list, strict=True):
            satellite = EarthSatellite(tle_lines[0], tle_lines[1], self.satellite_name)
            geocentric = satellite.at(timescale.from_datetime(tle_time))
            xyz = geocentric.xyz.km
            geo_coordinates.append(xyz)

        geo_coordinates = np.asarray(geo_coordinates, dtype=np.float64)

        if np.isnan(geo_coordinates).any():
            nan_indices = np.where(np.isnan(geo_coordinates).any(axis=1))[0]
            logger.warning(
                f"NaN values found in GEO coordinates at indices: {', '.join(str(idx) for idx in nan_indices)}. "
                "Check the TLE file at these indices."
            )

        return np.asarray(geo_coordinates, dtype=np.float64)
