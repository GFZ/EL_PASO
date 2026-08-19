# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

import csv
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from skyfield import api as sf_api

import el_paso as ep

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def calculate_geo_coords_from_tle(
    tle_filename: str | Path,
) -> tuple[str, list[datetime], ep.Variable]:
    """Calculate GEO coordinates (x, y, z) in kilometers from a TLE file.

    Args:
        tle_filename (str | Path): The file path containing the TLE data.

    Returns:
        tuple[str, list[datetime], ep.Variable]: A tuple of:
            - satellite_name: The name of the satellite.
            - tle_times: UTC datetime for each TLE epoch.
            - geo_coordinate variable.
    """
    lines = Path(tle_filename).read_text().splitlines()
    tle_data = [(lines[i].strip(), lines[i + 1].strip()) for i in range(0, len(lines), 2)]
    satellite_name = tle_data[0][0].split()[1]

    timescale = sf_api.load.timescale()
    tle_times: list[datetime] = []
    geo_coordinates: list[NDArray[np.float64]] = []

    for line1, line2 in tle_data:
        year = int(line1.split()[3][:2])
        doy = float(line1.split()[3][2:])
        tle_time = datetime(2000 + year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy)

        satellite = sf_api.EarthSatellite(line1, line2, satellite_name)
        geocentric = satellite.at(timescale.from_datetime(tle_time))

        tle_times.append(tle_time)
        geo_coordinates.append(geocentric.xyz.km)

    result = np.asarray(geo_coordinates, dtype=np.float64)

    if np.isnan(result).any():
        nan_indices = np.where(np.isnan(result).any(axis=1))[0]
        logger.warning(
            f"NaN values found in GEO coordinates at indices: {', '.join(str(i) for i in nan_indices)}. "
            "Check the TLE file at these indices."
        )

    xgeo_var = ep.Variable(data=result, original_unit=u.km)
    xgeo_var.metadata.add_processing_note("Created from TLE file.")
    xgeo_var.metadata.source_files.append(Path(tle_filename).name)

    return satellite_name, tle_times, xgeo_var


def calculate_geo_coords_from_omm(
    omm_line: dict[str, str],
    target_times: list[datetime],
) -> ep.Variable:
    """Calculate GEO coordinates (x, y, z) in kilometers from an OMM orbital element set.

    Args:
        omm_line (dict[str, str]): Dictionary containing the OMM (Orbit Mean-Elements Message)
            orbital element data for the satellite, as consumed by
            `skyfield.api.EarthSatellite.from_omm`.
        target_times (list[datetime]): UTC datetimes at which to evaluate the satellite position.

    Returns:
        ep.Variable: The GEO coordinate variable (x, y, z) in kilometers, evaluated at `target_times`.
    """
    timescale = sf_api.load.timescale()
    geo_coordinates: list[NDArray[np.float64]] = []

    satellite = sf_api.EarthSatellite.from_omm(timescale, omm_line)

    for t in target_times:
        geocentric = satellite.at(timescale.from_datetime(t))
        geo_coordinates.append(geocentric.itrf_xyz().km)

    result = np.asarray(geo_coordinates, dtype=np.float64)

    if np.isnan(result).any():
        nan_indices = np.where(np.isnan(result).any(axis=1))[0]
        logger.warning(
            f"NaN values found in GEO coordinates at indices: {', '.join(str(i) for i in nan_indices)}. "
            "Check the CSV file at these indices."
        )

    xgeo_var = ep.Variable(data=result, original_unit=u.km)
    xgeo_var.metadata.add_processing_note("Created from OMM CSV file.")

    return xgeo_var
