# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import typing
from datetime import datetime
from datetime import timezone as tz
from typing import Any

import numpy as np
from astropy import units as u  # type: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray
from sscws.sscws import CoordinateSystem, SscWs  # type: ignore[reportMissingTypeStubs]

import el_paso as ep


def get_real_time_tipsod(timestamps: NDArray[np.floating], sat_name: str, coord_system: str = "GEO") -> ep.Variable:
    """Gets real-time satellite position data from the TIPSOD service.

    This function queries the TIPSOD (Tool For Interactive Plotting, Sonification, and 3D Orbit Display)
    web service to retrieve the satellite's position (X, Y, Z) in a specified coordinate
    system at a given set of timestamps. The function then bins the retrieved data and
    computes the median position for each bin.

    Args:
        timestamps (NDArray[np.floating]): An array of timestamps in Unix time
            for which to retrieve satellite data. At least two timestamps are
            required to determine the time interval for data retrieval.
        sat_name (str): The name of the satellite (e.g., 'LANL-01A', 'GOES-15').
        coord_system (str, optional): The coordinate system for the returned data.
            Defaults to "GEO". Supported systems include "GEO", "GSE", "GSM", and "SM".

    Returns:
        ep.Variable: A variable containing the satellite's median position
            (X, Y, Z) for each time interval, converted to Earth Radii (RE).

    Raises:
        ValueError: If fewer than two timestamps are provided, as the time interval
                    cannot be determined.
        ValueError: If the SSCWS query fails or returns an unexpected format.
    """
    minimum_datetimes = 2

    if len(timestamps) < minimum_datetimes:
        msg = "At least two datetimes are required."
        raise ValueError(msg)

    datetimes = [datetime.fromtimestamp(t, tz=tz.utc) for t in timestamps]

    def _convert_to_sscws_compatible_time(dt: datetime) -> str:
        """Add milliseconds to the datetime objects.

        Args:
            dt (datetime): The datetime to be converted.

        Returns:
            str: The datetime in SSCWS compatible format with milliseconds.

        """
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    original_datetimes = datetimes
    extra_interval = original_datetimes[-1] - original_datetimes[-2]
    datetimes_str = [_convert_to_sscws_compatible_time(dt) for dt in datetimes]

    ssc = SscWs()

    # Retrieve data for the entire time range
    start_time = datetimes_str[0]
    end_time = _convert_to_sscws_compatible_time(original_datetimes[-1] + extra_interval)
    time_range = [start_time, end_time]

    coord_systems_dict = {"GEO": "Geo", "GM": "Gm", "GSE": "Gse", "GSM": "Gsm", "SM": "Sm", "J2000": "GeiJ2000"}
    result = typing.cast(
        "Any", ssc.get_locations([sat_name], time_range, coords=[CoordinateSystem(coord_systems_dict[coord_system])])
    )  # type: ignore[reportUnknownMemberType]

    try:
        # Extract X, Y, Z coordinates and corresponding times
        times = np.array(result["Data"][0]["Time"])
        x_coords = np.array([entry["X"] for entry in result["Data"][0]["Coordinates"]]).flatten()
        y_coords = np.array([entry["Y"] for entry in result["Data"][0]["Coordinates"]]).flatten()
        z_coords = np.array([entry["Z"] for entry in result["Data"][0]["Coordinates"]]).flatten()
    except Exception as e:
        raise ValueError(str(result)) from e

    # Bin the data according to the given datetimes grid and compute the median of the points in each bin
    all_xyz: list[list[np.floating]] = []
    for i in range(len(datetimes) - 1):
        bin_mask = (times >= datetimes[i]) & (times < datetimes[i + 1])
        if bin_mask.any():
            x_median = np.median(x_coords[bin_mask])
            y_median = np.median(y_coords[bin_mask])
            z_median = np.median(z_coords[bin_mask])
            all_xyz.append([x_median, y_median, z_median])

    # Handle the last entry with a small extra interval
    last_bin_mask = (times >= datetimes[-1]) & (times <= original_datetimes[-1] + extra_interval)
    if last_bin_mask.any():
        x_median = np.median(x_coords[last_bin_mask])
        y_median = np.median(y_coords[last_bin_mask])
        z_median = np.median(z_coords[last_bin_mask])
        all_xyz.append([x_median, y_median, z_median])

    var = ep.Variable(data=np.asarray(all_xyz), original_unit=u.km)
    var.convert_to_unit(ep.units.RE)

    return var
