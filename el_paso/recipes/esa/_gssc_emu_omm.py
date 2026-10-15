# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""Position reconstruction from Space-Track OMM data, shared by the GSSC EMU recipes."""

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from astropy import units as u
from numpy.typing import NDArray

import el_paso as ep
from el_paso.recipes.esa import ESAGSSCSatellite

_GSSC_EMU_SATELLITE_TO_NORAD_ID: dict[ESAGSSCSatellite, int] = {
    "gsat0207": 41859,
    "gsat0215": 43055,
}

_OMM_SEARCH_RANGE = timedelta(days=7)


def fill_invalid_positions_from_omm(
    xgeo_data: NDArray[np.float64],
    invalid_position_mask: NDArray[np.bool_],
    obstimes: list[datetime],
    satellite: ESAGSSCSatellite,
    raw_data_path: str | Path,
    start_time: datetime,
    end_time: datetime,
    *,
    skip_existing: bool = True,
) -> None:
    """Replace the invalid GEO positions of a GSSC EMU satellite by positions propagated from OMM data.

    Downloads the satellite's OMM element sets from Space-Track (see `el_paso.download_omm`) for
    the time range plus a margin of a few days, and propagates each invalid sample from the
    element set whose epoch is closest to it.

    Args:
        xgeo_data (NDArray[np.float64]): GEO positions in km, shape (n, 3). Modified in place.
        invalid_position_mask (NDArray[np.bool_]): True where the position is invalid, shape (n,).
        obstimes (list[datetime]): UTC time stamp of each sample.
        satellite (ESAGSSCSatellite): The Galileo satellite the positions belong to.
        raw_data_path (str | Path): Base directory the OMM files are downloaded into.
        start_time (datetime): Start of the processed time range.
        end_time (datetime): End of the processed time range.
        skip_existing (bool): Passed on to `el_paso.download_omm`.

    Raises:
        FileNotFoundError: If Space-Track has no element sets for the satellite around the time range.
    """
    norad_id = str(_GSSC_EMU_SATELLITE_TO_NORAD_ID[satellite])
    omm_path = Path(raw_data_path) / "OMM"
    ep.download_omm(
        norad_id,
        start_time - _OMM_SEARCH_RANGE,
        end_time + _OMM_SEARCH_RANGE,
        save_path=omm_path,
        skip_existing=skip_existing,
    )

    omm_records: list[dict[str, str]] = []
    for omm_file in omm_path.glob("*/omm_*.csv"):
        with omm_file.open(newline="") as f:
            omm_records += [row for row in csv.DictReader(f) if row["NORAD_CAT_ID"] == norad_id]

    if not omm_records:
        msg = f"No OMM data found on Space-Track for {satellite} (NORAD {norad_id}) around {start_time}."
        raise FileNotFoundError(msg)

    # propagate each invalid sample from the element set with the epoch closest to it
    omm_epochs = np.array(
        [datetime.fromisoformat(r["EPOCH"]).replace(tzinfo=timezone.utc).timestamp() for r in omm_records]
    )
    invalid_indices = np.flatnonzero(invalid_position_mask)
    invalid_times = np.array([obstimes[i].timestamp() for i in invalid_indices])
    nearest_record = np.abs(omm_epochs[None, :] - invalid_times[:, None]).argmin(axis=1)

    for record_index, omm_record in enumerate(omm_records):
        sample_indices = invalid_indices[nearest_record == record_index]
        if sample_indices.size:
            omm_xgeo_var = ep.processing.calculate_geo_coords_from_omm(
                omm_record, [obstimes[i] for i in sample_indices]
            )
            xgeo_data[sample_indices] = omm_xgeo_var.get_data(u.km)
