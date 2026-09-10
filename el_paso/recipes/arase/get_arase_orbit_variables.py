# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing
from pathlib import Path
from typing import Literal

from astropy import units as u

import el_paso as ep

if typing.TYPE_CHECKING:
    from datetime import datetime


def get_arase_orbit_level_2_variables(
    start_time: datetime,
    end_time: datetime,
    raw_data_path: str | Path = ".",
    *,
    skip_existing: bool = True,
) -> dict[str, ep.Variable]:
    """Download and extract Arase Level 2 orbit (definitive) data.

    Downloads the daily Arase ``orb/def`` Level 2 CDF files covering the requested time range
    and extracts the epoch and SM position variables from them.

    Args:
        start_time (datetime): Start of the time range to retrieve data for.
        end_time (datetime): End of the time range to retrieve data for.
        raw_data_path (str | Path, optional): Directory where the downloaded raw data files are
                                            stored. Defaults to ".".
        skip_existing (bool): If True, skip downloading files that already exist locally.
                                            Defaults to True.

    Returns:
        dict[str, ep.Variable]: A dictionary containing the extracted "Epoch" and "pos_sm" variables.
    """
    raw_data_path = Path(raw_data_path)

    file_name_stem = "erg_orb_l2_YYYYMMDD_.{3}.cdf"
    url = "https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/orb/def/YYYY/"

    ep.download(
        start_time,
        end_time,
        save_path=raw_data_path,
        download_url=url,
        file_name_stem=file_name_stem,
        file_cadence="daily",
        method="request",
        skip_existing=skip_existing,
    )

    extraction_infos = [
        ep.ExtractionInfo(
            result_key="Epoch",
            name_or_column="epoch",
            unit=ep.units.tt2000,
        ),
        ep.ExtractionInfo(
            result_key="pos_sm",
            name_or_column="pos_sm",
            unit=u.dimensionless_unscaled,
        ),
    ]

    variables = ep.extract_variables_from_files(
        start_time,
        end_time,
        "daily",
        data_path=raw_data_path,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )
    return variables


def get_arase_orbit_level_3_variables(
    start_time: datetime,
    end_time: datetime,
    mag_field: Literal["OP77Q", "T89", "TS04"],
    raw_data_path: str | Path = ".",
    *,
    skip_existing: bool = True,
) -> dict[str, ep.Variable]:
    """Download and extract Arase Level 3 orbit data for a given magnetic field model.

    Downloads the daily Arase ``orb/l3`` Level 3 CDF files corresponding to the given `mag_field`
    model, extracts the epoch, local and equatorial magnetic field magnitude, Lm, Lstar and
    equatorial position variables, truncates all variables to the requested time range, and
    derives "MLT" and "R0" variables from the equatorial position.

    Args:
        start_time (datetime): Start of the time range to retrieve data for.
        end_time (datetime): End of the time range to retrieve data for.
        mag_field (Literal["OP77Q", "T89", "TS04"]): The magnetic field model whose Level 3 orbit
                                                    data should be downloaded and extracted.
        raw_data_path (str | Path, optional): Directory where the downloaded raw data files are
                                            stored. Defaults to ".".
        skip_existing (bool): If True, skip downloading files that already exist locally.
                                            Defaults to True.

    Returns:
        dict[str, ep.Variable]: A dictionary containing the extracted and derived variables
                            ("Epoch", "B_local", "B_eq", "Lm", "Lstar", "MLT", "R0").

    Raises:
        ValueError: If `mag_field` is not one of "OP77Q", "T89" or "TS04".
    """
    raw_data_path = Path(raw_data_path)

    match mag_field:
        case "OP77Q":
            mag_field_label = "opq"
        case "T89":
            mag_field_label = "t89"
        case "TS04":
            mag_field_label = "ts04"
        case _:
            msg = "Encountered invalid mag field for Arase orbit: {mag_field}!"
            raise ValueError(msg)

    url = f"https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/orb/l3/{mag_field_label}/YYYY/MM/"

    if mag_field_label == "opq":
        mag_field_label = "op"
    file_name_stem = "erg_orb_l3_" + mag_field_label + "_YYYYMMDD_.{3}.cdf"

    ep.download(
        start_time,
        end_time,
        save_path=raw_data_path,
        download_url=url,
        file_name_stem=file_name_stem,
        file_cadence="daily",
        method="request",
        skip_existing=skip_existing,
    )

    match mag_field:
        case "OP77Q":
            mag_field_label = "op"
        case "T89":
            mag_field_label = "t89"
        case "TS04":
            mag_field_label = "TS04"
        case _:
            msg = "Encountered invalid mag field for Arase orbit: {mag_field}!"
            raise ValueError(msg)

    extraction_infos = [
        ep.ExtractionInfo(
            result_key="Epoch",
            name_or_column="epoch",
            unit=ep.units.tt2000,
        ),
        ep.ExtractionInfo(
            result_key="B_local",
            name_or_column=f"pos_blocal_{mag_field_label}",
            unit=u.nT,
        ),
        ep.ExtractionInfo(
            result_key="B_eq",
            name_or_column=f"pos_beq_{mag_field_label}",
            unit=u.nT,
        ),
        ep.ExtractionInfo(
            result_key="Lm",
            name_or_column=f"pos_lmc_{mag_field_label}",
            unit=u.dimensionless_unscaled,
        ),
        ep.ExtractionInfo(
            result_key="Lstar",
            name_or_column=f"pos_lstar_{mag_field_label}",
            unit=u.dimensionless_unscaled,
        ),
        ep.ExtractionInfo(
            result_key="pos_eq",
            name_or_column=f"pos_eq_{mag_field_label}",
            unit=u.dimensionless_unscaled,
        ),
    ]

    variables = ep.extract_variables_from_files(
        start_time,
        end_time,
        "daily",
        data_path=raw_data_path,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )

    for key in variables:
        if key == "Epoch":
            continue
        variables[key].truncate(variables["Epoch"], start_time, end_time)
    variables["Epoch"].truncate(variables["Epoch"], start_time, end_time)

    mlt_data = variables["pos_eq"].get_data()[:, 1]
    R_data = variables["pos_eq"].get_data()[:, 0]

    variables["MLT"] = ep.Variable(data=mlt_data, original_unit=u.hour)
    variables["R0"] = ep.Variable(data=R_data, original_unit=ep.units.RE)

    del variables["pos_eq"]

    return variables
