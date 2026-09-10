# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import logging
import os
import typing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import requests
from astropy import units as u
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation

import el_paso as ep
from el_paso.utils import timed_function

logger = logging.getLogger(__name__)

DEFAULT_BIN_CADENCE = timedelta(minutes=1)

_GSSC_EMU_SATELLITE_TO_ID: dict[str, str] = {
    "gsat0207": "galileo_gssc_emu_gsat0207_sd_l1",
    "gsat0215": "galileo_gssc_emu_gsat0215_sd_l1",
}

# NORAD catalog IDs, used to query Celestrak for orbital elements when ODI_Position is invalid
# (see _download_gssc_emu_omm).
_GSSC_EMU_SATELLITE_TO_NORAD_ID: dict[str, int] = {
    "gsat0207": 41859,
    "gsat0215": 43055,
}

_CELESTRAK_TIMEOUT_SECONDS = 30


def gssc_emu_proton_strategy(
    base_data_path: str | Path,
    satellite: str,
    *,
    file_format: ep.typing.MFSFormats = ".nc",
) -> ep.SavingStrategy:
    """Monthly RB saving strategy for GSSC EMU protons."""
    return ep.saving_strategies.MonthlyRBStrategy(
        base_data_path=Path(base_data_path),
        mission="ESA",
        satellite=satellite,
        instrument="emu-proton",
        mag_field="T89",
        file_format=file_format,
        data_standard=ep.data_standards.GFZStandard(),
    )


def _download_gssc_emu_omm(
    satellite: Literal["gsat0207", "gsat0215"],
    raw_data_path: str | Path,
    skip_existing: bool = True,  # noqa: FBT001, FBT002
) -> dict[str, str]:
    """Download today's orbital elements (OMM) for a GSSC EMU satellite from Celestrak.

    Used as a fallback for samples where `ODI_Position` (the CDF's own spacecraft position) is
    zero-filled or NaN. Celestrak's `gp.php` endpoint only serves the *current* element set, not
    a historical archive, so this caches one CSV per satellite per day under
    `<raw_data_path>/OMM/<satellite>/<satellite>_omm_YYYYMMDD.csv` and reuses it for the rest of
    the day instead of re-querying (Celestrak rate-limits/IP-blocks repeated requests).

    Args:
        satellite (Literal["gsat0207", "gsat0215"]): Which Galileo satellite's orbital elements
            to download.
        raw_data_path (str | Path): Base directory the OMM CSV is downloaded into.
        skip_existing (bool, optional): If True, reuse today's already-downloaded OMM file
            instead of re-querying Celestrak. Defaults to True.

    Returns:
        dict[str, str]: The OMM record as CCSDS OMM field name -> value, ready to pass to
        `el_paso.processing.calculate_geo_coords_from_omm`.

    Raises:
        FileNotFoundError: If Celestrak has no current element set for the satellite.
        ValueError: If the downloaded OMM CSV is empty.
    """
    norad_id = _GSSC_EMU_SATELLITE_TO_NORAD_ID[satellite]
    time_now = datetime.now(timezone.utc)

    sat_dir = Path(raw_data_path) / "OMM" / satellite
    sat_dir.mkdir(parents=True, exist_ok=True)
    omm_file_path = sat_dir / f"{satellite}_omm_{time_now:%Y%m%d}.csv"

    if skip_existing and omm_file_path.exists():
        logger.info(f"OMM file for {satellite} already downloaded today, reusing {omm_file_path}.")
    else:
        celestrak_url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=CSV"
        response = requests.get(celestrak_url, stream=True, timeout=_CELESTRAK_TIMEOUT_SECONDS)

        if response.status_code == requests.codes.not_found:
            msg = f"No current OMM data found on Celestrak for {satellite} (NORAD {norad_id})."
            raise FileNotFoundError(msg)

        response.raise_for_status()

        with omm_file_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        logger.info(f"Downloaded OMM data for {satellite} from Celestrak.")

    with omm_file_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        msg = f"OMM file {omm_file_path} is empty!"
        raise ValueError(msg)

    return rows[0]


@timed_function("process_gssc_emu_protons")
def process_gssc_emu_protons(
    start_time: datetime,
    end_time: datetime,
    satellite: Literal["gsat0207", "gsat0215"] = "gsat0207",
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    bin_cadence: timedelta = DEFAULT_BIN_CADENCE,
    num_cores: int = 16,
    username: str | None = None,
    password: str | None = None,
    saving_strategy: ep.SavingStrategy | None = None,
    skip_existing: bool = True,  # noqa: FBT001, FBT002
    *,
    calculate_Lstar: bool = True,
) -> None:
    """Process ESA GSSC EMU (Galileo) proton data into pitch-angle resolved flux and PSD.

    Galileo satellites are in MEO (~23,222 km altitude). Downloads the raw EMU L1 CDF files for
    the given satellite from the ESA GSSC FTP server, extracts the differential proton flux
    (FPDO) and spacecraft position (J2000, treated as ECI), converts the position to GEO. For
    samples where the CDF's own position (`ODI_Position`) is zero-filled or NaN, position is
    instead reconstructed from Celestrak's current orbital elements for the satellite (see
    `_download_gssc_emu_omm`). T89 magnetic field quantities (B_Calc, B_Eq, MLT, R_Eq, Alpha_Eq,
    L_m, and optionally L_star) are computed using the IRBEM library, the pitch angle distribution
    (FPDU) and phase space density (proton PSD) are derived from the omnidirectional flux, and
    the resulting variables are saved to disk (appending to existing files) using either the
    provided `saving_strategy` or a default monthly strategy.

    Args:
        start_time (datetime): Start of the time range to process.
        end_time (datetime): End of the time range to process.
        satellite (Literal["gsat0207", "gsat0215"]): Which Galileo satellite's EMU data to process.
        raw_data_path (str | Path): Base directory used for downloading and locating the raw EMU
            CDF files.
        processed_data_path (str | Path): Base directory in which the processed output files are
            saved.
        bin_cadence (timedelta, optional): Time binning cadence applied to the extracted
            variables. Defaults to timedelta(minutes=1) (EMU's native cadence).
        num_cores (int, optional): Number of CPU cores used for the magnetic field computations.
            Defaults to 16.
        username (str | None, optional): FTP username for GSSC. If None, read from the
            `GSSC_USER` environment variable. Defaults to None.
        password (str | None, optional): FTP password for GSSC. If None, read from the
            `GSSC_PASS` environment variable. Defaults to None.
        saving_strategy (ep.SavingStrategy | None, optional): Strategy used to save the
            processed variables. If None, a `MonthlyRBStrategy` (instrument "emu-proton") is
            used. Defaults to None.
        skip_existing (bool, optional): If True, skip downloading files that already exist
            locally. Defaults to True.
        calculate_Lstar (bool, optional): If True, also compute the L* magnetic field quantity.
            Defaults to True.

    Raises:
        ValueError: If `username` or `password` is not provided and not available via the
            `GSSC_USER`/`GSSC_PASS` environment variables.
    """
    if username is None:
        username = os.environ.get("GSSC_USER")
    if password is None:
        password = os.environ.get("GSSC_PASS")

    if username is None:
        msg = "GSSC username not found! Either load it from environment variables or pass it as an argument."
        raise ValueError(msg)

    if password is None:
        msg = "GSSC password not found! Either load it from environment variables or pass it as an argument."
        raise ValueError(msg)

    remote_dir_name = _GSSC_EMU_SATELLITE_TO_ID[satellite]
    data_path_stem = f"{raw_data_path}/GSSC_EMU/{satellite}/YYYY/"
    file_name_stem = rf"{remote_dir_name}_YYYYMMDD_V\d+\.cdf"

    ep.download(
        start_time,
        end_time,
        save_path=data_path_stem,
        file_cadence="daily",
        download_url=f"ftp://gssc.esa.int/emu/{remote_dir_name}/YYYY",
        file_name_stem=f"{remote_dir_name}_YYYYMMDD_V01.cdf.gz",
        method="ftp",
        authentication_info=(username, password),
        skip_existing=skip_existing,
    )

    # Native CDF units for FPDO are cm^-2 s^-1 sr^-1 MeV^-1. construct_pitch_angle_distribution()
    # below expects an sr-free omnidirectional flux and introduces the "/sr" factor itself, so the
    # sr^-1 is converted to an equivalent 4*pi-integrated omnidirectional flux, same as
    # process_ngrm_satellite.py does for NGRM's differential flux channels.
    flux_unit_per_sr = typing.cast("u.Unit", (u.cm**2 * u.s * u.sr * u.MeV) ** (-1))
    flux_unit_omni = typing.cast("u.Unit", (u.cm**2 * u.s * u.MeV) ** (-1))

    extraction_infos = [
        ep.ExtractionInfo(result_key="Epoch", name_or_column="Epoch", unit=ep.units.cdf_epoch),
        ep.ExtractionInfo(result_key="FPDO", name_or_column="FPDO", unit=flux_unit_per_sr),
        ep.ExtractionInfo(result_key="Energy_FPDO", name_or_column="FPDO_Energy", unit=u.MeV, is_time_dependent=False),
        ep.ExtractionInfo(result_key="x_ECI", name_or_column="ODI_Position", unit=u.km),
    ]

    variables = ep.extract_variables_from_files(
        start_time,
        end_time,
        file_cadence="daily",
        data_path=data_path_stem,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )

    variables["Epoch"].convert_to_unit(ep.units.posixtime)
    obstimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in variables["Epoch"].get_data()]

    # convert ECI coordinates to GEO using astropy
    xeci_data = variables["x_ECI"].get_data()
    coords_ECI = GCRS(
        CartesianRepresentation(x=xeci_data[:, 0], y=xeci_data[:, 1], z=xeci_data[:, 2]),
        obstime=obstimes,
    )
    coords_ITRS = coords_ECI.transform_to(ITRS(obstime=obstimes))
    xgeo_data = np.stack((coords_ITRS.x, coords_ITRS.y, coords_ITRS.z)).T.value  # ty:ignore[unresolved-attribute]

    # ODI_Position is zero-filled or NaN for some/all samples in some GSSC EMU files; for those
    # samples only, position is reconstructed from Celestrak's current orbital elements instead.
    invalid_position_mask = np.all(xeci_data == 0, axis=1) | np.any(np.isnan(xeci_data), axis=1)

    if invalid_position_mask.any():
        omm_line = _download_gssc_emu_omm(satellite=satellite, raw_data_path=raw_data_path, skip_existing=skip_existing)
        invalid_obstimes = [t for t, invalid in zip(obstimes, invalid_position_mask, strict=True) if invalid]
        omm_xgeo_var = ep.processing.calculate_geo_coords_from_omm(omm_line, invalid_obstimes)
        xgeo_data[invalid_position_mask] = omm_xgeo_var.get_data(u.km)

    variables["xGEO"] = ep.Variable(data=xgeo_data, original_unit=u.km)
    del variables["x_ECI"]

    variables["FPDO"].apply_thresholds_on_data(lower_threshold=0)
    variables["FPDO"] = ep.Variable(
        data=variables["FPDO"].get_data(flux_unit_per_sr) * 4 * np.pi, original_unit=flux_unit_omni
    )

    time_bin_methods = {
        "Energy_FPDO": ep.TimeBinMethod.Repeat,
        "FPDO": ep.TimeBinMethod.NanMedian,
        "xGEO": ep.TimeBinMethod.NanMean,
    }

    binned_time_var = ep.processing.bin_by_time(
        variables["Epoch"], variables, time_bin_methods, bin_cadence, start_time=start_time, end_time=end_time
    )

    pa_local_data = np.tile(np.arange(5, 91, 5), (len(binned_time_var.get_data()), 1)).astype(np.float64)
    variables["PA_local"] = ep.Variable(data=pa_local_data, original_unit=u.deg)

    variables_to_compute: ep.processing.VariableRequest = [
        ("B_Calc", "T89"),
        ("B_Eq", "T89"),
        ("MLT_Eq", "T89"),
        ("R_Eq", "T89"),
        ("Alpha_Eq", "T89"),
    ]

    if calculate_Lstar:
        variables_to_compute.append(("L_star", "T89"))
        variables_to_compute.append(("L_m", "T89"))

    irbem_options = ep.processing.magnetic_field_utils.IrbemOptions(
        lstar_quantity=ep.processing.magnetic_field_utils.LstarQuantity.LSTAR
        if calculate_Lstar
        else ep.processing.magnetic_field_utils.LstarQuantity.NONE,
    )

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=binned_time_var,
        xgeo_var=variables["xGEO"],
        energy_var=variables["Energy_FPDO"],
        pa_local_var=variables["PA_local"],
        particle_species="proton",
        variables_to_compute=variables_to_compute,
        irbem_options=irbem_options,
        num_cores=num_cores,
    )

    FPDU_var = ep.processing.construct_pitch_angle_distribution(
        variables["FPDO"], variables["PA_local"], magnetic_field_variables["Alpha_Eq_T89"], flux_type="omni"
    )
    FPDU_var.apply_thresholds_on_data(lower_threshold=0)

    psd_var = ep.processing.compute_phase_space_density(FPDU_var, variables["Energy_FPDO"], particle_species="proton")

    variables_to_save: dict[ep.typing.InternalName, ep.Variable] = {
        "Epoch": binned_time_var,
        "FEDU": FPDU_var,
        "Energy_FEDU": variables["Energy_FPDO"],
        "Alpha": variables["PA_local"],
        "Alpha_Eq": magnetic_field_variables["Alpha_Eq_T89"],
        "R_Eq": magnetic_field_variables["R_Eq_T89"],
        "MLT": magnetic_field_variables["MLT_Eq_T89"],
        "B_Calc": magnetic_field_variables["B_Calc_T89"],
        "B_Eq": magnetic_field_variables["B_Eq_T89"],
        "Position": variables["xGEO"],
        "PSD": psd_var,
    }

    if calculate_Lstar:
        variables_to_save["L_star"] = magnetic_field_variables["L_star_T89"]
        variables_to_save["L_m"] = magnetic_field_variables["L_m_T89"]

    if saving_strategy is None:
        saving_strategy = gssc_emu_proton_strategy(processed_data_path, satellite)

    ep.save(variables_to_save, saving_strategy, start_time, end_time, time_var=binned_time_var, append=True)


CLI_DEFAULTS = {
    "raw_data_path": "raw_GSSC_EMU",
    "processed_data_path": "processed_GSSC_EMU",
}

if __name__ == "__main__":
    ep.run_recipe_cli(process_gssc_emu_protons, defaults=CLI_DEFAULTS)
