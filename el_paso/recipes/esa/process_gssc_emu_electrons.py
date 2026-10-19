# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import typing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from astropy import units as u
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation

import el_paso as ep
from el_paso.recipes.esa import ESAGSSCSatellite
from el_paso.recipes.esa._gssc_emu_omm import fill_invalid_positions_from_omm
from el_paso.recipes.solar_wind_indices.process_solar_wind_indices import process_solar_wind_indices
from el_paso.utils import timed_function

logger = logging.getLogger(__name__)

DEFAULT_BIN_CADENCE = timedelta(minutes=1)

_GSSC_EMU_SATELLITE_TO_ID: dict[ESAGSSCSatellite, str] = {
    "gsat0207": "galileo_gssc_emu_gsat0207_sd_l1",
    "gsat0215": "galileo_gssc_emu_gsat0215_sd_l1",
}


def gssc_emu_electron_strategy(
    base_data_path: str | Path,
    mag_field: ep.typing.MagneticFieldLiteral,
    satellite: ESAGSSCSatellite,
    *,
    file_format: ep.typing.MFSFormats = ".nc",
) -> ep.SavingStrategy:
    """Monthly RB saving strategy for GSSC EMU electrons."""
    return ep.saving_strategies.MonthlyRBStrategy(
        Path(base_data_path),
        "ESA",
        satellite,
        "emu-electron",
        mag_field,
        data_standard=ep.data_standards.GFZStandard(),
        file_format=file_format,
    )


@timed_function("process_gssc_emu_electrons")
def process_gssc_emu_electrons(
    start_time: datetime,
    end_time: datetime,
    satellite: ESAGSSCSatellite = "gsat0207",
    mag_field: ep.typing.MagneticFieldLiteral = "T89",
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    bin_cadence: timedelta = DEFAULT_BIN_CADENCE,
    num_cores: int = 16,
    username: str | None = None,
    password: str | None = None,
    save_strategy: Literal["netcdf"] = "netcdf",
    skip_existing: bool = True,  # noqa: FBT001, FBT002
    *,
    calculate_Lstar: bool = True,
    save_indices_sw: bool = False,
) -> None:
    """Process ESA GSSC EMU (Galileo) electron data into pitch-angle resolved flux and PSD.

    Galileo satellites are in MEO (~23,222 km altitude). Downloads the raw EMU L1 CDF files for
    the given satellite from the ESA GSSC FTP server, extracts the differential electron flux
    (FEDO) and spacecraft position (J2000, treated as ECI), converts the position to GEO. For
    samples where the CDF's own position (`ODI_Position`) is zero-filled or NaN, position is
    instead reconstructed from the satellite's OMM orbital elements, downloaded from Space-Track
    (see `el_paso.download_omm`). Magnetic field quantities (B_Calc, B_Eq, MLT, R_Eq, Alpha_Eq,
    L_m, and optionally L_star) are computed using `mag_field` via the IRBEM library, the pitch angle distribution
    (FEDU) and phase space density (electron PSD) are derived from the omnidirectional flux, and
    the resulting variables are saved to disk (appending to existing files) using the
    `gssc_emu_electron_strategy`.

    Args:
        start_time (datetime): Start of the time range to process.
        end_time (datetime): End of the time range to process.
        satellite (ESAGSSCSatellite): Which Galileo satellite's EMU data to process.
        mag_field (MagneticFieldLiteral): Magnetic field model used for the derived quantities.
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
        save_strategy (Literal["netcdf"]): The saving strategy used to write the processed
            variables. GSSC EMU data only supports a single netCDF-based strategy.
        skip_existing (bool, optional): If True, skip downloading files that already exist
            locally. Defaults to True.
        calculate_Lstar (bool, optional): If True, also compute the L* magnetic field quantity.
            Defaults to True.
        save_indices_sw (bool): If True, also save all solar wind indices and parameters,
            independent of this recipe's own output. Defaults to False.

    Raises:
        ValueError: If `username` or `password` is not provided and not available via the
            `GSSC_USER`/`GSSC_PASS` environment variables, or if the Space-Track credentials
            (`SPACETRACK_USER`/`SPACETRACK_PASS`) are missing while OMM data is needed.
        FileNotFoundError: If OMM data is needed but Space-Track has no element sets for the satellite.
    """
    del save_strategy

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

    ep.exit_if_download_only()

    # Native CDF units for FEDO are cm^-2 s^-1 sr^-1 MeV^-1. construct_pitch_angle_distribution()
    # below expects an sr-free omnidirectional flux and introduces the "/sr" factor itself, so the
    # sr^-1 is converted to an equivalent 4*pi-integrated omnidirectional flux, same as
    # process_ngrm_satellite.py does for NGRM's differential flux channels.
    flux_unit_per_sr = typing.cast("u.Unit", (u.cm**2 * u.s * u.sr * u.MeV) ** (-1))
    flux_unit_omni = typing.cast("u.Unit", (u.cm**2 * u.s * u.MeV) ** (-1))

    extraction_infos = [
        ep.ExtractionInfo(result_key="Epoch", name_or_column="Epoch", unit=ep.units.cdf_epoch),
        ep.ExtractionInfo(result_key="FEDO", name_or_column="FEDO", unit=flux_unit_per_sr),
        ep.ExtractionInfo(result_key="Energy_FEDO", name_or_column="FEDO_Energy", unit=u.MeV, is_time_dependent=False),
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
    # samples only, position is reconstructed from OMM element sets downloaded from Space-Track.
    invalid_position_mask = np.all(xeci_data == 0, axis=1) | np.any(np.isnan(xeci_data), axis=1)

    if invalid_position_mask.any():
        fill_invalid_positions_from_omm(
            xgeo_data,
            invalid_position_mask,
            obstimes,
            satellite,
            raw_data_path,
            start_time,
            end_time,
            skip_existing=skip_existing,
        )

    variables["xGEO"] = ep.Variable(data=xgeo_data, original_unit=u.km)
    del variables["x_ECI"]

    variables["FEDO"].apply_thresholds_on_data(lower_threshold=0)
    variables["FEDO"] = ep.Variable(
        data=variables["FEDO"].get_data(flux_unit_per_sr) * 4 * np.pi, original_unit=flux_unit_omni
    )

    time_bin_methods = {
        "Energy_FEDO": ep.TimeBinMethod.Repeat,
        "FEDO": ep.TimeBinMethod.NanMedian,
        "xGEO": ep.TimeBinMethod.NanMean,
    }

    binned_time_var = ep.processing.bin_by_time(
        variables["Epoch"], variables, time_bin_methods, bin_cadence, start_time=start_time, end_time=end_time
    )

    pa_local_data = np.tile(np.arange(5, 91, 5), (len(binned_time_var.get_data()), 1)).astype(np.float64)
    variables["PA_local"] = ep.Variable(data=pa_local_data, original_unit=u.deg)

    variables_to_compute: ep.processing.VariableRequest = [
        ("B_Calc", mag_field),
        ("B_Eq", mag_field),
        ("MLT_Eq", mag_field),
        ("R_Eq", mag_field),
        ("Alpha_Eq", mag_field),
        ("L_m", mag_field),
    ]

    if calculate_Lstar:
        variables_to_compute.append(("L_star", mag_field))

    irbem_options = ep.processing.magnetic_field_utils.IrbemOptions(
        lstar_quantity=ep.processing.magnetic_field_utils.LstarQuantity.LSTAR
        if calculate_Lstar
        else ep.processing.magnetic_field_utils.LstarQuantity.NONE,
    )

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=binned_time_var,
        xgeo_var=variables["xGEO"],
        energy_var=variables["Energy_FEDO"],
        pa_local_var=variables["PA_local"],
        particle_species="electron",
        variables_to_compute=variables_to_compute,
        irbem_options=irbem_options,
        num_cores=num_cores,
    )

    FEDU_var = ep.processing.construct_pitch_angle_distribution(
        variables["FEDO"], variables["PA_local"], magnetic_field_variables[f"Alpha_Eq_{mag_field}"], flux_type="omni"
    )
    FEDU_var.apply_thresholds_on_data(lower_threshold=0)

    psd_var = ep.processing.compute_phase_space_density(FEDU_var, variables["Energy_FEDO"], particle_species="electron")

    variables_to_save: dict[ep.typing.InternalName, ep.Variable] = {
        "Epoch": binned_time_var,
        "FEDU": FEDU_var,
        "Energy_FEDU": variables["Energy_FEDO"],
        "Alpha": variables["PA_local"],
        "Alpha_Eq": magnetic_field_variables[f"Alpha_Eq_{mag_field}"],
        "R_Eq": magnetic_field_variables[f"R_Eq_{mag_field}"],
        "MLT": magnetic_field_variables[f"MLT_Eq_{mag_field}"],
        "B_Calc": magnetic_field_variables[f"B_Calc_{mag_field}"],
        "B_Eq": magnetic_field_variables[f"B_Eq_{mag_field}"],
        "Position": variables["xGEO"],
        "PSD": psd_var,
        "L_m": magnetic_field_variables[f"L_m_{mag_field}"],
    }

    if calculate_Lstar:
        variables_to_save["L_star"] = magnetic_field_variables[f"L_star_{mag_field}"]

    saving_strategy = gssc_emu_electron_strategy(processed_data_path, mag_field, satellite)

    ep.save(
        variables_to_save,
        saving_strategy,
        start_time,
        end_time,
        time_var=binned_time_var,
        append=True,
    )

    if save_indices_sw:
        process_solar_wind_indices(
            start_time, end_time, processed_data_path=processed_data_path, bin_cadence=bin_cadence
        )


CLI_DEFAULTS = {
    "raw_data_path": "raw_GSSC_EMU",
    "processed_data_path": "processed_GSSC_EMU",
}

if __name__ == "__main__":
    ep.run_recipe_cli(process_gssc_emu_electrons, defaults=CLI_DEFAULTS)
