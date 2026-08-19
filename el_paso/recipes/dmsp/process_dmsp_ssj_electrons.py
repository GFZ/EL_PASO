# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from astropy import units as u

import el_paso as ep

logging.captureWarnings(capture=True)
logger = logging.getLogger(__name__)

TELE_ALPHA_ANGLES = np.array([0.0, 0.0])
TELE_BETA_ANGLES = np.array([-180.0, 90.0])

DMSPSatellites = Literal["f17"]


def process_dmsp_ssj_electrons(
    sat_str: DMSPSatellites,
    processed_data_path: str | Path,
    raw_data_path: str | Path,
    start_time: datetime,
    end_time: datetime,
    num_cores: int = 32,
) -> None:
    """Process DMSP SSJ precipitating electron data into omnidirectional fluxes with magnetic field coordinates.

    This downloads and extracts SSM magnetometer data and SSJ precipitating electron/ion data for the
    given DMSP satellite and time range, bins both to a common 10-second cadence, and converts the SSJ
    differential energy flux into an omnidirectional differential flux (assuming a 90-degree SSJ-5
    field of view). It computes the spacecraft position in GEO coordinates, derives local pitch angles
    for the two SSJ telescopes from the SSM magnetic field, folds them around 90 degrees, and computes
    T89 magnetic field model quantities (B_Calc, B_Eq, MLT, B_fofl, R_Eq, Alpha_Eq, Alpha_LC_Eq,
    Alpha_LC). The resulting variables are saved to disk using a daily LEO/RB saving strategy.

    Args:
        sat_str (DMSPSatellites): Identifier of the DMSP satellite to process (e.g. "f17").
        processed_data_path (str | Path): Base directory in which the processed output files are saved.
        raw_data_path (str | Path): Base directory used for downloading and locating the raw SSM/SSJ CDF files.
        start_time (datetime): Start of the time range to process.
        end_time (datetime): End of the time range to process.
        num_cores (int, optional): Number of CPU cores used for the magnetic field computations. Defaults to 32.
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    data_path_stem = f"{raw_data_path}/DMSP/{sat_str}/YYYY/MM/"

    ssm_vars = _get_ssm_variables(sat_str, data_path_stem, start_time, end_time)
    ssj_vars = _get_ssj_variables(sat_str, data_path_stem, start_time, end_time)

    time_bin_methods_ssm = {
        "b_brf": ep.TimeBinMethod.NanMean,
    }
    time_bin_cadence = timedelta(seconds=10)

    binned_time_var = ep.processing.bin_by_time(
        time_variable=ssm_vars["time"],
        variables=ssm_vars,
        time_bin_method_dict=time_bin_methods_ssm,
        time_binning_cadence=time_bin_cadence,
        start_time=start_time,
        end_time=end_time,
    )

    time_bin_methods_ssj = {
        "diff_energy_flux": ep.TimeBinMethod.NanMedian,
        "diff_energy": ep.TimeBinMethod.Repeat,
        "R_geo": ep.TimeBinMethod.NanMean,
        "lat_geo": ep.TimeBinMethod.NanMean,
        "lon_geo": ep.TimeBinMethod.NanMean,
    }

    _ = ep.processing.bin_by_time(
        time_variable=ssj_vars["time"],
        variables=ssj_vars,
        time_bin_method_dict=time_bin_methods_ssj,
        time_binning_cadence=time_bin_cadence,
        start_time=start_time,
        end_time=end_time,
    )

    # calculate differential omnidirectional flux
    ssj_vars["diff_energy_flux"].apply_thresholds_on_data(lower_threshold=0)

    en = ssj_vars["diff_energy"].get_data(u.eV)
    sort_idx = np.argsort(en[0, :])
    ssj_vars["diff_energy"].set_data(en[:, sort_idx], unit=u.eV)

    diff_flux = ssj_vars["diff_energy_flux"].get_data().astype(np.float64)
    diff_flux = diff_flux[:, sort_idx]
    diff_flux /= ssj_vars["diff_energy"].get_data(u.eV)

    diff_omni_flux = 4 * np.pi * diff_flux  # SSJ-5 observes 90 deg FOV

    diff_omni_flux = diff_omni_flux[:, :, np.newaxis]  # add pitch angle dimension
    ssj_vars["diff_omni_flux"] = ep.Variable(data=diff_omni_flux, original_unit=(u.cm**2 * u.s * u.eV) ** (-1))

    del ssj_vars["diff_energy_flux"]

    # calculate xGEO
    geo_spherical = np.vstack(
        (
            ssj_vars["R_geo"].get_data(ep.units.RE).astype(np.float64),
            ssj_vars["lat_geo"].get_data(u.deg).astype(np.float64),
            ssj_vars["lon_geo"].get_data(u.deg).astype(np.float64),
        )
    ).T

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in binned_time_var.get_data(ep.units.posixtime)]
    xgeo_data = ep.processing.magnetic_field_utils.Coords().transform(
        datetimes,
        geo_spherical,
        ep.IRBEM_SYSAXIS_SPH,
        ep.IRBEM_SYSAXIS_GEO,
    )
    xgeo_var = ep.Variable(data=xgeo_data, original_unit=ep.units.RE)

    # calculate pitch angles
    tele_alpha_angles_var = ep.Variable(data=TELE_ALPHA_ANGLES, original_unit=u.deg)
    tele_beta_angles_var = ep.Variable(data=TELE_BETA_ANGLES, original_unit=u.deg)
    local_pa_var = ep.processing.compute_pitch_angles_for_telescopes(
        ssm_vars["b_brf"],
        tele_alpha_angles_var,
        tele_beta_angles_var,
    )
    local_pa_var.set_data(local_pa_var.get_data()[:, np.newaxis, :], unit="same")

    # fold pitch angles around 90 degree
    local_pa = local_pa_var.get_data(u.degree)
    local_pa_folded = np.where(local_pa > 90, 180 - local_pa, local_pa)
    local_pa_var.set_data(local_pa_folded, unit=u.degree)

    # Calculate magnetic field variables
    variables_to_compute: ep.processing.VariableRequest = [
        ("B_Calc", "T89"),
        ("B_Eq", "T89"),
        ("MLT", "T89"),
        ("B_fofl", "T89"),
        ("R_Eq", "T89"),
        ("Alpha_Eq", "T89"),
        ("Alpha_LC_Eq", "T89"),
        ("Alpha_LC", "T89"),
    ]

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=binned_time_var,
        xgeo_var=xgeo_var,
        energy_var=ssj_vars["diff_energy"],
        pa_local_var=local_pa_var,
        particle_species="electron",
        variables_to_compute=variables_to_compute,
        irbem_options=ep.processing.magnetic_field_utils.IrbemOptions(),
        num_cores=num_cores,
    )

    variables_to_save: dict[ep.typing.InternalName, ep.Variable] = {
        "Epoch": binned_time_var,
        "FEDO": ssj_vars["diff_omni_flux"],
        "Energy_FEDO": ssj_vars["diff_energy"],
        "Alpha_range": local_pa_var,
        "Alpha_Eq_range": magnetic_field_variables["Alpha_Eq_T89"],
        "R_Eq": magnetic_field_variables["R_Eq_T89"],
        "MLT": magnetic_field_variables["MLT_T89"],
        "B_Calc": magnetic_field_variables["B_Calc_T89"],
        "B_Eq": magnetic_field_variables["B_Eq_T89"],
        "Position": xgeo_var,
        "Position_geo_alt": ssj_vars["R_geo"],
        "Position_geo_lat": ssj_vars["lat_geo"],
        "Position_geo_lon": ssj_vars["lon_geo"],
        "Alpha_LC": magnetic_field_variables["Alpha_LC_T89"],
        "Alpha_LC_Eq": magnetic_field_variables["Alpha_LC_Eq_T89"],
    }

    saving_strategy = ep.saving_strategies.DailyLEORBStrategy(
        base_data_path=Path(processed_data_path),
        mission="DMSP",
        satellite=sat_str,
        instrument="SSJ",
        mag_field="T89",
        data_standard=ep.data_standards.GFZStandard(),
    )

    ep.save(variables_to_save, saving_strategy, start_time, end_time, time_var=binned_time_var)


def _get_ssm_variables(
    sat_str: DMSPSatellites,
    data_path_stem: str | Path,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, ep.Variable]:
    url = f"https://cdaweb.gsfc.nasa.gov/pub/data/dmsp/dmsp{sat_str}/ssm/magnetometer/YYYY/"

    file_name_stem = "dmsp-" + sat_str + "_ssm_magnetometer_YYYYMMDD_.{6}.cdf"

    ep.download(
        start_time,
        end_time,
        save_path=data_path_stem,
        file_cadence="daily",
        download_url=url,
        file_name_stem=file_name_stem,
        skip_existing=True,
    )

    extraction_infos = [
        ep.ExtractionInfo(name_or_column="Epoch", unit=ep.units.cdf_epoch, result_key="time"),
        ep.ExtractionInfo(name_or_column="B_SC_OBS_ORIG", unit=u.nT, result_key="b_brf"),
    ]

    return ep.extract_variables_from_files(
        start_time,
        end_time,
        file_cadence="daily",
        data_path=data_path_stem,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )


def _get_ssj_variables(
    sat_str: DMSPSatellites,
    data_path_stem: str | Path,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, ep.Variable]:
    url = f"https://cdaweb.gsfc.nasa.gov/pub/data/dmsp/dmsp{sat_str}/ssj/precipitating-electrons-ions/YYYY/"

    file_name_stem = "dmsp-" + sat_str + "_ssj_precipitating-electrons-ions_YYYYMMDD_.{6}.cdf"

    ep.download(
        start_time,
        end_time,
        save_path=data_path_stem,
        file_cadence="daily",
        download_url=url,
        file_name_stem=file_name_stem,
        skip_existing=True,
    )

    extraction_infos = [
        ep.ExtractionInfo(name_or_column="Epoch", unit=ep.units.cdf_epoch, result_key="time"),
        ep.ExtractionInfo(
            name_or_column="ELE_DIFF_ENERGY_FLUX",
            unit=u.eV * (u.cm**2 * u.s * u.eV * u.sr) ** (-1),
            result_key="diff_energy_flux",
        ),
        ep.ExtractionInfo(name_or_column="CHANNEL_ENERGIES", unit=u.eV, result_key="diff_energy"),
        ep.ExtractionInfo(name_or_column="SC_GEOCENTRIC_R", unit=u.km, result_key="R_geo"),
        ep.ExtractionInfo(name_or_column="SC_GEOCENTRIC_LAT", unit=u.deg, result_key="lat_geo"),
        ep.ExtractionInfo(name_or_column="SC_GEOCENTRIC_LON", unit=u.deg, result_key="lon_geo"),
    ]

    return ep.extract_variables_from_files(
        start_time,
        end_time,
        file_cadence="daily",
        data_path=data_path_stem,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )


if __name__ == "__main__":
    start_time = datetime(2013, 12, 14, tzinfo=timezone.utc)
    end_time = datetime(2013, 12, 14, 11, 59, tzinfo=timezone.utc)

    satellites = ["f17"]

    for sat in satellites:
        process_dmsp_ssj_electrons(
            sat_str=sat,  # ty:ignore[invalid-argument-type]
            raw_data_path="dmsp/raw/",
            processed_data_path="dmsp/processed/",
            start_time=start_time,
            end_time=end_time,
            num_cores=64,
        )
