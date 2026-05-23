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
TELE_BETA_ANGLES = np.array([0.0, 90.0])

DMSPSatellites = Literal["f18"]


def process_dmsp_ssj_electrons(
    sat_str: DMSPSatellites,
    processed_data_path: str | Path,
    raw_data_path: str | Path,
    start_time: datetime,
    end_time: datetime,
    save_strategy: Literal["dataorg", "netcdf"] = "netcdf",
    num_cores: int = 32,
) -> None:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    data_path_stem = f"{raw_data_path}/YYYY/MM/{sat_str}/"

    ssm_vars = _get_ssm_variables(sat_str, data_path_stem, start_time, end_time)
    ssj_vars = _get_ssj_variables(sat_str, data_path_stem, start_time, end_time)

    time_bin_methods_ssm = {
        "b_brf": ep.TimeBinMethod.NanMean,
    }
    time_bin_cadence = timedelta(minutes=1)

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
        "diff_energy": ep.TimeBinMethod.NanMean,
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

    ssj_vars["diff_energy_flux"].apply_thresholds_on_data(lower_threshold=0)

    geo_spherical = np.vstack((
        ssj_vars["R_geo"].get_data(ep.units.RE).astype(np.float64),
        ssj_vars["lat_geo"].get_data(u.deg).astype(np.float64),
        ssj_vars["lon_geo"].get_data(u.deg).astype(np.float64),
    )).T

    # calculate xGEO
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

    pa_data = local_pa_var.get_data(u.deg)
    lat_data = ssj_vars["lat_geo"].get_data(u.deg)

    print(datetimes)
    print(lat_data)
    print(pa_data[:,0])

    from matplotlib import pyplot as plt
    plt.scatter(datetimes, lat_data, s=10, c=pa_data[:,0])
    plt.colorbar()
    plt.savefig("test_0.png")

    plt.clf()
    plt.scatter(datetimes, lat_data, s=10, c=pa_data[:,1])
    plt.colorbar()
    plt.savefig("test_1.png")


    asdf

    # fold pitch angles around 90 degree
    local_pa = local_pa_var.get_data(u.degree)
    local_pa_folded = np.where(local_pa > 90, local_pa - 90, local_pa)  # noqa: PLR2004
    local_pa_var.set_data(local_pa_folded, unit=u.degree)

    # sort pitch angles in ascending order and apply to fluxes
    idx_sorted = np.argsort(local_pa_var.get_data(), axis=1)
    sorted_local_pa = np.take_along_axis(local_pa_var.get_data(), idx_sorted, axis=1)
    n_energy = mps_vars["diff_flux"].get_data().shape[1]
    sorted_diff_flux = np.take_along_axis(
        mps_vars["diff_flux"].get_data(), np.tile(idx_sorted[:, np.newaxis, :], [1, n_energy, 1]), axis=2
    )

    local_pa_var.set_data(sorted_local_pa, unit="same")
    mps_vars["diff_flux"].set_data(sorted_diff_flux, unit="same")

    # average energies over pitch angles
    diff_energy_avg = np.squeeze(np.mean(mps_vars["diff_energy"].get_data(u.MeV), axis=1))
    mps_vars["diff_energy"].set_data(diff_energy_avg, unit=u.MeV)

    # Calculate magnetic field variables
    variables_to_compute: ep.processing.VariableRequest = [
        ("B_local", "T89"),
        ("B_eq", "T89"),
        ("MLT", "T89"),
        ("B_eq", "T89"),
        ("R_eq", "T89"),
        ("PA_eq", "T89"),
        ("Lstar", "T89"),
        ("Lm", "T89"),
        ("invMu", "T89"),
        ("invK", "T89"),
    ]

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=binned_time_var,
        xgeo_var=xgeo_var,
        energy_var=mps_vars["diff_energy"],
        pa_local_var=local_pa_var,
        particle_species="electron",
        variables_to_compute=variables_to_compute,
        irbem_lib_path=str(irbem_lib_path),
        irbem_options=[1, 1, 4, 4, 0],
        num_cores=num_cores,
    )

    psd_var = ep.processing.compute_phase_space_density(
        mps_vars["diff_flux"], mps_vars["diff_energy"], particle_species="electron"
    )

    if save_strategy == "dataorg":
        variables_to_save = {
            "time": binned_time_var,
            "Flux": mps_vars["diff_flux"],
            "xGEO": xgeo_var,
            "energy_channels": mps_vars["diff_energy"],
            "alpha_local": local_pa_var,
            "PSD": psd_var,
            "alpha_eq_model": magnetic_field_variables["PA_eq_T89"],
            "MLT": magnetic_field_variables["MLT_T89"],
            "Lstar": magnetic_field_variables["Lstar_T89"],
            "R0": magnetic_field_variables["R_eq_T89"],
            "B_eq": magnetic_field_variables["B_eq_T89"],
            "B_local": magnetic_field_variables["B_local_T89"],
            "InvMu": magnetic_field_variables["invMu_T89"],
            "InvK": magnetic_field_variables["invK_T89"],
        }

        saving_strategy = ep.saving_strategies.GFZStrategy(
            processed_data_path,
            mission="GOES",
            satellite=sat_str,
            instrument="MAGED",
            kext="T89",
            file_format=".pickle",
        )
        append = True

    elif save_strategy == "netcdf":
        variables_to_save = {
            "time": binned_time_var,
            "flux/FEDU": mps_vars["diff_flux"],
            "flux/energy": mps_vars["diff_energy"],
            "flux/alpha_local": local_pa_var,
            "flux/alpha_eq": magnetic_field_variables["PA_eq_T89"],
            "position/T89/R0": magnetic_field_variables["R_eq_T89"],
            "position/T89/MLT": magnetic_field_variables["MLT_T89"],
            "position/T89/Lm": magnetic_field_variables["Lm_T89"],
            "position/T89/Lstar": magnetic_field_variables["Lstar_T89"],
            "mag_field/T89/B_local": magnetic_field_variables["B_local_T89"],
            "mag_field/T89/B_eq": magnetic_field_variables["B_eq_T89"],
            "psd/PSD": psd_var,
            "psd/T89/inv_mu": magnetic_field_variables["invMu_T89"],
            "psd/T89/inv_K": magnetic_field_variables["invK_T89"],
            "position/xGEO": xgeo_var,
        }

        saving_strategy = ep.saving_strategies.MonthlyNetCDFStrategy(
            base_data_path=Path(processed_data_path) / "GOES" / sat_str,
            file_name_stem=f"{sat_str}_mps_high",
            mag_field="T89",
        )
        append = False

    ep.save(variables_to_save, saving_strategy, start_time, end_time, time_var=binned_time_var, append=append)


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
    )

    extraction_infos = [
        ep.ExtractionInfo(name_or_column="Epoch", unit=ep.units.cdf_epoch, result_key="time"),
        ep.ExtractionInfo(
            name_or_column="ELE_DIFF_ENERGY_FLUX", unit=u.eV*(u.cm**2 * u.s * u.keV * u.sr) ** (-1), result_key="diff_energy_flux"
        ),
        ep.ExtractionInfo(name_or_column="ELE_AVG_ENERGY", unit=u.eV, result_key="diff_energy"),
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
    start_time = datetime(2013, 3, 17, tzinfo=timezone.utc)
    end_time = datetime(2013, 3, 17, 11, 59, tzinfo=timezone.utc)

    satellites = ["f16"]

    for sat in satellites:
        process_dmsp_ssj_electrons(
            sat_str=sat,
            raw_data_path="dmsp/raw/",
            processed_data_path="dmsp/processed/",
            start_time=start_time,
            end_time=end_time,
            num_cores=64,
        )
