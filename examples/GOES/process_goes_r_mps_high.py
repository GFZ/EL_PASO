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

TELE_ALPHA_ANGLES = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
TELE_BETA_ANGLES = np.array([-35.0, 35.0, -70.0, 0, 70.0])


def process_goes_r_mps_high(
    sat_str: Literal["goes18", "goes19"],
    processed_data_path: str | Path,
    raw_data_path: str | Path,
    irbem_lib_path: str | Path,
    start_time: datetime,
    end_time: datetime,
    save_strategy: Literal["dataorg", "netcdf"] = "netcdf",
    num_cores: int = 32,
) -> None:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    data_path_stem = f"{raw_data_path}/YYYY/MM/{sat_str}/"

    magn_vars = _get_magn_variables(sat_str, data_path_stem, start_time, end_time)
    mps_vars = _get_mps_high_variables(sat_str, data_path_stem, start_time, end_time)
    ephe_vars = _get_ephe_variables(sat_str, data_path_stem, start_time, end_time)

    time_bin_methods_magn = {
        "b_brf": ep.TimeBinMethod.NanMean,
    }

    binned_time_var = ep.processing.bin_by_time(
        time_variable=magn_vars["time"],
        variables=magn_vars,
        time_bin_method_dict=time_bin_methods_magn,
        time_binning_cadence=timedelta(minutes=5),
        start_time=start_time,
        end_time=end_time,
    )

    time_bin_methods_mps = {
        "diff_flux": ep.TimeBinMethod.NanMedian,
        "diff_flux_uncert": ep.TimeBinMethod.NanMedian,
        "diff_energy": ep.TimeBinMethod.Repeat,
        "int_flux": ep.TimeBinMethod.NanMedian,
        "int_flux_uncert": ep.TimeBinMethod.NanMedian,
        "int_energy": ep.TimeBinMethod.Repeat,
    }

    _ = ep.processing.bin_by_time(
        time_variable=mps_vars["time"],
        variables=mps_vars,
        time_bin_method_dict=time_bin_methods_mps,
        time_binning_cadence=timedelta(minutes=5),
        start_time=start_time,
        end_time=end_time,
    )

    time_bin_methods_ephe = {
        "xgse": ep.TimeBinMethod.NanMean,
    }

    _ = ep.processing.bin_by_time(
        time_variable=ephe_vars["time"],
        variables=ephe_vars,
        time_bin_method_dict=time_bin_methods_ephe,
        time_binning_cadence=timedelta(minutes=5),
        start_time=start_time,
        end_time=end_time,
    )

    mps_vars["diff_flux"].transpose_data((0, 2, 1))
    mps_vars["diff_flux"].apply_thresholds_on_data(lower_threshold=0)

    # calculate xGEO
    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in binned_time_var.get_data(ep.units.posixtime)]
    xgeo_data = ep.processing.magnetic_field_utils.Coords(lib_path=irbem_lib_path).transform(
        datetimes,
        ephe_vars["xgse"].get_data(ep.units.RE).astype(np.float64),
        ep.IRBEM_SYSAXIS_GSE,
        ep.IRBEM_SYSAXIS_GEO,
    )
    xgeo_var = ep.Variable(data=xgeo_data, original_unit=ep.units.RE)

    # calculate pitch angles
    tele_alpha_angles_var = ep.Variable(data=TELE_ALPHA_ANGLES, original_unit=u.deg)
    tele_beta_angles_var = ep.Variable(data=TELE_BETA_ANGLES, original_unit=u.deg)
    local_pa_var = ep.processing.compute_pitch_angles_for_telescopes(
        magn_vars["b_brf"],
        tele_alpha_angles_var,
        tele_beta_angles_var,
    )

    # fold pitch angles around 90 degree
    local_pa = local_pa_var.get_data(u.degree)
    local_pa_folded = np.where(local_pa > 90, local_pa - 90, local_pa)
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

        saving_strategy = ep.saving_strategies.DataOrgStrategy(
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


def _get_magn_variables(
    sat_str: Literal["goes18", "goes19"],
    data_path_stem: str | Path,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, ep.Variable]:
    url = f"https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/{sat_str}/l2/data/magn-l2-avg1m/YYYY/MM/"

    sat_stem = "g18" if sat_str == "goes18" else "g19"
    file_name_stem = f"dn_magn-l2-avg1m_{sat_stem}_dYYYYMMDD_.{r'{6}'}.nc"

    ep.download(
        start_time,
        end_time,
        save_path=data_path_stem,
        file_cadence="daily",
        download_url=url,
        file_name_stem=file_name_stem,
    )

    extraction_infos = [
        ep.ExtractionInfo(name_or_column="time", unit=ep.units.j2k, result_key="time"),
        ep.ExtractionInfo(name_or_column="DQF", unit=u.dimensionless_unscaled, result_key="dqf"),
        ep.ExtractionInfo(name_or_column="b_brf", unit=u.nT, result_key="b_brf"),
    ]

    return ep.processing.extract_variables_from_files(
        start_time,
        end_time,
        file_cadence="daily",
        data_path=data_path_stem,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )


def _get_ephe_variables(
    sat_str: Literal["goes18", "goes19"],
    data_path_stem: str | Path,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, ep.Variable]:
    url = f"https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/{sat_str}/l2/data/ephe-l2-orb1m/YYYY/MM/"

    sat_stem = "g18" if sat_str == "goes18" else "g19"
    file_name_stem = f"dn_ephe-l2-orb1m_{sat_stem}_dYYYYMMDD_.{r'{6}'}.nc"

    ep.download(
        start_time,
        end_time,
        save_path=data_path_stem,
        file_cadence="daily",
        download_url=url,
        file_name_stem=file_name_stem,
    )

    extraction_infos = [
        ep.ExtractionInfo(name_or_column="time", unit=ep.units.j2k, result_key="time"),
        ep.ExtractionInfo(name_or_column="gse_xyz", unit=u.km, result_key="xgse"),
    ]

    return ep.processing.extract_variables_from_files(
        start_time,
        end_time,
        file_cadence="daily",
        data_path=data_path_stem,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )


def _get_mps_high_variables(
    sat_str: Literal["goes18", "goes19"],
    data_path_stem: str | Path,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, ep.Variable]:
    url = f"https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/{sat_str}/l2/data/mpsh-l2-avg5m/YYYY/MM/"

    sat_stem = "g18" if sat_str == "goes18" else "g19"
    file_name_stem = f"sci_mpsh-l2-avg5m_{sat_stem}_dYYYYMMDD_.{r'{6}'}.nc"

    ep.download(
        start_time,
        end_time,
        save_path=data_path_stem,
        file_cadence="daily",
        download_url=url,
        file_name_stem=file_name_stem,
    )

    extraction_infos = [
        ep.ExtractionInfo(name_or_column="time", unit=ep.units.j2k, result_key="time"),
        ep.ExtractionInfo(
            name_or_column="AvgDiffElectronFlux", unit=(u.cm**2 * u.s * u.keV * u.sr) ** (-1), result_key="diff_flux"
        ),
        ep.ExtractionInfo(
            name_or_column="AvgDiffElectronFluxUncert",
            unit=(u.cm**2 * u.s * u.keV * u.sr) ** (-1),
            result_key="diff_flux_uncert",
        ),
        ep.ExtractionInfo(name_or_column="DiffElectronEffectiveEnergy", unit=u.keV, result_key="diff_energy"),
        ep.ExtractionInfo(
            name_or_column="AvgIntElectronFlux", unit=(u.cm**2 * u.s * u.sr) ** (-1), result_key="int_flux"
        ),
        ep.ExtractionInfo(
            name_or_column="AvgIntElectronFluxUncert", unit=(u.cm**2 * u.s * u.sr) ** (-1), result_key="int_flux_uncert"
        ),
        ep.ExtractionInfo(name_or_column="IntElectronEffectiveEnergy", unit=u.keV, result_key="int_energy"),
    ]

    return ep.processing.extract_variables_from_files(
        start_time,
        end_time,
        file_cadence="daily",
        data_path=data_path_stem,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )


if __name__ == "__main__":
    start_time = datetime(2025, 9, 25, tzinfo=timezone.utc)
    end_time = datetime(2025, 10, 13, 23, 59, tzinfo=timezone.utc)

    satellites = ["goes18", "goes19"]

    for sat in satellites:
        process_goes_r_mps_high(
            sat_str=sat,
            raw_data_path="goes/raw/",
            processed_data_path="goes/processed/",
            irbem_lib_path="../../libirbem.so",
            start_time=start_time,
            end_time=end_time,
            num_cores=64,
        )
