# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from astropy import units as u
from numpy.typing import NDArray

import el_paso as ep

if TYPE_CHECKING:
    from el_paso.typing import InternalName

logging.captureWarnings(capture=True)
logger = logging.getLogger(__name__)

LONG_FROM_SAT = {
    "goes13": 74.5,
    "goes15": 135.1,
}

# thresholds of the EPEAD integral electron channels according to the CDF metadata
EPEAD_E1_THRESHOLD_MEV = 0.8
EPEAD_E2_THRESHOLD_MEV = 2.0


def process_goes_eps_maged_and_epead(
    sat_str: Literal["goes13", "goes15"],
    processed_data_path: str | Path,
    raw_data_path: str | Path,
    start_time: datetime,
    end_time: datetime,
    num_cores: int = 32,
) -> None:

    data_path_stem = f"{raw_data_path}/YYYY/MM/{sat_str}/"

    maged_vars = _get_and_prepare_eps_maged_variables(sat_str, data_path_stem, start_time, end_time)
    epead_vars = _get_and_prepare_epead_variables(sat_str, data_path_stem, start_time, end_time)
    pa_vars = _get_and_prepare_eps_pitch_angles_variables(sat_str, data_path_stem, start_time, end_time)

    # sort pitch angles in ascending order and apply to fluxes
    local_pa = pa_vars["AlphaLocal"].get_data(u.degree)
    local_pa[local_pa > 90] = 180 - local_pa[local_pa > 90]

    idx_sorted = np.argsort(local_pa, axis=1)
    sorted_local_pa = np.take_along_axis(local_pa, idx_sorted, axis=1)
    n_energy = maged_vars["Energy"].get_data().shape[1]
    sorted_diff_flux = np.take_along_axis(
        maged_vars["FEDU"].get_data(), np.tile(idx_sorted[:, np.newaxis, :], [1, n_energy, 1]), axis=2
    )

    pa_vars["AlphaLocal"].set_data(sorted_local_pa, unit="same")
    pa_vars["AlphaLocal"].metadata.add_processing_note("Folded around 90 deg and sorted in ascending order")

    maged_vars["FEDU"].set_data(sorted_diff_flux, unit="same")

    # apply PAD from maged to epead
    maged_flux = maged_vars["FEDU"].get_data((u.cm**2 * u.s * u.sr * u.keV) ** (-1)).astype(np.float64)
    pad = maged_flux[:, -1, :] / np.nanmean(maged_flux[:, -1, :], axis=-1)[:, np.newaxis]

    epead_flux = epead_vars["FEDU"].get_data((u.cm**2 * u.s * u.sr * u.keV) ** (-1)).astype(np.float64)
    epead_flux = epead_flux[:, :, np.newaxis] * pad[:, np.newaxis, :]

    combined_flux = np.concatenate((maged_flux, epead_flux), axis=1)
    combined_energies = np.concatenate(
        (maged_vars["Energy"].get_data(u.keV), epead_vars["Energy_FEDU"].get_data(u.keV)), axis=1,
    )

    maged_vars["FEDU"].set_data(combined_flux, unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1))
    maged_vars["Energy"].set_data(combined_energies, unit=u.keV)

    # Calculate magnetic field variables
    variables_to_compute: ep.processing.VariableRequest = [
        ("B_Calc", "T89"),
        ("B_Eq", "T89"),
        ("MLT", "T89"),
        ("R_Eq", "T89"),
        ("Alpha_Eq", "T89"),
        ("L_star", "T89"),
        ("L_m", "T89"),
        ("InvMu", "T89"),
        ("InvK", "T89"),
    ]

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=maged_vars["Epoch"],
        xgeo_var=maged_vars["xGEO"],
        energy_var=maged_vars["Energy"],
        pa_local_var=pa_vars["AlphaLocal"],
        particle_species="electron",
        variables_to_compute=variables_to_compute,
        irbem_options=ep.processing.magnetic_field_utils.IrbemOptions(),
        num_cores=num_cores,
    )

    psd_var = ep.processing.compute_phase_space_density(
        maged_vars["FEDU"], maged_vars["Energy"], particle_species="electron"
    )

    variables_to_save: dict[InternalName, ep.Variable] = {
        "Epoch": maged_vars["Epoch"],
        "FEDU": maged_vars["FEDU"],
        "Position": maged_vars["xGEO"],
        "Energy_FEDU": maged_vars["Energy"],
        "Alpha": pa_vars["AlphaLocal"],
        "PSD": psd_var,
        "Alpha_Eq": magnetic_field_variables["Alpha_Eq_T89"],
        "MLT": magnetic_field_variables["MLT_T89"],
        "L_star": magnetic_field_variables["L_star_T89"],
        "R_Eq": magnetic_field_variables["R_Eq_T89"],
        "B_Eq": magnetic_field_variables["B_Eq_T89"],
        "B_Calc": magnetic_field_variables["B_Calc_T89"],
        "InvMu": magnetic_field_variables["InvMu_T89"],
        "InvK": magnetic_field_variables["InvK_T89"],
    }

    saving_strategy = ep.saving_strategies.MonthlyRBStrategy(
        Path(processed_data_path),
        mission="GOES",
        satellite=sat_str,
        instrument="MAGED_EPEAD",
        mag_field="T89",
        file_format="nc",
        data_standard=ep.data_standards.GFZStandard(),
    )

    ep.save(variables_to_save, saving_strategy, start_time, end_time, time_var=maged_vars["Epoch"], append=True)


def _get_and_prepare_eps_maged_variables(
    sat_str: Literal["goes13", "goes15"],
    data_path_stem: str | Path,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, ep.Variable]:
    url = f"https://spdf.gsfc.nasa.gov/pub/data/goes/{sat_str}/eps-maged/19me15_5min/YYYY/"

    file_name_stem = f"{sat_str}_eps-maged_5min_YYYYMMDD_.{r'{3}'}.cdf"

    ep.download(
        start_time,
        end_time,
        save_path=data_path_stem,
        file_cadence="monthly",
        download_url=url,
        file_name_stem=file_name_stem,
    )

    extraction_infos_maged = [
        ep.ExtractionInfo(name_or_column="Epoch", unit=ep.units.tt2000, result_key="Epoch"),
        ep.ExtractionInfo(
            name_or_column="dtc_cor_eflux", unit=(u.keV * u.cm**2 * u.s * u.sr) ** (-1), result_key="FEDU"
        ),
        ep.ExtractionInfo(name_or_column="energy", unit=u.keV, result_key="Energy", is_time_dependent=False),
        ep.ExtractionInfo(name_or_column="qual_flag", unit=u.dimensionless_unscaled, result_key="qual_flag"),
    ]

    maged_vars = ep.extract_variables_from_files(
        start_time,
        end_time,
        file_cadence="monthly",
        data_path=data_path_stem,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos_maged,
    )

    maged_vars["FEDU"] = _remove_bad_points(maged_vars["FEDU"], maged_vars["qual_flag"], "MAGED FEDU")

    # calculate xGEO
    xgeo_data = _geo_cartesian_from_west_lon_incl(LONG_FROM_SAT[sat_str], 0)
    maged_vars["xGEO"] = ep.Variable(data=xgeo_data, original_unit=u.km)

    time_bin_methods_maged = {
        "FEDU": ep.TimeBinMethod.NanMedian,
        "Energy": ep.TimeBinMethod.Repeat,
        "xGEO": ep.TimeBinMethod.Repeat,
    }

    maged_vars["FEDU"].transpose_data((0, 2, 1))
    maged_vars["FEDU"].apply_thresholds_on_data(lower_threshold=0)

    maged_vars["Epoch"] = ep.processing.bin_by_time(
        time_variable=maged_vars["Epoch"],
        variables=maged_vars,
        time_bin_method_dict=time_bin_methods_maged,
        time_binning_cadence=timedelta(minutes=5),
        start_time=start_time,
        end_time=end_time,
    )

    return maged_vars


def _get_and_prepare_epead_variables(
    sat_str: Literal["goes13", "goes15"],
    data_path_stem: str | Path,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, ep.Variable]:
    url = f"https://spdf.gsfc.nasa.gov/pub/data/goes/{sat_str}/epead-electrons/e13ew_1min/YYYY/"

    ep.setup_logging()

    file_name_stem = f"{sat_str}_epead-science-electrons-e13ew_1min_YYYYMMDD_.{r'{3}'}.cdf"

    ep.download(
        start_time,
        end_time,
        save_path=data_path_stem,
        file_cadence="monthly",
        download_url=url,
        file_name_stem=file_name_stem,
    )

    extraction_infos_maged = [
        ep.ExtractionInfo(name_or_column="Epoch", unit=ep.units.tt2000, result_key="Epoch"),
        ep.ExtractionInfo(
            name_or_column="E1W_COR_FLUX", unit=(u.cm**2 * u.s * u.sr) ** (-1), result_key="FEIU_0_8_west"
        ),
        ep.ExtractionInfo(
            name_or_column="E1E_COR_FLUX", unit=(u.cm**2 * u.s * u.sr) ** (-1), result_key="FEIU_0_8_east"
        ),
        ep.ExtractionInfo(
            name_or_column="E2W_COR_FLUX", unit=(u.cm**2 * u.s * u.sr) ** (-1), result_key="FEIU_2_0_west"
        ),
        ep.ExtractionInfo(
            name_or_column="E2E_COR_FLUX", unit=(u.cm**2 * u.s * u.sr) ** (-1), result_key="FEIU_2_0_east"
        ),
        ep.ExtractionInfo(name_or_column="E1W_DQF", unit=u.dimensionless_unscaled, result_key="qual_flag_0_8_west"),
        ep.ExtractionInfo(name_or_column="E1E_DQF", unit=u.dimensionless_unscaled, result_key="qual_flag_0_8_east"),
        ep.ExtractionInfo(name_or_column="E2W_DQF", unit=u.dimensionless_unscaled, result_key="qual_flag_2_0_west"),
        ep.ExtractionInfo(name_or_column="E2E_DQF", unit=u.dimensionless_unscaled, result_key="qual_flag_2_0_east"),
    ]

    epead_vars = ep.extract_variables_from_files(
        start_time,
        end_time,
        file_cadence="monthly",
        data_path=data_path_stem,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos_maged,
    )

    # apply maged quality flags
    epead_vars["FEIU_0_8_west"] = _remove_bad_points(
        epead_vars["FEIU_0_8_west"], epead_vars["qual_flag_0_8_west"], "EPEAD E1W"
    )
    epead_vars["FEIU_0_8_east"] = _remove_bad_points(
        epead_vars["FEIU_0_8_east"], epead_vars["qual_flag_0_8_east"], "EPEAD E1E"
    )
    epead_vars["FEIU_2_0_west"] = _remove_bad_points(
        epead_vars["FEIU_2_0_west"], epead_vars["qual_flag_2_0_west"], "EPEAD E2W"
    )
    epead_vars["FEIU_2_0_east"] = _remove_bad_points(
        epead_vars["FEIU_2_0_east"], epead_vars["qual_flag_2_0_east"], "EPEAD E2E"
    )

    # merge epead data
    feiu_0_8 = np.nanmean(
        [epead_vars["FEIU_0_8_west"].get_data(), epead_vars["FEIU_0_8_east"].get_data()],
        axis=0,
    )
    feiu_2_0 = np.nanmean(
        [epead_vars["FEIU_2_0_west"].get_data(), epead_vars["FEIU_2_0_east"].get_data()],
        axis=0,
    )
    epead_vars["FEIU"] = ep.Variable(
        data=np.vstack([feiu_0_8, feiu_2_0]).T,
        original_unit=(u.cm**2 * u.s * u.sr) ** (-1),
    )

    # same energy grid as in the original MATLAB processing (A. Kellerman)
    epead_vars["Energy_FEDU"] = ep.Variable(
        data=np.concatenate((np.arange(0.8, 2.01, 0.4), np.arange(3.0, 10.1, 1.0))),
        original_unit=u.MeV,
    )

    del (
        epead_vars["FEIU_0_8_west"],
        epead_vars["FEIU_0_8_east"],
        epead_vars["FEIU_2_0_west"],
        epead_vars["FEIU_2_0_east"],
    )

    time_bin_methods_epead = {"FEIU": ep.TimeBinMethod.NanMedian, "Energy_FEDU": ep.TimeBinMethod.Repeat}

    epead_vars["Epoch"] = ep.processing.bin_by_time(
        time_variable=epead_vars["Epoch"],
        variables=epead_vars,
        time_bin_method_dict=time_bin_methods_epead,
        time_binning_cadence=timedelta(minutes=5),
        start_time=start_time,
        end_time=end_time,
    )

    # fit exponential F(E) = A exp(-B E) to the integral channels as in the original MATLAB processing:
    # J(>E) = A / B exp(-B E), which gives B = (log(J1) - log(J2)) / (E2 - E1) and A = J B / exp(-B E)
    feiu = epead_vars["FEIU"].get_data()
    energy_fedu = epead_vars["Energy_FEDU"].get_data(u.MeV)
    B = (np.log(feiu[:, 0]) - np.log(feiu[:, 1])) / (EPEAD_E2_THRESHOLD_MEV - EPEAD_E1_THRESHOLD_MEV)
    B[B < 0] = np.nan

    A1 = feiu[:, 0] * B / np.exp(-B * EPEAD_E1_THRESHOLD_MEV)
    A2 = feiu[:, 1] * B / np.exp(-B * EPEAD_E2_THRESHOLD_MEV)
    A = np.mean([A1, A2], axis=0)

    fedu = A[:, np.newaxis] * np.exp(-B[:, np.newaxis] * energy_fedu)

    epead_vars["FEDU"] = ep.Variable(data=fedu, original_unit=(u.cm**2 * u.s * u.sr * u.MeV) ** (-1))

    return epead_vars


def _get_and_prepare_eps_pitch_angles_variables(
    sat_str: Literal["goes13", "goes15"],
    data_path_stem: str | Path,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, ep.Variable]:
    url = f"https://spdf.gsfc.nasa.gov/pub/data/goes/{sat_str}/eps-pitch-angles_1min/YYYY/"

    file_name_stem = f"{sat_str}_eps-pitch-angles_1min_YYYYMMDD_.{r'{3}'}.cdf"

    ep.download(
        start_time,
        end_time,
        save_path=data_path_stem,
        file_cadence="monthly",
        download_url=url,
        file_name_stem=file_name_stem,
    )

    extraction_infos = [
        ep.ExtractionInfo(name_or_column="Epoch", unit=ep.units.tt2000, result_key="Epoch"),
        ep.ExtractionInfo(name_or_column="pitch_angles", unit=u.degree, result_key="AlphaLocal"),
    ]

    pa_vars = ep.extract_variables_from_files(
        start_time,
        end_time,
        file_cadence="monthly",
        data_path=data_path_stem,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )

    # the CDF files use -99999 as fill value, which is not the declared FILLVAL (-1e31)
    local_pa = pa_vars["AlphaLocal"].get_data(u.degree)
    pa_vars["AlphaLocal"].apply_mask((local_pa >= 0) & (local_pa <= 180))
    pa_vars["AlphaLocal"].metadata.add_processing_note("Removed pitch angles outside of the valid range [0, 180] deg")

    time_bin_methods_pa = {
        "AlphaLocal": ep.TimeBinMethod.NanMean,
    }

    pa_vars["Epoch"] = ep.processing.bin_by_time(
        time_variable=pa_vars["Epoch"],
        variables=pa_vars,
        time_bin_method_dict=time_bin_methods_pa,
        time_binning_cadence=timedelta(minutes=5),
        start_time=start_time,
        end_time=end_time,
    )

    return pa_vars


def _geo_cartesian_from_west_lon_incl(
    west_longitude_deg: float, inclination_deg: float, radius_km: float = 42164.0
) -> NDArray[np.float64]:
    lam_w = np.deg2rad(west_longitude_deg)
    i = np.deg2rad(inclination_deg)

    x = radius_km * np.cos(i) * np.cos(lam_w)
    y = -radius_km * np.cos(i) * np.sin(lam_w)
    z = -radius_km * np.sin(i)

    return np.stack([x, y, z], axis=-1)


def _remove_bad_points(flux_var: ep.Variable, qf_var: ep.Variable, channel_name: str) -> ep.Variable:

    qual_flag = qf_var.get_data()
    n_bad_points = np.sum(qual_flag != 0)

    logger.info(f"Removed {n_bad_points} bad points ({n_bad_points / qual_flag.size * 100:0.1f}%) from {channel_name}!")

    flux_var.apply_mask(qual_flag == 0)
    flux_var.metadata.add_processing_note("Removed bad points according to whether qual_flag is non-zero")

    return flux_var


if __name__ == "__main__":

    ep.setup_logging()

    start_time = datetime(2017, 4, 20, tzinfo=timezone.utc)
    end_time = datetime(2017, 4, 25, 23, 59, tzinfo=timezone.utc)

    Satellite = Literal["goes13", "goes15"]
    sats: list[Satellite] = ["goes13"]

    for sat in sats:
        process_goes_eps_maged_and_epead(
            sat_str=sat,
            raw_data_path="goes/raw/",
            processed_data_path="goes/processed/",
            start_time=start_time,
            end_time=end_time,
            num_cores=64,
        )
