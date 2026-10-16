# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from astropy import units as u
from numpy.typing import NDArray

import el_paso as ep
from el_paso.recipes.goes import GOESNOPSatellite
from el_paso.recipes.solarwind.process_solar_wind_indices import process_solar_wind_indices

if TYPE_CHECKING:
    from el_paso.typing import InternalName

logging.captureWarnings(capture=True)
logger = logging.getLogger(__name__)


def goes_eps_maged_and_epead_gfz_strategy(
    base_data_path: str | Path, mag_field: ep.typing.MagneticFieldLiteral, satellite: GOESNOPSatellite
) -> ep.SavingStrategy:
    """Legacy GFZ .mat saving strategy for combined GOES-13/15 EPS MAGED and EPEAD."""
    return ep.saving_strategies.GFZStrategy(Path(base_data_path), "GOES", satellite, "MAGEDandEPEAD", mag_field)


def goes_eps_maged_and_epead_netcdf_strategy(
    base_data_path: str | Path,
    mag_field: ep.typing.MagneticFieldLiteral,
    satellite: GOESNOPSatellite,
    *,
    file_format: ep.typing.MFSFormats = "nc",
) -> ep.SavingStrategy:
    """Monthly NetCDF saving strategy for combined GOES-13/15 EPS MAGED and EPEAD."""
    return ep.saving_strategies.MonthlyRBStrategy(
        Path(base_data_path),
        "GOES",
        satellite,
        "MAGED_EPEAD",
        mag_field,
        data_standard=ep.data_standards.GFZStandard(),
        file_format=file_format,
    )


LONG_FROM_SAT: dict[GOESNOPSatellite, float] = {
    "goes13": 74.5,
    "goes15": 135.1,
}

# thresholds of the EPEAD integral electron channels according to the CDF metadata
EPEAD_E1_THRESHOLD_MEV = 0.8
EPEAD_E2_THRESHOLD_MEV = 2.0


def process_goes_eps_maged_and_epead(
    start_time: datetime,
    end_time: datetime,
    satellite: GOESNOPSatellite = "goes13",
    mag_field: ep.typing.MagneticFieldLiteral = "T89",
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    bin_cadence: timedelta = timedelta(minutes=5),
    num_cores: int = 16,
    save_strategy: Literal["gfz", "netcdf", "both"] = "netcdf",
    skip_existing: bool = True,  # noqa: FBT001, FBT002,
    *,
    save_sw: bool = False,
) -> None:
    """Process combined GOES-13/15 EPS MAGED and EPEAD electron data into phase space densities.

    Downloads and extracts the 5-minute EPS MAGED differential electron fluxes, the 1-minute
    EPEAD integral electron fluxes (>0.8 MeV and >2 MeV, east and west telescopes) and the
    1-minute EPS telescope pitch angles from CDAWeb for the given GOES-13/15 satellite. Points
    with a non-zero quality flag are removed and all data are binned onto a common time cadence.
    The local pitch angles are folded around 90 degrees and the MAGED fluxes sorted by ascending
    pitch angle. The EPEAD integral channels are converted to differential fluxes on a fixed
    energy grid (0.8 to 10 MeV) by fitting an exponential spectrum, and the pitch angle
    distribution of the highest MAGED channel is applied to them. A fixed geostationary
    spacecraft position is assigned, magnetic field quantities (B_Calc,
    B_Eq, MLT, R_Eq, Alpha_Eq, L_star, L_m, InvMu, InvK) are computed, and the electron phase
    space density is derived. The resulting variables are saved using the requested saving
    strategy.

    Args:
        start_time (datetime): Start of the time interval to process.
        end_time (datetime): End of the time interval to process.
        satellite (Literal["goes13", "goes15"]): The GOES satellite to process.
        mag_field (MagneticFieldLiteral): Magnetic field model used for the derived quantities.
        raw_data_path (str | Path): Directory where the raw downloaded data files are stored.
        processed_data_path (str | Path): Directory where the processed output files are saved.
        bin_cadence (timedelta): Time cadence used to bin the extracted variables.
        num_cores (int): Number of CPU cores used for the magnetic field computations.
        save_strategy (Literal["gfz", "netcdf", "both"]): Strategy used to save the processed
            data. "gfz" saves using the GFZ format, "netcdf" saves monthly NetCDF files, and
            "both" saves using both strategies.
        skip_existing (bool): If True, skip downloading files that already exist on disk.
        save_sw (bool): If True, also save all solar wind indices and parameters,
            independent of this recipe's own output. Defaults to False.
    """
    data_path_stem = f"{raw_data_path}/YYYY/MM/{satellite}/"

    maged_url = f"https://spdf.gsfc.nasa.gov/pub/data/goes/{satellite}/eps-maged/19me15_5min/YYYY/"
    maged_file_name_stem = f"{satellite}_eps-maged_5min_YYYYMMDD_.{r'{3}'}.cdf"

    ep.download(
        start_time,
        end_time,
        save_path=data_path_stem,
        file_cadence="monthly",
        download_url=maged_url,
        file_name_stem=maged_file_name_stem,
        skip_existing=skip_existing,
    )

    epead_url = f"https://spdf.gsfc.nasa.gov/pub/data/goes/{satellite}/epead-electrons/e13ew_1min/YYYY/"
    epead_file_name_stem = f"{satellite}_epead-science-electrons-e13ew_1min_YYYYMMDD_.{r'{3}'}.cdf"

    ep.download(
        start_time,
        end_time,
        save_path=data_path_stem,
        file_cadence="monthly",
        download_url=epead_url,
        file_name_stem=epead_file_name_stem,
        skip_existing=skip_existing,
    )

    pa_url = f"https://spdf.gsfc.nasa.gov/pub/data/goes/{satellite}/eps-pitch-angles_1min/YYYY/"
    pa_file_name_stem = f"{satellite}_eps-pitch-angles_1min_YYYYMMDD_.{r'{3}'}.cdf"

    ep.download(
        start_time,
        end_time,
        save_path=data_path_stem,
        file_cadence="monthly",
        download_url=pa_url,
        file_name_stem=pa_file_name_stem,
        skip_existing=skip_existing,
    )

    ep.exit_if_download_only()

    maged_vars = _get_and_prepare_eps_maged_variables(
        maged_file_name_stem, data_path_stem, start_time, end_time, satellite, bin_cadence
    )
    epead_vars = _get_and_prepare_epead_variables(
        epead_file_name_stem, data_path_stem, start_time, end_time, bin_cadence
    )
    pa_vars = _get_and_prepare_eps_pitch_angles_variables(
        pa_file_name_stem, data_path_stem, start_time, end_time, bin_cadence
    )

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
        (maged_vars["Energy"].get_data(u.keV), epead_vars["Energy_FEDU"].get_data(u.keV)),
        axis=1,
    )

    maged_vars["FEDU"].set_data(combined_flux, unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1))
    maged_vars["Energy"].set_data(combined_energies, unit=u.keV)

    # Calculate magnetic field variables
    variables_to_compute: ep.processing.VariableRequest = [
        ("B_Calc", mag_field),
        ("B_Eq", mag_field),
        ("MLT", mag_field),
        ("R_Eq", mag_field),
        ("Alpha_Eq", mag_field),
        ("L_star", mag_field),
        ("L_m", mag_field),
        ("InvMu", mag_field),
        ("InvK", mag_field),
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
        "Alpha_Eq": magnetic_field_variables[f"Alpha_Eq_{mag_field}"],
        "MLT": magnetic_field_variables[f"MLT_{mag_field}"],
        "L_star": magnetic_field_variables[f"L_star_{mag_field}"],
        "R_Eq": magnetic_field_variables[f"R_Eq_{mag_field}"],
        "B_Eq": magnetic_field_variables[f"B_Eq_{mag_field}"],
        "B_Calc": magnetic_field_variables[f"B_Calc_{mag_field}"],
        "InvMu": magnetic_field_variables[f"InvMu_{mag_field}"],
        "InvK": magnetic_field_variables[f"InvK_{mag_field}"],
    }

    if save_strategy in ("gfz", "both"):
        saving_strategy = goes_eps_maged_and_epead_gfz_strategy(processed_data_path, mag_field, satellite)
        ep.save(
            variables_to_save,
            saving_strategy,
            start_time,
            end_time,
            time_var=maged_vars["Epoch"],
            append=True,
        )

    if save_strategy in ("netcdf", "both"):
        saving_strategy = goes_eps_maged_and_epead_netcdf_strategy(processed_data_path, mag_field, satellite)
        ep.save(
            variables_to_save,
            saving_strategy,
            start_time,
            end_time,
            time_var=maged_vars["Epoch"],
            append=True,
        )

    if save_sw:
        process_solar_wind_indices(
            start_time, end_time, processed_data_path=processed_data_path, bin_cadence=bin_cadence
        )


def _get_and_prepare_eps_maged_variables(
    file_name_stem: str,
    data_path_stem: str | Path,
    start_time: datetime,
    end_time: datetime,
    satellite: GOESNOPSatellite,
    bin_cadence: timedelta,
) -> dict[str, ep.Variable]:
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
    xgeo_data = _geo_cartesian_from_west_lon_incl(LONG_FROM_SAT[satellite], 0)
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
        time_binning_cadence=bin_cadence,
        start_time=start_time,
        end_time=end_time,
    )

    return maged_vars


def _get_and_prepare_epead_variables(
    file_name_stem: str,
    data_path_stem: str | Path,
    start_time: datetime,
    end_time: datetime,
    bin_cadence: timedelta,
) -> dict[str, ep.Variable]:
    extraction_infos_epead = [
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
        extraction_infos=extraction_infos_epead,
    )

    # apply epead quality flags
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
        time_binning_cadence=bin_cadence,
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
    file_name_stem: str,
    data_path_stem: str | Path,
    start_time: datetime,
    end_time: datetime,
    bin_cadence: timedelta,
) -> dict[str, ep.Variable]:
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
        time_binning_cadence=bin_cadence,
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


CLI_DEFAULTS = {
    "raw_data_path": "goes/raw/",
    "processed_data_path": "goes/processed/",
}

if __name__ == "__main__":
    ep.run_recipe_cli(process_goes_eps_maged_and_epead, defaults=CLI_DEFAULTS)
