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
from el_paso.recipes.solar_wind_indices.process_solar_wind_indices import process_solar_wind_indices

if TYPE_CHECKING:
    from el_paso.typing import InternalName

logging.captureWarnings(capture=True)
logger = logging.getLogger(__name__)


def goes_eps_maged_and_epead_gfz_strategy(
    base_data_path: str | Path, mag_field: ep.typing.MagneticFieldLiteral, satellite: GOESNOPSatellite
) -> ep.SavingStrategy:
    """Legacy GFZ .mat saving strategy for combined GOES-13/15 EPS MAGED and EPEAD."""
    return ep.saving_strategies.GFZStrategy(Path(base_data_path), "GOES", satellite, "MAGEDandEPEAD", mag_field)


class _MonthlyMagedEpeadStrategy(ep.saving_strategies.MonthlyRBStrategy):
    """Monthly strategy that also saves the EPEAD integral channels.

    These are not part of the shared monthly variable list, since missing variables are filled with empty
    placeholders there, which would end up in the output of every other recipe.
    """

    def _get_output_file_entries(self) -> list[ep.typing.InternalName | ep.typing.OneOf]:
        return [*super()._get_output_file_entries(), "FEIU_SA", "Energy_FEIU"]


def goes_eps_maged_and_epead_netcdf_strategy(
    base_data_path: str | Path,
    mag_field: ep.typing.MagneticFieldLiteral,
    satellite: GOESNOPSatellite,
    *,
    file_format: ep.typing.MFSFormats = "nc",
    legacy_layout: bool = False,
) -> ep.SavingStrategy:
    """Monthly NetCDF saving strategy for combined GOES-13/15 EPS MAGED and EPEAD."""
    return _MonthlyMagedEpeadStrategy(
        Path(base_data_path),
        "GOES",
        satellite,
        "MAGED_EPEAD",
        mag_field,
        data_standard=ep.data_standards.GFZStandard(),
        file_format=file_format,
        legacy_layout=legacy_layout,
    )


LONG_FROM_SAT: dict[GOESNOPSatellite, float] = {
    "goes13": 74.5,
    "goes15": 135.1,
}

# thresholds of the EPEAD integral electron channels according to the CDF metadata
EPEAD_E1_THRESHOLD_MEV = 0.8
EPEAD_E2_THRESHOLD_MEV = 2.0

# Flux in 1/(cm^2 s sr) that is removed as background from the >2 MeV channel (east sensor, difference between the
# dead-time-corrected flux before and after the background correction). It is the median over December 2016 and varies
# by only about 3% over that month. The >2 MeV channel is flagged when it is not significantly above this background,
# so it is assumed to measure only background there and J(>2 MeV) is set to this value.
EPEAD_E2_BACKGROUND_FLUX: dict[GOESNOPSatellite, float] = {"goes13": 20.7, "goes15": 19.5}


def goes_eps_maged_and_epead_strategy(
    processed_data_path: str | Path,
    mag_field: ep.typing.MagneticFieldLiteral,
    satellite: GOESNOPSatellite,
    save_strategy: Literal["gfz", "netcdf"] = "netcdf",
    *,
    legacy_layout: bool = False,
) -> ep.SavingStrategy:
    """Saving strategy of the GOES EPS MAGED and EPEAD recipe.

    Args:
        processed_data_path (str | Path): Base directory of the processed data.
        mag_field (ep.typing.MagneticFieldLiteral): Magnetic field model the data was processed with.
        satellite (GOESNOPSatellite): Satellite the data belongs to.
        save_strategy (Literal["gfz", "netcdf"]): Saving strategy of the output files. Defaults to "netcdf".
        legacy_layout (bool): Deprecated. Set to True to load data saved without instrument folders. Defaults to False.

    Returns:
        ep.SavingStrategy: Legacy GFZ strategy for "gfz", otherwise a monthly or daily NetCDF strategy.
    """
    if save_strategy == "gfz":
        return goes_eps_maged_and_epead_gfz_strategy(processed_data_path, mag_field, satellite)

    return goes_eps_maged_and_epead_netcdf_strategy(
        processed_data_path, mag_field, satellite, legacy_layout=legacy_layout
    )


def process_goes_eps_maged_and_epead(
    start_time: datetime,
    end_time: datetime,
    satellite: GOESNOPSatellite = "goes13",
    mag_field: ep.typing.MagneticFieldLiteral = "T89",
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    bin_cadence: timedelta = timedelta(minutes=5),
    num_cores: int = 16,
    save_strategy: Literal["gfz", "netcdf"] = "netcdf",
    skip_existing: bool = True,  # noqa: FBT001, FBT002,
    *,
    add_B_obs: bool = False,
    calculate_Lstar: bool = True,
    save_indices_sw: bool = False,
    epead_pad_method: Literal["maged", "Smirnov2022"] = "maged",
    append: bool = True,
) -> None:
    """Process combined GOES-13/15 EPS MAGED and EPEAD electron data into phase space densities.

    Downloads and extracts the 5-minute EPS MAGED differential electron fluxes, the 1-minute
    EPEAD integral electron fluxes (>0.8 MeV and >2 MeV, east and west telescopes; only the east one for
    GOES-15, whose west sensor degrades with local time and season) and the
    1-minute EPS telescope pitch angles from CDAWeb for the given GOES-13/15 satellite. Points
    with a non-zero quality flag are removed and all data are binned onto a common time cadence.
    The local pitch angles are folded around 90 degrees and the MAGED fluxes sorted by ascending
    pitch angle. A fixed geostationary spacecraft position is assigned and magnetic field
    quantities (B_Calc, B_Eq, MLT, R_Eq, Alpha_Eq, L_star, L_m, InvMu, InvK) are computed. The
    EPEAD integral channels are converted to differential fluxes on a fixed energy grid (0.8 to
    10 MeV) by fitting an exponential spectrum, and a pitch angle distribution is applied to them,
    either that of the highest MAGED channel or one from the Smirnov et al. (2022) model (see
    `epead_pad_method`). Where the >2 MeV channel is missing, the spectrum is built from the >0.8 MeV
    channel and the background level of the >2 MeV channel and only kept up to 2 MeV. Finally, the electron phase
    space density is derived and the resulting variables are saved using the requested saving strategy. The measured
    EPEAD integral fluxes
    (`FEIU_SA`, mean of the east and west telescopes or only the east one for GOES-15, with their
    thresholds in `Energy_FEIU`) are
    saved as well by the "netcdf" strategy; they are NaN wherever a channel is flagged as bad, in
    particular the >2 MeV channel during solar proton contamination.

    Args:
        start_time (datetime): Start of the time interval to process.
        end_time (datetime): End of the time interval to process.
        satellite (Literal["goes13", "goes15"]): The GOES satellite to process.
        mag_field (MagneticFieldLiteral): Magnetic field model used for the derived quantities.
        raw_data_path (str | Path): Directory where the raw downloaded data files are stored.
        processed_data_path (str | Path): Directory where the processed output files are saved.
        bin_cadence (timedelta): Time cadence used to bin the extracted variables.
        num_cores (int): Number of CPU cores used for the magnetic field computations.
        save_strategy (Literal["gfz", "netcdf"]): Strategy used to save the processed
            data. "gfz" saves using the GFZ format, "netcdf" saves monthly NetCDF files.
        skip_existing (bool): If True, skip downloading files that already exist on disk.
        add_B_obs (bool): If True, also download the high-resolution L2 magnetometer data and save the
            observed magnetic field magnitude (B_total_obs) and InvMu_obs computed from it. Defaults to False.
        calculate_Lstar (bool): If True, also compute the L* magnetic field quantity. If False, the
            expensive drift-shell tracing is skipped and L* is not saved, while L_m and InvK are still
            computed. Defaults to True.
        save_indices_sw (bool): If True, also save all solar wind indices and parameters,
            independent of this recipe's own output. Defaults to False.
        epead_pad_method (Literal["maged", "Smirnov2022"]): Source of the pitch angle distribution
            applied to the EPEAD channels. "maged" uses the distribution measured by the highest MAGED
            channel. "Smirnov2022" uses the empirical Smirnov et al. (2022) model, evaluated at the
            equatorial pitch angles of the MAGED telescopes with R_Eq as L-shell, MLT and the solar wind
            dynamic pressure. The model is only fitted up to 1650 keV, so EPEAD energies above that use
            the shape of the 1650 keV channel. In both cases the EPEAD flux is treated as spin-averaged
            over the MAGED telescope pitch angles, i.e. the PAD is normalized to a trapezoidal mean of one
            over the sampled local pitch angle range. Defaults to "maged".
        append (bool): If True, append to existing output files instead of overwriting them.
            Defaults to True.
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

    if add_B_obs:
        magn_url = f"https://spdf.gsfc.nasa.gov/pub/data/goes/{satellite}/mag_l2_netcdf/YYYY/"
        magn_file_name_stem = f"dn_magn-l2-hires_g{satellite[-2:]}_dYYYYMMDD_.{r'{6}'}.nc"

        ep.download(
            start_time,
            end_time,
            save_path=data_path_stem,
            file_cadence="daily",
            download_url=magn_url,
            file_name_stem=magn_file_name_stem,
            skip_existing=skip_existing,
        )

    ep.exit_if_download_only()

    maged_vars = _get_and_prepare_eps_maged_variables(
        maged_file_name_stem, data_path_stem, start_time, end_time, satellite, bin_cadence
    )
    epead_vars = _get_and_prepare_epead_variables(
        epead_file_name_stem, data_path_stem, start_time, end_time, satellite, bin_cadence
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

    combined_energies = np.concatenate(
        (maged_vars["Energy"].get_data(u.keV), epead_vars["Energy_FEDU"].get_data(u.keV)),
        axis=1,
    )
    maged_vars["Energy"].set_data(combined_energies, unit=u.keV)

    # Calculate magnetic field variables
    variables_to_compute: ep.processing.VariableRequest = [
        ("B_Calc", mag_field),
        ("B_Eq", mag_field),
        ("MLT", mag_field),
        ("R_Eq", mag_field),
        ("Alpha_Eq", mag_field),
        ("L_m", mag_field),
        ("InvMu", mag_field),
        ("InvK", mag_field),
    ]

    if calculate_Lstar:
        variables_to_compute.append(("L_star", mag_field))

    irbem_options = ep.processing.magnetic_field_utils.IrbemOptions(
        lstar_quantity=ep.processing.magnetic_field_utils.LstarQuantity.LSTAR
        if calculate_Lstar
        else ep.processing.magnetic_field_utils.LstarQuantity.NONE,
    )

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=maged_vars["Epoch"],
        xgeo_var=maged_vars["xGEO"],
        energy_var=maged_vars["Energy"],
        pa_local_var=pa_vars["AlphaLocal"],
        particle_species="electron",
        variables_to_compute=variables_to_compute,
        irbem_options=irbem_options,
        num_cores=num_cores,
    )

    # apply PAD to epead; the epead flux is treated as spin-averaged over the maged pitch angles
    maged_flux = maged_vars["FEDU"].get_data((u.cm**2 * u.s * u.sr * u.keV) ** (-1)).astype(np.float64)

    match epead_pad_method:
        case "maged":
            # same normalization as construct_pitch_angle_distribution with flux_type="spin_average": trapezoidal
            # mean over the sampled local pitch angle range (the pitch angles are sorted in ascending order above)
            pa_local = pa_vars["AlphaLocal"].get_data(u.rad).astype(np.float64)
            pad_shape = maged_flux[:, -1, :]
            pad_mean = np.trapezoid(pad_shape, x=pa_local, axis=-1) / (pa_local[:, -1] - pa_local[:, 0])
            pad = pad_shape / pad_mean[:, np.newaxis]

            epead_flux = epead_vars["FEDU"].get_data((u.cm**2 * u.s * u.sr * u.keV) ** (-1)).astype(np.float64)
            epead_flux = epead_flux[:, :, np.newaxis] * pad[:, np.newaxis, :]
            pad_note = "the pitch angle distribution of the highest MAGED channel"
        case "Smirnov2022":
            epead_fedu_var = ep.processing.construct_pitch_angle_distribution(
                epead_vars["FEDU"],
                pa_vars["AlphaLocal"],
                magnetic_field_variables[f"Alpha_Eq_{mag_field}"],
                flux_type="spin_average",
                method="Smirnov2022",
                time_var=maged_vars["Epoch"],
                L_var=magnetic_field_variables[f"R_Eq_{mag_field}"],
                MLT_var=magnetic_field_variables[f"MLT_{mag_field}"],
                energy_var=epead_vars["Energy_FEDU"],
            )

            epead_flux = epead_fedu_var.get_data((u.cm**2 * u.s * u.sr * u.keV) ** (-1)).astype(np.float64)
            pad_note = (
                "the empirical Smirnov et al. (2022) pitch angle distribution model, using R_Eq as L-shell, "
                "MLT and the solar wind dynamic pressure (energies above 1650 keV use the shape of the 1650 keV "
                "channel)"
            )

    combined_flux = np.concatenate((maged_flux, epead_flux), axis=1)
    maged_vars["FEDU"].set_data(combined_flux, unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1))
    maged_vars["FEDU"].metadata.add_processing_note(
        "Appended the EPEAD channels (0.8 to 10 MeV): the >0.8 MeV and >2 MeV integral fluxes (mean of the east "
        "and west telescopes, only the east one for GOES-15) were converted to differential fluxes by fitting an "
        "exponential spectrum. These are "
        f"treated as spin-averaged over the MAGED telescope pitch angles and multiplied with {pad_note}, "
        "normalized to a mean of one over the sampled local pitch angle range (trapezoidal integration)."
    )

    psd_var = ep.processing.compute_phase_space_density(
        maged_vars["FEDU"], maged_vars["Energy"], particle_species="electron"
    )

    variables_to_save: dict[InternalName, ep.Variable] = {
        "Epoch": maged_vars["Epoch"],
        "FEDU": maged_vars["FEDU"],
        "Position": maged_vars["xGEO"],
        "Energy_FEDU": maged_vars["Energy"],
        "FEIU_SA": epead_vars["FEIU"],
        "Energy_FEIU": epead_vars["Energy_FEIU"],
        "Alpha": pa_vars["AlphaLocal"],
        "PSD": psd_var,
        "Alpha_Eq": magnetic_field_variables[f"Alpha_Eq_{mag_field}"],
        "MLT": magnetic_field_variables[f"MLT_{mag_field}"],
        "R_Eq": magnetic_field_variables[f"R_Eq_{mag_field}"],
        "B_Eq": magnetic_field_variables[f"B_Eq_{mag_field}"],
        "B_Calc": magnetic_field_variables[f"B_Calc_{mag_field}"],
        "InvMu": magnetic_field_variables[f"InvMu_{mag_field}"],
        "InvK": magnetic_field_variables[f"InvK_{mag_field}"],
    }

    if calculate_Lstar:
        variables_to_save["L_star"] = magnetic_field_variables[f"L_star_{mag_field}"]

    if add_B_obs:
        magn_vars = _get_and_prepare_magnetometer_variables(
            magn_file_name_stem, data_path_stem, start_time, end_time, bin_cadence
        )
        variables_to_save["B_total_obs"] = magn_vars["B_total_obs"]
        variables_to_save["InvMu_obs"] = ep.processing.compute_invariant_mu(
            maged_vars["Energy"], pa_vars["AlphaLocal"], magn_vars["B_total_obs"], "electron"
        )

    saving_strategy = goes_eps_maged_and_epead_strategy(processed_data_path, mag_field, satellite, save_strategy)

    ep.save(variables_to_save, saving_strategy, start_time, end_time, time_var=maged_vars["Epoch"], append=append)

    if save_indices_sw:
        process_solar_wind_indices(
            start_time,
            end_time,
            processed_data_path=processed_data_path,
            bin_cadence=bin_cadence,
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
    satellite: GOESNOPSatellite,
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

    # The GOES-15 west sensor degrades with local time and season (Rodriguez et al., 2025,
    # https://doi.org/10.1029/2024SW004228, Appendix A), so only the east sensor is used there.
    # TODO(Bernhard Haas): Contact the authors of Rodriguez et al. (2025) whether the west sensor of GOES-13 is  # noqa: FIX002, TD003
    #  affected as well. Until then, the west and east telescopes of GOES-13 are averaged.
    telescopes = ("east",) if satellite == "goes15" else ("west", "east")

    # merge epead data
    feiu_0_8 = np.nanmean([epead_vars[f"FEIU_0_8_{telescope}"].get_data() for telescope in telescopes], axis=0)
    feiu_2_0 = np.nanmean([epead_vars[f"FEIU_2_0_{telescope}"].get_data() for telescope in telescopes], axis=0)
    epead_vars["FEIU"] = ep.Variable(
        data=np.vstack([feiu_0_8, feiu_2_0]).T,
        original_unit=(u.cm**2 * u.s * u.sr) ** (-1),
    )
    telescope_note = (
        "Only the east telescope (the west sensor degrades with local time and season)"
        if telescopes == ("east",)
        else "Mean of the east and west telescopes"
    )
    epead_vars["FEIU"].metadata.add_processing_note(
        f"{telescope_note} for the >0.8 MeV and >2 MeV integral channels; samples with a "
        "non-zero quality flag were removed beforehand."
    )

    # integral channels are labeled with their thresholds, not central energies
    epead_vars["Energy_FEIU"] = ep.Variable(
        data=np.array([EPEAD_E1_THRESHOLD_MEV, EPEAD_E2_THRESHOLD_MEV]),
        original_unit=u.MeV,
    )
    epead_vars["Energy_FEIU"].metadata.add_processing_note("Lower energy thresholds of the integral channels.")

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

    time_bin_methods_epead = {
        "FEIU": ep.TimeBinMethod.NanMedian,
        "Energy_FEDU": ep.TimeBinMethod.Repeat,
        "Energy_FEIU": ep.TimeBinMethod.Repeat,
    }

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
    feiu = np.array(epead_vars["FEIU"].get_data(), dtype=np.float64)
    energy_fedu = epead_vars["Energy_FEDU"].get_data(u.MeV)

    # Where the >2 MeV channel is missing but E1 is valid, assume that it measures only background. Its flux is then
    # set to the background level, which is an upper limit of the true flux, and the spectrum is only kept up to the
    # E2 threshold, above which it is unconstrained.
    use_background = np.isfinite(feiu[:, 0]) & (feiu[:, 0] > 0) & ~np.isfinite(feiu[:, 1])
    feiu[use_background, 1] = EPEAD_E2_BACKGROUND_FLUX[satellite]

    B = (np.log(feiu[:, 0]) - np.log(feiu[:, 1])) / (EPEAD_E2_THRESHOLD_MEV - EPEAD_E1_THRESHOLD_MEV)
    B[B < 0] = np.nan

    A1 = feiu[:, 0] * B / np.exp(-B * EPEAD_E1_THRESHOLD_MEV)
    A2 = feiu[:, 1] * B / np.exp(-B * EPEAD_E2_THRESHOLD_MEV)
    A = np.mean([A1, A2], axis=0)

    fedu = A[:, np.newaxis] * np.exp(-B[:, np.newaxis] * energy_fedu)
    fedu[use_background[:, np.newaxis] & (energy_fedu > EPEAD_E2_THRESHOLD_MEV + 1e-6)] = np.nan

    logger.info(
        "Assumed that the >2 MeV channel only measures background "
        f"({EPEAD_E2_BACKGROUND_FLUX[satellite]} 1/(cm^2 s sr)) "
        f"for {np.sum(use_background)} of {len(feiu)} time steps without a valid >2 MeV channel."
    )

    epead_vars["FEDU"] = ep.Variable(data=fedu, original_unit=(u.cm**2 * u.s * u.sr * u.MeV) ** (-1))
    epead_vars["FEDU"].metadata.add_processing_note(
        "Exponential spectrum fitted to the >0.8 MeV and >2 MeV integral channels. Where the >2 MeV channel is "
        "missing, it is assumed to measure only background, and its flux is set to the removed background of "
        f"{EPEAD_E2_BACKGROUND_FLUX[satellite]} 1/(cm^2 s sr). Energies above {EPEAD_E2_THRESHOLD_MEV} MeV are set to "
        "NaN in these steps."
    )

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


def _get_and_prepare_magnetometer_variables(
    file_name_stem: str,
    data_path_stem: str | Path,
    start_time: datetime,
    end_time: datetime,
    bin_cadence: timedelta,
) -> dict[str, ep.Variable]:
    extraction_infos = [
        ep.ExtractionInfo(name_or_column="time", unit=ep.units.j2k, result_key="Epoch"),
        ep.ExtractionInfo(name_or_column="b_epn", unit=u.nT, result_key="b_epn"),
        ep.ExtractionInfo(name_or_column="quality", unit=u.dimensionless_unscaled, result_key="quality"),
    ]

    magn_vars = ep.extract_variables_from_files(
        start_time,
        end_time,
        file_cadence="daily",
        data_path=data_path_stem,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )

    # b_total is fill in the hires files, so the magnitude is taken from the best-estimate field vector
    b_epn_var = magn_vars.pop("b_epn")
    b_total_obs = np.linalg.norm(b_epn_var.get_data(u.nT).astype(np.float64), axis=1)
    b_total_obs[magn_vars.pop("quality").get_data() != 0] = np.nan

    magn_vars["B_total_obs"] = ep.Variable(u.nT, b_total_obs, source_files=b_epn_var.metadata.source_files)
    magn_vars["B_total_obs"].metadata.add_processing_note(
        "Magnitude of the magn b_epn variable; samples with a non-zero quality flag set to NaN."
    )

    magn_vars["Epoch"] = ep.processing.bin_by_time(
        time_variable=magn_vars["Epoch"],
        variables=magn_vars,
        time_bin_method_dict={"B_total_obs": ep.TimeBinMethod.NanMean},
        time_binning_cadence=bin_cadence,
        start_time=start_time,
        end_time=end_time,
    )

    return magn_vars


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

PLOT_DEFAULTS = {
    "panels": ["Kp", "FEDU", "L_star", "MLT"],
    "FEDU": {"pitch_angle": 90 * u.deg},
}

if __name__ == "__main__":
    ep.run_recipe_cli(
        process_goes_eps_maged_and_epead,
        defaults=CLI_DEFAULTS,
        strategy=goes_eps_maged_and_epead_strategy,
        plot_defaults=PLOT_DEFAULTS,
    )
