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

import el_paso as ep

if TYPE_CHECKING:
    from el_paso.typing import InternalName

logging.captureWarnings(capture=True)
logger = logging.getLogger(__name__)

TELE_ALPHA_ANGLES = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
TELE_BETA_ANGLES = np.array([-35.0, 35.0, -70.0, 0, 70.0])


def process_goes_r_mps_high(
    start_time: datetime,
    end_time: datetime,
    satellite: Literal["goes18", "goes19"] = "goes18",
    mag_field: ep.typing.MagneticFieldLiteral = "T89",
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    bin_cadence: timedelta = timedelta(minutes=5),
    num_cores: int = 16,
    save_strategy: Literal["gfz", "netcdf", "both"] = "netcdf",
) -> None:
    """Process GOES-R MPS-HI MAGED electron data into pitch-angle resolved phase space densities.

    Downloads and extracts the magnetometer (MAGN), MPS-HI energetic particle (MPSH), and
    ephemeris (EPHE) L2 data products for the given GOES-R satellite, bins them onto a common
    5-minute time cadence, computes the local telescope pitch angles from the magnetic field
    direction, sorts the differential electron fluxes by ascending pitch angle, transforms the
    spacecraft position to GEO coordinates, computes magnetic field quantities (B_Calc, B_Eq,
    MLT, R_Eq, Alpha_Eq, L_star, L_m, InvMu, InvK), and derives the electron phase space density.
    The resulting variables are saved using the requested saving strategy.

    Args:
        start_time (datetime): Start of the time interval to process.
        end_time (datetime): End of the time interval to process.
        satellite (Literal["goes18", "goes19"]): The GOES-R satellite to process.
        mag_field (MagneticFieldLiteral): Magnetic field model used for the derived quantities.
        raw_data_path (str | Path): Directory where the raw downloaded data files are stored.
        processed_data_path (str | Path): Directory where the processed output files are saved.
        bin_cadence (timedelta): Time cadence used to bin the extracted variables.
        num_cores (int): Number of CPU cores used for the magnetic field computations.
        save_strategy (Literal["gfz", "netcdf", "both"]): Strategy used to save the processed
            data. "gfz" saves using the GFZ format, "netcdf" saves monthly NetCDF files, and
            "both" saves using both strategies.
    """
    data_path_stem = f"{raw_data_path}/YYYY/MM/{satellite}/"

    magn_vars = _get_magn_variables(satellite, data_path_stem, start_time, end_time)
    mps_vars = _get_mps_high_variables(satellite, data_path_stem, start_time, end_time)
    ephe_vars = _get_ephe_variables(satellite, data_path_stem, start_time, end_time)

    time_bin_methods_magn = {
        "b_brf": ep.TimeBinMethod.NanMean,
    }

    binned_time_var = ep.processing.bin_by_time(
        time_variable=magn_vars["time"],
        variables=magn_vars,
        time_bin_method_dict=time_bin_methods_magn,
        time_binning_cadence=bin_cadence,
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
        time_binning_cadence=bin_cadence,
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
        time_binning_cadence=bin_cadence,
        start_time=start_time,
        end_time=end_time,
    )

    mps_vars["diff_flux"].transpose_data((0, 2, 1))
    mps_vars["diff_flux"].apply_thresholds_on_data(lower_threshold=0)

    # calculate xGEO
    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in binned_time_var.get_data(ep.units.posixtime)]
    xgeo_data = ep.processing.magnetic_field_utils.Coords().transform(
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
        time_var=binned_time_var,
        xgeo_var=xgeo_var,
        energy_var=mps_vars["diff_energy"],
        pa_local_var=local_pa_var,
        particle_species="electron",
        variables_to_compute=variables_to_compute,
        irbem_options=ep.processing.magnetic_field_utils.IrbemOptions(),
        num_cores=num_cores,
    )

    psd_var = ep.processing.compute_phase_space_density(
        mps_vars["diff_flux"], mps_vars["diff_energy"], particle_species="electron"
    )

    variables_to_save: dict[InternalName, ep.Variable] = {
        "Epoch": binned_time_var,
        "FEDU": mps_vars["diff_flux"],
        "Position": xgeo_var,
        "Energy_FEDU": mps_vars["diff_energy"],
        "Alpha": local_pa_var,
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
        saving_strategy = ep.saving_strategies.GFZStrategy(
            Path(processed_data_path),
            mission="GOES",
            satellite=satellite,
            instrument="MAGED",
            mag_field=mag_field,
            data_standard=ep.data_standards.GFZStandard(),
        )
    if save_strategy in ("netcdf", "both"):
        saving_strategy = ep.saving_strategies.MonthlyRBStrategy(
            Path(processed_data_path),
            mission="GOES",
            satellite=satellite,
            instrument="MAGED",
            mag_field=mag_field,
            file_format="nc",
            data_standard=ep.data_standards.GFZStandard(),
        )

    ep.save(variables_to_save, saving_strategy, start_time, end_time, time_var=binned_time_var, append=True)


def _get_magn_variables(
    satellite: Literal["goes18", "goes19"],
    data_path_stem: str | Path,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, ep.Variable]:
    url = f"https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/{satellite}/l2/data/magn-l2-avg1m/YYYY/MM/"

    sat_stem = "g18" if satellite == "goes18" else "g19"
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

    return ep.extract_variables_from_files(
        start_time,
        end_time,
        file_cadence="daily",
        data_path=data_path_stem,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )


def _get_ephe_variables(
    satellite: Literal["goes18", "goes19"],
    data_path_stem: str | Path,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, ep.Variable]:
    url = f"https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/{satellite}/l2/data/ephe-l2-orb1m/YYYY/MM/"

    sat_stem = "g18" if satellite == "goes18" else "g19"
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

    return ep.extract_variables_from_files(
        start_time,
        end_time,
        file_cadence="daily",
        data_path=data_path_stem,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )


def _get_mps_high_variables(
    satellite: Literal["goes18", "goes19"],
    data_path_stem: str | Path,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, ep.Variable]:
    url = f"https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/{satellite}/l2/data/mpsh-l2-avg5m_science/YYYY/MM/"

    sat_stem = "g18" if satellite == "goes18" else "g19"
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

    return ep.extract_variables_from_files(
        start_time,
        end_time,
        file_cadence="daily",
        data_path=data_path_stem,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )


CLI_DEFAULTS = {
    "raw_data_path": "goes/raw/",
    "processed_data_path": "goes/processed/",
}

if __name__ == "__main__":
    ep.run_recipe_cli(process_goes_r_mps_high, defaults=CLI_DEFAULTS)
