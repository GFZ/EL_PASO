# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Jaskirat Singh
#
# SPDX-License-Identifier: Apache-2.0

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from astropy import units as u

import el_paso as ep
from el_paso.processing.magnetic_field_utils import IrbemOptions, LstarQuantity

logging.captureWarnings(capture=True)
logger = logging.getLogger(__name__)

ELFINSatellites = Literal["ela", "elb"]


def elfin_epd_electron_strategy(
    processed_data_path: str | Path,
    mag_field: ep.typing.MagneticFieldLiteral,
    satellite: ELFINSatellites,
) -> ep.SavingStrategy:
    """Daily NetCDF saving strategy for ELFIN EPD electrons."""
    return ep.saving_strategies.DailyLEORBStrategy(
        base_data_path=Path(processed_data_path),
        mission="ELFIN",
        satellite=satellite,
        instrument="EPD",
        mag_field=mag_field,
        data_standard=ep.data_standards.GFZStandard(),
    )


def process_elfin_epd_electrons(
    start_time: datetime,
    end_time: datetime,
    satellite: ELFINSatellites = "ela",
    mag_field: ep.typing.MagneticFieldLiteral = "Dip",
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    bin_cadence: timedelta = timedelta(seconds=1.4),
    num_cores: int = 32,
    *,
    skip_existing: bool = True,
) -> None:
    """Process ELFIN EPD electron flux data into the EL-PASO data standard.

    Downloads the daily ELFIN state (l1) and EPD electron (l2) CDF files for the given time range
    and satellite, extracts the energy/pitch-angle-resolved energy flux (half-spin resolution),
    the loss-cone angle and the GEI position, time-bins all variables to a common cadence,
    applies a lower bound threshold to the flux, computes the xGEO and geographic positions,
    computes magnetic-field-related quantities (B field, equatorial pitch angle, L_m, MLT,
    loss-cone angles, etc.) with the given magnetic field model via IRBEM, and saves all
    resulting variables using the daily saving strategy.

    Args:
        start_time (datetime): Start of the time range to process.
        end_time (datetime): End of the time range to process.
        satellite (Literal["ela", "elb"]): ELFIN satellite identifier.
        mag_field (ep.typing.MagneticFieldLiteral): Magnetic field model used to compute the
            magnetic-field-related variables.
        raw_data_path (str | Path): Directory where raw CDF files are downloaded to and
            read from. Defaults to ".".
        processed_data_path (str | Path): Directory where the processed output files are
            written to. Defaults to ".".
        bin_cadence (timedelta): Time-binning cadence applied to all variables.
        num_cores (int): Number of CPU cores used for the magnetic field computations.
            Defaults to 32.
        skip_existing (bool): If True, skip downloading files that already exist on disk.
    """
    data_path_stem = f"{raw_data_path}/ELFIN/{satellite}/YYYY/MM/"

    state_name_stem = satellite + "_l1_state_defn_YYYYMMDD_.{3}.cdf"
    epd_name_stem = satellite + "_l2_epdef_YYYYMMDD_.{3}.cdf"

    ep.download(
        start_time,
        end_time,
        save_path=data_path_stem,
        file_cadence="daily",
        download_url=f"https://data.elfin.ucla.edu/{satellite}/l1/state/defn/YYYY/",
        file_name_stem=state_name_stem,
        skip_existing=skip_existing,
    )

    ep.download(
        start_time,
        end_time,
        save_path=data_path_stem,
        file_cadence="daily",
        download_url=f"https://data.elfin.ucla.edu/{satellite}/l2/epd/fast/electron/YYYY/",
        file_name_stem=epd_name_stem,
        skip_existing=skip_existing,
    )

    ep.exit_if_download_only()

    state_vars = _get_state_variables(satellite, data_path_stem, state_name_stem, start_time, end_time)
    epd_vars = _get_epd_variables(satellite, data_path_stem, epd_name_stem, start_time, end_time)

    time_bin_methods_state = {
        "pos_gei": ep.TimeBinMethod.NanMean,
    }

    binned_time_var = ep.processing.bin_by_time(
        time_variable=state_vars["time"],
        variables=state_vars,
        time_bin_method_dict=time_bin_methods_state,
        time_binning_cadence=bin_cadence,
        start_time=start_time,
        end_time=end_time,
    )

    time_bin_methods_epd = {
        "diff_eflux": ep.TimeBinMethod.NanMedian,
        "diff_energy": ep.TimeBinMethod.Repeat,
        "pitch_angle": ep.TimeBinMethod.NanMean,
        "LC_deg": ep.TimeBinMethod.NanMean,
    }

    _ = ep.processing.bin_by_time(
        time_variable=epd_vars["time"],
        variables=epd_vars,
        time_bin_method_dict=time_bin_methods_epd,
        time_binning_cadence=bin_cadence,
        start_time=start_time,
        end_time=end_time,
    )

    # clean flux data
    epd_vars["diff_eflux"].apply_thresholds_on_data(lower_threshold=0)
    epd_vars["diff_eflux"].transpose_data([0, 2, 1])  # making it have dimensions (time, energy, pitch angle)

    # local pitch angle sectored-center
    local_pa_var = epd_vars["pitch_angle"]
    local_pa_var.set_data(local_pa_var.get_data(u.deg), unit=u.deg)
    logger.info("local_pa_var shape: %s", local_pa_var.get_data(u.deg).shape)

    # calculate xGEO from GEI position
    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in binned_time_var.get_data(ep.units.posixtime)]
    pos_gei_km = state_vars["pos_gei"].get_data(u.km).astype(np.float64)

    xgeo_data = ep.processing.magnetic_field_utils.Coords().transform(
        datetimes,
        pos_gei_km,
        ep.IRBEM_SYSAXIS_GEI,
        ep.IRBEM_SYSAXIS_GEO,
    )
    xgeo_var = ep.Variable(data=xgeo_data, original_unit=u.km)
    xgeo_var.set_data(xgeo_var.get_data(ep.units.RE), unit=ep.units.RE)

    # Converting cartesian units to Re first and then to spherical-geocentric.
    pos_gei_re = state_vars["pos_gei"].get_data(ep.units.RE).astype(np.float64)

    geo_sph_data = ep.processing.magnetic_field_utils.Coords().transform(
        datetimes,
        pos_gei_re,
        ep.IRBEM_SYSAXIS_GEI,
        "SPH",
    )
    geo_R_var = ep.Variable(data=geo_sph_data[:, 0], original_unit=ep.units.RE)
    geo_lat_var = ep.Variable(data=geo_sph_data[:, 1], original_unit=u.deg)
    geo_lon_var = ep.Variable(data=geo_sph_data[:, 2], original_unit=u.deg)

    # Calculate magnetic field variables
    vars_to_compute: ep.typing.VariableRequest = [
        ("B_Calc", mag_field),
        ("B_Eq", mag_field),
        ("MLT", mag_field),
        ("B_fofl", mag_field),
        ("R_Eq", mag_field),
        ("L_m", mag_field),
        ("Alpha_Eq", mag_field),
        ("Alpha_LC_Eq", mag_field),
        ("Alpha_LC", mag_field),
        ("InvK", mag_field),
        ("InvMu", mag_field),
    ]

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=binned_time_var,
        xgeo_var=xgeo_var,
        energy_var=epd_vars["diff_energy"],
        pa_local_var=local_pa_var,
        particle_species="electron",
        variables_to_compute=vars_to_compute,
        irbem_options=IrbemOptions(lstar_quantity=LstarQuantity.NONE),
        num_cores=num_cores,
        overwrite_cache=True,
    )

    #psd_var = ep.processing.compute_phase_space_density(
    #epd_vars["diff_eflux"], epd_vars["diff_energy"], "electron")


    variables_to_save: ep.typing.VariablesDict = {
        "Epoch": binned_time_var,
        "FEDU": epd_vars["diff_eflux"],
        "Energy_FEDU": epd_vars["diff_energy"],
        "Alpha": local_pa_var,
        "Alpha_Eq": magnetic_field_variables["Alpha_Eq_" + mag_field],
        #"Alpha_LC": magnetic_field_variables["Alpha_LC_" + mag_field],
        "Alpha_LC": epd_vars["LC_deg"],
        "Alpha_LC_Eq": magnetic_field_variables["Alpha_LC_Eq_" + mag_field],
        "R_Eq": magnetic_field_variables["R_Eq_" + mag_field],
        "L_m": magnetic_field_variables["L_m_" + mag_field],
        "MLT": magnetic_field_variables["MLT_" + mag_field],
        "B_Calc": magnetic_field_variables["B_Calc_" + mag_field],
        "B_Eq": magnetic_field_variables["B_Eq_" + mag_field],
        "Position": xgeo_var,
        "Position_geo_lat": geo_lat_var,
        "Position_geo_lon": geo_lon_var,
        "Position_geo_alt": geo_R_var,
        "InvK": magnetic_field_variables["InvK_" + mag_field],
        "InvMu": magnetic_field_variables["InvMu_" + mag_field],
        #"PSD": psd_var,
    }

    strategy = elfin_epd_electron_strategy(processed_data_path, mag_field, satellite)

    ep.save(variables_to_save, strategy, start_time, end_time, time_var=binned_time_var)


def _get_state_variables(
    satellite: ELFINSatellites,
    data_path_stem: str | Path,
    file_name_stem: str,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, ep.Variable]:
    extraction_infos = [
        ep.ExtractionInfo(name_or_column=f"{satellite}_state_time", unit=ep.units.tt2000, result_key="time"),
        ep.ExtractionInfo(name_or_column=f"{satellite}_pos_gei", unit=u.km, result_key="pos_gei"),
    ]

    return ep.extract_variables_from_files(
        start_time,
        end_time,
        file_cadence="daily",
        data_path=data_path_stem,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )


def _get_epd_variables(
    satellite: ELFINSatellites,
    data_path_stem: str | Path,
    file_name_stem: str,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, ep.Variable]:
    extraction_infos = [
        ep.ExtractionInfo(name_or_column=f"{satellite}_pef_hs_time", unit=ep.units.tt2000, result_key="time"),
        ep.ExtractionInfo(name_or_column=f"{satellite}_pef_hs_Epat_nflux", unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),result_key="diff_eflux",),
        ep.ExtractionInfo(name_or_column=f"{satellite}_pef_energies_mean", unit=u.keV, result_key="diff_energy", is_time_dependent=False),
        ep.ExtractionInfo(name_or_column=f"{satellite}_pef_hs_epa_spec", unit=u.deg, result_key="pitch_angle", is_time_dependent=True),
        ep.ExtractionInfo(name_or_column=f"{satellite}_pef_hs_LCdeg", unit=u.deg, result_key="LC_deg"),
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
    "raw_data_path": "elfin_raw",
    "processed_data_path": "elfin_processed",
}

PLOT_DEFAULTS = {
    "panels": ["FEDU", "R_Eq", "MLT"],
    "FEDU": {"pitch_angle": 90 * u.deg},
}

if __name__ == "__main__":
    ep.run_recipe_cli(
        process_elfin_epd_electrons,
        defaults=CLI_DEFAULTS,
        strategy=elfin_epd_electron_strategy,
        plot_defaults=PLOT_DEFAULTS,
    )
