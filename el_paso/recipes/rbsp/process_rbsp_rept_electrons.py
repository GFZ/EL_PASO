# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
from astropy import units as u

import el_paso as ep
from el_paso.recipes.rbsp import RBSPSatellite
from el_paso.recipes.rbsp._emfisis_magnetometer import compute_observed_B_variables, download_emfisis_magnetometer
from el_paso.recipes.solar_wind_indices.process_solar_wind_indices import process_solar_wind_indices


def rbsp_rept_electron_gfz_strategy(
    base_data_path: str | Path, mag_field: ep.typing.MagneticFieldLiteral, satellite: RBSPSatellite
) -> ep.SavingStrategy:
    """Legacy GFZ .mat saving strategy for RBSP REPT electrons."""
    return ep.saving_strategies.GFZStrategy(Path(base_data_path), "RBSP", "rbsp" + satellite, "rept", mag_field)


def rbsp_rept_electron_netcdf_strategy(
    base_data_path: str | Path,
    mag_field: ep.typing.MagneticFieldLiteral,
    satellite: RBSPSatellite,
    *,
    file_format: ep.typing.MFSFormats = "nc",
    legacy_layout: bool = False,
) -> ep.SavingStrategy:
    """Monthly NetCDF saving strategy for RBSP REPT electrons."""
    return ep.saving_strategies.MonthlyRBStrategy(
        Path(base_data_path),
        "RBSP",
        "rbsp" + satellite,
        "rept",
        mag_field,
        data_standard=ep.data_standards.GFZStandard(),
        file_format=file_format,
        legacy_layout=legacy_layout,
    )


def rbsp_rept_electron_strategy(
    processed_data_path: str | Path,
    mag_field: ep.typing.MagneticFieldLiteral,
    satellite: RBSPSatellite,
    save_strategy: Literal["gfz", "netcdf"] = "netcdf",
    *,
    legacy_layout: bool = False,
) -> ep.SavingStrategy:
    """Saving strategy of the RBSP REPT electron recipe.

    Args:
        processed_data_path (str | Path): Base directory of the processed data.
        mag_field (ep.typing.MagneticFieldLiteral): Magnetic field model the data was processed with.
        satellite (RBSPSatellite): Satellite the data belongs to.
        save_strategy (Literal["gfz", "netcdf"]): Saving strategy of the output files. Defaults to "netcdf".
        legacy_layout (bool): Deprecated. Set to True to load data saved without instrument folders. Defaults to False.

    Returns:
        ep.SavingStrategy: Legacy GFZ strategy for "gfz", otherwise a monthly NetCDF strategy.
    """
    if save_strategy == "gfz":
        return rbsp_rept_electron_gfz_strategy(processed_data_path, mag_field, satellite)

    return rbsp_rept_electron_netcdf_strategy(processed_data_path, mag_field, satellite, legacy_layout=legacy_layout)


def process_rbsp_rept_electrons(
    start_time: datetime,
    end_time: datetime,
    satellite: RBSPSatellite = "a",
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
    append: bool = False,
) -> None:
    """Process RBSP ECT/REPT electron flux data into the EL-PASO data standard.

    Downloads the daily RBSP REPT level-3 electron CDF files for the given time range and
    satellite and extracts the energy/pitch-angle-resolved flux (FEDU) and position.
    Non-positive and fill fluxes are set to NaN, the flux is folded to the
    [0, 90] degree pitch angle range per measurement (NaN-mean of the two mirrored pitch angles),
    and all variables are afterwards time-binned with a NaN-mean to a common cadence. Then
    magnetic-field-related quantities (B field, equatorial pitch angle, L*, L_m, MLT, InvK, InvMu,
    etc.) are computed with IRBEM, the electron phase space density is computed from FEDU, and all
    resulting variables are saved using the requested saving strategy.
    If `add_B_obs` is set, the EMFISIS level-3 4-second magnetometer CDF files are downloaded as well,
    and the observed magnetic field magnitude (B_total_obs) and the first adiabatic invariant computed
    from it (InvMu_obs) are saved in addition.
    Validated against legacy GFZ data and achieved a match within 1% for all variables.

    Args:
        start_time (datetime): Start of the time range to process.
        end_time (datetime): End of the time range to process.
        satellite (Literal["a", "b"]): RBSP satellite identifier ("a" or "b").
        mag_field (ep.typing.MagneticFieldLiteral): Magnetic field model used to compute the
            magnetic-field-related variables.
        raw_data_path (str | Path): Directory where raw CDF files are downloaded to and
            read from. Defaults to ".".
        processed_data_path (str | Path): Directory where the processed output files are
            written to. Defaults to ".".
        bin_cadence (timedelta): Time-binning cadence applied to all variables.
        num_cores (int): Number of CPU cores used for the magnetic field computations.
            Defaults to 16.
        save_strategy (Literal["gfz", "netcdf"]): Which saving strategy to use for writing the
            processed output. Defaults to "netcdf".
        skip_existing (bool): If True, skip downloading files that already exist in
            raw_data_path. Defaults to True.
        add_B_obs (bool): If True, also download the EMFISIS magnetometer data and save the observed
            magnetic field magnitude (B_total_obs) and InvMu_obs computed from it. Defaults to False.
        calculate_Lstar (bool): If True, also compute the L* magnetic field quantity.
            Defaults to True.
        save_indices_sw (bool): If True, also save all solar wind indices and parameters,
            independent of this recipe's own output. Defaults to False.
        append (bool): If True, append to existing output files instead of overwriting them.
            Defaults to False.
    """
    raw_data_path = Path(raw_data_path)
    processed_data_path = Path(processed_data_path)

    file_name_stem = "rbsp" + satellite + "_rel03_ect-rept-sci-l3_YYYYMMDD_.{6}.cdf"

    ep.download(
        start_time,
        end_time,
        save_path=raw_data_path,
        download_url=f"https://spdf.gsfc.nasa.gov/pub/data/rbsp/rbsp{satellite}/l3/ect/rept/sectors/rel03/YYYY/",
        file_name_stem=file_name_stem,
        file_cadence="daily",
        method="request",
        skip_existing=skip_existing,
    )

    if add_B_obs:
        download_emfisis_magnetometer(start_time, end_time, satellite, raw_data_path, skip_existing=skip_existing)

    ep.exit_if_download_only()

    extraction_infos = [
        ep.ExtractionInfo(
            result_key="Epoch",
            name_or_column="Epoch",
            unit=ep.units.cdf_epoch,
        ),
        ep.ExtractionInfo(
            result_key="Energy",
            name_or_column="FEDU_Energy",
            unit=u.MeV,
            is_time_dependent=False,
        ),
        ep.ExtractionInfo(
            result_key="Pitch_angle",
            name_or_column="FEDU_Alpha",
            unit=u.deg,
            is_time_dependent=False,
        ),
        ep.ExtractionInfo(
            result_key="FEDU",
            name_or_column="FEDU",
            unit=(u.cm**2 * u.s * u.sr * u.MeV) ** (-1),
        ),
        ep.ExtractionInfo(
            result_key="xGEO",
            name_or_column="Position",
            unit=u.km,
        ),
    ]

    variables = ep.extract_variables_from_files(
        start_time,
        end_time,
        "daily",
        data_path=raw_data_path,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )

    # zeros are invalid (fill values are already NaN)
    variables["FEDU"].apply_thresholds_on_data(lower_threshold=0)

    # fold before time binning. The folding needs the (constant) pitch angles for every time step
    pitch_angle_channels = variables["Pitch_angle"].get_data()
    n_time = len(variables["FEDU"].get_data())
    variables["Pitch_angle"].set_data(np.tile(pitch_angle_channels, (n_time, 1)), "same")

    variables["FEDU"].transpose_data([0, 2, 1])  # making it have dimensions (time, energy, pitch angle)
    ep.processing.fold_pitch_angles_and_flux(variables["FEDU"], variables["Pitch_angle"])

    # back to time independent, such that it is repeated for the binned time
    variables["Pitch_angle"].set_data(variables["Pitch_angle"].get_data()[0], "same")

    time_bin_methods = {
        "Energy": ep.TimeBinMethod.Repeat,
        "FEDU": ep.TimeBinMethod.NanMean,
        "Pitch_angle": ep.TimeBinMethod.Repeat,
        "xGEO": ep.TimeBinMethod.NanMean,
    }

    binned_time_variable = ep.processing.bin_by_time(
        variables["Epoch"],
        variables=variables,
        time_bin_method_dict=time_bin_methods,
        time_binning_cadence=bin_cadence,
        start_time=start_time,
        end_time=end_time,
    )

    # not needed anymore
    del variables["Epoch"]

    vars_to_compute: ep.typing.VariableRequest = [
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
        vars_to_compute.append(("L_star", mag_field))

    irbem_options = ep.processing.magnetic_field_utils.IrbemOptions(
        lstar_quantity=ep.processing.magnetic_field_utils.LstarQuantity.LSTAR
        if calculate_Lstar
        else ep.processing.magnetic_field_utils.LstarQuantity.NONE,
    )

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=binned_time_variable,
        xgeo_var=variables["xGEO"],
        variables_to_compute=vars_to_compute,
        irbem_options=irbem_options,
        num_cores=num_cores,
        pa_local_var=variables["Pitch_angle"],
        energy_var=variables["Energy"],
        particle_species="electron",
    )

    psd_var = ep.processing.compute_phase_space_density(variables["FEDU"], variables["Energy"], "electron")

    variables_to_save: ep.typing.VariablesDict = {
        "Epoch": binned_time_variable,
        "FEDU": variables["FEDU"],
        "Energy_FEDU": variables["Energy"],
        "Alpha": variables["Pitch_angle"],
        "Alpha_Eq": magnetic_field_variables["Alpha_Eq_" + mag_field],
        "R_Eq": magnetic_field_variables["R_Eq_" + mag_field],
        "MLT": magnetic_field_variables["MLT_" + mag_field],
        "L_m": magnetic_field_variables["L_m_" + mag_field],
        "B_Calc": magnetic_field_variables["B_Calc_" + mag_field],
        "B_Eq": magnetic_field_variables["B_Eq_" + mag_field],
        "PSD": psd_var,
        "InvMu": magnetic_field_variables["InvMu_" + mag_field],
        "InvK": magnetic_field_variables["InvK_" + mag_field],
        "Position": variables["xGEO"],
    }

    if calculate_Lstar:
        variables_to_save["L_star"] = magnetic_field_variables["L_star_" + mag_field]

    if add_B_obs:
        variables_to_save |= compute_observed_B_variables(
            start_time,
            end_time,
            satellite,
            raw_data_path,
            binned_time_variable,
            bin_cadence,
            variables["Energy"],
            variables["Pitch_angle"],
            "electron",
        )

    strategy = rbsp_rept_electron_strategy(processed_data_path, mag_field, satellite, save_strategy)

    ep.save(variables_to_save, strategy, start_time, end_time, time_var=binned_time_variable, append=append)

    if save_indices_sw:
        process_solar_wind_indices(
            start_time,
            end_time,
            processed_data_path=processed_data_path,
            bin_cadence=bin_cadence,
        )


PLOT_DEFAULTS = {
    "panels": ["Kp", "FEDU"],
    "FEDU": {
        "pitch_angle": 90 * u.deg,
        "energies": [2.1, 3.4, 5.2] * u.MeV,
        "vmin": 1e-2,
        "vmax": 1e4,
        "along": "L_star",
    },
}

if __name__ == "__main__":
    ep.run_recipe_cli(
        process_rbsp_rept_electrons,
        strategy=rbsp_rept_electron_strategy,
        plot_defaults=PLOT_DEFAULTS,
    )
