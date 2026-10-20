# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Jaskirat Singh
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
from astropy import units as u
from numpy.typing import NDArray

import el_paso as ep
from el_paso.processing.magnetic_field_utils import IrbemOptions
from el_paso.recipes.rbsp import RBSPSatellite
from el_paso.recipes.rbsp._emfisis_magnetometer import compute_observed_B_variables, download_emfisis_magnetometer

logger = logging.getLogger(__name__)

# FEDU_Energy and FEDU_Alpha entries after these are fill values
N_ENERGY_CHANNELS = 21
N_PITCH_ANGLES = 11

# FEDU_Energy of a complete rel04 file (rbspa_rel04_ect-mageis-l3_20140101_v8.1.0.cdf). Files with missing
# channels have the remaining ones shifted down; these are mapped back onto this list.
STANDARD_ENERGIES_KEV = np.array(
    [33, 54, 80, 108, 143, 184, 226, 235, 346, 470, 597, 749, 909, 1064, 1079, 1575, 1728, 2280, 2619, 3618, 4062],
    dtype=np.float64,
)


def rbsp_mageis_electron_strategy(
    base_data_path: str | Path,
    mag_field: ep.typing.MagneticFieldLiteral,
    satellite: RBSPSatellite,
    *,
    file_format: ep.typing.MFSFormats = "nc",
) -> ep.SavingStrategy:
    """Monthly NetCDF saving strategy for RBSP MagEIS electrons."""
    return ep.saving_strategies.MonthlyRBStrategy(
        Path(base_data_path),
        "RBSP",
        "rbsp" + satellite,
        "mageis",
        mag_field,
        data_standard=ep.data_standards.GFZStandard(),
        file_format=file_format,
    )


def process_rbsp_mageis_electrons(
    start_time: datetime,
    end_time: datetime,
    satellite: RBSPSatellite = "a",
    mag_field: Literal["T89", "T96", "TS04"] = "T89",
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    bin_cadence: timedelta = timedelta(minutes=5),
    num_cores: int = 16,
    skip_existing: bool = True,  # noqa: FBT001, FBT002,
    save_strategy: Literal["netcdf"] = "netcdf",
    add_B_obs: bool = False,  # noqa: FBT001, FBT002
) -> None:
    """Process RBSP ECT/MagEIS electron flux data into the EL-PASO data standard.

    Downloads the daily RBSP MagEIS level-3 electron CDF files for the given time range and
    satellite and extracts the energy/pitch-angle-resolved flux (FEDU) and position. Each file is
    treated as in the GFZ MATLAB reader `mageis_read_one_file`: fill energy channels and pitch angles
    are removed, files with missing energy channels are mapped onto the standard rel04 energies,
    non-positive fluxes are set to NaN, and positions are converted to Earth radii; energies are kept
    per file, so they are time dependent. The pitch angles and flux are then folded to the [0, 90]
    degree range with a NaN-mean, all variables are time-binned to a common cadence, magnetic-field-related
    quantities (B field, equatorial pitch angle, L*, L_m, MLT, InvK, InvMu, etc.) using the position
    time grid to ensure consistent time/position pairing with IRBEM, computes the electron phase space
    density from FEDU, and saves all resulting variables using the requested saving strategy.
    If `add_B_obs` is set, the EMFISIS level-3 4-second magnetometer CDF files are downloaded as well,
    and the observed magnetic field magnitude (B_total_obs) and the first adiabatic invariant computed
    from it (InvMu_obs) are saved in addition.

    Args:
        start_time (datetime): Start of the time range to process.
        end_time (datetime): End of the time range to process.
        satellite (Literal["a", "b"]): RBSP satellite identifier ("a" or "b").
        mag_field (Literal["T89", "T96", "TS04"]): Magnetic field model used to compute the
            magnetic-field-related variables.
        raw_data_path (str | Path): Directory where raw CDF files are downloaded to and
            read from. Defaults to ".".
        processed_data_path (str | Path): Directory where the processed output files are
            written to. Defaults to ".".
        bin_cadence (timedelta): Time-binning cadence applied to all variables.
        num_cores (int): Number of CPU cores used for the magnetic field computations.
            Defaults to 16.
        skip_existing (bool): If True, skip downloading files that already exist in
            raw_data_path. Defaults to True.
        save_strategy (Literal["netcdf"]): Saving strategy used for the output files.
            Only "netcdf" is currently supported for this recipe. Defaults to "netcdf".
        add_B_obs (bool): If True, also download the EMFISIS magnetometer data and save the observed
            magnetic field magnitude (B_total_obs) and InvMu_obs computed from it. Defaults to False.
    """
    del save_strategy
    raw_data_path = Path(raw_data_path)
    processed_data_path = Path(processed_data_path)

    file_name_stem = "rbsp" + satellite + "_rel04_ect-mageis-l3_YYYYMMDD_.{6}.cdf"

    ep.download(
        start_time,
        end_time,
        save_path=raw_data_path,
        download_url=f"https://spdf.gsfc.nasa.gov/pub/data/rbsp/rbsp{satellite}/l3/ect/mageis/sectors/rel04/YYYY/",
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
        # made time dependent per file by _mageis_data_modifier, since the energy channels differ between files
        ep.ExtractionInfo(
            result_key="Energy",
            name_or_column="FEDU_Energy",
            unit=u.keV,
        ),
        ep.ExtractionInfo(
            result_key="Pitch_angle",
            name_or_column="FEDU_Alpha",
            unit=u.deg,
        ),
        ep.ExtractionInfo(
            result_key="FEDU",
            name_or_column="FEDU",
            unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
        ),
        ep.ExtractionInfo(
            result_key="xGEO",
            name_or_column="Position",
            unit=ep.units.RE,
        ),
    ]

    variables = ep.extract_variables_from_files(
        start_time,
        end_time,
        "daily",
        data_path=raw_data_path,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
        data_modifier=_mageis_data_modifier,
    )

    # fold before time binning, as in the GFZ MATLAB processing
    variables["FEDU"].transpose_data([0, 2, 1])  # making it have dimensions (time, energy, pitch angle)
    ep.processing.fold_pitch_angles_and_flux(variables["FEDU"], variables["Pitch_angle"])

    time_bin_methods = {
        "Energy": ep.TimeBinMethod.NanMean,
        "FEDU": ep.TimeBinMethod.NanMedian,
        "Pitch_angle": ep.TimeBinMethod.NanMean,
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
        ("MLT", mag_field),
        ("B_Eq", mag_field),
        ("R_Eq", mag_field),
        ("Alpha_Eq", mag_field),
        ("L_star", mag_field),
        ("L_m", mag_field),
        ("InvK", mag_field),
        ("InvMu", mag_field),
    ]

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=binned_time_variable,
        xgeo_var=variables["xGEO"],
        variables_to_compute=vars_to_compute,
        irbem_options=IrbemOptions(),
        num_cores=num_cores,
        pa_local_var=variables["Pitch_angle"],
        energy_var=variables["Energy"],
        particle_species="electron",
    )

    psd_var = ep.processing.compute_phase_space_density(variables["FEDU"], variables["Energy"], "electron")

    variables_to_save: dict[ep.typing.InternalName, ep.Variable] = {
        "Epoch": binned_time_variable,
        "FEDU": variables["FEDU"],
        "Position": variables["xGEO"],
        "Energy_FEDU": variables["Energy"],
        "Alpha": variables["Pitch_angle"],
        "Alpha_Eq": magnetic_field_variables["Alpha_Eq_" + mag_field],
        "L_star": magnetic_field_variables["L_star_" + mag_field],
        "MLT": magnetic_field_variables["MLT_" + mag_field],
        "L_m": magnetic_field_variables["L_m_" + mag_field],
        "R_Eq": magnetic_field_variables["R_Eq_" + mag_field],
        "InvK": magnetic_field_variables["InvK_" + mag_field],
        "InvMu": magnetic_field_variables["InvMu_" + mag_field],
        "B_Eq": magnetic_field_variables["B_Eq_" + mag_field],
        "B_Calc": magnetic_field_variables["B_Calc_" + mag_field],
        "PSD": psd_var,
    }

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

    strategy = rbsp_mageis_electron_strategy(processed_data_path, mag_field, satellite)
    ep.save(variables_to_save, strategy, start_time, end_time, time_var=binned_time_variable, append=False)


def _mageis_data_modifier(
    variable_data: dict[str | int, NDArray[np.generic]],
    extraction_infos: Iterable[ep.ExtractionInfo],  # noqa: ARG001
) -> dict[str | int, NDArray[np.generic]]:
    """Apply the per-file treatment of the GFZ MATLAB reader `mageis_read_one_file`."""
    energy = variable_data["FEDU_Energy"].astype(np.float64)
    alpha = variable_data["FEDU_Alpha"].astype(np.float64)
    flux = variable_data["FEDU"].astype(np.float64)  # (time, pitch angle, energy)

    energy[energy < 0] = np.nan
    alpha[alpha < 0] = np.nan

    if np.isfinite(energy[N_ENERGY_CHANNELS:]).any() or np.isfinite(alpha[N_PITCH_ANGLES:]).any():
        msg = "MagEIS fill channels contain data; the channel layout of this file is not the expected one!"
        raise ValueError(msg)

    energy = energy[:N_ENERGY_CHANNELS]
    alpha = alpha[:N_PITCH_ANGLES]
    flux = flux[:, :N_PITCH_ANGLES, :N_ENERGY_CHANNELS]

    # some files miss channels and shift the remaining ones down; map them back onto the standard energies
    is_valid = np.isfinite(energy)
    if not is_valid.all():
        matches = np.isclose(energy[is_valid, np.newaxis], STANDARD_ENERGIES_KEV[np.newaxis, :], rtol=1e-6, atol=0)

        if matches.any(axis=1).all():
            shifted_flux = np.full_like(flux, np.nan)
            shifted_flux[:, :, matches.argmax(axis=1)] = flux[:, :, is_valid]
            flux = shifted_flux
            energy = STANDARD_ENERGIES_KEV.copy()
            logger.info(f"Shifted MagEIS energy channels onto standard energies at {matches.argmax(axis=1)}.")
        else:
            logger.warning("MagEIS energy channel definitions are different from usual in this file!")

    flux[flux <= 0] = np.nan

    position = variable_data["Position"].astype(np.float64)
    position[np.abs(position) > 1e10] = np.nan
    is_km = ~(np.linalg.norm(position, axis=1) < 16)  # rows not already in Earth radii
    position[is_km] = (position[is_km] * u.km).to_value(ep.units.RE)

    n_time = flux.shape[0]
    variable_data["FEDU_Energy"] = np.tile(energy, (n_time, 1))
    variable_data["FEDU_Alpha"] = np.tile(alpha, (n_time, 1))
    variable_data["FEDU"] = flux
    variable_data["Position"] = position

    return variable_data


if __name__ == "__main__":
    ep.run_recipe_cli(process_rbsp_mageis_electrons)
