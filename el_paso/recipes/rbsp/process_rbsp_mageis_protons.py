# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Jaskirat Singh
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import dateutil
import numpy as np
from astropy import units as u

import el_paso as ep
from el_paso.processing.magnetic_field_utils import InternalFieldModel, IrbemOptions, LstarQuantity


def process_rbsp_mageis_protons(
    start_time: datetime,
    end_time: datetime,
    satellite: Literal["a", "b"] = "a",
    mag_field: Literal["T89", "T96", "TS04"] = "T89",
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    bin_cadence: timedelta = timedelta(minutes=5),
    num_cores: int = 16,
    save_strategy: Literal["gfz", "netcdf", "both"] = "both",
) -> None:
    """Process RBSP ECT/MagEIS proton flux data into the EL-PASO data standard.

    Downloads the daily RBSP MagEIS level-3 proton CDF files for the given time range and
    satellite, extracts the energy/pitch-angle-resolved flux (FPDU) and position on their
    respective time grids, applies a lower bound threshold to the flux, filters out non-positive
    energy channels, truncates all variables to the requested time range, time-bins all variables
    to a common cadence (FPDU and position are binned independently due to differing time grids),
    folds the pitch angles and flux to the [0, 90] degree range, computes magnetic-field-related
    quantities (B field, equatorial pitch angle, L*, L_m, MLT, InvK, InvMu, etc.) using the position
    time grid to ensure consistent time/position pairing with IRBEM, computes the proton phase space
    density from FPDU, and saves all resulting variables using the requested saving strategy/strategies.

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
            Defaults to 32.
        save_strategy (Literal["gfz", "netcdf", "both"]): Which saving strategy/strategies
            to use for writing the processed output. Defaults to "both".
    """
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
        skip_existing=True,
    )

    extraction_infos_fpdu = [
        ep.ExtractionInfo(
            result_key="FPDU_Epoch",
            name_or_column="FPDU_Epoch",
            unit=ep.units.cdf_epoch,
        ),
        ep.ExtractionInfo(
            result_key="Energy",
            name_or_column="FPDU_Energy",
            unit=u.keV,
            is_time_dependent=False,
        ),
        ep.ExtractionInfo(
            result_key="Pitch_angle",
            name_or_column="FPDU_Alpha",
            unit=u.deg,
            is_time_dependent=False,
        ),
        ep.ExtractionInfo(
            result_key="FPDU",
            name_or_column="FPDU",
            unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
        ),
    ]

    extraction_infos_geo = [
        ep.ExtractionInfo(
            result_key="Epoch",
            name_or_column="Epoch",
            unit=ep.units.cdf_epoch,
        ),
        ep.ExtractionInfo(
            result_key="xGEO",
            name_or_column="Position",
            unit=u.km,
        ),
    ]

    variables_fpdu = ep.extract_variables_from_files(
        start_time,
        end_time,
        "daily",
        data_path=raw_data_path,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos_fpdu,
    )

    variables_geo = ep.extract_variables_from_files(
        start_time,
        end_time,
        "daily",
        data_path=raw_data_path,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos_geo,
    )

    variables = {**variables_fpdu, **variables_geo}

    variables["xGEO"].truncate(variables["Epoch"], start_time, end_time)
    variables["FPDU"].truncate(variables["FPDU_Epoch"], start_time, end_time)
    variables["Epoch"].truncate(variables["Epoch"], start_time, end_time)
    variables["FPDU_Epoch"].truncate(variables["FPDU_Epoch"], start_time, end_time)

    variables["FPDU"].apply_thresholds_on_data(1e-21)
    variables["Energy"].apply_thresholds_on_data(0)

    # Remove non-positive energy channels and the corresponding FPDU entries
    energy_data = variables["Energy"].get_data()
    valid_mask = np.isfinite(energy_data) & (energy_data > 0)  # ty:ignore[unsupported-operator]
    variables["Energy"]._data = energy_data[valid_mask]
    fpdu_data = variables["FPDU"].get_data()
    variables["FPDU"]._data = fpdu_data[:, :, valid_mask]

    time_bin_methods = {
        "Energy": ep.TimeBinMethod.Repeat,  # making energy time dependent
        "FPDU": ep.TimeBinMethod.NanMedian,
        "Pitch_angle": ep.TimeBinMethod.Repeat,
    }

    time_bin_methods_xGEO = {
        "xGEO": ep.TimeBinMethod.NanMean,
    }

    binned_time_variable = ep.processing.bin_by_time(
        variables["FPDU_Epoch"],
        variables=variables,
        time_bin_method_dict=time_bin_methods,
        time_binning_cadence=bin_cadence,
    )

    binned_time_variable_xGEO = ep.processing.bin_by_time(
        variables["Epoch"],
        variables=variables,
        time_bin_method_dict=time_bin_methods_xGEO,
        time_binning_cadence=bin_cadence,
    )

    variables["FPDU"].transpose_data([0, 2, 1])  # making it have dimensions (time, energy, pitch angle)
    ep.processing.fold_pitch_angles_and_flux(variables["FPDU"], variables["Pitch_angle"])

    # not needed anymore
    del variables["Epoch"]
    del variables["FPDU_Epoch"]

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
        time_var=binned_time_variable_xGEO,
        xgeo_var=variables["xGEO"],
        variables_to_compute=vars_to_compute,
        irbem_options=IrbemOptions(LstarQuantity.LSTAR, 1, 4, 4, InternalFieldModel.IGRF),
        num_cores=num_cores,
        pa_local_var=variables["Pitch_angle"],
        energy_var=variables["Energy"],
        particle_species="proton",
    )

    psd_var = ep.processing.compute_phase_space_density(variables["FPDU"], variables["Energy"], "proton")

    variables_to_save: dict[ep.typing.InternalName, ep.Variable] = {
        "Epoch": binned_time_variable,
        "FPDU": variables["FPDU"],
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

    if save_strategy in ("gfz", "both"):
        strategy = ep.saving_strategies.GFZStrategy(
            processed_data_path,
            mission="RBSP",
            satellite="rbsp" + satellite,
            instrument="mageis",
            mag_field=mag_field,
            data_standard=ep.data_standards.GFZStandard(),
        )
        ep.save(variables_to_save, strategy, start_time, end_time, time_var=binned_time_variable, append=True)

    if save_strategy in ("netcdf", "both"):
        strategy = ep.saving_strategies.MonthlyRBStrategy(
            base_data_path=Path(processed_data_path),
            mission="RBSP",
            satellite="rbsp" + satellite,
            instrument="mageis",
            mag_field=mag_field,
            file_format=".nc",
            data_standard=ep.data_standards.GFZStandard(),
        )
        ep.save(variables_to_save, strategy, start_time, end_time, time_var=binned_time_variable, append=True)


CLI_DEFAULTS = {
    "raw_data_path": "raw_MagEIS",
    "processed_data_path": "processed_MagEIS",
}

if __name__ == "__main__":
    ep.run_recipe_cli(process_rbsp_mageis_protons, defaults=CLI_DEFAULTS)
