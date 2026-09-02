# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Jaskirat Singh
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import sys
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import dateutil
import numpy as np
from astropy import units as u
from numpy.typing import NDArray

import el_paso as ep
from el_paso.processing.magnetic_field_utils import InternalFieldModel, IrbemOptions, LstarQuantity

BAD_CHANNELS = (13, 21, 22, 23, 24)

def process_rbsp_mageis_electrons(
    start_time: datetime,
    end_time: datetime,
    satellite: Literal["a", "b"] = "a",
    mag_field: Literal["T89", "T96", "TS04"] = "T89",
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    bin_cadence: timedelta = timedelta(minutes=5),
    num_cores: int = 16,
) -> None:
    """Process RBSP ECT/MagEIS electron flux data into the EL-PASO data standard.

    Downloads the daily RBSP MagEIS level-3 electron CDF files for the given time range and
    satellite, extracts the energy/pitch-angle-resolved flux (FEDU) and position on their
    respective time grids, applies a lower bound threshold to the flux, filters out non-positive
    energy channels, truncates all variables to the requested time range, time-bins all variables
    to a common cadence (FEDU and position are binned independently due to differing time grids),
    folds the pitch angles and flux to the [0, 90] degree range, computes magnetic-field-related
    quantities (B field, equatorial pitch angle, L*, L_m, MLT, InvK, InvMu, etc.) using the position
    time grid to ensure consistent time/position pairing with IRBEM, computes the electron phase space
    density from FEDU, and saves all resulting variables using the requested saving strategy.

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

    extraction_infos = [
        ep.ExtractionInfo(
            result_key="Epoch",
            name_or_column="Epoch",
            unit=ep.units.cdf_epoch,
        ),
        ep.ExtractionInfo(
            result_key="Energy",
            name_or_column="FEDU_Energy",
            unit=u.keV,
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
        data_modifier=xgeo_data_modifier,
    )

    variables["FEDU"].apply_thresholds_on_data(1e-21)
    variables["Energy"].apply_thresholds_on_data(1e-21)

    # delete bad channels; these are always nan
    flux = variables["FEDU"].get_data()

    flux = np.delete(flux, BAD_CHANNELS, axis=2)
    variables["FEDU"].set_data(flux, unit="same")

    energy = variables["Energy"].get_data()
    energy = np.delete(energy, BAD_CHANNELS, axis=0)
    variables["Energy"].set_data(energy, unit="same")

    time_bin_methods = {
        "Energy": ep.TimeBinMethod.Repeat,  # making energy time dependent
        "FEDU": ep.TimeBinMethod.NanMedian,
        "Pitch_angle": ep.TimeBinMethod.Repeat,
        "xGEO": ep.TimeBinMethod.NanMean,
    }

    binned_time_variable = ep.processing.bin_by_time(
        variables["Epoch"],
        variables=variables,
        time_bin_method_dict=time_bin_methods,
        time_binning_cadence=bin_cadence,
    )

    variables["FEDU"].transpose_data([0, 2, 1])  # making it have dimensions (time, energy, pitch angle)
    ep.processing.fold_pitch_angles_and_flux(variables["FEDU"], variables["Pitch_angle"])

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
        irbem_options=IrbemOptions(LstarQuantity.LSTAR, 1, 0, 0, InternalFieldModel.IGRF),
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

    strategy = ep.saving_strategies.MonthlyRBStrategy(
        base_data_path=Path(processed_data_path),
        mission="RBSP",
        satellite="rbsp" + satellite,
        instrument="mageis",
        mag_field=mag_field,
        file_format=".nc",
        data_standard=ep.data_standards.GFZStandard(),
    )
    ep.save(variables_to_save, strategy, start_time, end_time, time_var=binned_time_variable, append=False)


def xgeo_data_modifier(variable_data: dict[str | int, NDArray[np.generic]], extraction_infos: Iterable[ep.ExtractionInfo]) -> dict[str | int, NDArray[np.generic]]:

    xgeo_data = variable_data["Position"].astype(np.float64)

    if np.max(xgeo_data) > 100:  # unit is km
        xgeo_data = (xgeo_data * u.km).to_value(ep.units.RE)
        variable_data["Position"] = xgeo_data

    return variable_data


if __name__ == "__main__":
    ep.run_recipe_cli(process_rbsp_mageis_electrons)
