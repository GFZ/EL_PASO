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


def process_mageis_electrons(
    start_time: datetime,
    end_time: datetime,
    sat_str: Literal["a", "b"],
    mag_field: Literal["T89", "T96", "TS04"],
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    num_cores: int = 32,
) -> None:
    """Process RBSP ECT/MagEIS electron flux data into the EL-PASO data standard.

    Downloads the daily RBSP MagEIS level-3 electron CDF files for the given time range and
    satellite, extracts the energy/pitch-angle-resolved flux (FEDU) and position on their
    respective time grids, applies a lower bound threshold to the flux, filters out non-positive
    energy channels, truncates all variables to the requested time range, time-bins all variables
    to a 5-minute cadence (FEDU and position are binned independently due to differing time grids),
    folds the pitch angles and flux to the [0, 90] degree range, computes magnetic-field-related
    quantities (B field, equatorial pitch angle, L*, L_m, MLT, InvK, InvMu, etc.) using the position
    time grid to ensure consistent time/position pairing with IRBEM, computes the electron phase space
    density from FEDU, and saves all resulting variables using the requested saving strategy.

    Args:
        start_time (datetime): Start of the time range to process.
        end_time (datetime): End of the time range to process.
        sat_str (Literal["a", "b"]): RBSP satellite identifier ("a" or "b").
        mag_field (Literal["T89", "T96", "TS04"]): Magnetic field model used to compute the
            magnetic-field-related variables.
        raw_data_path (str | Path, optional): Directory where raw CDF files are downloaded to and
            read from. Defaults to ".".
        processed_data_path (str | Path, optional): Directory where the processed output files are
            written to. Defaults to ".".
        num_cores (int, optional): Number of CPU cores used for the magnetic field computations.
            Defaults to 32.
    """
    raw_data_path = Path(raw_data_path)
    processed_data_path = Path(processed_data_path)

    file_name_stem = "rbsp" + sat_str + "_rel04_ect-mageis-l3_YYYYMMDD_.{6}.cdf"

    ep.download(
        start_time,
        end_time,
        save_path=raw_data_path,
        download_url=f"https://spdf.gsfc.nasa.gov/pub/data/rbsp/rbsp{sat_str}/l3/ect/mageis/sectors/rel04/YYYY/",
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
    ]

    variables = ep.extract_variables_from_files(
        start_time,
        end_time,
        "daily",
        data_path=raw_data_path,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )

    variables["FEDU"].apply_thresholds_on_data(1e-21)
    variables["Energy"].apply_thresholds_on_data(0)

    # delete bad channels; these are always nan
    flux = variables["FEDU"].get_data()
    nan_rows = np.all(np.isnan(flux), axis=(0,1))
    variables["FEDU"].set_data(flux[:, :, ~nan_rows], unit="same")

    energy = variables["Energy"].get_data()
    variables["Energy"].set_data(energy[~nan_rows], unit="same")

    time_bin_methods = {
        "Energy": ep.TimeBinMethod.Repeat,  # making energy time dependent
        "FEDU": ep.TimeBinMethod.NanMedian,
        "Pitch_angle": ep.TimeBinMethod.Repeat,
    }

    binned_time_variable = ep.processing.bin_by_time(
        variables["Epoch"],
        variables=variables,
        time_bin_method_dict=time_bin_methods,
        time_binning_cadence=timedelta(minutes=5),
    )

    variables["xGEO"] = _get_xgeo_var(sat_str, start_time, end_time, raw_data_path)

    assert variables["xGEO"].get_data().shape[0] == binned_time_variable.get_data().shape[0]


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
        satellite="rbsp" + sat_str,
        instrument="mageis",
        mag_field=mag_field,
        file_format=".nc",
        data_standard=ep.data_standards.GFZStandard(),
    )
    ep.save(variables_to_save, strategy, start_time, end_time, time_var=binned_time_variable, append=False)


def _get_xgeo_var(sat_str: str, start_time: datetime, end_time: datetime, raw_data_path: str | Path) -> ep.Variable:

    file_name_stem_v8_1 = "rbsp" + sat_str + "_rel04_ect-mageis-l3_YYYYMMDD_v8.1.0.cdf"
    file_name_stem_v8_4 = "rbsp" + sat_str + "_rel04_ect-mageis-l3_YYYYMMDD_v8.4.0.cdf"

    infos_v8_1 = [
        ep.ExtractionInfo(
            result_key="xGEO",
            name_or_column="Position",
            unit=u.km,
        ),
        ep.ExtractionInfo(
            result_key="Epoch",
            name_or_column="Epoch",
            unit=ep.units.cdf_epoch,
        ),
    ]

    infos_v8_4 = [
        ep.ExtractionInfo(
            result_key="xGEO",
            name_or_column="Position",
            unit=ep.units.RE,
        ),
        ep.ExtractionInfo(
            result_key="Epoch",
            name_or_column="Epoch",
            unit=ep.units.cdf_epoch,
        ),
    ]

    variables_xgeo_v8_4 = ep.extract_variables_from_files(
        start_time,
        end_time,
        "daily",
        data_path=raw_data_path,
        file_name_stem=file_name_stem_v8_4,
        extraction_infos=infos_v8_4,
    )

    xgeo_var_v8_4 = variables_xgeo_v8_4["xGEO"]

    try:
        variables_xgeo_v8_1 = ep.extract_variables_from_files(
            start_time,
            end_time,
            "daily",
            data_path=raw_data_path,
            file_name_stem=file_name_stem_v8_1,
            extraction_infos=infos_v8_1,
        )

        xgeo_var_v8_1 = variables_xgeo_v8_1["xGEO"]
        xgeo_time_var = xgeo_var_v8_1.merge(variables_xgeo_v8_1["Epoch"], xgeo_var_v8_4, variables_xgeo_v8_4["Epoch"])

    except ValueError:
        xgeo_time_var = variables_xgeo_v8_4["Epoch"]
        xgeo_var_v8_1 = xgeo_var_v8_4


    time_bin_methods = {
        "xGEO": ep.TimeBinMethod.NanMean,
    }
    variables = {"xGEO": xgeo_var_v8_1}

    _ = ep.processing.bin_by_time(
        xgeo_time_var,
        variables=variables,
        time_bin_method_dict=time_bin_methods,
        time_binning_cadence=timedelta(minutes=5),
        start_time=start_time,
        end_time=end_time,
    )

    return variables["xGEO"]


if __name__ == "__main__":
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Process flux flux data from ECT/MagEIS instrument on VanAllenProbes."
    )
    parser.add_argument(
        "--start_time",
        type=str,
        help="Start time in valid dateparse format. Example: YYYY-MM-DDTHH:MM:SS.",
        default=datetime(2017, 9, 6, tzinfo=timezone.utc).isoformat(),
        required=False,
    )
    parser.add_argument(
        "--end_time",
        type=str,
        help="End time in valid dateparse format. Example: YYYY-MM-DDTHH:MM:SS.",
        default=datetime(2017, 9, 12, 23, 59, 59, tzinfo=timezone.utc).isoformat(),
        required=False,
    )

    args = parser.parse_args()

    dt_start = dateutil.parser.parse(args.start_time)
    dt_end = dateutil.parser.parse(args.end_time)

    for sat_str in ["b"]:
        process_mageis_electrons(
            dt_start,
            dt_end,
            sat_str,
            "T89",
            raw_data_path="raw_MagEIS",
            processed_data_path="processed_MagEIS",
            num_cores=64,
        )
