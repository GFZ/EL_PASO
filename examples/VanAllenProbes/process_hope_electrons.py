# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import dateutil
from astropy import units as u

import el_paso as ep

if TYPE_CHECKING:
    from el_paso.processing import MagFieldVarTypes


def process_hope_electrons(
    start_time: datetime,
    end_time: datetime,
    sat_str: Literal["a", "b"],
    irbem_lib_path: str | Path,
    mag_field: Literal["T89", "T96", "TS04"] | list[Literal["T89", "T96", "TS04"]],
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    num_cores: int = 4,
) -> None:
    if not isinstance(mag_field, list):
        mag_field = [mag_field]

    irbem_lib_path = Path(irbem_lib_path)
    raw_data_path = Path(raw_data_path)
    processed_data_path = Path(processed_data_path)

    file_name_stem = "rbsp" + sat_str + "_rel04_ect-hope-pa-l3_YYYYMMDD_.{6}.cdf"

    ep.download(
        start_time,
        end_time,
        save_path=raw_data_path,
        download_url=f"https://spdf.gsfc.nasa.gov/pub/data/rbsp/rbsp{sat_str}/l3/ect/hope/pitchangle/rel04/YYYY/",
        file_name_stem=file_name_stem,
        file_cadence="daily",
        method="request",
        skip_existing=True,
    )

    extraction_infos = [
        ep.ExtractionInfo(
            result_key="Epoch",
            name_or_column="Epoch_Ele",
            unit=ep.units.cdf_epoch,
        ),
        ep.ExtractionInfo(
            result_key="Energy",
            name_or_column="HOPE_ENERGY_Ele",
            unit=u.eV,
        ),
        ep.ExtractionInfo(
            result_key="Pitch_angle",
            name_or_column="PITCH_ANGLE",
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
            name_or_column="Position_Ele",
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

    variables["xGEO"].truncate(variables["Epoch"], start_time, end_time)
    variables["Energy"].truncate(variables["Epoch"], start_time, end_time)
    variables["FEDU"].truncate(variables["Epoch"], start_time, end_time)
    variables["Epoch"].truncate(variables["Epoch"], start_time, end_time)

    time_bin_methods = {
        "xGEO": ep.TimeBinMethod.NanMean,
        "Energy": ep.TimeBinMethod.NanMedian,
        "FEDU": ep.TimeBinMethod.NanMedian,
        "Pitch_angle": ep.TimeBinMethod.Repeat,
    }

    binned_time_variable = ep.processing.bin_by_time(
        variables["Epoch"],
        variables=variables,
        time_bin_method_dict=time_bin_methods,
        time_binning_cadence=timedelta(minutes=5),
    )

    variables["FEDU"].transpose_data([0, 2, 1])  # making it having dimensions (time, energy, pitch angle)
    ep.processing.fold_pitch_angles_and_flux(variables["FEDU"], variables["Pitch_angle"])

    # not needed anymore
    del variables["Epoch"]

    # Calculate magnetic field variables
    irbem_options = [1, 1, 4, 4, 0]

    vars_to_compute: list[tuple[MagFieldVarTypes, str]] = []

    for single_mag_field in mag_field:
        vars_to_compute.extend(
            [
                ("B_local", single_mag_field),
                ("MLT", single_mag_field),
                ("B_eq", single_mag_field),
                ("R_eq", single_mag_field),
                ("PA_eq", single_mag_field),
                ("Lstar", single_mag_field),
                ("Lm", single_mag_field),
                ("invK", single_mag_field),
                ("invMu", single_mag_field),
            ]
        )

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=binned_time_variable,
        xgeo_var=variables["xGEO"],
        variables_to_compute=vars_to_compute,
        irbem_lib_path=str(irbem_lib_path),
        irbem_options=irbem_options,
        num_cores=num_cores,
        pa_local_var=variables["Pitch_angle"],
        energy_var=variables["Energy"],
        particle_species="electron",
    )

    psd_var = ep.processing.compute_phase_space_density(variables["FEDU"], variables["Energy"], "electron")

    for single_mag_field in mag_field:
        saving_strategy = ep.saving_strategies.GFZStrategy(
            processed_data_path, "RBSP", "rbsp" + sat_str, "hope", single_mag_field, ".mat"
        )

        variables_to_save = {
            "time": binned_time_variable,
            "Flux": variables["FEDU"],
            "xGEO": variables["xGEO"],
            "energy_channels": variables["Energy"],
            "alpha_local": variables["Pitch_angle"],
            "alpha_eq_model": magnetic_field_variables["PA_eq_" + single_mag_field],
            "Lstar": magnetic_field_variables["Lstar_" + single_mag_field],
            "MLT": magnetic_field_variables["MLT_" + single_mag_field],
            "Lm": magnetic_field_variables["Lm_" + single_mag_field],
            "R0": magnetic_field_variables["R_eq_" + single_mag_field],
            "InvK": magnetic_field_variables["invK_" + single_mag_field],
            "InvMu": magnetic_field_variables["invMu_" + single_mag_field],
            "B_eq": magnetic_field_variables["B_eq_" + single_mag_field],
            "B_local": magnetic_field_variables["B_local_" + single_mag_field],
            "PSD": psd_var,
        }

        ep.save(variables_to_save, saving_strategy, start_time, end_time, binned_time_variable)

    saving_strategy = ep.saving_strategies.MonthlyNetCDFStrategy(
        base_data_path=processed_data_path, file_name_stem=f"rbsp{sat_str}_hope", mag_field=mag_field
    )

    variables_to_save = {
        "time": binned_time_variable,
        "flux/FEDU": variables["FEDU"],
        "position/xGEO": variables["xGEO"],
        "flux/energy": variables["Energy"],
        "flux/alpha_local": variables["Pitch_angle"],
        "psd/PSD": psd_var,
    }

    for single_mag_field in mag_field:
        variables_to_save |= {
            "flux/alpha_eq": magnetic_field_variables["PA_eq_" + single_mag_field],
            f"position/{single_mag_field}/Lstar": magnetic_field_variables["Lstar_" + single_mag_field],
            f"position/{single_mag_field}/MLT": magnetic_field_variables["MLT_" + single_mag_field],
            f"position/{single_mag_field}/Lm": magnetic_field_variables["Lm_" + single_mag_field],
            f"position/{single_mag_field}/R0": magnetic_field_variables["R_eq_" + single_mag_field],
            f"psd/{single_mag_field}/inv_K": magnetic_field_variables["invK_" + single_mag_field],
            f"psd/{single_mag_field}/inv_mu": magnetic_field_variables["invMu_" + single_mag_field],
            f"mag_field/{single_mag_field}/B_eq": magnetic_field_variables["B_eq_" + single_mag_field],
            f"mag_field/{single_mag_field}/B_local": magnetic_field_variables["B_local_" + single_mag_field],
        }

    ep.save(variables_to_save, saving_strategy, start_time, end_time, binned_time_variable)


if __name__ == "__main__":
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Process density data from EFW and EMFISIS instrument on VanAllenProbes."
    )
    parser.add_argument(
        "--start_time",
        type=str,
        help="Start time in valid dateparse format. Example: YYYY-MM-DDTHH:MM:SS.",
        default=datetime(2017, 4, 1, tzinfo=timezone.utc).isoformat(),
        required=False,
    )
    parser.add_argument(
        "--end_time",
        type=str,
        help="End time in valid dateparse format. Example: YYYY-MM-DDTHH:MM:SS.",
        default=datetime(2017, 4, 1, 4, 30, 59, tzinfo=timezone.utc).isoformat(),
        required=False,
    )
    parser.add_argument(
        "--irbem_lib_path",
        type=str,
        help="Path towards the compiled IRBEM library..",
        default="../../libirbem.so",
        required=False,
    )

    args = parser.parse_args()

    dt_start = dateutil.parser.parse(args.start_time)
    dt_end = dateutil.parser.parse(args.end_time)

    #    with tempfile.TemporaryDirectory() as tmpdir:
    for sat_str in ["a", "b"]:
        process_hope_electrons(
            dt_start,
            dt_end,
            sat_str,
            args.irbem_lib_path,
            ["T89", "OP77"],
            raw_data_path=".",
            processed_data_path=".",
            num_cores=8,
        )
