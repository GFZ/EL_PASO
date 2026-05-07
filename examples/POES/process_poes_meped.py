# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, get_args

import dateutil
import numpy as np
from astropy import units as u

import el_paso as ep

poes_satellite_literal = Literal[
    "metop1",
    "metop2",
    "metop3",
    "noaa05",
    "noaa06",
    "noaa07",
    "noaa08",
    "noaa10",
    "noaa12",
    "noaa14",
    "noaa15",
    "noaa16",
    "noaa17",
    "noaa18",
    "noaa19",
]


def process_poes_meped_electron(
    satellite_str: poes_satellite_literal,
    raw_data_path: str | Path,
    processed_data_path: str | Path,
    irbem_lib_path: str | Path,
    start_time: datetime,
    end_time: datetime,
    num_cores: int = 32,
    bin_cadence: timedelta = timedelta(minutes=5),
) -> None:
    data_path_stem = f"{raw_data_path}/YYYY/MM/{satellite_str}/"
    url = f"https://spdf.gsfc.nasa.gov/pub/data/noaa/{satellite_str}/sem2_fluxes-2sec/YYYY/"
    file_name_stem = satellite_str + "_poes-sem2_fluxes-2sec_YYYYMMDD_.{3}.cdf"

    ep.download(
        start_time,
        end_time,
        save_path=data_path_stem,
        file_cadence="daily",
        download_url=url,
        file_name_stem=file_name_stem,
    )

    extraction_infos = [
        ep.processing.ExtractionInfo(
            result_key="Epoch",
            name_or_column="Epoch",
            unit=ep.units.tt2000,
        ),
        ep.processing.ExtractionInfo(
            result_key="Energy",
            name_or_column="mep_ele_int_energies",
            unit=u.keV,
            is_time_dependent=False,
        ),
        ep.processing.ExtractionInfo(
            result_key="FEIU",
            name_or_column="mep_ele_flux",
            unit=(u.cm**2 * u.s * u.sr) ** (-1),
        ),
        ep.processing.ExtractionInfo(
            result_key="PA_local_t0",
            name_or_column="meped_alpha_0_sat",
            unit=u.deg,
        ),
        ep.processing.ExtractionInfo(
            result_key="PA_local_t90",
            name_or_column="meped_alpha_90_sat",
            unit=u.deg,
        ),
        ep.processing.ExtractionInfo(
            result_key="alt",
            name_or_column="alt",
            unit=u.km,
        ),
        ep.processing.ExtractionInfo(
            result_key="lon",
            name_or_column="lon",
            unit=u.deg,
        ),
        ep.processing.ExtractionInfo(
            result_key="lat",
            name_or_column="lat",
            unit=u.deg,
        ),
    ]

    variables = ep.processing.extract_variables_from_files(
        start_time,
        end_time,
        file_cadence="daily",
        data_path=data_path_stem,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )

    time_bin_methods = {
        "Energy": ep.TimeBinMethod.Repeat,
        "alt": ep.TimeBinMethod.NanMean,
        "lat": ep.TimeBinMethod.NanMean,
        "lon": ep.TimeBinMethod.NanMean,
        "PA_local_t0": ep.TimeBinMethod.NanMean,
        "PA_local_t90": ep.TimeBinMethod.NanMean,
        "FEIU": ep.TimeBinMethod.NanMedian,
    }

    binned_time_var = ep.processing.bin_by_time(
        variables["Epoch"], variables, time_bin_methods, bin_cadence, start_time=start_time, end_time=end_time
    )

    variables["FEIU"].apply_thresholds_on_data(lower_threshold=0)
    variables["FEIU"].transpose_data((0, 2, 1))

    # stack pitch angles
    pa_arr = np.stack((variables["PA_local_t0"].get_data(u.deg), variables["PA_local_t90"].get_data(u.deg))).T.astype(
        np.float64
    )
    variables["PA_local"] = ep.Variable(data=pa_arr, original_unit=u.deg)

    del variables["PA_local_t0"]
    del variables["PA_local_t90"]

    xGDZ_arr = np.stack(
        (variables["alt"].get_data(), variables["lat"].get_data(), variables["lon"].get_data())
    ).T.astype(np.float64)
    model_coord = ep.processing.magnetic_field_utils.Coords(lib_path=irbem_lib_path)

    # convert time_array to datetimes for transform function
    time_var_datetime = [
        datetime.fromtimestamp(t, tz=timezone.utc) for t in binned_time_var.get_data(ep.units.posixtime)
    ]
    xgeo_arr = model_coord.transform(time_var_datetime, xGDZ_arr, ep.IRBEM_SYSAXIS_GDZ, ep.IRBEM_SYSAXIS_GEO)
    variables["xGEO"] = ep.Variable(data=xgeo_arr, original_unit=ep.units.RE)

    del variables["alt"]
    del variables["lon"]
    del variables["lat"]

    variables_to_compute: ep.processing.VariableRequest = [
        ("B_local", "T89"),
        ("B_eq", "T89"),
        ("MLT_eq", "T89"),
        ("B_eq", "T89"),
        ("R_eq", "T89"),
        ("PA_eq", "T89"),
        ("Lstar", "T89"),
        ("Lm", "T89"),
    ]

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=binned_time_var,
        xgeo_var=variables["xGEO"],
        energy_var=variables["Energy"],
        pa_local_var=variables["PA_local"],
        particle_species="electron",
        variables_to_compute=variables_to_compute,
        irbem_lib_path=str(irbem_lib_path),
        irbem_options=[1, 1, 4, 4, 0],
        num_cores=num_cores,
    )

    variables |= magnetic_field_variables

    variables_to_save = {
        "time": binned_time_var,
        "flux/FEIU": variables["FEIU"],
        "flux/energy": variables["Energy"],
        "flux/alpha_local": variables["PA_local"],
        "flux/alpha_eq": magnetic_field_variables["PA_eq_T89"],
        "position/T89/R0": magnetic_field_variables["R_eq_T89"],
        "position/T89/MLT": magnetic_field_variables["MLT_eq_T89"],
        "position/T89/Lm": magnetic_field_variables["Lm_T89"],
        "position/T89/Lstar": magnetic_field_variables["Lstar_T89"],
        "mag_field/T89/B_local": magnetic_field_variables["B_local_T89"],
        "mag_field/T89/B_eq": magnetic_field_variables["B_eq_T89"],
        "position/xGEO": variables["xGEO"],
    }

    saving_strategy = ep.saving_strategies.MonthlyNetCDFStrategy(
        base_data_path=Path(processed_data_path) / "POES" / sat_str,
        file_name_stem=f"{sat_str}_MEPED",
        mag_field="T89",
        data_standard=ep.data_standards.PRBEMStandard(),
    )
    append = False

    ep.save(variables_to_save, saving_strategy, start_time, end_time, time_var=binned_time_var, append=append)


if __name__ == "__main__":
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Process MEPED data from POES satellites.")
    parser.add_argument(
        "--start_time",
        type=str,
        help="Start time in valid dateparse format. Example: YYYY-MM-DDTHH:MM:SS.",
        default=datetime(2025, 9, 28, tzinfo=timezone.utc).isoformat(),
        required=False,
    )
    parser.add_argument(
        "--end_time",
        type=str,
        help="End time in valid dateparse format. Example: YYYY-MM-DDTHH:MM:SS.",
        default=datetime(2025, 10, 6, 23, 59, 59, tzinfo=timezone.utc).isoformat(),
        required=False,
    )
    parser.add_argument(
        "--irbem_lib_path",
        type=str,
        help="Path towards the compiled IRBEM library..",
        default="../../IRBEM/libirbem.so",
        required=False,
    )

    args = parser.parse_args()

    dt_start = dateutil.parser.parse(args.start_time)
    dt_end = dateutil.parser.parse(args.end_time)

    #    with tempfile.TemporaryDirectory() as tmpdir:
    for sat_str in get_args(poes_satellite_literal):
        try:
            process_poes_meped_electron(
                start_time=dt_start,
                end_time=dt_end,
                satellite_str=sat_str,
                irbem_lib_path=args.irbem_lib_path,
                raw_data_path=".",
                processed_data_path=".",
                num_cores=64,
                bin_cadence=timedelta(seconds=10),
            )
        except:  # noqa: E722, S112
            continue
