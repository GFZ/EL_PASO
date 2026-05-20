# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
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

logger = logging.getLogger(__name__)


def process_efw_emfisis_density_combined(
    start_time: datetime,
    end_time: datetime,
    sat_str: Literal["a", "b"],
    irbem_lib_path: str | Path,
    mag_field: Literal["T89", "T96", "TS04"],
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    num_cores: int = 4,
    *,
    add_hiss_derived_densitites: bool = True,
    hiss_derived_densities_data_path: str | Path = ".",
) -> None:
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel(logging.INFO)

    irbem_lib_path = Path(irbem_lib_path)
    raw_data_path = Path(raw_data_path)
    processed_data_path = Path(processed_data_path)

    efw_variables = _get_efw_variables(
        start_time - timedelta(minutes=10), end_time + timedelta(minutes=10), sat_str, raw_data_path
    )
    emfisis_variables = _get_emfisis_variables(
        start_time - timedelta(minutes=10), end_time + timedelta(minutes=10), sat_str, raw_data_path
    )

    efw_time_bin_methods = {
        "xGSE": ep.TimeBinMethod.NanMean,
        "Density": ep.TimeBinMethod.NanMedian,
    }

    binned_time_variable = ep.processing.bin_by_time(
        efw_variables["Epoch"],
        variables=efw_variables,
        time_bin_method_dict=efw_time_bin_methods,
        time_binning_cadence=timedelta(minutes=1),
        start_time=start_time,
        end_time=end_time,
    )

    emfisis_time_bin_methods = {
        "Digi_type": ep.TimeBinMethod.Unique,
        "Density": ep.TimeBinMethod.NanMedian,
    }

    _ = ep.processing.bin_by_time(
        emfisis_variables["Epoch"],
        variables=emfisis_variables,
        time_bin_method_dict=emfisis_time_bin_methods,
        time_binning_cadence=timedelta(minutes=1),
        start_time=start_time,
        end_time=end_time,
    )

    digi_type_cleaned = np.asarray([s.strip() for s in emfisis_variables["Digi_type"].get_data()])
    digi_type_cleaned = digi_type_cleaned.astype("S")
    emfisis_variables["Digi_type"].set_data(digi_type_cleaned, "same")

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in binned_time_variable.get_data(ep.units.posixtime)]

    xgeo_data = ep.processing.magnetic_field_utils.Coords(lib_path=irbem_lib_path).transform(
        datetimes,
        efw_variables["xGSE"].get_data(ep.units.RE).astype(np.float64),
        ep.IRBEM_SYSAXIS_GSE,
        ep.IRBEM_SYSAXIS_GEO,
    )

    efw_variables["xGEO"] = ep.Variable(data=xgeo_data, original_unit=ep.units.RE)

    # Calculate magnetic field variables
    irbem_options = [1, 1, 4, 4, 0]

    variables_to_compute: ep.processing.VariableRequest = [
        ("MLT", mag_field),
        ("R_eq", mag_field),
        ("xGEO_eq", mag_field),
    ]

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=binned_time_variable,
        xgeo_var=efw_variables["xGEO"],
        variables_to_compute=variables_to_compute,
        irbem_lib_path=str(irbem_lib_path),
        irbem_options=irbem_options,
        num_cores=num_cores,
    )

    efw_variables["Density_mapped"] = ep.processing.compute_equatorial_plasmaspheric_density(
        efw_variables["Density"],
        efw_variables["xGEO"],
        magnetic_field_variables["xGEO_eq_" + mag_field],
        method="Denton_average",
    )
    emfisis_variables["Density_mapped"] = ep.processing.compute_equatorial_plasmaspheric_density(
        emfisis_variables["Density"],
        efw_variables["xGEO"],
        magnetic_field_variables["xGEO_eq_" + mag_field],
        method="Denton_average",
    )

    if add_hiss_derived_densitites:
        hiss_derived_densities_vars = _get_and_time_bin_hiss_derived_densities(
            hiss_derived_densities_data_path, start_time, end_time, sat_str
        )
        hiss_derived_densities_vars["Density_mapped"] = ep.processing.compute_equatorial_plasmaspheric_density(
            hiss_derived_densities_vars["Density"],
            efw_variables["xGEO"],
            magnetic_field_variables["xGEO_eq_" + mag_field],
            method="Denton_average",
        )

    variables_to_save = {
        "time": binned_time_variable,
        "density_efw_local": efw_variables["Density"],
        "density_emfisis_local": emfisis_variables["Density"],
        "density_efw_eq": efw_variables["Density_mapped"],
        "density_emfisis_eq": emfisis_variables["Density_mapped"],
        "MLT": magnetic_field_variables["MLT_" + mag_field],
        "R_eq": magnetic_field_variables["R_eq_" + mag_field],
        "density_emfisis_digi_type": emfisis_variables["Digi_type"],
        "xGEO": efw_variables["xGEO"],
        "xGEO_eq": magnetic_field_variables["xGEO_eq_" + mag_field],
    }

    if add_hiss_derived_densitites:
        variables_to_save |= {
            "density_hiss_derived_local": hiss_derived_densities_vars["Density"],
            "density_hiss_derived_eq": hiss_derived_densities_vars["Density_mapped"],
        }

    saving_strategy = ep.saving_strategies.DensityNetCDFStrategy(
        base_data_path=processed_data_path,
        file_name_stem=f"rbsp_{sat_str}_densities_combined",
        mag_field=mag_field,
        satellite="RBSP",
    )

    ep.save(variables_to_save, saving_strategy, start_time, end_time, binned_time_variable)


def _get_efw_variables(
    start_time: datetime, end_time: datetime, sat_str: Literal["a", "b"], raw_data_path: Path
) -> dict[str, ep.Variable]:
    file_name_stem = "rbsp" + sat_str + "_efw-l3_YYYYMMDD_.{3}.cdf"

    ep.download(
        start_time,
        end_time,
        save_path=raw_data_path,
        download_url=f"https://spdf.gsfc.nasa.gov/pub/data/rbsp/rbsp{sat_str}/l3/efw/YYYY/",
        file_name_stem=file_name_stem,
        file_cadence="daily",
        method="request",
        skip_existing=True,
    )

    extraction_infos = [
        ep.ExtractionInfo(
            result_key="Epoch",
            name_or_column="epoch",
            unit=ep.units.cdf_epoch,
        ),
        ep.ExtractionInfo(
            result_key="Density",
            name_or_column="density",
            unit=u.cm**-3,
        ),
        ep.ExtractionInfo(
            result_key="xGSE",
            name_or_column="position_gse",
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

    variables["xGSE"].truncate(variables["Epoch"], start_time, end_time)
    variables["Density"].truncate(variables["Epoch"], start_time, end_time)
    variables["Epoch"].truncate(variables["Epoch"], start_time, end_time)

    return variables


def _get_emfisis_variables(
    start_time: datetime, end_time: datetime, sat_str: Literal["a", "b"], raw_data_path: Path
) -> dict[str, ep.Variable]:
    file_name_stem = "rbsp-" + sat_str + "_density_emfisis-l4_YYYYMMDD_.{6,7}.cdf"

    ep.download(
        start_time,
        end_time,
        save_path=raw_data_path,
        download_url=f"https://spdf.gsfc.nasa.gov/pub/data/rbsp/rbsp{sat_str}/l4/emfisis/density/YYYY/",
        file_name_stem=file_name_stem,
        file_cadence="daily",
        method="request",
        skip_existing=True,
    )

    extraction_infos = [
        ep.ExtractionInfo(
            result_key="Epoch",
            name_or_column="Epoch",
            unit=ep.units.tt2000,
        ),
        ep.ExtractionInfo(
            result_key="Density",
            name_or_column="density",
            unit=u.cm**-3,
        ),
        ep.ExtractionInfo(
            result_key="Digi_type",
            name_or_column="digi_type",
            unit=u.dimensionless_unscaled,
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

    variables["Density"].truncate(variables["Epoch"], start_time, end_time)
    variables["Digi_type"].truncate(variables["Epoch"], start_time, end_time)
    variables["Epoch"].truncate(variables["Epoch"], start_time, end_time)

    digi_type = variables["Digi_type"].get_data()
    density = variables["Density"].get_data()

    is_fpe = ["fpe" in dt for dt in digi_type]
    density[is_fpe] = np.nan

    variables["Density"].set_data(density, "same")

    return variables


def _get_and_time_bin_hiss_derived_densities(
    hiss_derived_densities_data_path: str | Path,
    start_time: datetime,
    end_time: datetime,
    sat_str: Literal["a", "b"],
) -> dict[str, ep.Variable]:
    logger.info("Processing hiss-derived densities!")

    if sat_str == "a":
        file_name_stem = "rbsp-a_hiss_density_arase_recalibrated.txt"
    else:
        file_name_stem = "rbsp-b_hiss_density_arase_recalibrated_v2.txt"

    extraction_infos = [
        ep.ExtractionInfo(
            result_key="Epoch",
            name_or_column=0,
            unit=u.dimensionless_unscaled,
        ),
        ep.ExtractionInfo(
            result_key="Density",
            name_or_column=1,
            unit=u.cm**-3,
        ),
    ]

    hiss_derived_vars = ep.extract_variables_from_files(
        start_time=start_time,
        end_time=end_time,
        file_cadence="single_file",
        data_path=hiss_derived_densities_data_path,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
        pd_read_csv_kwargs={"skiprows": 4, "sep": "\t", "dtype": {0: str, 1: np.float64}},
    )

    datetimes = ep.processing.convert_string_to_datetime(hiss_derived_vars["Epoch"], time_format="%Y-%m-%dT%H:%M:%S.%f")
    timestamps = np.asarray([dt.timestamp() for dt in datetimes])
    hiss_derived_vars["Epoch"].set_data(timestamps, unit=ep.units.posixtime)

    time_bin_methods = {
        "Density": ep.TimeBinMethod.NanMean,
    }

    _ = ep.processing.bin_by_time(
        hiss_derived_vars["Epoch"],
        variables=hiss_derived_vars,
        time_bin_method_dict=time_bin_methods,
        time_binning_cadence=timedelta(minutes=1),
        start_time=start_time,
        end_time=end_time,
    )

    return hiss_derived_vars


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process density data from EFW and EMFISIS instrument on VanAllenProbes."
    )
    parser.add_argument(
        "start_time",
        type=str,
        nargs="?",
        help="Start time in valid dateparse format. Example: YYYY-MM-DDTHH:MM:SS.",
        default=datetime(2013, 3, 1, tzinfo=timezone.utc).isoformat(),
    )
    parser.add_argument(
        "end_time",
        type=str,
        nargs="?",
        help="End time in valid dateparse format. Example: YYYY-MM-DDTHH:MM:SS.",
        default=datetime(2013, 3, 31, 23, 59, 59, tzinfo=timezone.utc).isoformat(),
    )
    parser.add_argument(
        "irbem_lib_path",
        type=str,
        nargs="?",
        help="Path towards the compiled IRBEM library..",
        default="../../libirbem.so",
    )

    args = parser.parse_args()

    dt_start = dateutil.parser.parse(args.start_time)
    dt_end = dateutil.parser.parse(args.end_time)

    #    with tempfile.TemporaryDirectory() as tmpdir:
    for sat_str in ["b"]:
        process_efw_emfisis_density_combined(
            dt_start,
            dt_end,
            sat_str,
            args.irbem_lib_path,
            "T89",
            raw_data_path=".",
            processed_data_path=".",
            num_cores=32,
            add_hiss_derived_densitites=True,
            hiss_derived_densities_data_path="/home/bhaas/data/rbsp_hiss_densities/",
        )
