# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Parvathy Santhini
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import dateutil.parser
import numpy as np
from astropy import units as u

import el_paso as ep

LANL_SAT = Literal[
    "ns41",
    "ns48",
    "ns53",
    "ns54",
    "ns55",
    "ns56",
    "ns57",
    "ns58",
    "ns59",
    "ns60",
    "ns61",
    "ns62",
    "ns63",
    "ns64",
    "ns65",
    "ns66",
    "ns67",
    "ns68",
    "ns69",
    "ns70",
    "ns71",
    "ns72",
    "ns73",
    "ns74",
    "ns75",
    "ns76",
    "ns77",
    "ns78",
    "ns79",
    "ns80",
    "ns81",
]

#  If you add a satellite, add it to BOTH: here, and to the Literal.

SATELLITE_ANCHOR_DATES: dict[LANL_SAT, datetime] = {
    "ns41": datetime(2000, 12, 10, tzinfo=timezone.utc),
    "ns48": datetime(2008, 3, 23, tzinfo=timezone.utc),
    "ns53": datetime(2005, 10, 2, tzinfo=timezone.utc),
    "ns54": datetime(2001, 2, 18, tzinfo=timezone.utc),
    "ns55": datetime(2007, 11, 4, tzinfo=timezone.utc),
    "ns56": datetime(2003, 2, 9, tzinfo=timezone.utc),
    "ns57": datetime(2008, 1, 13, tzinfo=timezone.utc),
    "ns58": datetime(2006, 12, 3, tzinfo=timezone.utc),
    "ns59": datetime(2004, 3, 28, tzinfo=timezone.utc),
    "ns60": datetime(2004, 7, 11, tzinfo=timezone.utc),
    "ns61": datetime(2004, 11, 14, tzinfo=timezone.utc),
    "ns62": datetime(2010, 6, 13, tzinfo=timezone.utc),
    "ns63": datetime(2011, 7, 17, tzinfo=timezone.utc),
    "ns64": datetime(2014, 2, 23, tzinfo=timezone.utc),
    "ns65": datetime(2012, 10, 14, tzinfo=timezone.utc),
    "ns66": datetime(2013, 6, 2, tzinfo=timezone.utc),
    "ns67": datetime(2014, 5, 18, tzinfo=timezone.utc),
    "ns68": datetime(2014, 8, 10, tzinfo=timezone.utc),
    "ns69": datetime(2014, 11, 16, tzinfo=timezone.utc),
    "ns70": datetime(2016, 2, 7, tzinfo=timezone.utc),
    "ns71": datetime(2015, 3, 29, tzinfo=timezone.utc),
    "ns72": datetime(2015, 7, 19, tzinfo=timezone.utc),
    "ns73": datetime(2015, 11, 1, tzinfo=timezone.utc),
    "ns74": datetime(2019, 1, 13, tzinfo=timezone.utc),
    "ns75": datetime(2020, 3, 22, tzinfo=timezone.utc),
    "ns76": datetime(2020, 7, 26, tzinfo=timezone.utc),
    "ns77": datetime(2020, 12, 6, tzinfo=timezone.utc),
    "ns78": datetime(2021, 7, 18, tzinfo=timezone.utc),
    "ns79": datetime(2023, 1, 22, tzinfo=timezone.utc),
    "ns80": datetime(2024, 12, 29, tzinfo=timezone.utc),
    "ns81": datetime(2025, 6, 1, tzinfo=timezone.utc),
}


def snap_to_weekly_grid(satellite_str: LANL_SAT, target_date: datetime) -> datetime:
    """Snap target_date onto the satellite's fixed weekly (Sunday) grid -- floor:
    nearest grid date <= target_date. Missing weeks in the archive don't shift this
    grid, so no live directory listing is needed once start_time is correctly phased.
    """  # noqa: D205
    anchor = SATELLITE_ANCHOR_DATES[satellite_str]
    n_days = (target_date - anchor).days
    n_weeks = n_days // 7
    return anchor + timedelta(weeks=n_weeks)


def weekly_cadence(curr_time: datetime) -> datetime:  # noqa: D103
    return curr_time + timedelta(days=7)


def _parse_lanl_gps_header(file_path: str) -> dict:
    header_lines = []
    with Path(file_path).open("r") as f:
        for line in f:
            if not line.startswith("#"):
                break
            header_lines.append(line[1:])
    return json.loads("".join(header_lines))


def extract_data_from_lanl_gps_ascii(file_path, extraction_infos):  # noqa: ANN001, ANN201
    """Custom extractor for LANL GPS ns41-style ASCII files (JSON header + no-header data block)."""
    header = _parse_lanl_gps_header(file_path)
    data_block = np.loadtxt(file_path, comments="#")

    data = {}
    for info in extraction_infos:
        name = info.name_or_column
        if name not in header:
            msg = f"Variable {name!r} not found in header of {file_path}!"
            raise ValueError(msg)
        start_column = header[name]["START_COLUMN"]
        (dimension,) = header[name]["DIMENSION"]

        if info.is_time_dependent:
            data[name] = (
                data_block[:, start_column]
                if dimension == 1
                else data_block[:, start_column : start_column + dimension]
            )
        else:
            # fixed grid, identical every row -- just take the first row
            data[name] = (
                data_block[0, start_column]
                if dimension == 1
                else data_block[0, start_column : start_column + dimension]
            )

    return data


def process_gps_data(
    satellite_str: LANL_SAT,
    raw_data_path: str | Path,
    processed_data_path: str | Path,
    start_time: datetime,
    end_time: datetime,
    num_cores: int = 32,
    bin_cadence: timedelta = timedelta(minutes=4),
) -> None:
    """Process LANL GPS electron flux data into magnetic-field-resolved data products.

    Downloads and extracts the omnidirectional electron differential flux from ascii files
    for the given LANL GPS satellite, converts the geocentric position to GEO coordinates,
    computes T89 magnetic field quantities (B_Calc, B_Eq, MLT, R_Eq, L_m, L_star, Alpha_Eq)
    at sampled local pitch angles (5-90 deg, step 5), and expands the omnidirectional flux
    into a full pitch-angle-resolved flux via construct_pitch_angle_distribution (sine PAD
    shape, "omni" normalization). Results are saved using a monthly RB saving strategy.

    Note: this instrument has no measured local pitch-angle telescope information (CXD
    dosimeter is omnidirectional). The local pitch angles used here are a fixed synthetic
    grid (5-90 deg), same convention as the GOES recipe, not a measured quantity -- this
    should be documented for downstream users of the saved data.

    Note on file discovery: file dates are computed via a fixed weekly (Sunday) grid
    anchored per-satellite in SATELLITE_ANCHOR_DATES (no live directory listing needed).
    Missing weeks in the real archive are handled gracefully -- download() and
    extract_variables_from_files() log and skip any date that has no corresponding file.

    Args:
        satellite_str (LANL_SAT): The LANL GPS satellite to process.
        raw_data_path (str | Path): Directory where the raw downloaded data files are stored.
        processed_data_path (str | Path): Directory where the processed output files are saved.
        start_time (datetime): Start of the time interval to process.
        end_time (datetime): End of the time interval to process.
        num_cores (int, optional): Number of CPU cores used for the magnetic field computations.
            Defaults to 32.
        bin_cadence (timedelta, optional): Time cadence used to bin the extracted variables.
            Defaults to timedelta(minutes=4).
    """
    data_path_stem = f"{raw_data_path}/GPS/{satellite_str}/YY/MM/DD"
    url = f"https://www.ngdc.noaa.gov/stp/space-weather/satellite-data/satellite-systems/lanl_gps/version_v1.10r2/{satellite_str}"
    file_name_stem = satellite_str + "_YYMMDD_v1.10.ascii"

    real_start = snap_to_weekly_grid(satellite_str, start_time)
    weekly = weekly_cadence

    download_end_time = end_time + timedelta(days=7)

    ep.download(
        real_start,
        download_end_time,
        save_path=data_path_stem,
        file_cadence=weekly,
        download_url=url,
        file_name_stem=file_name_stem,
    )

    extraction_infos = [
        ep.ExtractionInfo(result_key="decimal_day", name_or_column="decimal_day", unit=u.day),
        ep.ExtractionInfo(result_key="year", name_or_column="year", unit=u.dimensionless_unscaled),
        ep.ExtractionInfo(result_key="lat", name_or_column="Geographic_Latitude", unit=u.deg),
        ep.ExtractionInfo(result_key="lon", name_or_column="Geographic_Longitude", unit=u.deg),
        ep.ExtractionInfo(result_key="rad_re", name_or_column="Rad_Re", unit=ep.units.RE),
        ep.ExtractionInfo(
            result_key="FEDO",
            name_or_column="electron_diff_flux",
            unit=(u.cm**2 * u.s * u.sr * u.MeV) ** (-1),
        ),
        ep.ExtractionInfo(
            result_key="Energy_FEDO",
            name_or_column="electron_diff_flux_energy",
            unit=u.MeV,
            is_time_dependent=False,
        ),
    ]

    variables = ep.extract_variables_from_files(
        real_start,
        end_time,
        file_cadence=weekly,
        data_path=data_path_stem,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
        custom_extractors={".ascii": extract_data_from_lanl_gps_ascii},
    )

    variables["FEDO"].set_data(
        variables["FEDO"].get_data() * (4 * np.pi),  # ty: ignore[unsupported-operator]
        unit=(u.cm**2 * u.s * u.MeV) ** (-1),
    )

    year = variables["year"].get_data().astype(int)
    doy_frac = variables["decimal_day"].get_data()
    dt = np.array(
        [datetime(y, 1, 1, tzinfo=timezone.utc) + timedelta(days=d - 1) for y, d in zip(year, doy_frac, strict=False)]
    )
    variables["Epoch"] = ep.Variable(data=np.array([t.timestamp() for t in dt]), original_unit=ep.units.posixtime)

    variables["FEDO"].apply_thresholds_on_data(lower_threshold=0)

    time_bin_methods_pre = {
        "FEDO": ep.TimeBinMethod.NanMedian,
        "Energy_FEDO": ep.TimeBinMethod.Repeat,
        "rad_re": ep.TimeBinMethod.NanMedian,
        "lat": ep.TimeBinMethod.NanMedian,
        "lon": ep.TimeBinMethod.NanMedian,
    }

    binned_time_var = ep.processing.bin_by_time(
        variables["Epoch"],
        variables,
        time_bin_methods_pre,
        bin_cadence,
        start_time=start_time,
        end_time=end_time,
    )

    pa_local_data = np.tile(np.arange(5, 91, 5), (len(binned_time_var.get_data()), 1)).astype(np.float64)
    variables["PA_local_FEDO"] = ep.Variable(data=pa_local_data, original_unit=u.deg)

    geo_spherical = np.vstack(
        (
            variables["rad_re"].get_data(ep.units.RE),
            variables["lat"].get_data(u.deg),
            variables["lon"].get_data(u.deg),
        )
    ).T.astype(np.float64)

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in binned_time_var.get_data(ep.units.posixtime)]

    xgeo_arr = ep.processing.magnetic_field_utils.Coords().transform(
        datetimes,
        geo_spherical,
        ep.IRBEM_SYSAXIS_SPH,
        ep.IRBEM_SYSAXIS_GEO,
    )
    variables["xGEO"] = ep.Variable(data=xgeo_arr, original_unit=ep.units.RE)

    del variables["lon"]
    del variables["lat"]

    variables_to_compute: ep.processing.VariableRequest = [
        ("B_Calc", "T89"),
        ("B_Eq", "T89"),
        ("MLT", "T89"),
        ("R_Eq", "T89"),
        ("L_m", "T89"),
        ("L_star", "T89"),
        ("Alpha_Eq", "T89"),
        ("InvMu", "T89"),
        ("InvK", "T89"),
    ]

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=binned_time_var,
        xgeo_var=variables["xGEO"],
        energy_var=variables["Energy_FEDO"],
        pa_local_var=variables["PA_local_FEDO"],
        particle_species="electron",
        variables_to_compute=variables_to_compute,
        irbem_options=ep.processing.magnetic_field_utils.IrbemOptions(),
        num_cores=num_cores,
    )

    FEDU_var = ep.processing.construct_pitch_angle_distribution(
        variables["FEDO"],
        variables["PA_local_FEDO"],
        magnetic_field_variables["Alpha_Eq_T89"],
        flux_type="omni",
    )
    FEDU_var.apply_thresholds_on_data(lower_threshold=0)

    psd_var = ep.processing.compute_phase_space_density(FEDU_var, variables["Energy_FEDO"], particle_species="electron")

    variables_to_save = {
        "Epoch": binned_time_var,
        "FEDU": FEDU_var,
        "Energy_FEDU": variables["Energy_FEDO"],
        "Alpha": variables["PA_local_FEDO"],
        "R_Eq": magnetic_field_variables["R_Eq_T89"],
        "MLT": magnetic_field_variables["MLT_T89"],
        "B_Calc": magnetic_field_variables["B_Calc_T89"],
        "B_Eq": magnetic_field_variables["B_Eq_T89"],
        "L_m": magnetic_field_variables["L_m_T89"],
        "L_star": magnetic_field_variables["L_star_T89"],
        "Alpha_Eq": magnetic_field_variables["Alpha_Eq_T89"],
        "Position": variables["xGEO"],
        "PSD": psd_var,
        "InvMu": magnetic_field_variables["InvMu_T89"],
        "InvK": magnetic_field_variables["InvK_T89"],
    }

    saving_strategy = ep.saving_strategies.MonthlyRBStrategy(
        base_data_path=Path(processed_data_path),
        mission="GPS",
        satellite=satellite_str,
        instrument="cxd",
        mag_field="T89",
        data_standard=ep.data_standards.GFZStandard(),
    )

    ep.save(variables_to_save, saving_strategy, start_time, end_time, time_var=binned_time_var)  # ty:ignore[invalid-argument-type]


if __name__ == "__main__":
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Process LANL GPS data.")
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
        default=datetime(2017, 4, 30, tzinfo=timezone.utc).isoformat(),
        required=False,
    )
    parser.add_argument(
        "--satellite",
        type=str,
        default="ns41",
        choices=[*SATELLITE_ANCHOR_DATES, "all"],
        required=False,
    )
    args = parser.parse_args()

    dt_start = dateutil.parser.parse(args.start_time)
    dt_end = dateutil.parser.parse(args.end_time)

    satellites_to_run = list(SATELLITE_ANCHOR_DATES) if args.satellite == "all" else [args.satellite]

    for sat_str in satellites_to_run:
        process_gps_data(
            start_time=dt_start,
            end_time=dt_end,
            satellite_str=sat_str,
            raw_data_path=".",
            processed_data_path=".",
            num_cores=64,
            bin_cadence=timedelta(minutes=4),
        )
