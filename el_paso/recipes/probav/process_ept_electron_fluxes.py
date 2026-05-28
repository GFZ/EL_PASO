# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import logging
import os
import sys
import typing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import dateutil
import numpy as np
from astropy import units as u
from dotenv import load_dotenv

import el_paso as ep
from el_paso.utils import timed_function

CHI2_BAD_QUALITY_THRESHOLD = 2
EPT_ENERGY_LIMITS = [0.5, 0.6, 0.7, 0.8, 1.0, 2.4, 8.0]

logger = logging.getLogger(__name__)


load_dotenv()


@timed_function("process_ept_electron_fluxes")
def process_ept_electron_fluxes(  # noqa: D103
    raw_data_path: str | Path,
    processed_data_path: str | Path,
    start_time: datetime,
    end_time: datetime,
    num_cores: int = 32,
    bin_cadence: timedelta = timedelta(seconds=10),
    skip_existing: bool = True,  # noqa: FBT001, FBT002,
    client_id: str | None = None,
    client_secret: str | None = None,
    save_strategy: typing.Literal["gfz", "netcdf", "both"] = "netcdf",
) -> None:
    if client_id is None:
        client_id = os.environ.get("CLIENT_ID")
    if client_secret is None:
        client_secret = os.environ.get("CLIENT_SECRET")

    if client_id is None:
        msg = "Client ID not found! Either load it from environment variables or pass it as an argument."
        raise ValueError(msg)

    if client_secret is None:
        msg = "Client secret not found! Either load it from environment variables or pass it as an argument."
        raise ValueError(msg)

    data_path_stem = f"{raw_data_path}/PROBAV/YYYY/MM/"

    url = "https://sso-csr-ucl-ac-be.content.swe.s2p.esa.int/r109_111/ascii/YYYYMM/PROBAV_EPT_YYYYMMDD_L1d.dat.gz"
    rename_file_name_stem = "PROBAV_ept_YYYYMMDD_L1d.csv"

    ep.download(
        start_time,
        end_time,
        save_path=data_path_stem,
        method="esa_swe",
        file_cadence="daily",
        download_url=url,
        file_name_stem="",
        rename_file_name_stem=rename_file_name_stem,
        authentication_info=(client_id, client_secret),
        skip_existing=skip_existing,
    )

    flux_unit = typing.cast("u.Unit", (u.cm**2 * u.s * u.sr * u.MeV) ** (-1))

    extraction_infos = [
        ep.ExtractionInfo(result_key="year", name_or_column="Y", unit=u.dimensionless_unscaled, np_dtype=np.int32),
        ep.ExtractionInfo(result_key="month", name_or_column="M", unit=u.dimensionless_unscaled, np_dtype=np.int32),
        ep.ExtractionInfo(result_key="day", name_or_column="D", unit=u.dimensionless_unscaled, np_dtype=np.int32),
        ep.ExtractionInfo(result_key="hour", name_or_column="H", unit=u.dimensionless_unscaled, np_dtype=np.int32),
        ep.ExtractionInfo(result_key="minute", name_or_column="MI", unit=u.dimensionless_unscaled, np_dtype=np.int32),
        ep.ExtractionInfo(result_key="second", name_or_column="S", unit=u.dimensionless_unscaled, np_dtype=np.int32),
        ep.ExtractionInfo(
            result_key="millisecond", name_or_column="mS", unit=u.dimensionless_unscaled, np_dtype=np.int32
        ),
        ep.ExtractionInfo(result_key="flag", name_or_column="FLAG", unit=u.dimensionless_unscaled),
        ep.ExtractionInfo(result_key="chi2", name_or_column="e-Chi2", unit=u.dimensionless_unscaled),
        ep.ExtractionInfo(result_key="ch0", name_or_column="e-fl-00", unit=flux_unit),
        ep.ExtractionInfo(result_key="ch1", name_or_column="e-fl-01", unit=flux_unit),
        ep.ExtractionInfo(result_key="ch2", name_or_column="e-fl-02", unit=flux_unit),
        ep.ExtractionInfo(result_key="ch3", name_or_column="e-fl-03", unit=flux_unit),
        ep.ExtractionInfo(result_key="ch4", name_or_column="e-fl-04", unit=flux_unit),
        ep.ExtractionInfo(result_key="ch5", name_or_column="e-fl-05", unit=flux_unit),
        ep.ExtractionInfo(result_key="PA_local", name_or_column="Pitch", unit=u.deg),
        ep.ExtractionInfo(result_key="rad", name_or_column="Rad", unit=u.km),
        ep.ExtractionInfo(result_key="lon", name_or_column="Long", unit=u.deg),
        ep.ExtractionInfo(result_key="lat", name_or_column="Lat", unit=u.deg),
    ]

    variables = ep.extract_variables_from_files(
        start_time,
        end_time,
        file_cadence="daily",
        data_path=data_path_stem,
        file_name_stem=rename_file_name_stem,
        extraction_infos=extraction_infos,
        pd_read_csv_kwargs={"sep": r"\s+", "header": 24},
    )

    # create flux variable
    flux_data = np.stack(
        [
            variables["ch0"].get_data(),
            variables["ch1"].get_data(),
            variables["ch2"].get_data(),
            variables["ch3"].get_data(),
            variables["ch4"].get_data(),
            variables["ch5"].get_data(),
        ]
    ).T
    flux_data = flux_data[:, :, np.newaxis]
    variables["FEDU"] = ep.Variable(data=flux_data, original_unit=flux_unit)
    del variables["ch0"], variables["ch1"], variables["ch2"], variables["ch3"], variables["ch4"], variables["ch5"]

    # apply chi-2 quality check
    variables["FEDU"].apply_mask(variables["chi2"].get_data().astype(np.float64) < CHI2_BAD_QUALITY_THRESHOLD)
    variables["FEDU"].metadata.add_processing_note(
        f"Values with CHI2 >= {CHI2_BAD_QUALITY_THRESHOLD:0.1f} are set to NaN."
    )

    # expand PA variable
    variables["PA_local"].set_data(variables["PA_local"].get_data()[:, np.newaxis], unit="same")
    pa_arr = variables["PA_local"].get_data(u.deg)
    pa_arr = np.where(pa_arr > 90, 180 - pa_arr, pa_arr)
    variables["PA_local"].set_data(pa_arr, unit=u.deg)

    # create Epoch variable
    epoch_datetime = [
        datetime(y, m, d, h, mi, s, int(ms), tzinfo=timezone.utc)
        for (y, m, d, h, mi, s, ms) in zip(
            variables["year"].get_data(),
            variables["month"].get_data(),
            variables["day"].get_data(),
            variables["hour"].get_data(),
            variables["minute"].get_data(),
            variables["second"].get_data(),
            variables["millisecond"].get_data().astype(np.int32) * 1e3,
            strict=True,
        )
    ]
    epoch_data = [t.timestamp() for t in epoch_datetime]

    variables["Epoch"] = ep.Variable(data=np.asarray(epoch_data), original_unit=ep.units.posixtime)
    del variables["year"], variables["month"], variables["day"], variables["hour"], variables["minute"]
    del variables["second"], variables["millisecond"]

    # calculate mean of energy limits to get center energies
    energy_data = np.convolve(EPT_ENERGY_LIMITS, np.ones(2), "valid") / 2
    variables["Energy_FEDU"] = ep.Variable(data=energy_data, original_unit=u.MeV)
    variables["Energy_FEDU"].metadata.add_processing_note(
        f"Created by calculating center energies from {', '.join(map(str, EPT_ENERGY_LIMITS))}."
    )

    time_bin_methods = {
        "Energy_FEDU": ep.TimeBinMethod.Repeat,
        "rad": ep.TimeBinMethod.NanMean,
        "lat": ep.TimeBinMethod.NanMean,
        "lon": ep.TimeBinMethod.NanMean,
        "PA_local": ep.TimeBinMethod.NanMean,
        "FEDU": ep.TimeBinMethod.NanMedian,
    }

    binned_time_var = ep.processing.bin_by_time(
        variables["Epoch"], variables, time_bin_methods, bin_cadence, start_time=start_time, end_time=end_time
    )

    xsph_arr = np.stack(
        (
            variables["rad"].get_data(ep.units.RE),
            variables["lat"].get_data(u.degree),
            variables["lon"].get_data(u.degree),
        )
    ).T.astype(np.float64)
    model_coord = ep.processing.magnetic_field_utils.Coords()

    epoch_datetime = [datetime.fromtimestamp(t, tz=timezone.utc) for t in binned_time_var.get_data()]
    xgeo_arr = model_coord.transform(epoch_datetime, xsph_arr, ep.IRBEM_SYSAXIS_SPH, ep.IRBEM_SYSAXIS_GEO)
    variables["xGEO"] = ep.Variable(data=xgeo_arr, original_unit=ep.units.RE)

    del variables["rad"], variables["lon"], variables["lat"]

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
        energy_var=variables["Energy_FEDU"],
        pa_local_var=variables["PA_local"],
        particle_species="electron",
        variables_to_compute=variables_to_compute,
        irbem_options=[1, 1, 4, 4, 0],
        num_cores=num_cores,
    )

    variables |= magnetic_field_variables

    variables_to_save: dict[ep.typing.InternalName, ep.Variable] = {
        "Epoch": binned_time_var,
        "FEDU": variables["FEDU"],
        "Energy_FEDU": variables["Energy_FEDU"],
        "Alpha": variables["PA_local"],
        "Alpha_Eq": magnetic_field_variables["PA_eq_T89"],
        "R_Eq": magnetic_field_variables["R_eq_T89"],
        "MLT": magnetic_field_variables["MLT_eq_T89"],
        "L_m": magnetic_field_variables["Lm_T89"],
        "L_star": magnetic_field_variables["Lstar_T89"],
        "B_Calc": magnetic_field_variables["B_local_T89"],
        "B_Eq": magnetic_field_variables["B_eq_T89"],
        "Position": variables["xGEO"],
    }

    if save_strategy in ("gfz", "both"):
        strategy = ep.saving_strategies.GFZStrategy(
            processed_data_path,
            mission="PROBAV",
            satellite="probav_EPT",
            instrument="PROBAV",
            mag_field="T89",
            data_standard=ep.data_standards.GFZStandard(),
        )

    if save_strategy in ("netcdf", "both"):
        strategy = ep.saving_strategies.MonthlyRBStrategy(
            base_data_path=Path(processed_data_path),
            mission="PROBAV",
            satellite="probav_EPT",
            instrument="PROBAV",
            mag_field="T89",
            file_format=".nc",
            data_standard=ep.data_standards.GFZStandard(),
        )
    ep.save(variables_to_save, strategy, start_time, end_time, time_var=binned_time_var, append=True)


if __name__ == "__main__":
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Process EPT electron fulx data.")
    parser.add_argument(
        "--start_time",
        type=str,
        help="Start time in valid dateparse format. Example: YYYY-MM-DDTHH:MM:SS.",
        default=datetime(2024, 5, 8, tzinfo=timezone.utc).isoformat(),
        required=False,
    )
    parser.add_argument(
        "--end_time",
        type=str,
        help="End time in valid dateparse format. Example: YYYY-MM-DDTHH:MM:SS.",
        default=datetime(2024, 5, 8, 23, 59, 59, tzinfo=timezone.utc).isoformat(),
        required=False,
    )

    args = parser.parse_args()

    dt_start = dateutil.parser.parse(args.start_time)
    dt_end = dateutil.parser.parse(args.end_time)

    #    with tempfile.TemporaryDirectory() as tmpdir:
    process_ept_electron_fluxes(
        start_time=dt_start,
        end_time=dt_end,
        raw_data_path=".",
        processed_data_path=".",
        num_cores=64,
        bin_cadence=timedelta(seconds=10),
    )
