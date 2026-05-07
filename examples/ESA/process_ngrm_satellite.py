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
from typing import Literal

import dateutil
import numpy as np
from astropy import units as u
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation

import el_paso as ep
from el_paso.utils import timed_function

CHI2_BAD_QUALITY_THRESHOLD = 2
EPT_ENERGY_LIMITS = [0.5, 0.6, 0.7, 0.8, 1.0, 2.4, 8.0]

SATELLITE_TO_ID = {
    "EDRS-C": "https://swe.ssa.esa.int/hapi/data?id=spase://SSA/NumericalData/D3S/d3s_edrsc_ngrm_spid204030252_science_ep_l1_gc_v3",
    "S6-MF": "https://swe.ssa.esa.int/hapi/data?id=spase://SSA/NumericalData/D3S/d3s_sentinel6mf_ngrm_science_ep_l1_gc_v1",
    "MTG-S1": "https://swe.ssa.esa.int/hapi/data?id=spase://SSA/NumericalData/D3S/d3s_mtgs1_ngrm_science_ep_l1_gc_v1",
    "MTG-I1": "https://swe.ssa.esa.int/hapi/data?id=spase://SSA/NumericalData/D3S/d3s_mtgi1_ngrm_science_ep_l1_gc_v1",
}
NGRM_ENERGIES = [0.18, 0.27, 0.40, 0.60, 0.88, 1.30, 1.93, 2.90, 3.40, 4.00]


@timed_function("process_ngrm_electron_fluxes")
def process_ngrm_electron_fluxes(
    satellite: Literal["EDRS-C", "S6-MF", "MTG-S1", "MTG-I1"],
    raw_data_path: str | Path,
    processed_data_path: str | Path,
    irbem_lib_path: str | Path,
    start_time: datetime,
    end_time: datetime,
    num_cores: int = 32,
    bin_cadence: timedelta = timedelta(seconds=10),
    skip_existing: bool = True,  # noqa: FBT001, FBT002,
    client_id: str | None = None,
    client_secret: str | None = None,
) -> None:
    data_path_stem = f"{raw_data_path}/{satellite}/YYYY/MM/"
    file_name_stem = f"{satellite}_ngrm_YYYYMMDD_L1d.csv"

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

    ep.download(
        start_time,
        end_time,
        save_path=data_path_stem,
        file_cadence="daily",
        download_url=SATELLITE_TO_ID[satellite],
        file_name_stem="",
        rename_file_name_stem=file_name_stem,
        authentification_info=(client_id, client_secret),
        method="esa_swe",
        skip_existing=skip_existing,
    )

    flux_unit = typing.cast("u.Unit", (u.cm**2 * u.s * u.sr * u.MeV) ** (-1))

    extraction_infos = [
        ep.ExtractionInfo(result_key="Epoch_iso", name_or_column="Time", unit=u.dimensionless_unscaled),
        ep.ExtractionInfo(
            result_key="FEDO_ch1", name_or_column="Differential electron flux (0.18 MeV)", unit=flux_unit
        ),
        ep.ExtractionInfo(
            result_key="FEDO_ch2", name_or_column="Differential electron flux (0.27 MeV)", unit=flux_unit
        ),
        ep.ExtractionInfo(
            result_key="FEDO_ch3", name_or_column="Differential electron flux (0.40 MeV)", unit=flux_unit
        ),
        ep.ExtractionInfo(
            result_key="FEDO_ch4", name_or_column="Differential electron flux (0.60 MeV)", unit=flux_unit
        ),
        ep.ExtractionInfo(
            result_key="FEDO_ch5", name_or_column="Differential electron flux (0.88 MeV)", unit=flux_unit
        ),
        ep.ExtractionInfo(
            result_key="FEDO_ch6", name_or_column="Differential electron flux (1.30 MeV)", unit=flux_unit
        ),
        ep.ExtractionInfo(
            result_key="FEDO_ch7", name_or_column="Differential electron flux (1.93 MeV)", unit=flux_unit
        ),
        ep.ExtractionInfo(
            result_key="FEDO_ch8", name_or_column="Differential electron flux (2.90 MeV)", unit=flux_unit
        ),
        ep.ExtractionInfo(
            result_key="FEDO_ch9", name_or_column="Differential electron flux (3.40 MeV)", unit=flux_unit
        ),
        ep.ExtractionInfo(
            result_key="FEDO_ch10", name_or_column="Differential electron flux (4.00 MeV)", unit=flux_unit
        ),
        ep.ExtractionInfo(result_key="x_ECI", name_or_column="X", unit=u.km),
        ep.ExtractionInfo(result_key="y_ECI", name_or_column="Y", unit=u.km),
        ep.ExtractionInfo(result_key="z_ECI", name_or_column="Z", unit=u.km),
        ep.ExtractionInfo(result_key="L", name_or_column="L", unit=ep.units.RE),
    ]

    variables = ep.processing.extract_variables_from_files(
        start_time,
        end_time,
        file_cadence="daily",
        data_path=data_path_stem,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
        pd_read_csv_kwargs={"index_col": False},
    )

    time_format = "%Y-%m-%dT%H:%M:%S.%fZ" if satellite in ["MTG-I1", "MTG-S1"] else "%Y-%m-%dT%H:%M:%SZ"

    # convert iso strings to posixtime
    datetimes = ep.utils.convert_string_to_datetime(variables["Epoch_iso"], time_format=time_format)
    variables["Epoch"] = ep.Variable(
        data=np.asarray([t.timestamp() for t in datetimes]), original_unit=ep.units.posixtime
    )
    del variables["Epoch_iso"]

    # convert ECI coordinates to GEO using astropy
    coords_ECI = GCRS(
        CartesianRepresentation(
            x=variables["x_ECI"].get_data(), y=variables["y_ECI"].get_data(), z=variables["z_ECI"].get_data()
        ),
        obstime=datetimes,
    )
    coords_ITRS = coords_ECI.transform_to(ITRS(obstime=datetimes))
    xgeo_data = np.stack((coords_ITRS.x, coords_ITRS.y, coords_ITRS.z)).T

    variables["xGEO"] = ep.Variable(data=xgeo_data.value, original_unit=u.km)
    del variables["x_ECI"], variables["y_ECI"], variables["z_ECI"]

    # create flux variable
    flux_data = np.stack(
        [
            variables["FEDO_ch1"].get_data(),
            variables["FEDO_ch2"].get_data(),
            variables["FEDO_ch3"].get_data(),
            variables["FEDO_ch4"].get_data(),
            variables["FEDO_ch5"].get_data(),
            variables["FEDO_ch6"].get_data(),
            variables["FEDO_ch7"].get_data(),
            variables["FEDO_ch8"].get_data(),
            variables["FEDO_ch9"].get_data(),
            variables["FEDO_ch10"].get_data(),
        ]
    ).T.astype(np.float64)

    # convert to proper omnidirectional units
    flux_data = flux_data * 4 * np.pi

    variables["FEDO"] = ep.Variable(data=flux_data, original_unit=flux_unit * u.sr)
    for i in range(1, 11):
        del variables[f"FEDO_ch{i}"]
    variables["FEDO"].apply_thresholds_on_data(lower_threshold=0)
    variables["FEDO"].convert_to_unit((u.cm**2 * u.s * u.keV) ** (-1))

    # get energies
    variables["Energy"] = ep.Variable(data=np.asarray(NGRM_ENERGIES), original_unit=u.MeV)

    time_bin_methods = {
        "Energy": ep.TimeBinMethod.Repeat,
        "FEDO": ep.TimeBinMethod.NanMedian,
        "FEDU": ep.TimeBinMethod.NanMedian,
        "xGEO": ep.TimeBinMethod.NanMean,
        "L": ep.TimeBinMethod.NanMean,
    }

    binned_time_var = ep.processing.bin_by_time(
        variables["Epoch"], variables, time_bin_methods, bin_cadence, start_time=start_time, end_time=end_time
    )

    pa_local_data = np.tile(np.arange(5, 91, 5), (len(binned_time_var.get_data()), 1)).astype(np.float64)
    variables["PA_local_FEDU"] = ep.Variable(data=pa_local_data, original_unit=u.deg)

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
        pa_local_var=variables["PA_local_FEDU"],
        particle_species="electron",
        variables_to_compute=variables_to_compute,
        irbem_lib_path=str(irbem_lib_path),
        irbem_options=[1, 1, 4, 4, 0],
        num_cores=num_cores,
    )

    variables |= magnetic_field_variables

    FEDU_var = ep.processing.construct_pitch_angle_distribution(
        variables["FEDO"], variables["PA_local_FEDU"], magnetic_field_variables["PA_eq_T89"]
    )
    FEDU_var.apply_thresholds_on_data(lower_threshold=0)

    psd_var = ep.processing.compute_phase_space_density(FEDU_var, variables["Energy"], particle_species="electron")

    variables_to_save = {
        "time": binned_time_var,
        "flux/FEDU": FEDU_var,
        "flux/FEDO": variables["FEDO"],
        "flux/energy": variables["Energy"],
        "flux/alpha_local": variables["PA_local_FEDU"],
        "flux/alpha_eq": magnetic_field_variables["PA_eq_T89"],
        "position/T89/R0": magnetic_field_variables["R_eq_T89"],
        "position/T89/MLT": magnetic_field_variables["MLT_eq_T89"],
        "position/T89/Lm": magnetic_field_variables["Lm_T89"],
        "position/T89/Lstar": magnetic_field_variables["Lstar_T89"],
        "mag_field/T89/B_local": magnetic_field_variables["B_local_T89"],
        "mag_field/T89/B_eq": magnetic_field_variables["B_eq_T89"],
        "position/xGEO": variables["xGEO"],
        "psd/PSD": psd_var,
    }

    saving_strategy = ep.saving_strategies.MonthlyNetCDFStrategy(
        base_data_path=Path(processed_data_path) / "NGRM" / satellite.lower(),
        file_name_stem=f"{satellite.lower()}_NGRM",
        mag_field="T89",
        data_standard=ep.data_standards.PRBEMStandard(),
    )
    append = True

    ep.save(variables_to_save, saving_strategy, start_time, end_time, time_var=binned_time_var, append=append)


if __name__ == "__main__":
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Process EPT electron fulx data.")
    parser.add_argument(
        "--start_time",
        type=str,
        help="Start time in valid dateparse format. Example: YYYY-MM-DDTHH:MM:SS.",
        default=datetime(2026, 3, 18, tzinfo=timezone.utc).isoformat(),
        required=False,
    )
    parser.add_argument(
        "--end_time",
        type=str,
        help="End time in valid dateparse format. Example: YYYY-MM-DDTHH:MM:SS.",
        default=datetime(2026, 3, 20, 23, 59, 59, tzinfo=timezone.utc).isoformat(),
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
    process_ngrm_electron_fluxes(
        satellite="EDRS-C",
        start_time=dt_start,
        end_time=dt_end,
        irbem_lib_path=args.irbem_lib_path,
        raw_data_path=".",
        processed_data_path=".",
        num_cores=64,
        bin_cadence=timedelta(minutes=5),
    )
