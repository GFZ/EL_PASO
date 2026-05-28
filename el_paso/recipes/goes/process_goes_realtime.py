# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from astropy import units as u
from numpy.typing import NDArray

import el_paso as ep
from el_paso.utils import timed_function

logging.captureWarnings(capture=True)
logger = logging.getLogger(__name__)


LONGITUDES_DICT: dict[Literal["primary", "secondary"], float] = {
    "primary": 72.5,  # goes19
    "secondary": 137.0,  # goes18
}


GEOCOORDS_DICT: dict[Literal["primary", "secondary"], np.ndarray] = {
    "primary": np.array([1.690, -6.391, 0]),  # goes19
    "secondary": np.array([-4.83367734, -4.50943888, 0]),  # goes18
}


def _remove_unit_from_energy_channels(energy_channels: NDArray[np.generic]) -> NDArray[np.int32]:
    """Remove the unit from the energy ranges."""
    return np.asarray([int(i.replace(" keV", "")) for i in energy_channels if "keV" in i])


@timed_function("process_goes_real_time")
def process_goes_real_time(  # noqa: D103
    sat_str: Literal["primary", "secondary"],
    processed_data_path: str | Path,
    raw_data_path: str | Path,
    start_time: datetime,
    end_time: datetime,
    save_strategy: Literal["gfz", "netcdf", "both"] = "netcdf",
    num_cores: int = 32,
    skip_existing: bool = True,  # noqa: FBT001, FBT002,
) -> None:
    # Part 1: specify source files to extract variables
    data_path_stem = f"{raw_data_path}/GOES/YYYY/MM/{sat_str}/"
    rename_file_name_stem = f"{sat_str}_YYYYMMDD.json"
    url = f"https://services.swpc.noaa.gov/json/goes/{sat_str}/"

    ep.download(
        start_time,
        end_time,
        save_path=data_path_stem,
        file_cadence="daily",
        download_url=url,
        file_name_stem="differential-electrons-3-day.json",
        rename_file_name_stem=rename_file_name_stem,
        skip_existing=skip_existing,
    )

    extraction_infos = [
        ep.ExtractionInfo(
            result_key="Epoch",
            name_or_column="time_tag",
            unit=u.dimensionless_unscaled,
        ),
        ep.ExtractionInfo(
            result_key="Energy",
            name_or_column="energy",
            unit=u.keV,
            is_time_dependent=False,
        ),
        ep.ExtractionInfo(
            result_key="FEDO",
            name_or_column="flux",
            unit=(u.cm**2 * u.s * u.keV) ** (-1),
            dependent_variables=["time_tag", "energy"],
        ),
        ep.ExtractionInfo(
            result_key="sat_id",
            name_or_column="satellite",
            unit=u.dimensionless_unscaled,
            is_time_dependent=False,
        ),
    ]

    variables = ep.extract_variables_from_files(
        start_time,
        end_time,
        file_cadence="daily",
        data_path=data_path_stem,
        file_name_stem=rename_file_name_stem,
        extraction_infos=extraction_infos,
    )

    sat_name = "goes" + str(variables["sat_id"].get_data()[0])
    logger.info(f"Processing satellite: {sat_name}")

    # parse time strings
    datetimes = ep.processing.convert_string_to_datetime(variables["Epoch"], time_format="%Y-%m-%dT%H:%M:%SZ")
    variables["Epoch"].set_data(np.asarray([t.timestamp() for t in datetimes]), ep.units.posixtime)

    # generated weighted energy channels
    variables["Energy"].set_data(_remove_unit_from_energy_channels(variables["Energy"].get_data()), "same")

    # Get the sorting order based on the row
    sorting_order = np.argsort(variables["Energy"].get_data())

    # Apply the sorting order to all rows
    variables["Energy"].set_data(variables["Energy"].get_data()[sorting_order], "same")
    variables["FEDO"].set_data(variables["FEDO"].get_data()[:, sorting_order], "same")
    variables["FEDO"].apply_thresholds_on_data(lower_threshold=0)

    time_bin_methods = {
        "FEDO": ep.TimeBinMethod.NanMedian,
        "Energy": ep.TimeBinMethod.Repeat,
    }

    binned_time_var = ep.processing.bin_by_time(
        time_variable=variables["Epoch"],
        variables=variables,
        time_bin_method_dict=time_bin_methods,
        time_binning_cadence=timedelta(minutes=5),
        start_time=start_time,
        end_time=end_time,
    )

    binned_datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in binned_time_var.get_data()]
    geo_coords = GEOCOORDS_DICT[sat_str]
    variables["xGEO"] = ep.Variable(data=np.tile(geo_coords, (len(binned_datetimes), 1)), original_unit=ep.units.RE)

    # Local pitch angles from 5 to 90 deg
    pa_local_data = np.tile(np.arange(5, 91, 5), (len(binned_time_var.get_data()), 1)).astype(np.float64)
    variables["PA_local_FEDU"] = ep.Variable(data=pa_local_data, original_unit=u.deg)

    # Calculate magnetic field variables
    variables_to_compute: ep.processing.VariableRequest = [
        ("B_local", "T89"),
        ("B_eq", "T89"),
        ("MLT", "T89"),
        ("B_eq", "T89"),
        ("R_eq", "T89"),
        ("PA_eq", "T89"),
        ("Lstar", "T89"),
        ("Lm", "T89"),
        ("invMu", "T89"),
        ("invK", "T89"),
    ]

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=binned_time_var,
        xgeo_var=variables["xGEO"],
        energy_var=variables["Energy"],
        pa_local_var=variables["PA_local_FEDU"],
        particle_species="electron",
        variables_to_compute=variables_to_compute,
        irbem_options=[1, 1, 4, 4, 0],
        num_cores=num_cores,
    )

    FEDU_var = ep.processing.construct_pitch_angle_distribution(
        variables["FEDO"], variables["PA_local_FEDU"], magnetic_field_variables["PA_eq_T89"]
    )
    FEDU_var.apply_thresholds_on_data(lower_threshold=0)

    psd_var = ep.processing.compute_phase_space_density(FEDU_var, variables["Energy"], particle_species="electron")

    vars_to_save: dict[ep.typing.InternalName, ep.Variable] = {
        "Epoch": binned_time_var,
        "FEDU": FEDU_var,
        "Position": variables["xGEO"],
        "Energy_FEDU": variables["Energy"],
        "Alpha": variables["PA_local_FEDU"],
        "PSD": psd_var,
        "Alpha_Eq": magnetic_field_variables["PA_eq_T89"],
        "MLT": magnetic_field_variables["MLT_T89"],
        "L_star": magnetic_field_variables["Lstar_T89"],
        "R_Eq": magnetic_field_variables["R_eq_T89"],
        "B_Eq": magnetic_field_variables["B_eq_T89"],
        "B_Calc": magnetic_field_variables["B_local_T89"],
        "InvMu": magnetic_field_variables["invMu_T89"],
        "InvK": magnetic_field_variables["invK_T89"],
    }

    if save_strategy in ("gfz", "both"):
        strategy = ep.saving_strategies.GFZStrategy(
            processed_data_path,
            mission="GOES",
            satellite="goes_" + sat_str,
            instrument="mps-high",
            mag_field="T89",
            data_standard=ep.data_standards.GFZStandard(),
        )

    if save_strategy in ("netcdf", "both"):
        strategy = ep.saving_strategies.MonthlyRBStrategy(
            base_data_path=Path(processed_data_path),
            mission="GOES",
            satellite="goes_" + sat_str,
            instrument="mps-high",
            mag_field="T89",
            file_format=".nc",
            data_standard=ep.data_standards.GFZStandard(),
        )

    ep.save(vars_to_save, strategy, start_time, end_time, time_var=binned_time_var, append=True)


if __name__ == "__main__":
    start_time = (datetime.now(timezone.utc)).replace(hour=0, minute=0, second=0, microsecond=0)
    end_time = start_time + timedelta(hours=1)

    for sat in ["primary", "secondary"]:
        process_goes_real_time(
            sat_str=sat,  # ty:ignore[invalid-argument-type]
            raw_data_path="goes/raw/",
            processed_data_path="goes/processed/",
            start_time=start_time,
            end_time=end_time,
            num_cores=64,
            skip_existing=True,
        )
