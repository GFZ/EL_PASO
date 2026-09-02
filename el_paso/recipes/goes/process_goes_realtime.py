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
    return np.asarray([int(i.replace(" keV", "")) for i in energy_channels if "keV" in i])  # ty:ignore[invalid-return-type]


@timed_function("process_goes_real_time")
def process_goes_real_time(
    start_time: datetime,
    end_time: datetime,
    satellite: Literal["primary", "secondary"] = "primary",
    mag_field: ep.typing.MagneticFieldLiteral = "T89",
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    bin_cadence: timedelta = timedelta(minutes=5),
    num_cores: int = 16,
    save_strategy: Literal["gfz", "netcdf", "both"] = "netcdf",
    skip_existing: bool = True,  # noqa: FBT001, FBT002,
) -> None:
    """Process GOES real-time differential electron flux data into pitch-angle resolved phase space densities.

    Downloads and extracts the real-time "differential-electrons-3-day" JSON product for the
    given GOES satellite, converts the timestamps and energy channel labels, sorts the energy
    channels and fluxes in ascending order, and bins the data onto a 5-minute time cadence.
    A fixed spacecraft position (from `GEOCOORDS_DICT`) and a fixed set of local pitch angles
    (5 to 90 degrees in 5-degree steps) are assigned, magnetic field quantities (B_Calc, B_Eq,
    MLT, R_Eq, Alpha_Eq, L_star, L_m, InvMu, InvK) are computed, the omnidirectional flux is
    converted to a pitch-angle distribution, and the electron phase space density is derived.
    The resulting variables are saved using the requested saving strategy.

    Args:
        start_time (datetime): Start of the time interval to process.
        end_time (datetime): End of the time interval to process.
        satellite (Literal["primary", "secondary"]): Which GOES real-time satellite to process
            ("primary" corresponds to GOES19, "secondary" to GOES18).
        mag_field (MagneticFieldLiteral): Magnetic field model used for the derived quantities.
        raw_data_path (str | Path): Directory where the raw downloaded data files are stored.
        processed_data_path (str | Path): Directory where the processed output files are saved.
        bin_cadence (timedelta): Time cadence used to bin the extracted variables.
        num_cores (int): Number of CPU cores used for the magnetic field computations.
        save_strategy (Literal["gfz", "netcdf", "both"]): Strategy used to save the
            processed data. "gfz" saves using the GFZ format, "netcdf" saves monthly NetCDF files,
            and "both" saves using both strategies.
        skip_existing (bool): If True, skip downloading files that already exist on disk.
    """
    # Part 1: specify source files to extract variables
    data_path_stem = f"{raw_data_path}/GOES/YYYY/MM/{satellite}/"
    rename_file_name_stem = f"{satellite}_YYYYMMDD.json"
    url = f"https://services.swpc.noaa.gov/json/goes/{satellite}/"

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
        time_binning_cadence=bin_cadence,
        start_time=start_time,
        end_time=end_time,
    )

    binned_datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in binned_time_var.get_data()]
    geo_coords = GEOCOORDS_DICT[satellite]
    variables["xGEO"] = ep.Variable(data=np.tile(geo_coords, (len(binned_datetimes), 1)), original_unit=ep.units.RE)

    # Local pitch angles from 5 to 90 deg
    pa_local_data = np.tile(np.arange(5, 91, 5), (len(binned_time_var.get_data()), 1)).astype(np.float64)
    variables["PA_local_FEDU"] = ep.Variable(data=pa_local_data, original_unit=u.deg)

    # Calculate magnetic field variables
    variables_to_compute: ep.processing.VariableRequest = [
        ("B_Calc", mag_field),
        ("B_Eq", mag_field),
        ("MLT", mag_field),
        ("R_Eq", mag_field),
        ("Alpha_Eq", mag_field),
        ("L_star", mag_field),
        ("L_m", mag_field),
        ("InvMu", mag_field),
        ("InvK", mag_field),
    ]

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=binned_time_var,
        xgeo_var=variables["xGEO"],
        energy_var=variables["Energy"],
        pa_local_var=variables["PA_local_FEDU"],
        particle_species="electron",
        variables_to_compute=variables_to_compute,
        irbem_options=ep.processing.magnetic_field_utils.IrbemOptions(),
        num_cores=num_cores,
    )

    FEDU_var = ep.processing.construct_pitch_angle_distribution(
        variables["FEDO"],
        variables["PA_local_FEDU"],
        magnetic_field_variables[f"Alpha_Eq_{mag_field}"],
        flux_type="spin_average",
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
        "Alpha_Eq": magnetic_field_variables[f"Alpha_Eq_{mag_field}"],
        "MLT": magnetic_field_variables[f"MLT_{mag_field}"],
        "L_star": magnetic_field_variables[f"L_star_{mag_field}"],
        "R_Eq": magnetic_field_variables[f"R_Eq_{mag_field}"],
        "B_Eq": magnetic_field_variables[f"B_Eq_{mag_field}"],
        "B_Calc": magnetic_field_variables[f"B_Calc_{mag_field}"],
        "InvMu": magnetic_field_variables[f"InvMu_{mag_field}"],
        "InvK": magnetic_field_variables[f"InvK_{mag_field}"],
    }

    if save_strategy in ("gfz", "both"):
        strategy = ep.saving_strategies.GFZStrategy(
            processed_data_path,
            mission="GOES",
            satellite="goes_" + satellite,
            instrument="mps-high",
            mag_field=mag_field,
            data_standard=ep.data_standards.GFZStandard(),
        )

    if save_strategy in ("netcdf", "both"):
        strategy = ep.saving_strategies.MonthlyRBStrategy(
            base_data_path=Path(processed_data_path),
            mission="GOES",
            satellite="goes_" + satellite,
            instrument="mps-high",
            mag_field=mag_field,
            file_format=".nc",
            data_standard=ep.data_standards.GFZStandard(),
        )

    ep.save(vars_to_save, strategy, start_time, end_time, time_var=binned_time_var, append=True)


CLI_DEFAULTS = {
    "raw_data_path": "goes/raw/",
    "processed_data_path": "goes/processed/",
}

if __name__ == "__main__":
    ep.run_recipe_cli(process_goes_real_time, defaults=CLI_DEFAULTS)
