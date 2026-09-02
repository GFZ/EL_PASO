# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from astropy import units as u

import el_paso as ep
from el_paso.recipes.poes import poes_satellite_literal


def process_poes_ted_electron(
    start_time: datetime,
    end_time: datetime,
    satellite: poes_satellite_literal = "noaa15",
    mag_field: ep.typing.MagneticFieldLiteral = "T89",
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    bin_cadence: timedelta = timedelta(minutes=5),
    num_cores: int = 16,
    *,
    calculate_Lm_Lstar: bool = False,
) -> None:
    """Process POES/MetOp TED electron flux data into magnetic-field-resolved data products.

    Downloads and extracts the SEM-2 "fluxes-2sec" CDF files for the given POES/MetOp satellite,
    bins the differential electron flux, energy channels, local pitch angles, and ephemeris onto
    the given time cadence. The two local pitch angles (0 and 30 degrees relative to the
    spacecraft) are stacked into a single pitch-angle variable and folded into the 0-90 degree
    range, and the geodetic position is converted to GEO and spherical GEO coordinates. The
    magnetic field quantities (B_Calc, MLT, B_Eq, R_Eq, Alpha_Eq, Alpha_LC, Alpha_LC_Eq, and
    optionally L_star and L_m) are computed and the resulting variables are saved using a daily
    LEO saving strategy.

    Args:
        start_time (datetime): Start of the time interval to process.
        end_time (datetime): End of the time interval to process.
        satellite (poes_satellite_literal): The POES/MetOp satellite to process.
        mag_field (MagneticFieldLiteral): Magnetic field model used for the derived quantities.
        raw_data_path (str | Path): Directory where the raw downloaded data files are stored.
        processed_data_path (str | Path): Directory where the processed output files are saved.
        bin_cadence (timedelta): Time cadence used to bin the extracted variables.
        num_cores (int): Number of CPU cores used for the magnetic field computations.
        calculate_Lm_Lstar (bool): If True, additionally compute and save the L_m and
            L_star magnetic field quantities.
    """
    data_path_stem = f"{raw_data_path}/POES/{satellite}/YYYY/MM/"
    url = f"https://spdf.gsfc.nasa.gov/pub/data/noaa/{satellite}/sem2_fluxes-2sec/YYYY/"
    file_name_stem = satellite + "_poes-sem2_fluxes-2sec_YYYYMMDD_.{3}.cdf"

    ep.download(
        start_time,
        end_time,
        save_path=data_path_stem,
        file_cadence="daily",
        download_url=url,
        file_name_stem=file_name_stem,
    )

    extraction_infos = [
        ep.ExtractionInfo(
            result_key="Epoch",
            name_or_column="Epoch",
            unit=ep.units.tt2000,
        ),
        ep.ExtractionInfo(
            result_key="Energy",
            name_or_column="ted_ele_diff_energies",
            unit=u.eV,
            is_time_dependent=False,
        ),
        ep.ExtractionInfo(
            result_key="FEDU",
            name_or_column="ted_ele_flux",
            unit=(u.cm**2 * u.s * u.sr * u.eV) ** (-1),
        ),
        ep.ExtractionInfo(
            result_key="PA_local_t0",
            name_or_column="ted_alpha_0_sat",
            unit=u.deg,
        ),
        ep.ExtractionInfo(
            result_key="PA_local_t30",
            name_or_column="ted_alpha_30_sat",
            unit=u.deg,
        ),
        ep.ExtractionInfo(
            result_key="alt",
            name_or_column="alt",
            unit=u.km,
        ),
        ep.ExtractionInfo(
            result_key="lon",
            name_or_column="lon",
            unit=u.deg,
        ),
        ep.ExtractionInfo(
            result_key="lat",
            name_or_column="lat",
            unit=u.deg,
        ),
    ]

    variables = ep.extract_variables_from_files(
        start_time,
        end_time,
        file_cadence="daily",
        data_path=data_path_stem,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )

    variables["FEDU"].apply_thresholds_on_data(lower_threshold=0)

    time_bin_methods = {
        "Energy": ep.TimeBinMethod.Repeat,
        "alt": ep.TimeBinMethod.NanMean,
        "lat": ep.TimeBinMethod.NanMean,
        "lon": ep.TimeBinMethod.NanMean,
        "PA_local_t0": ep.TimeBinMethod.NanMean,
        "PA_local_t30": ep.TimeBinMethod.NanMean,
        "FEDU": ep.TimeBinMethod.NanMean,
    }

    binned_time_var = ep.processing.bin_by_time(
        variables["Epoch"], variables, time_bin_methods, bin_cadence, start_time=start_time, end_time=end_time
    )

    variables["FEDU"].transpose_data((0, 2, 1))
    # stack pitch angles
    pa_arr = np.stack((variables["PA_local_t0"].get_data(u.deg), variables["PA_local_t30"].get_data(u.deg))).T.astype(
        np.float64
    )
    pa_arr = np.where(pa_arr > 90, 180 - pa_arr, pa_arr)

    variables["PA_local"] = ep.Variable(data=pa_arr, original_unit=u.deg)

    del variables["PA_local_t0"], variables["PA_local_t30"]

    xGDZ_arr = np.stack(
        (variables["alt"].get_data(), variables["lat"].get_data(), variables["lon"].get_data())
    ).T.astype(np.float64)
    model_coord = ep.processing.magnetic_field_utils.Coords()

    # convert time_array to datetimes for transform function
    time_var_datetime = [
        datetime.fromtimestamp(t, tz=timezone.utc) for t in binned_time_var.get_data(ep.units.posixtime)
    ]
    xgeo_arr = model_coord.transform(time_var_datetime, xGDZ_arr, ep.IRBEM_SYSAXIS_GDZ, ep.IRBEM_SYSAXIS_GEO)
    variables["xGEO"] = ep.Variable(data=xgeo_arr, original_unit=ep.units.RE)

    geo_sph_arr = model_coord.transform(time_var_datetime, xGDZ_arr, ep.IRBEM_SYSAXIS_GDZ, ep.IRBEM_SYSAXIS_SPH)
    variables["geo_alt"] = ep.Variable(data=geo_sph_arr[:, 0], original_unit=ep.units.RE)
    variables["geo_lat"] = ep.Variable(data=geo_sph_arr[:, 1], original_unit=u.deg)
    variables["geo_lon"] = ep.Variable(data=geo_sph_arr[:, 2], original_unit=u.deg)

    variables_to_compute: ep.processing.VariableRequest = [
        ("B_Calc", mag_field),
        ("MLT", mag_field),
        ("B_Eq", mag_field),
        ("R_Eq", mag_field),
        ("Alpha_Eq", mag_field),
        ("Alpha_LC", mag_field),
        ("Alpha_LC_Eq", mag_field),
    ]

    if calculate_Lm_Lstar:
        variables_to_compute.extend([("L_star", mag_field), ("L_m", mag_field)])  # ty:ignore[invalid-argument-type]

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=binned_time_var,
        xgeo_var=variables["xGEO"],
        energy_var=variables["Energy"],
        pa_local_var=variables["PA_local"],
        particle_species="electron",
        variables_to_compute=variables_to_compute,
        irbem_options=ep.processing.magnetic_field_utils.IrbemOptions(),
        num_cores=num_cores,
    )

    variables |= magnetic_field_variables

    variables_to_save = {
        "Epoch": binned_time_var,
        "FEDU": variables["FEDU"],
        "Energy_FEDU": variables["Energy"],
        "Alpha": variables["PA_local"],
        "Alpha_Eq": magnetic_field_variables[f"Alpha_Eq_{mag_field}"],
        "R_Eq": magnetic_field_variables[f"R_Eq_{mag_field}"],
        "MLT": magnetic_field_variables[f"MLT_{mag_field}"],
        "B_Calc": magnetic_field_variables[f"B_Calc_{mag_field}"],
        "B_Eq": magnetic_field_variables[f"B_Eq_{mag_field}"],
        "Position": variables["xGEO"],
        "Position_geo_alt": variables["geo_alt"],
        "Position_geo_lat": variables["geo_lat"],
        "Position_geo_lon": variables["geo_lon"],
        "Alpha_LC": magnetic_field_variables[f"Alpha_LC_{mag_field}"],
        "Alpha_LC_Eq": magnetic_field_variables[f"Alpha_LC_Eq_{mag_field}"],
    }

    if calculate_Lm_Lstar:
        variables_to_save |= {
            "L_m": magnetic_field_variables[f"L_m_{mag_field}"],
            "L_star": magnetic_field_variables[f"L_star_{mag_field}"],
        }

    saving_strategy = ep.saving_strategies.DailyLEORBStrategy(
        base_data_path=Path(processed_data_path),
        mission="POES",
        satellite=satellite,
        instrument="TED",
        mag_field=mag_field,
        data_standard=ep.data_standards.GFZStandard(),
    )

    ep.save(variables_to_save, saving_strategy, start_time, end_time, time_var=binned_time_var)  # ty:ignore[invalid-argument-type]


CLI_DEFAULTS = {
    "bin_cadence": timedelta(seconds=10),
}

if __name__ == "__main__":
    ep.run_recipe_cli(process_poes_ted_electron, defaults=CLI_DEFAULTS)
