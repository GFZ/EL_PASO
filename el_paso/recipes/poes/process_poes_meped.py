# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from astropy import units as u

import el_paso as ep
from el_paso.recipes.poes import POESSatellite
from el_paso.recipes.solar_wind_indices.process_solar_wind_indices import process_solar_wind_indices

MEPED_FOV_HALF_ANGLE = 15.0  # degrees; the MEPED telescopes have a 30 degree full opening angle


def _local_pitch_angle_range(pa_local_deg: np.ndarray, half_angle_deg: float = MEPED_FOV_HALF_ANGLE) -> np.ndarray:
    """Local pitch angle range covered by the field of view of each telescope, folded into [0, 90] degrees.

    The range reaches ``half_angle_deg`` to either side of the pitch angle of the telescope axis, folded
    around 90 degrees, and is limited to [0, 90]: a telescope that looks within ``half_angle_deg`` of the
    field direction (or perpendicular to it) covers pitch angles down to 0 (or up to 90) degrees.

    Args:
        pa_local_deg (np.ndarray): Local pitch angles of the telescope axes in degrees, shape ``(n_time, n_alpha)``.
        half_angle_deg (float): Half opening angle of the telescopes in degrees.

    Returns:
        np.ndarray: Minimum and maximum local pitch angle in degrees, shape ``(n_time, n_alpha, 2)``.
    """
    folded = np.minimum(pa_local_deg, 180.0 - pa_local_deg)

    return np.stack((np.maximum(folded - half_angle_deg, 0.0), np.minimum(folded + half_angle_deg, 90.0)), axis=-1)


def _equatorial_pitch_angle_range(pa_local_range_deg: np.ndarray, b_eq: np.ndarray, b_local: np.ndarray) -> np.ndarray:
    """Map a local pitch angle range to the magnetic equator, ``sin(alpha_eq) = sin(alpha) sqrt(B_eq / B_local)``.

    The same mapping as the one of ``Alpha_Eq``, applied to the bounds of the range.

    Args:
        pa_local_range_deg (np.ndarray): Local pitch angle range in degrees, shape ``(n_time, n_alpha, 2)``.
        b_eq (np.ndarray): Equatorial magnetic field strength, shape ``(n_time,)``, same unit as ``b_local``.
        b_local (np.ndarray): Local magnetic field strength, shape ``(n_time,)``.

    Returns:
        np.ndarray: Equatorial pitch angle range in radians, shape ``(n_time, n_alpha, 2)``.
    """
    # B_eq is the minimum along the field line, so the ratio cannot exceed 1 beyond round-off
    b_ratio = np.sqrt(np.clip(b_eq / b_local, 0.0, 1.0)).reshape(-1, 1, 1)

    return np.arcsin(np.clip(np.sin(np.deg2rad(pa_local_range_deg)) * b_ratio, -1.0, 1.0))


def poes_meped_strategy(
    processed_data_path: str | Path,
    mag_field: ep.typing.MagneticFieldLiteral,
    satellite: POESSatellite,
    save_strategy: Literal["netcdf"] = "netcdf",
) -> ep.SavingStrategy:
    """Saving strategy of the POES MEPED recipe.

    Args:
        processed_data_path (str | Path): Base directory of the processed data.
        mag_field (ep.typing.MagneticFieldLiteral): Magnetic field model the data was processed with.
        satellite (POESSatellite): Satellite the data belongs to.
        save_strategy (Literal["netcdf"]): Saving strategy of the output files. Defaults to "netcdf".

    Returns:
        ep.SavingStrategy: Daily NetCDF saving strategy.
    """
    file_format: ep.typing.MFSFormats = {"netcdf": "nc"}[save_strategy]
    return ep.saving_strategies.DailyLEORBStrategy(
        Path(processed_data_path),
        "POES",
        satellite,
        "MEPED",
        mag_field,
        data_standard=ep.data_standards.GFZStandard(),
        file_format=file_format,
    )


def process_poes_meped_electron(
    start_time: datetime,
    end_time: datetime,
    satellite: POESSatellite = "noaa15",
    mag_field: ep.typing.MagneticFieldLiteral = "T89",
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    bin_cadence: timedelta = timedelta(minutes=5),
    num_cores: int = 16,
    save_strategy: Literal["netcdf"] = "netcdf",
    *,
    skip_existing: bool = True,
    calculate_Lstar: bool = True,
    save_indices_sw: bool = False,
    append: bool = False,
) -> None:
    """Process POES/MetOp MEPED electron flux data into magnetic-field-resolved data products.

    Downloads and extracts the SEM-2 "fluxes-2sec" CDF files for the given POES/MetOp satellite,
    bins the integral electron flux, energy channels, local pitch angles, and ephemeris onto the
    given time cadence. The two local pitch angles (0 and 90 degrees relative to the spacecraft)
    are stacked into a single pitch-angle variable, and the geodetic position is converted to GEO
    coordinates. Magnetic field quantities (B_Calc, B_Eq, MLT_Eq, R_Eq, Alpha_Eq, L_star, L_m, Alpha_LC,
    Alpha_LC_Eq, Alpha_DLC, Alpha_DLC_Eq) are computed and the resulting variables are saved using a daily LEO
    saving strategy. The local and equatorial pitch angle ranges (Alpha_range, Alpha_Eq_range) are the field of
    view of the telescopes, MEPED_FOV_HALF_ANGLE to either side of their axes, folded into [0, 90] degrees, and the
    geographic position is saved as Position_geo_alt, Position_geo_lat and Position_geo_lon.

    The drift loss cone is taken from the precomputed IGRF lookup table of each year (see
    `scripts/compute_dlc_lookup_table.py`); years without a table are traced per time step, which is slow.
    Alpha_DLC is calculated in the IGRF field without external field model, independent of `mag_field`, while
    Alpha_DLC_Eq is mapped to the equator with `mag_field`, consistently with Alpha_Eq.

    Args:
        start_time (datetime): Start of the time interval to process.
        end_time (datetime): End of the time interval to process.
        satellite (POESSatellite): The POES/MetOp satellite to process.
        mag_field (MagneticFieldLiteral): Magnetic field model used for the derived quantities.
        raw_data_path (str | Path): Directory where the raw downloaded data files are stored.
        processed_data_path (str | Path): Directory where the processed output files are saved.
        bin_cadence (timedelta): Time cadence used to bin the extracted variables.
        num_cores (int): Number of CPU cores used for the magnetic field computations.
        save_strategy (Literal["netcdf"]): The saving strategy used to write the processed
                                                    data. POES MEPED data only supports a single
                                                    netCDF-based strategy.
        skip_existing (bool): If True, skip downloading files that already exist locally.
                                            Defaults to True.
        calculate_Lstar (bool): If True, also compute the L* magnetic field quantity.
            Defaults to True.
        save_indices_sw (bool): If True, also save all solar wind indices and parameters,
            independent of this recipe's own output. Defaults to False.
        append (bool): If True, append to existing output files instead of overwriting them.
            Defaults to False.
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
        skip_existing=skip_existing,
    )

    ep.exit_if_download_only()

    extraction_infos = [
        ep.ExtractionInfo(
            result_key="Epoch",
            name_or_column="Epoch",
            unit=ep.units.tt2000,
        ),
        ep.ExtractionInfo(
            result_key="Energy",
            name_or_column="mep_ele_int_energies",
            unit=u.keV,
            is_time_dependent=False,
        ),
        ep.ExtractionInfo(
            result_key="FEIU",
            name_or_column="mep_ele_flux",
            unit=(u.cm**2 * u.s * u.sr) ** (-1),
        ),
        ep.ExtractionInfo(
            result_key="PA_local_t0",
            name_or_column="meped_alpha_0_sat",
            unit=u.deg,
        ),
        ep.ExtractionInfo(
            result_key="PA_local_t90",
            name_or_column="meped_alpha_90_sat",
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
    # fold pitch angles around 90 degrees
    pa_arr = np.where(pa_arr > 90, 180 - pa_arr, pa_arr)

    variables["PA_local"] = ep.Variable(data=pa_arr, original_unit=u.deg)
    variables["PA_local_range"] = ep.Variable(data=_local_pitch_angle_range(pa_arr), original_unit=u.deg)

    del variables["PA_local_t0"]
    del variables["PA_local_t90"]

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

    del variables["alt"]
    del variables["lon"]
    del variables["lat"]

    variables_to_compute: ep.processing.VariableRequest = [
        ("B_Calc", mag_field),
        ("MLT_Eq", mag_field),
        ("B_Eq", mag_field),
        ("R_Eq", mag_field),
        ("Alpha_Eq", mag_field),
        ("L_m", mag_field),
        ("Alpha_LC", mag_field),
        ("Alpha_LC_Eq", mag_field),
        ("Alpha_DLC", "Dip"),
        ("Alpha_DLC_Eq", mag_field),
    ]

    if calculate_Lstar:
        variables_to_compute.append(("L_star", mag_field))

    irbem_options = ep.processing.magnetic_field_utils.IrbemOptions(
        lstar_quantity=ep.processing.magnetic_field_utils.LstarQuantity.LSTAR
        if calculate_Lstar
        else ep.processing.magnetic_field_utils.LstarQuantity.NONE,
    )

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=binned_time_var,
        xgeo_var=variables["xGEO"],
        energy_var=variables["Energy"],
        pa_local_var=variables["PA_local"],
        particle_species="electron",
        variables_to_compute=variables_to_compute,
        irbem_options=irbem_options,
        num_cores=num_cores,
    )

    variables |= magnetic_field_variables

    variables["Alpha_Eq_range"] = ep.Variable(
        data=_equatorial_pitch_angle_range(
            variables["PA_local_range"].get_data(u.deg),
            magnetic_field_variables[f"B_Eq_{mag_field}"].get_data(u.nT),
            magnetic_field_variables[f"B_Calc_{mag_field}"].get_data(u.nT),
        ),
        original_unit=u.rad,
    )

    variables_to_save: ep.typing.VariablesDict = {
        "Epoch": binned_time_var,
        "FEIU": variables["FEIU"],
        "Energy_FEIU": variables["Energy"],
        "Alpha": variables["PA_local"],
        "Alpha_range": variables["PA_local_range"],
        "Alpha_Eq": magnetic_field_variables[f"Alpha_Eq_{mag_field}"],
        "Alpha_Eq_range": variables["Alpha_Eq_range"],
        "R_Eq": magnetic_field_variables[f"R_Eq_{mag_field}"],
        "L_m": magnetic_field_variables[f"L_m_{mag_field}"],
        "L_star": magnetic_field_variables[f"L_star_{mag_field}"],
        "MLT": magnetic_field_variables[f"MLT_Eq_{mag_field}"],
        "B_Calc": magnetic_field_variables[f"B_Calc_{mag_field}"],
        "B_Eq": magnetic_field_variables[f"B_Eq_{mag_field}"],
        "Position": variables["xGEO"],
        "Position_geo_alt": variables["geo_alt"],
        "Position_geo_lat": variables["geo_lat"],
        "Position_geo_lon": variables["geo_lon"],
        "Alpha_LC": magnetic_field_variables[f"Alpha_LC_{mag_field}"],
        "Alpha_LC_Eq": magnetic_field_variables[f"Alpha_LC_Eq_{mag_field}"],
        "Alpha_DLC": magnetic_field_variables["Alpha_DLC_Dip"],
        "Alpha_DLC_Eq": magnetic_field_variables[f"Alpha_DLC_Eq_{mag_field}"],
    }

    saving_strategy = poes_meped_strategy(processed_data_path, mag_field, satellite, save_strategy)

    ep.save(variables_to_save, saving_strategy, start_time, end_time, time_var=binned_time_var, append=append)

    if save_indices_sw:
        process_solar_wind_indices(
            start_time,
            end_time,
            processed_data_path=processed_data_path,
            bin_cadence=bin_cadence,
        )


CLI_DEFAULTS = {
    "bin_cadence": timedelta(seconds=10),
}

PLOT_DEFAULTS = {
    "panels": ["Kp", "FEIU"],
    "FEIU": {
        "pitch_angle": 90 * u.deg,
        "energies": [100, 500, 1000] * u.keV,  # placeholder energies, to be replaced by the instrument channels
        "along": "L_m",
    },
}

if __name__ == "__main__":
    ep.run_recipe_cli(
        process_poes_meped_electron, defaults=CLI_DEFAULTS, strategy=poes_meped_strategy, plot_defaults=PLOT_DEFAULTS
    )
