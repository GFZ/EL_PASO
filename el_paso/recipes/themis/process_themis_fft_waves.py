# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from astropy import units as u
from pyspedas.projects import themis

import el_paso as ep
from el_paso.pyspedas_utils import (
    build_trange,
    set_pyspedas_data_dir,
    tplot_to_bin_variable,
    tplot_to_time_variable,
    tplot_to_variable,
)
from el_paso.recipes.themis import ThemisProbe
from el_paso.recipes.themis.get_themis_pyspedas_variables import (
    get_themis_position_geo,
    get_themis_scpot_density,
)

if TYPE_CHECKING:
    from el_paso.processing.interpolate_in_time import InterpolationMethod

logger = logging.getLogger(__name__)


def themis_fft_waves_strategy(
    base_data_path: str | Path,
    satellite: ThemisProbe,
    data_standard: ep.typing.DataStandard[ep.typing.StandardName] | None = None,
) -> ep.SavingStrategy:
    """Daily NetCDF wave saving strategy for THEMIS FFT."""
    return ep.saving_strategies.DailyWaveStrategy(
        Path(base_data_path),
        "THEMIS",
        "th" + satellite,
        "FFT",
        data_standard or ep.data_standards.GFZStandard(),
    )


def process_themis_fft_waves(
    start_time: datetime,
    end_time: datetime,
    satellite: ThemisProbe = "a",
    mag_field: Literal["T89", "T96", "TS04"] = "T89",
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    bin_cadence: timedelta = timedelta(minutes=5),
    num_cores: int = 16,
    save_strategy: Literal["netcdf"] = "netcdf",
    *,
    skip_existing: bool = True,
) -> None:
    """Process THEMIS FFT wave data and save the magnetic power spectral density.

    Downloads the THEMIS Level 2 FFT, FGM, ESA and state data for the given probe and time range
    via pyspedas. The magnetic power spectral density is formed by summing the three search-coil
    magnetometer axes of the 32-bin FFT spectrum, which sets the master time cadence for this
    recipe. The electron density derived from the spacecraft potential, the total magnetic field,
    and the spacecraft position are all interpolated onto that cadence. Magnetic local time and
    the mapped equatorial radial distance are computed with IRBEM for the given `mag_field`, while
    the magnetic latitude is derived directly from the position. The electron gyrofrequency and
    its equatorial mapping are computed from the measured field, and the results are written
    with `DailyWaveStrategy`, one NetCDF file per day.

    Args:
        start_time (datetime): Start of the time range to process.
        end_time (datetime): End of the time range to process.
        satellite (Literal["a", "b", "c", "d", "e"]): THEMIS probe identifier.
        mag_field (Literal["T89", "T96", "TS04"]): The magnetic field model used to compute the
            magnetic local time and the mapped equatorial radial distance.
        raw_data_path (str | Path): Base directory pyspedas downloads the raw THEMIS files into
            and reads them from. Defaults to ".".
        processed_data_path (str | Path): Directory where the processed output files are
            written to. Defaults to ".".
        bin_cadence (timedelta): Unused by this recipe; accepted only for interface
            consistency with other EL-PASO recipes, since the wave processing uses the
            instrument's own FFT time grid instead of a configurable binning cadence.
        num_cores (int): Number of CPU cores used for the IRBEM magnetic field computations.
            Defaults to 16.
        save_strategy (Literal["netcdf"]): Unused by this recipe; accepted only for
            interface consistency with other EL-PASO recipes, since the THEMIS wave saving
            strategy factory only supports a single output format. Defaults to "netcdf".
        skip_existing (bool): Unused by this recipe; accepted only for interface consistency
            with other EL-PASO recipes, since pyspedas decides on its own whether a locally
            cached THEMIS file is still current. Defaults to True.
    """
    del bin_cadence
    del save_strategy
    del skip_existing

    set_pyspedas_data_dir("themis", raw_data_path)

    fft_vars = _get_fft_data(start_time, end_time, satellite)
    target_time_var = fft_vars["Epoch"]

    mag_vars = _get_magnetometer_data(start_time, end_time, satellite, target_time_var)
    density_vars = _get_density_data(start_time, end_time, satellite, target_time_var)
    orbit_vars = _get_orbit_vars(start_time, end_time, satellite, target_time_var, mag_field, num_cores)

    # Both the local gyrofrequency and its equatorial mapping come from the measured field.
    # Pairing a measured f_ce with the model-derived "f_ce_Eq" from
    # compute_magnetic_field_variables would make their ratio reflect field-model error
    # rather than field-line geometry.
    gyro_vars = {"f_ce": ep.processing.compute_electron_gyrofrequency(mag_vars["Bt"])}
    gyro_vars["f_ce_Eq"] = ep.processing.map_to_dipole_equator(gyro_vars["f_ce"], orbit_vars["MLat"])

    vars_to_save: dict[ep.typing.InternalName, ep.Variable] = {
        "Epoch": target_time_var,
        "Wave_frequency": fft_vars["freq"],
        "Magnetic_Power_Spectral_Density": fft_vars["BB"],
        "Number_density": density_vars["Density"],
        "B_total_obs": mag_vars["Bt"],
        "MLat": orbit_vars["MLat"],
        "MLT": orbit_vars["MLT_" + mag_field],
        "R_Eq": orbit_vars["R_Eq_" + mag_field],
        "f_ce": gyro_vars["f_ce"],
        "f_ce_Eq": gyro_vars["f_ce_Eq"],
    }

    saving_strat = themis_fft_waves_strategy(processed_data_path, satellite)

    ep.save(vars_to_save, saving_strat, start_time, end_time, time_var=target_time_var)


def _get_fft_data(
    start_time: datetime,
    end_time: datetime,
    satellite: ThemisProbe,
) -> dict[str, ep.Variable]:
    """Load the 32-bin FFT search-coil spectrum and sum it over the three SCM axes."""
    probe = str(satellite)
    axis_names = [f"th{probe}_fff_32_scm{axis}" for axis in (3,)]

    themis.fft(
        trange=build_trange(start_time, end_time),
        probe=probe,
        level="l2",
        varnames=axis_names,
    )

    psd_unit = (u.nT) ** 2 / u.Hz

    total_psd = None
    for axis_name in axis_names:
        axis_psd = np.asarray(tplot_to_variable(axis_name, psd_unit).get_data(psd_unit)).astype(np.float64)
        total_psd = axis_psd if total_psd is None else total_psd + axis_psd

    time_var = tplot_to_time_variable(axis_names[0])
    freq_var = tplot_to_bin_variable(axis_names[0], u.Hz)

    bb_var = ep.Variable(
        original_unit=psd_unit,
        data=total_psd,
        description="Total magnetic wave power spectral density.",
        processing_notes="Sum of the three search-coil axes of the THEMIS 32-bin FFT spectrum.",
    )

    bb_var.truncate(time_var, start_time, end_time)
    time_var.truncate(time_var, start_time, end_time)

    return {
        "Epoch": time_var,
        "freq": freq_var,
        "BB": bb_var,
    }


def _get_magnetometer_data(
    start_time: datetime,
    end_time: datetime,
    satellite: ThemisProbe,
    target_time_var: ep.Variable,
) -> dict[str, ep.Variable]:
    """Load the total magnetic field from FGM and interpolate it onto the FFT cadence."""
    probe = str(satellite)

    themis.fgm(
        trange=build_trange(start_time, end_time),
        probe=probe,
        level="l2",
    )

    btotal_name = f"th{probe}_fgs_btotal"

    try:
        time_var = tplot_to_time_variable(btotal_name)
        bt_data = np.asarray(tplot_to_variable(btotal_name, u.nT).get_data(u.nT)).astype(np.float64)
    except ValueError:
        # Some intervals only carry the field vector, not the precomputed magnitude.
        vector_name = f"th{probe}_fgs_gse"
        logger.info(f"'{btotal_name}' unavailable; deriving the magnitude from '{vector_name}'.")
        time_var = tplot_to_time_variable(vector_name)
        b_vec = np.asarray(tplot_to_variable(vector_name, u.nT).get_data(u.nT)).astype(np.float64)
        bt_data = np.linalg.norm(b_vec, axis=1)

    # FGM occasionally repeats timestamps across file boundaries, which the interpolation
    # below cannot handle.
    times = np.asarray(time_var.get_data(ep.units.posixtime))
    unique_mask = ~pd.Index(times).duplicated(keep="first")

    variables = {
        "Epoch": ep.Variable(original_unit=ep.units.posixtime, data=times[unique_mask]),
        "Bt": ep.Variable(
            original_unit=u.nT,
            data=bt_data[unique_mask],
            description="Observed total magnetic field at the satellite location.",
        ),
    }

    interp_methods: dict[str, InterpolationMethod] = {"Bt": "nearest"}

    _ = ep.processing.interpolate_in_time(
        variables["Epoch"],
        variables,
        interp_methods,
        target_time_variable=target_time_var,
    )

    del variables["Epoch"]

    return variables


def _get_density_data(
    start_time: datetime,
    end_time: datetime,
    satellite: ThemisProbe,
    target_time_var: ep.Variable,
) -> dict[str, ep.Variable]:
    """Derive the spacecraft-potential density and interpolate it onto the FFT cadence."""
    variables = get_themis_scpot_density(satellite, start_time, end_time)

    interp_methods: dict[str, InterpolationMethod] = {"Density": "linear"}

    _ = ep.processing.interpolate_in_time(
        variables["Epoch"],
        variables,
        interp_methods,
        target_time_variable=target_time_var,
    )

    del variables["Epoch"]

    return variables


def _get_orbit_vars(
    start_time: datetime,
    end_time: datetime,
    satellite: ThemisProbe,
    target_time_var: ep.Variable,
    mag_field: Literal["T89", "T96", "TS04"],
    num_cores: int,
) -> dict[str, ep.Variable]:
    """Interpolate the position onto the FFT cadence and derive the orbital quantities."""
    variables = get_themis_position_geo(satellite, start_time, end_time)

    interp_methods: dict[str, InterpolationMethod] = {"xGEO": "linear"}

    _ = ep.processing.interpolate_in_time(
        variables["Epoch"],
        variables,
        interp_methods,
        target_time_variable=target_time_var,
    )

    del variables["Epoch"]

    # MLat is not among the quantities IRBEM computes here, so it is derived from the position.
    variables["MLat"] = ep.processing.compute_magnetic_latitude(target_time_var, variables["xGEO"])

    variables_to_compute: ep.processing.VariableRequest = [
        ("MLT", mag_field),
        ("R_Eq", mag_field),
    ]

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=target_time_var,
        xgeo_var=variables["xGEO"],
        variables_to_compute=variables_to_compute,
        irbem_options=ep.processing.magnetic_field_utils.IrbemOptions(),
        num_cores=num_cores,
    )

    return variables | magnetic_field_variables


CLI_DEFAULTS = {
    "raw_data_path": "./data",
    "processed_data_path": "./data",
}

if __name__ == "__main__":
    ep.run_recipe_cli(process_themis_fft_waves, defaults=CLI_DEFAULTS)
