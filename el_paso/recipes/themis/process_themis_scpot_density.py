# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import el_paso as ep
from el_paso.pyspedas_utils import set_pyspedas_data_dir
from el_paso.recipes.themis import ThemisProbe
from el_paso.recipes.themis.get_themis_pyspedas_variables import (
    get_themis_position_geo,
    get_themis_scpot_density,
)

logger = logging.getLogger(__name__)

_MINIMUM_PHYSICAL_DENSITY = 1e-21
"""Lower threshold in cm^-3, below which a density sample is treated as a fill value."""


def themis_scpot_density_strategy(
    base_data_path: str | Path,
    satellite: ThemisProbe,
    mag_field: ep.typing.MagneticFieldLiteral,
) -> ep.SavingStrategy:
    """Monthly NetCDF density saving strategy for the THEMIS spacecraft-potential density."""
    return ep.saving_strategies.MonthlyDensityStrategy(
        base_data_path=base_data_path,
        mission="THEMIS",
        satellite="th" + satellite,
        instrument="SCPOT",
        mag_field=mag_field,
    )


def process_themis_scpot_density(
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
    """Process the THEMIS spacecraft-potential electron density and save the mapped equatorial density.

    Downloads the THEMIS ESA Level 2 moments and the Level 1 state data for the given probe and
    time range via pyspedas, derives the electron density from the spacecraft potential, and
    time-bins both the density and the position onto a common cadence. A lower density threshold
    removes fill values, the position is rotated to GEO coordinates, the magnetic-field-related
    quantities (MLT, Lstar, equatorial radial distance and equatorial position) are computed via
    IRBEM for the given `mag_field`, the local density is mapped to the magnetic equator, and the
    results are saved with a `MonthlyDensityStrategy`.

    Args:
        start_time (datetime): Start of the time range to process.
        end_time (datetime): End of the time range to process.
        satellite (Literal["a", "b", "c", "d", "e"]): THEMIS probe identifier.
        mag_field (Literal["T89", "T96", "TS04"]): The magnetic field model used for the
            magnetic-field-related output variables and the equatorial density mapping.
        raw_data_path (str | Path): Base directory pyspedas downloads the raw THEMIS files into
            and reads them from. Defaults to ".".
        processed_data_path (str | Path): Base directory where the processed output data is
            saved. Defaults to ".".
        bin_cadence (timedelta): Time binning cadence applied to the density and the position.
        num_cores (int): Number of CPU cores used for the IRBEM magnetic field computations.
            Defaults to 16.
        save_strategy (Literal["netcdf"]): Unused by this recipe; accepted only for interface
            consistency with other EL-PASO recipes, since the THEMIS density saving strategy
            factory only supports a single output format. Defaults to "netcdf".
        skip_existing (bool): If True, let pyspedas reuse raw files that already exist locally
            instead of re-downloading them. Defaults to True.
    """
    del skip_existing
    del save_strategy

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel(logging.INFO)

    set_pyspedas_data_dir("themis", raw_data_path)

    density_variables = get_themis_scpot_density(satellite, start_time, end_time)
    orbit_variables = get_themis_position_geo(satellite, start_time, end_time)

    # Both calls bin onto the same grid (same range and cadence), so either returned time
    # variable describes the binned density and the binned position alike.
    _ = ep.processing.bin_by_time(
        density_variables["Epoch"],
        variables=density_variables,
        time_bin_method_dict={"Density": ep.TimeBinMethod.NanMedian},
        time_binning_cadence=bin_cadence,
        start_time=start_time,
        end_time=end_time,
    )

    density_variables["Density"].apply_thresholds_on_data(lower_threshold=_MINIMUM_PHYSICAL_DENSITY)

    binned_time_variable = ep.processing.bin_by_time(
        orbit_variables["Epoch"],
        variables=orbit_variables,
        time_bin_method_dict={"xGEO": ep.TimeBinMethod.NanMean},
        time_binning_cadence=bin_cadence,
        start_time=start_time,
        end_time=end_time,
    )

    pos_geo_var = orbit_variables["xGEO"]

    variables_to_compute: ep.processing.VariableRequest = [
        ("MLT", mag_field),
        ("R_Eq", mag_field),
        ("xGEO_Eq", mag_field),
    ]

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=binned_time_variable,
        xgeo_var=pos_geo_var,
        variables_to_compute=variables_to_compute,
        irbem_options=ep.processing.magnetic_field_utils.IrbemOptions(),
        num_cores=num_cores,
    )

    density_eq_var = ep.processing.compute_equatorial_plasmaspheric_density(
        density_variables["Density"],
        pos_geo_var,
        magnetic_field_variables["xGEO_Eq_" + mag_field],
        method="Denton_average",
    )

    saving_strategy = themis_scpot_density_strategy(processed_data_path, satellite, mag_field)

    variables_to_save: dict[ep.typing.InternalName, ep.Variable] = {
        "Epoch": binned_time_variable,
        "Number_density": density_variables["Density"],
        "Number_density_Eq": density_eq_var,
        "MLT": magnetic_field_variables["MLT_" + mag_field],
        "R_Eq": magnetic_field_variables["R_Eq_" + mag_field],
        "Position": pos_geo_var,
        "xGEO_Eq": magnetic_field_variables["xGEO_Eq_" + mag_field],
    }

    ep.save(variables_to_save, saving_strategy, start_time, end_time, binned_time_variable)


CLI_DEFAULTS = {
    "raw_data_path": "./data",
    "processed_data_path": "./data",
}

if __name__ == "__main__":
    ep.run_recipe_cli(process_themis_scpot_density, defaults=CLI_DEFAULTS)
