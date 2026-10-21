# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np

import el_paso as ep
from el_paso.saving_strategies.solar_wind_indices_strategy import ALL_SW_INDICES


def solar_wind_indices_strategy(
    base_data_path: str | Path,
    *,
    file_format: ep.typing.MFSFormats = "nc",
) -> ep.SavingStrategy:
    """Monthly solar wind indices saving strategy following the GFZ data standard."""
    return ep.saving_strategies.SolarWindIndicesStrategy(
        base_data_path,
        ep.data_standards.GFZStandard(),
        file_format=file_format,
    )


def process_solar_wind_indices(
    start_time: datetime,
    end_time: datetime,
    satellite: Literal["none"] = "none",
    mag_field: ep.typing.MagneticFieldLiteral = "T89",
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    bin_cadence: timedelta = timedelta(minutes=5),
    num_cores: int = 16,
    save_strategy: Literal["gfz"] = "gfz",
    *,
    skip_existing: bool = True,
    save_indices_sw: bool = False,
    fill_sw: bool = False,
) -> None:
    """Save every solar wind and geomagnetic index EL-PASO can load over a time range.

    Builds a uniform time grid from `start_time` to `end_time` at `bin_cadence`, loads every
    index in `el_paso.saving_strategies.solar_wind_indices_strategy.ALL_SW_INDICES` (Kp, Dst,
    solar wind and IMF parameters, Tsyganenko G/W parameters), interpolates them onto the grid,
    and writes them to `processed_data_path` as monthly files using the GFZ data standard. The
    indices are independent of any magnetic field model and of any spacecraft, so this recipe
    takes no meaningful `mag_field` or satellite. Use it to keep the model inputs on disk
    without processing particle data, or call it from another recipe to save the indices
    alongside that recipe's own product.

    Args:
        start_time (datetime): Start of the time range (inclusive).
        end_time (datetime): End of the time range (exclusive).
        satellite (Literal["none"]): Unused.
        mag_field (ep.typing.MagneticFieldLiteral): Unused.
        raw_data_path (str | Path): Unused.
        processed_data_path (str | Path): Base directory in which the results are saved.
        bin_cadence (timedelta): Time step between consecutive samples of the time grid.
        num_cores (int): Unused. Loading the indices is not parallelized.
        save_strategy (Literal["gfz"]): Unused.
        skip_existing (bool): Unused.
        save_indices_sw (bool): Unused.
        fill_sw (bool): For the high-resolution OMNI parameters (Pdyn, BzIMF, ByIMF, Vsw, Nsw),
            whether leading/trailing gaps that linear interpolation cannot reach are forward/backward
            filled. If False, those edge gaps are left as NaN. Defaults to False; turn it on
            explicitly via the CLI or in a script.
    """
    del satellite, mag_field, raw_data_path, num_cores, save_strategy, skip_existing, save_indices_sw

    timestamps = []
    curr_time = start_time
    while curr_time < end_time:
        timestamps.append(curr_time.timestamp())
        curr_time += bin_cadence

    time_var = ep.Variable(data=np.asarray(timestamps), original_unit=ep.units.posixtime)
    time_var.metadata.original_cadence_seconds = bin_cadence.total_seconds()

    indices_solar_wind = ep.load_indices_solar_wind_parameters(
        start_time, end_time, ALL_SW_INDICES, time_var, fill_sw=fill_sw
    )

    variables_to_save: dict[ep.typing.InternalName, ep.Variable] = {
        "Epoch": time_var,
        **indices_solar_wind,
    }

    ep.save(
        variables_to_save,
        solar_wind_indices_strategy(processed_data_path),
        start_time,
        end_time,
        time_var,
    )


if __name__ == "__main__":
    ep.run_recipe_cli(process_solar_wind_indices)
