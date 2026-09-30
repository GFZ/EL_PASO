# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
from astropy import units as u

import el_paso as ep

DEFAULT_ALPHA_EQ_DEG = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
"""Equatorial pitch angles the LCDS is computed for unless the caller asks for others."""


def lcds_strategy(
    base_data_path: str | Path,
    mag_field: ep.typing.MagneticFieldLiteral,
    *,
    file_format: ep.typing.MFSFormats = "nc",
) -> ep.SavingStrategy:
    """Monthly LCDS saving strategy following the GFZ data standard."""
    return ep.saving_strategies.LCDSStrategy(
        base_data_path,
        mag_field,
        ep.data_standards.GFZStandard(),
        file_format=file_format,
    )


def compute_lcds(
    start_time: datetime,
    end_time: datetime,
    satellite: Literal["none"] = "none",
    mag_field: ep.typing.MagneticFieldLiteral = "T89",
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    bin_cadence: timedelta = timedelta(minutes=5),
    num_cores: int = 16,
    save_strategy: Literal["gfz"] = "gfz",
    alpha_eq: list[float] = DEFAULT_ALPHA_EQ_DEG,
    max_search_radius: float = 10,
    *,
    skip_existing: bool = True,
) -> None:
    """Compute the Last Closed Drift Shell (LCDS) over a time range and save it to disk.

    Builds a uniform time grid from `start_time` to `end_time` at `bin_cadence`, computes the
    LCDS L* and the corresponding second adiabatic invariant K for each equatorial pitch angle
    in `alpha_eq` at every time step, and writes the results (Epoch, LCDS, Alpha, InvK) to
    `processed_data_path` using the GFZ data standard. The LCDS is a property of the magnetic
    field model alone, so this recipe downloads no spacecraft data and takes no satellite.

    Args:
        start_time (datetime): Start of the time range (inclusive).
        end_time (datetime): End of the time range (exclusive).
        satellite (Literal["none"]): Unused. The LCDS does not belong to any spacecraft; the
                                     parameter only keeps the recipe interface uniform.
        mag_field (ep.typing.MagneticFieldLiteral): External magnetic field model to use
                                                    (e.g. "T89", "TS04").
        raw_data_path (str | Path): Unused. The recipe needs no raw spacecraft files; the
                                    magnetic field indices are fetched by the processing step.
        processed_data_path (str | Path): Base directory in which the results are saved.
        bin_cadence (timedelta): Time step between consecutive samples of the time grid.
        num_cores (int): Number of worker processes used to parallelize the computation
                         across time steps.
        save_strategy (Literal["gfz"]): The saving strategy used to write the results. The LCDS
                                        is only written in the GFZ monthly format.
        alpha_eq (list[float]): Equatorial pitch angles in degrees, applied identically at
                                every time step.
        max_search_radius (float): Outer ceiling for the radial search, in RE. A drift shell
                                   still closed at this radius is censored rather than
                                   reported as the LCDS.
        skip_existing (bool): Unused. The recipe downloads no files.
    """
    del satellite, raw_data_path, save_strategy, skip_existing

    datetimes = []
    curr_time = start_time
    while curr_time < end_time:
        datetimes.append(curr_time)
        curr_time += bin_cadence

    timestamps = [t.timestamp() for t in datetimes]

    time_var = ep.Variable(data=np.asarray(timestamps), original_unit=ep.units.posixtime)
    alpha_eq_var = ep.Variable(data=np.tile(list(alpha_eq), (len(timestamps), 1)), original_unit=u.deg)

    search_params = ep.processing.magnetic_field_utils.LCDSSearchParams(
        max_r=max_search_radius, start_r=max_search_radius
    )

    lcds_var, inv_K_var = ep.processing.compute_LCDS(
        time_var,
        alpha_eq_var,
        mag_field,
        ep.processing.magnetic_field_utils.IrbemOptions(drift_shell_resolution=0, field_line_resolution=0),
        num_cores=num_cores,
        search_params=search_params,
    )

    variables_to_save: dict[ep.typing.InternalName, ep.Variable] = {
        "Epoch": time_var,
        "LCDS": lcds_var,
        "Alpha": alpha_eq_var,
        "InvK": inv_K_var,
    }

    ep.save(variables_to_save, lcds_strategy(processed_data_path, mag_field), start_time, end_time, time_var)


if __name__ == "__main__":
    ep.run_recipe_cli(compute_lcds)
