import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import dateutil
import numpy as np
import pandas as pd
from astropy import units as u

import el_paso as ep
from el_paso import setup_logging


def compute_lcds(
    start_time: datetime,
    end_time: datetime,
    cadence: timedelta,
    alpha_eq: list[float],
    mag_field: ep.typing.MagneticFieldLiteral,
    processed_data_path: str | Path,
    *,
    num_cores: int = 10,
    max_search_radius: float = 10,
) -> None:
    """Computes the Last Closed Drift Shell (LCDS) over a time range and saves it to disk.

    Builds a uniform time grid from ``start_time`` to ``end_time`` at the given cadence,
    computes the LCDS L* for each equatorial pitch angle in ``alpha_eq`` at every time
    step, and writes the results (Epoch, LCDS, Alpha) to ``processed_data_path`` using
    the GFZ data standard.

    Args:
        start_time: Start of the time range (inclusive).
        end_time: End of the time range (exclusive).
        cadence: Time step between consecutive samples.
        alpha_eq: Equatorial pitch angles in degrees, applied identically at every
            time step.
        mag_field: External magnetic field model to use (e.g. ``"T89"``, ``"T04s"``).
        processed_data_path: Directory or file path where the output is saved.
        num_cores: Number of worker processes used to parallelize the computation
            across time steps. Defaults to 10.
        max_search_radius: Outer ceiling for the radial search, in RE. A drift shell
            still closed at this radius is censored rather than reported as the LCDS.
            Defaults to 10.

    Returns:
        None. Results are written to ``processed_data_path`` as a side effect.
    """
    datetimes = []
    curr_time = start_time
    while curr_time < end_time:
        datetimes.append(curr_time)
        curr_time += cadence

    timestamps = [t.timestamp() for t in datetimes]

    time_var = ep.Variable(data=np.asarray(timestamps), original_unit=ep.units.posixtime)
    alpha_eq_var = ep.Variable(data=np.tile(alpha_eq, (len(timestamps), 1)), original_unit=u.deg)

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
    saving_strategy = ep.saving_strategies.LCDSStrategy(processed_data_path, mag_field, ep.data_standards.GFZStandard())

    variables_to_save: dict[ep.typing.InternalName, ep.Variable] = {
        "Epoch": time_var,
        "LCDS": lcds_var,
        "Alpha": alpha_eq_var,
        "InvK": inv_K_var
    }

    ep.save(variables_to_save, saving_strategy, start_time, end_time, time_var)


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Compute Last Closed Drift Shell (LCDS) for an empirical magnetic field model."
    )
    parser.add_argument(
        "--start_time",
        type=str,
        help="Start time in valid dateparse format. Example: YYYY-MM-DDTHH:MM:SS.",
        default=datetime(2014, 1, 11, 0, 0, tzinfo=timezone.utc).isoformat(),
        required=False,
    )
    parser.add_argument(
        "--end_time",
        type=str,
        help="End time in valid dateparse format. Example: YYYY-MM-DDTHH:MM:SS.",
        default=datetime(2014, 1, 14, 0, 0, tzinfo=timezone.utc).isoformat(),
        required=False,
    )
    parser.add_argument(
        "mag_field",
        type=str,
        help="Magnetic field model. [T89, T96, T01s, TS04]",
        default="T89",
        required=False,
    )

    args = parser.parse_args()

    dt_start = dateutil.parser.parse(args.start_time)
    dt_end = dateutil.parser.parse(args.end_time)

    alpha_eq = list(np.arange(10, 91, 10))

    compute_lcds(
        dt_start,
        dt_end,
        timedelta(minutes=5),
        alpha_eq,
        args.mag_field,
        processed_data_path=".",
        num_cores=128,
    )
