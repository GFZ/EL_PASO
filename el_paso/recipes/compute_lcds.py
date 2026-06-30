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

    lcds_var = ep.processing.compute_LCDS(
        time_var,
        alpha_eq_var,
        mag_field,
        ep.processing.magnetic_field_utils.IrbemOptions(drift_shell_resolution=0, field_line_resolution=0),
        num_cores=num_cores,
        search_params=search_params,
    )

    saving_strategy = ep.saving_strategies.LCDSSTrategy(processed_data_path, mag_field, ep.data_standards.GFZStandard())

    variables_to_save: dict[ep.typing.InternalName, ep.Variable] = {
        "Epoch": time_var,
        "LCDS": lcds_var,
        "Alpha_Eq": alpha_eq_var,
    }

    ep.save(variables_to_save, saving_strategy, start_time, end_time)


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Process density data from EFW and EMFISIS instrument on VanAllenProbes."
    )
    parser.add_argument(
        "--start_time",
        type=str,
        help="Start time in valid dateparse format. Example: YYYY-MM-DDTHH:MM:SS.",
        default=datetime(2017, 4, 1, tzinfo=timezone.utc).isoformat(),
        required=False,
    )
    parser.add_argument(
        "--end_time",
        type=str,
        help="End time in valid dateparse format. Example: YYYY-MM-DDTHH:MM:SS.",
        default=datetime(2017, 4, 1, 0, 15, tzinfo=timezone.utc).isoformat(),
        required=False,
    )

    args = parser.parse_args()

    dt_start = dateutil.parser.parse(args.start_time)
    dt_end = dateutil.parser.parse(args.end_time)

    alpha_eq = list(np.arange(5, 91, 30))

    compute_lcds(
        dt_start,
        dt_end,
        timedelta(minutes=5),
        alpha_eq,
        "TS04",
        processed_data_path=".",
        num_cores=1,
    )
