# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: INP001, T201

import argparse
import calendar
import subprocess
from datetime import datetime, timedelta, timezone
from enum import Enum

import dateutil

from el_paso.typing import TimeInterval


class ChunkType(Enum):  # noqa: D101
    DAILY = "daily"
    MONTHLY = "monthly"
    YEARLY = "yearly"


def _get_time_intervals(start_time: datetime, end_time: datetime, chunk_type: ChunkType) -> list[TimeInterval]:
    time_intervals: list[TimeInterval] = []

    current_time = start_time.replace(day=1)
    while current_time <= end_time:
        match chunk_type:
            case ChunkType.DAILY:
                day_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                day_end = current_time.replace(hour=23, minute=59, second=59, microsecond=999999)
                time_intervals.append((day_start, day_end))
                current_time += timedelta(days=1)

            case ChunkType.MONTHLY:
                year = current_time.year
                month = current_time.month
                eom_day = calendar.monthrange(year, month)[1]

                month_start = datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc)
                month_end = datetime(year, month, eom_day, 23, 59, 59, tzinfo=timezone.utc)
                time_intervals.append((month_start, month_end))
                current_time = (
                    datetime(year + 1, 1, 1, tzinfo=timezone.utc)
                    if month == 12
                    else datetime(year, month + 1, 1, tzinfo=timezone.utc)
                )

            case ChunkType.YEARLY:
                year = current_time.year

                year_start = datetime(year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
                year_end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
                time_intervals.append((year_start, year_end))
                current_time = datetime(year + 1, 1, 1, tzinfo=timezone.utc)

    return time_intervals


def submit_slurm_jobs_in_chunks(
    start_time_str: str, end_time_str: str, chunk_type: ChunkType, job_script_path: str = "job_script_template.sh"
) -> None:
    """Submits HPC jobs in time-based chunks.

    This function divides a specified time range into smaller intervals (daily,
    monthly, or yearly) and submits a separate job for each interval to an HPC
    cluster using the `sbatch` command. It assumes a job script template named
    `job_script_template.sh` exists in the same directory. The chunk start and
    end times are passed to the job script as command-line arguments.

    Parameters:
        start_time_str (str): The start of the time range, in a format parsable
                              by `dateutil.parser`. Example: '2023-01-01T00:00:00'.
        end_time_str (str): The end of the time range, in a format parsable
                            by `dateutil.parser`. Example: '2023-03-31T23:59:59'.
        chunk_type (ChunkType): The type of time chunk to use for job submission.
                                Valid options are `ChunkType.DAILY`, `ChunkType.MONTHLY`,
                                or `ChunkType.YEARLY`.

    Raises:
        subprocess.CalledProcessError: If an `sbatch` command fails to execute
                                        with a non-zero exit code.
    """
    # Convert string times to datetime objects
    start_time = dateutil.parser.parse(start_time_str).replace(tzinfo=timezone.utc)
    end_time = dateutil.parser.parse(end_time_str).replace(tzinfo=timezone.utc)

    time_intervals = _get_time_intervals(start_time, end_time, chunk_type)

    for start_interval, end_interval in time_intervals:
        # Format the times for the command line arguments
        chunk_start_str = start_interval.strftime("%Y-%m-%dT%H:%M:%S")
        chunk_end_str = end_interval.strftime("%Y-%m-%dT%H:%M:%S")

        print(f"Submitting job for time range: {chunk_start_str} to {chunk_end_str}")

        # Construct the sbatch command
        command = [
            "sbatch",
            job_script_path,
            chunk_start_str,
            chunk_end_str,
        ]

        try:
            # Execute the sbatch command and check for errors
            subprocess.run(command, check=True)  # noqa: S603
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job: {e}")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit HPC jobs in time-based chunks.")
    parser.add_argument(
        "start_time",
        type=str,
        help="Start time in valid dateparse format. Example: YYYY-MM-DDTHH:MM:SS.",
    )
    parser.add_argument(
        "end_time",
        type=str,
        help="End time in valid dateparse format. Example: YYYY-MM-DDTHH:MM:SS.",
    )
    parser.add_argument(
        "chunk_type",
        type=str,
        help="Chunk type either daily|monthly|yearly.",
    )
    parser.add_argument(
        "job_script_path",
        type=str,
        help="Path towards the job script.",
        default="job_script_template.sh",
    )

    args = parser.parse_args()
    submit_slurm_jobs_in_chunks(
        args.start_time, args.end_time, ChunkType[args.chunk_type.upper()], job_script_path=args.job_script_path
    )
