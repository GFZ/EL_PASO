# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

import numpy as np
from dateutil import parser
from numpy.typing import NDArray

from el_paso import Variable


def convert_string_to_datetime(time_var: Variable, time_format: str | None = None) -> NDArray[np.generic]:
    """Converts a Variable's string-based time data to UTC datetime objects.

    This function transforms an array of time strings into Python datetime objects,
    automatically converting them to UTC. If time_format is provided, it first uses
    datetime.strptime for explicit parsing and falls back to dateutil.parser.parse
    when the timestamp does not match the provided format.

    Args:
        time_var (Variable): The variable containing string-based time data to be
            converted. Its data is accessed via time_var.get_data().
        time_format (str | None): The explicit format string (e.g., "%Y-%m-%d %H:%M:%S")
            used to parse the time data. If None (default), the function uses a
            flexible parser to infer the correct format.

    Returns:
        NDArray[np.generic]: A NumPy array of Python datetime objects that are all
            localized to UTC.
    """
    time_var.metadata.add_processing_note("Converting string-time to datetime")

    def to_utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def parse_time(t: str) -> datetime:
        if time_format is not None:
            try:
                return to_utc(datetime.strptime(t, time_format))  # noqa: DTZ007
            except ValueError:
                return to_utc(parser.parse(t))

        return to_utc(parser.parse(t))

    return np.asarray([parse_time(t) for t in time_var.get_data()])
