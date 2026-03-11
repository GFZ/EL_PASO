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
    automatically setting the timezone to UTC. If time_format is provided, it uses
    datetime.strptime for explicit parsing; otherwise, it uses a flexible parser
    (like dateutil.parser.parse) to infer the format.

    Args:
        time_var (Variable): The variable containing string-based time data to be
            converted. Its data is accessed via time_var.get_data().
        time_format (str | None): The explicit format string (e.g., "%Y-%m-%d %H:%M:%S")
            used to parse the time data. If None (default), the function uses a
            flexible parser to infer the correct format.

    Returns:
        NDArray[np.generic]: A NumPy array of Python datetime objects that are all
            localized to the UTC timezone.
    """
    time_var.metadata.add_processing_note("Converting string-time to datetime")

    if time_format is None:
        return np.asarray([parser.parse(t).replace(tzinfo=timezone.utc) for t in time_var.get_data()])

    return np.asarray([datetime.strptime(t, time_format).replace(tzinfo=timezone.utc) for t in time_var.get_data()])
