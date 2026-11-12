# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""Custom extractor for the LANL GeoGrid LowRes ASCII files (SOPA_ESP, MPA and MERGED)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

    from el_paso.extract_variables_from_files import ExtractionInfo

FILL_VALUE_THRESHOLD = -1e30
"""Values at or below this are fill values (the files use -1e31) and are set to NaN."""


def _parse_header(header_lines: list[str], file_path: str | Path) -> dict:
    """Parse the commented JSON header.

    The header ends with ``} end JSON``, so only the text between the first ``{`` and the
    last ``}`` is valid JSON.
    """
    header_text = "".join(header_lines)
    start = header_text.find("{")
    end = header_text.rfind("}")

    if start == -1 or end == -1:
        msg = f"No JSON header found in {file_path}!"
        raise ValueError(msg)

    try:
        return json.loads(header_text[start : end + 1])
    except json.JSONDecodeError as e:
        msg = f"Could not parse the JSON header of {file_path}: {e}"
        raise ValueError(msg) from e


def _number_of_columns(header: dict) -> int:
    """Return the number of data columns described by the header's START_COLUMN and DIMENSION entries."""
    return max(
        entry["START_COLUMN"] + entry.get("DIMENSION", [1])[0]
        for entry in header.values()
        if isinstance(entry, dict) and "START_COLUMN" in entry
    )


def extract_lanl_geogrid_ascii(
    file_path: str | Path, extraction_infos: Iterable[ExtractionInfo]
) -> dict[str | int, NDArray[np.floating]]:
    """Extract variables from a LANL GeoGrid LowRes ASCII file.

    The files carry a commented JSON header followed by a whitespace-delimited data block.
    Time-dependent variables are read from the data block, using the header's ``START_COLUMN``
    and ``DIMENSION``. Time-independent variables (e.g. energy grids) are read from the
    header's ``VALUES``. Column 0 (the ISO time string) is never read; use ``Date`` and ``UTC``
    instead. Fill values (-1e31), ``nan`` and ``-nan`` are returned as NaN.

    Pass it to `ep.extract_variables_from_files` via ``custom_extractors={".txt": extract_lanl_geogrid_ascii}``.

    Args:
        file_path (str | Path): Path to the LANL GeoGrid ASCII file.
        extraction_infos (Iterable[ExtractionInfo]): The variables to extract. ``name_or_column``
            must be a variable name from the file header.

    Returns:
        dict[str | int, NDArray[np.floating]]: The extracted data, keyed by ``name_or_column``.

    Raises:
        ValueError: If the header cannot be parsed, a variable is missing from the header or
            lacks ``START_COLUMN``/``VALUES``, a variable lives in the ISO time string column,
            or a data row has a different number of fields than the header describes.
    """
    header_lines: list[str] = []
    data_lines: list[str] = []
    with Path(file_path).open("r") as f:
        for line in f:
            if line.startswith("#"):
                header_lines.append(line[1:])
            elif line.strip():
                data_lines.append(line)

    header = _parse_header(header_lines, file_path)
    n_columns = _number_of_columns(header)

    for i_row, line in enumerate(data_lines):
        n_fields = len(line.split())
        if n_fields != n_columns:
            msg = (
                f"Data row {i_row} of {file_path} has {n_fields} fields, but the header describes {n_columns} columns!"
            )
            raise ValueError(msg)

    # column 0 holds the ISO time string, so the data block starts at column 1
    if data_lines:
        data_block = np.loadtxt(data_lines, usecols=range(1, n_columns), ndmin=2, dtype=np.float64)
    else:
        data_block = np.empty((0, n_columns - 1), dtype=np.float64)
    data_block[data_block <= FILL_VALUE_THRESHOLD] = np.nan

    data: dict[str | int, NDArray[np.floating]] = {}
    for info in extraction_infos:
        name = info.name_or_column
        if name not in header or not isinstance(header[name], dict):
            msg = f"Variable {name!r} not found in header of {file_path}!"
            raise ValueError(msg)

        entry = header[name]
        (dimension,) = entry.get("DIMENSION", [1])

        if info.is_time_dependent:
            if "START_COLUMN" not in entry:
                msg = f"Variable {name!r} has no START_COLUMN in header of {file_path}!"
                raise ValueError(msg)

            start_column = entry["START_COLUMN"] - 1
            if start_column < 0:
                msg = f"Variable {name!r} is the ISO time string column, which is not supported. Use Date and UTC!"
                raise ValueError(msg)

            data[name] = (
                data_block[:, start_column]
                if dimension == 1
                else data_block[:, start_column : start_column + dimension]
            )
        else:
            if "VALUES" not in entry:
                msg = f"Time-independent variable {name!r} has no VALUES in header of {file_path}!"
                raise ValueError(msg)

            values = np.asarray(entry["VALUES"], dtype=np.float64)
            values[values <= FILL_VALUE_THRESHOLD] = np.nan
            data[name] = values

    return data
