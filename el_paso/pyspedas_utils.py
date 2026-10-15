# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers for recipes that load their data through pyspedas instead of `el_paso.download`.

Some missions are easier to reach through pyspedas than through `el_paso.download`: it already
knows the archive layout, and several derived products (THEMIS' ``scpot2dens``, for example)
exist only as pyspedas routines. Such a recipe calls pyspedas, converts the resulting tplot
variables into `el_paso.Variable` objects with the functions here, and is an ordinary EL-PASO
pipeline from that point on.

tplot carries times as POSIX floats, which is exactly `el_paso.units.posixtime`, so a time axis
needs no conversion on the way in.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pyspedas

import el_paso as ep

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from astropy import units as u
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_MIN_TPLOT_ENTRIES_WITH_BINS = 3


def set_pyspedas_data_dir(project: str, data_path: str | Path) -> None:
    """Point a pyspedas project's downloader at `data_path`.

    Every pyspedas project resolves its download directory from a module-level ``CONFIG``
    dict, which it normally seeds from the ``SPEDAS_DATA_DIR`` environment variable at import
    time. The loaders read that dict on each call, so assigning into it here lets a recipe
    honour its own ``raw_data_path`` argument without the caller setting anything in the
    environment.

    Args:
        project (str): The pyspedas project name, as in ``pyspedas.projects.<project>``
            (e.g. ``"themis"``, ``"rbsp"``, ``"erg"``).
        data_path (str | Path): Directory pyspedas should download that project's files into.

    Raises:
        ValueError: If the project does not exist, or does not expose a configurable data
            directory (``mms`` is the notable exception).
    """
    try:
        config = importlib.import_module(f"pyspedas.projects.{project}.config")
    except ModuleNotFoundError as exc:
        msg = f"pyspedas project '{project}' has no configurable data directory."
        raise ValueError(msg) from exc

    if "local_data_dir" not in config.CONFIG:
        msg = f"pyspedas project '{project}' does not define a 'local_data_dir'."
        raise ValueError(msg)

    config.CONFIG["local_data_dir"] = str(data_path)
    logger.info(f"pyspedas '{project}' data directory set to: {data_path}")


def build_trange(start_time: datetime, end_time: datetime) -> list[str]:
    """Format a time range the way pyspedas' ``trange`` argument expects it.

    Args:
        start_time (datetime): Start of the time range.
        end_time (datetime): End of the time range.

    Returns:
        list[str]: The two bounds as ``"YYYY-MM-DD hh:mm:ss"`` strings.
    """
    return [start_time.strftime("%Y-%m-%d %H:%M:%S"), end_time.strftime("%Y-%m-%d %H:%M:%S")]


def unpack_tplot(tplot_name: str) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any] | None]:
    """Return ``(times, values, bins)`` for a tplot variable, whichever shape pyspedas returns.

    `pyspedas.get_data` returns either a namedtuple-like sequence or a plain dict depending on
    the variable, so both are unpacked here.

    Args:
        tplot_name (str): Name of the tplot variable to read.

    Returns:
        tuple: The time axis, the values, and the bin centres (``None`` when the variable
        carries no bin component).

    Raises:
        ValueError: If `tplot_name` does not exist, which is how a failed or empty download
            surfaces.
    """
    data = pyspedas.get_data(tplot_name)

    if data is None:
        msg = f"tplot variable '{tplot_name}' not found -- the download likely returned no data."
        raise ValueError(msg)

    if isinstance(data, dict):
        return np.asarray(data["times"]), np.asarray(data["y"]), (np.asarray(data["v"]) if "v" in data else None)

    bins = np.asarray(data[2]) if len(data) >= _MIN_TPLOT_ENTRIES_WITH_BINS else None
    return np.asarray(data[0]), np.asarray(data[1]), bins


def tplot_to_variable(tplot_name: str, unit: u.UnitBase) -> ep.Variable:
    """Read one tplot variable's values into an `el_paso.Variable`.

    Args:
        tplot_name (str): Name of the tplot variable to read.
        unit (u.UnitBase): The physical unit the tplot values are in.

    Returns:
        ep.Variable: The variable's values, tagged with `unit`.
    """
    _, values, _ = unpack_tplot(tplot_name)
    return ep.Variable(original_unit=unit, data=values, processing_notes=f"Loaded from tplot variable '{tplot_name}'.")


def tplot_to_time_variable(tplot_name: str) -> ep.Variable:
    """Read one tplot variable's time base into an `el_paso.Variable`.

    Args:
        tplot_name (str): Name of the tplot variable whose times should be read.

    Returns:
        ep.Variable: The time base, in `el_paso.units.posixtime`.
    """
    times, _, _ = unpack_tplot(tplot_name)
    return ep.Variable(
        original_unit=ep.units.posixtime,
        data=times,
        processing_notes=f"Loaded from tplot variable '{tplot_name}'.",
    )


def tplot_to_bin_variable(tplot_name: str, unit: u.UnitBase) -> ep.Variable:
    """Read one tplot variable's bin centres (its ``v`` component) into an `el_paso.Variable`.

    Used for a spectral frequency axis. Some products repeat the bin centres for every record;
    since such a grid is fixed, a 2-D result is collapsed to its first row.

    Args:
        tplot_name (str): Name of the tplot variable whose bins should be read.
        unit (u.UnitBase): The physical unit the bin centres are in.

    Returns:
        ep.Variable: The 1-D bin centres, tagged with `unit`.

    Raises:
        ValueError: If the variable carries no bin component.
    """
    _, _, bins = unpack_tplot(tplot_name)

    if bins is None:
        msg = f"tplot variable '{tplot_name}' has no bin ('v') component."
        raise ValueError(msg)

    if bins.ndim > 1:
        bins = bins[0, :]

    return ep.Variable(original_unit=unit, data=bins, processing_notes=f"Bin centres of tplot variable '{tplot_name}'.")
