# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""Shared pyspedas loaders that turn THEMIS tplot variables into EL-PASO variables.

Unlike most EL-PASO recipes, the THEMIS recipes do not go through `ep.download` and
`ep.extract_variables_from_files`. THEMIS data is served from the Berkeley SSL archive in a
layout pyspedas already knows how to walk, and several of the derived products used here
(notably the spacecraft-potential density from `scpot2dens`) only exist as pyspedas routines.
So the loaders below call pyspedas and convert its tplot output into `ep.Variable` objects,
after which the recipes are ordinary EL-PASO pipelines.

tplot carries times as POSIX floats, which is exactly `ep.units.posixtime`, so time needs no
conversion on the way in.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
import pyspedas
from astropy import units as u
from pyspedas.projects import themis
from pyspedas.projects.themis import config as themis_config

import el_paso as ep
from el_paso.processing.magnetic_field_utils.irbem import Coords

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from el_paso.recipes.themis import ThemisProbe

logger = logging.getLogger(__name__)


def set_pyspedas_data_dir(raw_data_path: str | Path) -> None:
    """Point pyspedas' THEMIS downloader at `raw_data_path`.

    pyspedas resolves its local data directory from a module-level ``CONFIG`` dict, which it
    normally seeds from the ``SPEDAS_DATA_DIR``/``THM_DATA_DIR`` environment variables at import
    time. The loaders read the dict on every call, so assigning into it here lets the recipes
    honour their own ``raw_data_path`` argument without the caller having to set anything in the
    environment.

    Args:
        raw_data_path (str | Path): Directory pyspedas should download THEMIS files into.
    """
    themis_config.CONFIG["local_data_dir"] = str(raw_data_path)
    logger.info(f"pyspedas THEMIS data directory set to: {raw_data_path}")


def build_trange(start_time: datetime, end_time: datetime) -> list[str]:
    """Format a time range the way pyspedas' ``trange`` argument expects it.

    Args:
        start_time (datetime): Start of the time range.
        end_time (datetime): End of the time range.

    Returns:
        list[str]: The two bounds as ``"YYYY-MM-DD hh:mm:ss"`` strings.
    """
    return [start_time.strftime("%Y-%m-%d %H:%M:%S"), end_time.strftime("%Y-%m-%d %H:%M:%S")]


def _unpack_tplot(tplot_name: str) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any] | None]:
    """Return ``(times, values, bins)`` for a tplot variable, whichever shape pyspedas returns.

    `pyspedas.get_data` returns either a namedtuple-like sequence or a plain dict depending on
    the variable, so both are unpacked here.

    Raises:
        ValueError: If `tplot_name` does not exist, which is how a failed or empty download
            surfaces.
    """
    data = pyspedas.get_data(tplot_name)

    if data is None:
        msg = f"THEMIS tplot variable '{tplot_name}' not found -- the download likely returned no data."
        raise ValueError(msg)

    if isinstance(data, dict):
        return np.asarray(data["times"]), np.asarray(data["y"]), (np.asarray(data["v"]) if "v" in data else None)

    bins = np.asarray(data[2]) if len(data) > 2 else None
    return np.asarray(data[0]), np.asarray(data[1]), bins


def tplot_to_variable(tplot_name: str, unit: u.UnitBase) -> ep.Variable:
    """Read one tplot variable's values into an `ep.Variable`.

    Args:
        tplot_name (str): Name of the tplot variable to read.
        unit (u.UnitBase): The physical unit the tplot values are in.

    Returns:
        ep.Variable: The variable's values, tagged with `unit`.
    """
    _, values, _ = _unpack_tplot(tplot_name)
    return ep.Variable(original_unit=unit, data=values, processing_notes=f"Loaded from tplot variable '{tplot_name}'.")


def tplot_to_time_variable(tplot_name: str) -> ep.Variable:
    """Read one tplot variable's time base into an `ep.Variable`.

    Args:
        tplot_name (str): Name of the tplot variable whose times should be read.

    Returns:
        ep.Variable: The time base, in `ep.units.posixtime`.
    """
    times, _, _ = _unpack_tplot(tplot_name)
    return ep.Variable(
        original_unit=ep.units.posixtime,
        data=times,
        processing_notes=f"Loaded from tplot variable '{tplot_name}'.",
    )


def tplot_to_bin_variable(tplot_name: str, unit: u.UnitBase) -> ep.Variable:
    """Read one tplot variable's bin centres (its ``v`` component) into an `ep.Variable`.

    Used for the spectral frequency axis. The bins come back per-record for some THEMIS
    products; since the FFT frequency grid is fixed, a 2-D result is collapsed to its first row.

    Args:
        tplot_name (str): Name of the tplot variable whose bins should be read.
        unit (u.UnitBase): The physical unit the bin centres are in.

    Returns:
        ep.Variable: The 1-D bin centres, tagged with `unit`.

    Raises:
        ValueError: If the variable carries no bin component.
    """
    _, _, bins = _unpack_tplot(tplot_name)

    if bins is None:
        msg = f"THEMIS tplot variable '{tplot_name}' has no bin ('v') component."
        raise ValueError(msg)

    if bins.ndim > 1:
        bins = bins[0, :]

    return ep.Variable(original_unit=unit, data=bins, processing_notes=f"Bin centres of tplot variable '{tplot_name}'.")


def get_themis_position_geo(
    satellite: ThemisProbe,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, ep.Variable]:
    """Load the THEMIS spacecraft position and convert it to GEO coordinates.

    Loads the Level 1 state data, reads the GSM position, and rotates it to GEO -- the frame
    every downstream IRBEM computation in EL-PASO expects.

    Args:
        satellite (ThemisProbe): The THEMIS probe to load ("a" through "e").
        start_time (datetime): Start of the time range to load.
        end_time (datetime): End of the time range to load.

    Returns:
        dict[str, ep.Variable]: The "Epoch" time base and the GEO position "xGEO", in Earth radii.
    """
    probe = str(satellite)
    themis.state(trange=build_trange(start_time, end_time), probe=probe, get_support_data=True)

    pos_var_name = f"th{probe}_pos_gsm"

    time_var = tplot_to_time_variable(pos_var_name)
    pos_gsm_var = tplot_to_variable(pos_var_name, u.km)

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in time_var.get_data(ep.units.posixtime)]

    pos_geo = Coords().transform(
        time=datetimes,
        pos=np.asarray(pos_gsm_var.get_data(ep.units.RE)).astype(np.float64),
        sysaxes_in=ep.IRBEM_SYSAXIS_GSM,
        sysaxes_out=ep.IRBEM_SYSAXIS_GEO,
    )

    return {
        "Epoch": time_var,
        "xGEO": ep.Variable(
            original_unit=ep.units.RE,
            data=pos_geo,
            processing_notes=f"Rotated from '{pos_var_name}' (GSM) to GEO.",
        ),
    }


def get_themis_scpot_density(
    satellite: ThemisProbe,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, ep.Variable]:
    """Derive the THEMIS electron density from the spacecraft potential.

    Loads the ESA Level 2 moments (electron and ion density, spacecraft potential, and electron
    temperature) and feeds them to `pyspedas.projects.themis.scpot2dens`, which calibrates the
    spacecraft potential against the ESA densities. The result covers the plasmasphere far better
    than the ESA electron moment alone, which undercounts cold electrons there.

    Args:
        satellite (ThemisProbe): The THEMIS probe to load ("a" through "e").
        start_time (datetime): Start of the time range to load.
        end_time (datetime): End of the time range to load.

    Returns:
        dict[str, ep.Variable]: The "Epoch" time base and the derived "Density", in cm^-3.
    """
    probe = str(satellite)
    prefix = f"th{probe}"

    themis.esa(
        trange=build_trange(start_time, end_time),
        probe=probe,
        level="l2",
        varnames=[
            f"{prefix}_peer_density",
            f"{prefix}_peir_density",
            f"{prefix}_peer_sc_pot",
            f"{prefix}_peer_avgtemp",
        ],
    )

    dens_e_time, dens_e, _ = _unpack_tplot(f"{prefix}_peer_density")
    dens_i_time, dens_i, _ = _unpack_tplot(f"{prefix}_peir_density")
    sc_pot_time, sc_pot, _ = _unpack_tplot(f"{prefix}_peer_sc_pot")
    temp_e_time, temp_e, _ = _unpack_tplot(f"{prefix}_peer_avgtemp")

    density = themis.scpot2dens(
        sc_pot,
        sc_pot_time,
        temp_e,
        temp_e_time,
        dens_e,
        dens_e_time,
        dens_i,
        dens_i_time,
        probe,
    )

    return {
        "Epoch": ep.Variable(original_unit=ep.units.posixtime, data=np.asarray(dens_e_time)),
        "Density": ep.Variable(
            original_unit=u.cm ** (-3),
            data=np.asarray(density),
            processing_notes="Derived from the spacecraft potential via pyspedas scpot2dens.",
        ),
    }
