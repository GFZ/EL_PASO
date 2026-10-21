# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""THEMIS-specific pyspedas loaders.

The generic tplot conversion helpers these build on live in `el_paso.pyspedas_utils`; what
remains here is the part that knows THEMIS variable names and instruments.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from pyspedas.projects import themis

import el_paso as ep
from el_paso.processing.magnetic_field_utils.irbem import Coords
from el_paso.pyspedas_utils import (
    build_trange,
    load_with_local_fallback,
    tplot_to_time_variable,
    tplot_to_variable,
    unpack_tplot,
)

if TYPE_CHECKING:
    from el_paso.recipes.themis import ThemisProbe

logger = logging.getLogger(__name__)

THEMIS_SPDF_URL = "https://spdf.gsfc.nasa.gov/pub/data/themis/"
"""SPDF mirror of the THEMIS data archive, with the same directory layout as the Berkeley server."""

_THEMIS_BERKELEY_HOST = "themis.ssl.berkeley.edu"


def use_spdf_for_themis_downloads() -> None:
    """Download THEMIS data from the SPDF mirror instead of the Berkeley server.

    pyspedas defaults to the Berkeley server, which is not always reachable. The THEMIS loaders
    read their server from the module-level ``CONFIG`` dict on each call, so it is switched to
    SPDF here. A server configured explicitly by the user (e.g. through the pyspedas preferences
    file) is kept.
    """
    from pyspedas.projects.themis.config import CONFIG  # noqa: PLC0415

    configured_url = str(CONFIG["remote_data_dir"])
    if _THEMIS_BERKELEY_HOST in configured_url:
        CONFIG["remote_data_dir"] = THEMIS_SPDF_URL
        logger.info(f"pyspedas 'themis' server set to: {THEMIS_SPDF_URL}")
    else:
        logger.info(f"Keeping the configured pyspedas 'themis' server: {configured_url}")


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
    load_with_local_fallback(
        themis.state, trange=build_trange(start_time, end_time), probe=probe, get_support_data=True
    )

    pos_var_name = f"th{probe}_pos_gsm"

    time_var = tplot_to_time_variable(pos_var_name)
    pos_gsm_var = tplot_to_variable(pos_var_name, u.km)

    pos_gsm_var.truncate(time_var, start_time, end_time)
    time_var.truncate(time_var, start_time, end_time)

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

    load_with_local_fallback(
        themis.esa,
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

    dens_e_time, dens_e, _ = unpack_tplot(f"{prefix}_peer_density")
    dens_i_time, dens_i, _ = unpack_tplot(f"{prefix}_peir_density")
    sc_pot_time, sc_pot, _ = unpack_tplot(f"{prefix}_peer_sc_pot")
    temp_e_time, temp_e, _ = unpack_tplot(f"{prefix}_peer_avgtemp")

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
