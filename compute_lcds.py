# SPDX-License-Identifier: Apache-2.0
"""Compute the Last Closed Drift Shell (LCDS, expressed as the maximum closed L*).

Python port of A. C. Kellerman's ``LCDS2`` Fortran routine, built on el_paso's IRBEM
wrapper. For a given equatorial pitch angle the algorithm marches along the SM x-axis
(noon meridian) and finds the largest radial distance at which a *closed* drift shell
still exists, i.e. the largest valid Roederer L*.

The search is done in three passes of increasing resolution:

    Pass 1 (coarse) : march *inward* from the start radius until a closed shell appears,
                      anchoring the search where the predicate is reliable.
    Pass 2 (medium) : march back *outward*, tracking the highest valid L*.
    Pass 3 (fine)   : continue outward to pin the boundary to ~``fine_step`` RE.

The closed/open test is "the traced field line has exactly one |B| minimum" — near the
magnetopause field lines develop multiple minima and L* is no longer defined.

CEILING / CENSORING
-------------------
``max_r`` is both the search ceiling and (for a cold start) where the coarse march
begins. If the shell is still closed at ``max_r``, the outward passes never observe an
open shell, so the true boundary is only known to be *at least* the returned value — it
is right-censored. ``LCDSResult.at_ceiling`` flags this so a censored lower bound is
never mistaken for a resolved boundary.

TIME-SERIES OPTIMIZATIONS (see ``compute_lcds_timeseries``)
  * Work is split into contiguous *chunks*, one per worker; each worker builds its IRBEM
    handle once and reuses it for every step in the chunk.
  * The LCDS moves smoothly in time, so each step *warm-starts* the coarse march from the
    previous step's solution instead of restarting at ``max_r`` every time.

Building blocks (from el_paso's IRBEM wrapper):
    trace_field_line2_1          -> MagFields.trace_field_line
    find_magequator1             -> MagFields.find_magequator
    Make_LSTAR_SHELL_SPLITTING1  -> MagFields.make_lstar_shell_splitting
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from el_paso.processing.magnetic_field_utils.construct_maginput import MagInputKeys
from el_paso.processing.magnetic_field_utils.irbem import (
    DEFAULT_LIBIRBEM_PATH,
    SYSAXES_STR_TO_INT,
    IrbemOptions,
    MagFields,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    import pandas as pd

logger = logging.getLogger(__name__)

# IRBEM/Fortran fill value used for "no result".
FORTRAN_BAD_VALUE = np.float64(-1.0e31)


def compute_lcds(
    time: datetime | str | pd.Timestamp,
    alpha_eq_deg: float,
    maginput: Mapping[MagInputKeys, float],
    *,
    kext: int | str = "T89",
    options: IrbemOptions | None = None,
    lib_path: str | Path = DEFAULT_LIBIRBEM_PATH,
    max_r: float = 10.0,
    coarse_step: float = 1.0,
    medium_step: float = 0.5,
    fine_step: float = 0.1,
    trace_r0: float = 0.8,
    search_start: float | None = None,
) -> LCDSResult:
    """Compute the LCDS (max closed L*) for one time and one equatorial pitch angle.

    Args:
        time: Epoch (datetime, ISO string, or pandas Timestamp).
        alpha_eq_deg: Equatorial pitch angle in degrees (LCDS is pitch-angle specific).
        maginput: Scalar magnetic-field-model inputs keyed by IRBEM name (e.g. ``{"Kp": 3.0}``).
        kext: External field model (int code or name, e.g. ``"T89"``).
        options: IRBEM options. Defaults reproduce LCDS2's settings.
        lib_path: Path to ``libirbem.so``.
        max_r: Outer ceiling for the search, in RE. If the shell is still closed here the
            result is flagged ``at_ceiling`` (a lower bound, not a resolved boundary).
        coarse_step, medium_step, fine_step: Radial step sizes (RE) for the three passes.
        trace_r0: Field-line trace stop radius in RE (LCDS2 used 0.8).
        search_start: Radius (RE) at which the coarse march begins. ``None`` => cold start
            at ``max_r``. Pass the previous solution's ``x_sm`` (plus margin) to warm-start.

    Returns:
        LCDSResult with the L* of the last closed drift shell (check ``at_ceiling``).
    """
    if options is None:
        options = IrbemOptions()

    mag_geo = MagFields(lib_path=lib_path, kext=kext, sysaxes=SYSAXES_STR_TO_INT["GEO"], options=options)

    search = _SearchParams(max_r, coarse_step, medium_step, fine_step, trace_r0, warm_start_buffer=coarse_step)
    start = max_r if search_start is None else search_start
    return _lcds_search(mag_geo, time, alpha_eq_deg, maginput, search, start)




if __name__ == "__main__":
    # Illustrative usage (requires libirbem.so and a valid maginput for the model).
    logging.basicConfig(level=logging.INFO)

    res = compute_lcds(
        datetime(2015, 3, 17, 12, 0, 0), alpha_eq_deg=90.0, maginput={"Kp": 6.0}, kext="T89", max_r=6.6
    )
    if not res.found:
        print("No closed drift shell found.")
    elif res.at_ceiling:
        print(f"Shell still closed at the ceiling: LCDS >= {res.lcds:.3f} (boundary beyond {res.x_sm:.2f} RE).")
    else:
        print(f"LCDS L* = {res.lcds:.3f} (SM x = {res.x_sm:.2f} RE)")
