# SPDX-FileCopyrightText: 2022 Mykhaylo Shumko
# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: LGPL-3.0-only

import copy
import ctypes
import itertools
import logging
import os
import pathlib
import shutil
import sys
import typing
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Literal, NamedTuple, Optional

import dateutil.parser
import numpy as np
import pandas as pd
from numpy.typing import NDArray

import el_paso as ep
from el_paso.processing.magnetic_field_utils.construct_maginput import MagInputKeys

__author__ = "Mykhaylo Shumko"
__last_modified__ = "2022-06-16"
__credit__ = "IRBEM-LIB development team"

"""
Copyright 2022, Mykhaylo Shumko

IRBEM magnetic coordinates and fields wrapper class for Python. Source code
credit goes to the IRBEM-LIB development team.

***************************************************************************
IRBEM-LIB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

IRBEM-LIB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with IRBEM-LIB.  If not, see <http://www.gnu.org/licenses/>.
***************************************************************************
"""

# External magnetic field model look up table.
EXT_MODELS = [
    "None",
    "MF75",
    "TS87",
    "TL87",
    "T89",
    "OPQ77",
    "OPD88",
    "T96",
    "OM97",
    "T01",
    "T01S",
    "T04",
    "A00",
    "T07",
    "MT",
]

DEFAULT_LIBIRBEM_PATH = Path(ep.__file__).parent / "libirbem.so"

logger = logging.getLogger(__name__)

BADDATA = -1e31


def _baddata_to_nan(x: object) -> NDArray[np.float64]:
    """Replaces IRBEM's baddata sentinel (-1e31) with NaN in an array or ctypes array."""
    arr = np.array(x, dtype=np.float64)
    return np.where(arr <= BADDATA / 10, np.nan, arr)


def _scalar_baddata_to_nan(x: float) -> float:
    """Replaces IRBEM's baddata sentinel (-1e31) with NaN in a single value."""
    return float(_baddata_to_nan(x))


class MakeLstarOutput(NamedTuple):
    """Container for outputs of L* calculations for a single shell.

    Attributes:
        lm (NDArray[np.float64]): The Mcllwain L parameter.
        mlt (NDArray[np.float64]): Magnetic Local Time (MLT) in hours.
        blocal (NDArray[np.float64]): The local magnetic field magnitude in nT.
        bmin (NDArray[np.float64]): The minimum magnetic field magnitude along the field line in nT.
        lstar (NDArray[np.float64]): The L* value.
        xj (NDArray[np.float64]): The third adiabatic invariant, J, scaled by the Earth's radius.
    """

    lm: NDArray[np.float64]
    mlt: NDArray[np.float64]
    blocal: NDArray[np.float64]
    bmin: NDArray[np.float64]
    lstar: NDArray[np.float64]
    xj: NDArray[np.float64]


class GetFieldMultiOutput(NamedTuple):
    """Container for vectorized magnetic field data.

    Attributes:
        bgeo (NDArray[np.float64]): The magnetic field vector in GEO coordinates.
        blocal (NDArray[np.float64]): The magnetic field magnitude in nT.
    """

    bgeo: NDArray[np.float64]
    blocal: NDArray[np.float64]


class MakeLstarShellSplittingOutput(NamedTuple):
    """Container for L* and related parameters for multiple pitch angles.

    This class is used when the calculation is performed for multiple pitch angles
    simultaneously, producing arrays of results.

    Attributes:
        lm (NDArray[np.float64]): The Mcllwain L parameter for each pitch angle.
        mlt (NDArray[np.float64]): Magnetic Local Time (MLT) in hours for each pitch angle.
        blocal (NDArray[np.float64]): The local magnetic field magnitude in nT for each pitch angle.
        bmin (NDArray[np.float64]): The minimum magnetic field magnitude along the field line in nT.
        lstar (NDArray[np.float64]): The L* value for each pitch angle.
        xj (NDArray[np.float64]): The third adiabatic invariant, J, scaled by the Earth's radius, for each pitch angle.
    """

    lm: NDArray[np.float64]
    mlt: NDArray[np.float64]
    blocal: NDArray[np.float64]
    bmin: NDArray[np.float64]
    lstar: NDArray[np.float64]
    xj: NDArray[np.float64]


class DriftLossConeStatus(IntEnum):
    """Status code returned by `drift_loss_cone`, i.e. the Fortran `iflag`.

    Attributes:
        OK: Both boundaries resolved normally.
        ALL_LOST: The whole local drift shell is inside the loss cone, so
            `alpha_dlc` is 90 degrees. The drift orbit dips below `stop_alt`
            even for equatorially mirroring particles. Common in low-altitude
            equatorial orbits.
        NO_DRIFT_CONE: The drift loss cone is not resolvably wider than the
            bounce loss cone. Typical when the spacecraft is itself over the
            South Atlantic Anomaly, where the local field line already sets
            the minimum mirror altitude of the whole drift orbit.
        NO_BMIN: `alpha_dlc` was found but Bmin on the local field line was
            not, so only the local pitch angles are valid; the equatorial
            ones are NaN.
        PARTIAL_TRACE_FAILURE: The boundary is bracketed to within
            `dlc_tol`, but one or more trial pitch angles inside the loss
            cone could not be traced and were treated as lost. Discard these
            points if you want the strict policy.
        SETUP_FAILED: Field model or coordinate setup failed.
        NO_FOOT_POINT: No foot point at `stop_alt` was found in either
            hemisphere; nothing was computed.
        SHELL_NOT_CLOSED: The drift-bounce orbit does not close even for
            equatorially mirroring particles (open field line, or
            magnetopause shadowing at high L); nothing was computed.
        BAD_STOP_ALT: `stop_alt` out of range.
        BAD_DLC_TOL: `dlc_tol` outside (0, 90) degrees.
    """

    OK = 0
    ALL_LOST = 1
    NO_DRIFT_CONE = 2
    NO_BMIN = 3
    PARTIAL_TRACE_FAILURE = 4
    SETUP_FAILED = -1
    NO_FOOT_POINT = -2
    SHELL_NOT_CLOSED = -3
    BAD_STOP_ALT = -4
    BAD_DLC_TOL = -5


class DriftLossConeOutput(NamedTuple):
    """Container for drift loss cones at a series of spacecraft locations.

    Every entry is an array with one element per input time step.

    The two boundaries partition pitch angle space into three classes: below `alpha_blc` a particle is
    lost within a quarter bounce; between `alpha_blc` and `alpha_dlc` it survives locally but mirrors
    below `stop_alt` somewhere else on its drift orbit (quasi-trapped); above `alpha_dlc` it is stably
    trapped. Because the drift shell depends only on the mirror field, the result is independent of
    species and energy, and it is symmetric about 90 degrees: the loss cone around the
    anti-field-aligned direction runs from 180 minus these values.

    Attributes:
        alpha_blc_eq (NDArray[np.float64]): Bounce loss cone, equatorial pitch angle, in degrees.
        alpha_dlc_eq (NDArray[np.float64]): Drift loss cone, equatorial pitch angle, in degrees.
        alpha_blc_loc (NDArray[np.float64]): Bounce loss cone, local pitch angle, in degrees.
        alpha_dlc_loc (NDArray[np.float64]): Drift loss cone, local pitch angle, in degrees.
        blocal (NDArray[np.float64]): The magnetic field magnitude at the spacecraft in nT.
        bmin (NDArray[np.float64]): The minimum magnetic field magnitude along the local field line in nT.
        lm (NDArray[np.float64]): The Mcllwain L parameter of the marginally trapped drift shell.
        lstar (NDArray[np.float64]): The L* value of the marginally trapped drift shell.
        hmin (NDArray[np.float64]): The lowest geodetic altitude in km on the marginally trapped drift orbit.
        hmin_lon (NDArray[np.float64]): The geodetic longitude in degrees at which `hmin` occurs.
        status (NDArray[np.int32]): `DriftLossConeStatus` codes, one per time step.
    """

    alpha_blc_eq: NDArray[np.float64]
    alpha_dlc_eq: NDArray[np.float64]
    alpha_blc_loc: NDArray[np.float64]
    alpha_dlc_loc: NDArray[np.float64]
    blocal: NDArray[np.float64]
    bmin: NDArray[np.float64]
    lm: NDArray[np.float64]
    lstar: NDArray[np.float64]
    hmin: NDArray[np.float64]
    hmin_lon: NDArray[np.float64]
    status: NDArray[np.int32]


class DriftShellOutput(NamedTuple):
    """Container for outputs related to a particle drift shell.

    Attributes:
        lm (float): The Mcllwain L parameter.
        lstar (float): The L* value.
        blocal (NDArray[np.float64]): The local magnetic field magnitude along the drift shell.
        bmin (float): The minimum magnetic field magnitude along the field line in nT.
        xj (float): The third adiabatic invariant, J, scaled by the Earth's radius.
        posit (NDArray[np.float64]): The coordinates of the drift shell.
        nposit (NDArray[np.int32]): The number of points in the `posit` array.
    """

    lm: float
    lstar: float
    blocal: NDArray[np.float64]
    bmin: float
    xj: float
    posit: NDArray[np.float64]
    nposit: NDArray[np.int32]


class FindMagEquatorOutput(NamedTuple):
    """Container for the magnetic equator's location and field strength.

    Attributes:
        xgeo (NDArray[np.float64]): The position of the magnetic equator in GEO coordinates.
        bmin (float): The minimum magnetic field magnitude at the equator in nT.
    """

    xgeo: NDArray[np.float64]
    bmin: float


class TraceFieldLineOutput(NamedTuple):
    """Container for outputs from tracing a magnetic field line.

    Attributes:
        posit (NDArray[np.float64]): The coordinates along the traced field line.
        n_posit (int): The number of points in the `posit` array.
        lm (float): The Mcllwain L parameter.
        blocal (NDArray[np.float64]): The local magnetic field magnitude along the field line in nT.
        bmin (float): The minimum magnetic field magnitude along the field line in nT.
        xj (float): The third adiabatic invariant, J, scaled by the Earth's radius.
    """

    posit: NDArray[np.float64]
    n_posit: int
    lm: float
    blocal: NDArray[np.float64]
    bmin: float
    xj: float


class FindFootPointOutput(NamedTuple):
    """Container for outputs of a magnetic field line foot point.

    Attributes:
        x_foot (NDArray[np.float64]): The foot point location in GDZ coordinates.
        b_foot (NDArray[np.float64]): The magnetic field vector at the foot point in GEO coordinates.
        b_foot_mag (float): The magnetic field magnitude at the foot point in nT.
    """

    x_foot: NDArray[np.float64]
    b_foot: NDArray[np.float64]
    b_foot_mag: float


class FindMirrorPointOutput(NamedTuple):
    """Container for outputs of a magnetic mirror point calculation.

    Attributes:
        blocal (float): The magnetic field magnitude at the local position in nT.
        bmirr (float): The magnetic field magnitude at the mirror point in nT. NaN if the particle has no
            mirror point on this field line, i.e. is in the loss cone.
        posit (NDArray[np.float64]): The location of the mirror point in GEO coordinates.
    """

    blocal: float
    bmirr: float
    posit: NDArray[np.float64]


class LstarQuantity(IntEnum):
    """Quantity computed via drift-shell tracing, i.e. IRBEM's `options(1)`.

    Drift-shell tracing is the most computationally expensive part of a call to
    `make_lstar`, so this should be set to `NONE` whenever L* or Phi are not needed.

    Attributes:
        NONE: Skip drift-shell tracing entirely. L* and Phi are not computed
            (returned as NaN), but Lm, Bmin and Blocal are still calculated.
        LSTAR: Compute the Roederer L* (the third adiabatic invariant expressed as
            an equivalent dipole L value).
        PHI: Compute the drift-shell magnetic flux Phi (the third adiabatic
            invariant itself) instead of L*.
    """

    NONE = 0
    LSTAR = 1
    PHI = 2


class InternalFieldModel(IntEnum):
    """Internal (Earth) magnetic field model, i.e. IRBEM's `options(5)`."""

    IGRF = 0
    ECCENTRIC_TILTED_DIPOLE = 1
    JENSEN_CAIN_1960 = 2
    GSFC_1266_1970 = 3
    USER_DEFINED = 4
    CENTERED_DIPOLE = 5


class IrbemOptions(NamedTuple):
    """Descriptive representation of IRBEM-LIB's 5-element `options` array.

    See https://prbem.github.io/IRBEM/api/general_information.html for details.

    Attributes:
        lstar_quantity (LstarQuantity): Whether and what drift-shell quantity to
            compute. Corresponds to `options(1)`. See `LstarQuantity` for the
            trade-off between the computed quantity and computation time.
        igrf_update_interval_days (int): How often the IGRF coefficients are
            updated, in days. Corresponds to `options(2)`. 0 means once per year,
            at the middle of the year.
        field_line_resolution (int): Resolution of the field line tracing used for
            L*, from 0 (fastest, ~2% error at L=6) to 9 (most accurate). Corresponds
            to `options(3)`.
        drift_shell_resolution (int): Resolution of the drift shell tracing used for
            L*, from 0 (sufficient outside LEO) to 9 (most accurate). Corresponds to
            `options(4)`.
        internal_field_model (InternalFieldModel): Internal magnetic field model
            used by IRBEM. Corresponds to `options(5)`.
    """

    lstar_quantity: LstarQuantity = LstarQuantity.LSTAR
    igrf_update_interval_days: int = 1
    field_line_resolution: int = 4
    drift_shell_resolution: int = 4
    internal_field_model: InternalFieldModel = InternalFieldModel.IGRF


class MagFields:
    """A class to interface with the IRBEM library for magnetic field calculations.

    This class provides a Pythonic wrapper around the Fortran-based IRBEM library,
    allowing users to perform a variety of magnetospheric calculations, such as
    tracing magnetic field lines, finding L*, and calculating Magnetic Local Time (MLT).

    Every floating point output has IRBEM's baddata sentinel (-1e31) replaced with NaN, so
    callers never see it. Other IRBEM conventions are passed through unchanged; in particular a
    negative Lm or L* is IRBEM's flag for a particle whose mirror point is in the loss cone,
    and its magnitude is still the L value.

    Attributes:
        irbem_obj_path (Path): The path to the IRBEM shared library object.
        kext (ctypes.c_int): The integer code for the selected external magnetic field model.
        sysaxes (ctypes.c_int): The coordinate system code (e.g., GEO, GSE, GSM).
        options (ctypes.Array[ctypes.c_int]): An array of five integers to set calculation options.
        ntime_max (ctypes.c_int): The maximum number of time steps the IRBEM library can process in a single call.
    """

    def __init__(
        self,
        *,
        lib_path: Optional[str | Path] = DEFAULT_LIBIRBEM_PATH,
        kext: int | str = 5,
        sysaxes: int = 0,
        options: IrbemOptions | None = None,
    ) -> None:
        """Initializes the MagFields object and loads the IRBEM shared library.

        Args:
            lib_path (str | Path, optional): The path to the IRBEM shared library file.
                                                Defaults to DEFAULT_LIBIRBEM_PATH.
            kext (int | str, optional): The code for the external magnetic field model. Can be an integer
                                        (e.g., 5 for OPQ77) or a string (e.g., 'T96'). Defaults to 5 (OPQ77).
            sysaxes (int, optional): The coordinate system for input positions.
                                     0 for GEI/GEO (default), 1 for GSE, 2 for GSM. Defaults to 0.
            options (IrbemOptions | None, optional): The IRBEM-LIB options to use.
                                                      Defaults to all zeros if None.

        Raises:
            ValueError: If an incorrect external magnetic field model string is provided.
        """
        if isinstance(lib_path, str):
            lib_path = Path(lib_path)

        self.irbem_obj_path = lib_path
        self.irbem_obj_path, self._irbem_obj = _load_shared_object(self.irbem_obj_path)

        # global model parameters, default is OPQ77 model with GDZ coordinate
        # system. If kext is a string, find the corresponding integer value.
        if isinstance(kext, str):
            try:
                self.kext = ctypes.c_int(EXT_MODELS.index(kext))
            except ValueError as err:
                msg = (
                    "Incorrect external model selected. Valid models are 'None', 'MF75',"
                    "'TS87', 'TL87', 'T89', 'OPQ77', 'OPD88', 'T96', 'OM97'"
                    "'T01', 'T04', 'A00'"
                )
                raise ValueError(msg) from err
        else:
            self.kext = ctypes.c_int(kext)

        self.sysaxes = ctypes.c_int(sysaxes)

        # If options are not supplied, assume they are all 0's.
        options_type = ctypes.c_int * 5
        if options is None:
            self.options = options_type(0, 0, 0, 0, 0)
        else:
            self.options = options_type()
            for i in range(5):
                self.options[i] = options[i]

        # Get the NTIME_MAX value
        self.ntime_max = ctypes.c_int(-1)
        self._irbem_obj.get_irbem_ntime_max1_(ctypes.byref(self.ntime_max))

    def make_lstar(
        self,
        time: Sequence[datetime | str] | datetime | str,
        position: Mapping[Literal["x1", "x2", "x3"], Sequence[np.floating] | NDArray[np.floating] | np.floating],
        maginput: Mapping[MagInputKeys, NDArray[np.number] | list[np.number] | np.number],
    ) -> MakeLstarOutput:
        """Calculates L* and related parameters for a series of time steps and positions.

        This method is a vectorized wrapper around the `make_lstar1_` Fortran subroutine,
        which efficiently handles multiple input points.

        Args:
            time (Sequence[datetime | str] | datetime | str): A single datetime object or a sequence
                                                              of datetime objects or ISO-formatted strings.
            position (Mapping): A dictionary containing 'x1', 'x2', and 'x3' keys with single values
                                or sequences of coordinates corresponding to each time step.
            maginput (Mapping): A dictionary of magnetic field model input parameters. The keys
                                correspond to different models (e.g., 'Kp', 'Dst', 'Pdyn').

        Returns:
            MakeLstarOutput: An object containing NumPy arrays of the calculated L*, Lm, B_local, B_min, XJ.
        """
        # Convert the satellite time and position into c objects.
        c_ntime, c_iyear, c_idoy, c_ut, c_x1, c_x2, c_x3 = self._prep_time_pos_array(time, position)

        # Convert the model parameters into c objects.
        c_maginput = self._prep_maginput(maginput)

        # Model outputs
        double_arr_type = ctypes.c_double * c_ntime.value
        c_lm, c_lstar, c_blocal, c_bmin, c_xj, c_mlt = [double_arr_type() for _ in range(6)]

        logger.debug("Running IRBEM-LIB make_lstar")

        self._irbem_obj.make_lstar1_(
            ctypes.byref(c_ntime),
            ctypes.byref(self.kext),
            ctypes.byref(self.options),
            ctypes.byref(self.sysaxes),
            ctypes.byref(c_iyear),
            ctypes.byref(c_idoy),
            ctypes.byref(c_ut),
            ctypes.byref(c_x1),
            ctypes.byref(c_x2),
            ctypes.byref(c_x3),
            ctypes.byref(c_maginput),
            ctypes.byref(c_lm),
            ctypes.byref(c_lstar),
            ctypes.byref(c_blocal),
            ctypes.byref(c_bmin),
            ctypes.byref(c_xj),
            ctypes.byref(c_mlt),
        )

        return MakeLstarOutput(
            lm=_baddata_to_nan(c_lm),
            lstar=_baddata_to_nan(c_lstar),
            blocal=_baddata_to_nan(c_blocal),
            bmin=_baddata_to_nan(c_bmin),
            mlt=_baddata_to_nan(c_mlt),
            xj=_baddata_to_nan(c_xj),
        )

    def make_lstar_shell_splitting(
        self,
        time: Sequence[datetime | str] | datetime | str,
        position: Mapping[Literal["x1", "x2", "x3"], Sequence[np.floating] | NDArray[np.floating] | np.floating],
        maginput: Mapping[MagInputKeys, NDArray[np.number] | list[np.number] | np.number],
        alpha: Sequence[np.floating] | NDArray[np.floating] | np.floating,
    ) -> MakeLstarShellSplittingOutput:
        """Calculates L* for multiple pitch angles for a series of time steps.

        This function handles the "shell splitting" calculation, where L* is computed
        for a range of pitch angles at each time step.

        Args:
            time (Sequence[datetime | str] | datetime | str): Time or sequence of times for the calculation.
            position (Mapping): Position or sequence of positions for the calculation.
            maginput (Mapping): Magnetic field model inputs for the calculation.
            alpha (Sequence[np.floating] | NDArray[np.floating] | np.floating): A single pitch angle
                                                                               or a sequence of pitch angles in degrees.

        Returns:
            MakeLstarShellSplittingOutput: An object containing 2D NumPy arrays
            of the calculated L* and related parameters.
        """
        c_ntime, c_iyear, c_idoy, c_ut, c_x1, c_x2, c_x3 = self._prep_time_pos_array(time, position)

        if not isinstance(alpha, Sequence | np.ndarray):
            alpha = [alpha]

        # Convert the model parameters into c objects.
        c_maginput = self._prep_maginput(maginput)

        logger.debug("Running IRBEM-LIB make_lstar_shell_splitting")

        # The Fortran routine declares its outputs as (NTIME_MAX, 25), so pitch angle k starts NTIME_MAX
        # elements after pitch angle k-1. Buffers of that size would take hundreds of MB, and buffers of
        # ntime * n_alpha are overrun (and the process crashes) as soon as there is a second pitch angle.
        # One call per pitch angle only ever writes the first column, which fits in ntime elements.
        c_n_alpha = ctypes.c_int(1)
        double_arr_type = ctypes.c_double * c_ntime.value
        outputs: dict[str, list[NDArray[np.float64]]] = {
            key: [] for key in ("lm", "lstar", "blocal", "bmin", "xj", "mlt")
        }

        for pitch_angle in alpha:
            c_alpha = (ctypes.c_double * 1)(float(pitch_angle))
            c_lm, c_lstar, c_blocal, c_bmin, c_xj, c_mlt = [double_arr_type() for _ in range(6)]

            self._irbem_obj.make_lstar_shell_splitting1_(
                ctypes.byref(c_ntime),
                ctypes.byref(c_n_alpha),
                ctypes.byref(self.kext),
                ctypes.byref(self.options),
                ctypes.byref(self.sysaxes),
                ctypes.byref(c_iyear),
                ctypes.byref(c_idoy),
                ctypes.byref(c_ut),
                ctypes.byref(c_x1),
                ctypes.byref(c_x2),
                ctypes.byref(c_x3),
                ctypes.byref(c_alpha),
                ctypes.byref(c_maginput),
                ctypes.byref(c_lm),
                ctypes.byref(c_lstar),
                ctypes.byref(c_blocal),
                ctypes.byref(c_bmin),
                ctypes.byref(c_xj),
                ctypes.byref(c_mlt),
            )

            for key, c_arr in zip(outputs, (c_lm, c_lstar, c_blocal, c_bmin, c_xj, c_mlt), strict=True):
                outputs[key].append(_baddata_to_nan(c_arr))

        # Bmin and MLT do not depend on the pitch angle; they are repeated per row to keep one shape.
        return MakeLstarShellSplittingOutput(**{key: np.stack(rows) for key, rows in outputs.items()})

    def drift_shell(
        self,
        time: datetime | str | pd.Timestamp,
        position: Mapping[Literal["x1", "x2", "x3"], np.floating],
        maginput: Mapping[MagInputKeys, NDArray[np.number] | list[np.number] | np.number],
    ) -> DriftShellOutput:
        """Traces a particle drift shell for a single time and position.

        Args:
            time (datetime | str | pd.Timestamp): The time of the calculation.
            position (Mapping): The starting position of the drift shell tracing.
            maginput (Mapping): Magnetic field model inputs.

        Returns:
            DriftShellOutput: An object containing the traced drift shell coordinates and associated parameters.
        """
        c_iyear, c_idoy, c_ut, c_x1, c_x2, c_x3 = self._prep_time_pos(time, position)
        c_maginput = self._prep_maginput(maginput)

        logger.debug("Running IRBEM-LIB drift_shell for multiple time steps")

        c_posit = (((ctypes.c_double * 3) * 1000) * 48)()
        c_nposit = (48 * ctypes.c_int)()
        c_lm, c_lstar, c_bmin, c_xj = [ctypes.c_double() for _ in range(4)]
        c_blocal = ((ctypes.c_double * 1000) * 48)()

        self._irbem_obj.drift_shell1_(
            ctypes.byref(self.kext),
            ctypes.byref(self.options),
            ctypes.byref(self.sysaxes),
            ctypes.byref(c_iyear),
            ctypes.byref(c_idoy),
            ctypes.byref(c_ut),
            ctypes.byref(c_x1),
            ctypes.byref(c_x2),
            ctypes.byref(c_x3),
            ctypes.byref(c_maginput),
            ctypes.byref(c_lm),
            ctypes.byref(c_lstar),
            ctypes.byref(c_blocal),
            ctypes.byref(c_bmin),
            ctypes.byref(c_xj),
            ctypes.byref(c_posit),
            ctypes.byref(c_nposit),
        )

        posit = _baddata_to_nan(c_posit)
        nposit = np.array(c_nposit)
        for i, n in enumerate(nposit):
            posit[i, n:, :] = np.nan

        return DriftShellOutput(
            lm=_scalar_baddata_to_nan(c_lm.value),
            lstar=_scalar_baddata_to_nan(c_lstar.value),
            blocal=_baddata_to_nan(c_blocal),
            bmin=_scalar_baddata_to_nan(c_bmin.value),
            xj=_scalar_baddata_to_nan(c_xj.value),
            posit=posit,
            nposit=nposit,
        )

    _DLC_N_OUT = len(DriftLossConeOutput._fields) - 1

    def drift_loss_cone(
        self,
        time: Sequence[datetime | str] | datetime | str,
        position: Mapping[Literal["x1", "x2", "x3"], Sequence[np.floating] | NDArray[np.floating] | np.floating],
        maginput: Mapping[MagInputKeys, NDArray[np.number] | list[np.number] | np.number],
        stop_alt: float = 100,
        dlc_tol: float = 0.1,
    ) -> DriftLossConeOutput:
        """Calculates the bounce and drift loss cones for a series of time steps and positions.

        Unlike most of the other calls, this one takes no pitch angle: it returns the two pitch angles
        that bound the loss cones, so a measured pitch angle can be classified against them.

        The bounce loss cone comes from the local field line, tracing to `stop_alt` in both hemispheres
        and taking the weaker of the two foot point fields. The drift loss cone is found by bisecting
        on pitch angle, tracing the full drift-bounce orbit at each trial angle and comparing its
        minimum altitude against `stop_alt`, until the bracket is narrower than `dlc_tol`.

        The bisection returns the midpoint of its final bracket, so `alpha_dlc_eq` is within
        `dlc_tol / 2` of the true boundary and lies on a grid of that spacing: where the true boundary
        varies smoothly between neighbouring locations, the returned one can still jump by up to
        `dlc_tol`. Lower it when comparing the boundary across locations; each halving costs one more
        drift shell trace per location.

        This costs roughly a dozen drift shell traces per location, so pass whole orbits in one call
        rather than looping. `stop_alt` is common to all time steps. If `maginput` is given as scalars
        rather than sequences, the same values are used at every time step. Note that `options[3]`
        controls how finely the drift orbit samples longitude and therefore how well the South Atlantic
        Anomaly is resolved; 0 gives 14.4 degree steps, which can step over the bottom of the anomaly.

        Args:
            time (Sequence[datetime | str] | datetime | str): A single datetime object or a sequence of
                datetime objects or ISO-formatted strings.
            position (Mapping): A dictionary containing 'x1', 'x2' and 'x3' keys with single values or
                sequences of coordinates corresponding to each time step.
            maginput (Mapping): Magnetic field model inputs, either scalars or one value per time step.
            stop_alt (float, optional): The geodetic altitude in km below which a particle counts as
                lost to the atmosphere. Defaults to 100.
            dlc_tol (float, optional): The bisection tolerance on `alpha_dlc_eq`, in degrees. Must lie
                in (0, 90); otherwise every time step comes back with status `BAD_DLC_TOL`. Defaults
                to 0.1.

        Returns:
            DriftLossConeOutput: Arrays of the two loss cone boundaries and the properties of the
            marginally trapped drift shell, one element per time step. Quantities that could not be
            computed come back as NaN; check `status` before using them.
        """
        c_ntime, c_iyear, c_idoy, c_ut, c_x1, c_x2, c_x3 = self._prep_time_pos_array(time, position)
        ntime = c_ntime.value

        c_maginput = self._prep_maginput(maginput)
        if not isinstance(c_maginput[0], ctypes.Array):
            # Scalar model inputs: broadcast the single 25-element row to every time step,
            # since the Fortran routine indexes maginput as (25, ntime).
            c_maginput_scalar = c_maginput
            c_maginput = ((ctypes.c_double * 25) * ntime)()
            for it, i in itertools.product(range(ntime), range(25)):
                c_maginput[it][i] = c_maginput_scalar[i]

        c_stop_alt = ctypes.c_double(stop_alt)
        c_dlc_tol = ctypes.c_double(dlc_tol)

        logger.debug("Running IRBEM-LIB drift_loss_cone for multiple time steps")

        double_arr_type = ctypes.c_double * ntime
        c_out = [double_arr_type() for _ in range(self._DLC_N_OUT)]
        c_iflag = (ctypes.c_int * ntime)()

        self._irbem_obj.drift_loss_cone_multi_(
            ctypes.byref(c_ntime),
            ctypes.byref(self.kext),
            ctypes.byref(self.options),
            ctypes.byref(self.sysaxes),
            ctypes.byref(c_iyear),
            ctypes.byref(c_idoy),
            ctypes.byref(c_ut),
            ctypes.byref(c_x1),
            ctypes.byref(c_x2),
            ctypes.byref(c_x3),
            ctypes.byref(c_maginput),
            ctypes.byref(c_stop_alt),
            ctypes.byref(c_dlc_tol),
            *[ctypes.byref(o) for o in c_out],
            ctypes.byref(c_iflag),
        )

        return DriftLossConeOutput(
            *[_baddata_to_nan(o) for o in c_out],
            status=np.array(c_iflag, dtype=np.int32),
        )

    def find_mirror_point(
        self,
        time: datetime | str | pd.Timestamp,
        position: Mapping[Literal["x1", "x2", "x3"], np.floating],
        maginput: Mapping[MagInputKeys, NDArray[np.number] | list[np.number] | np.number],
        alpha: float,
    ) -> FindMirrorPointOutput:
        """Finds the magnetic mirror point for a given location and pitch angle.

        This method identifies the point along a magnetic field line where a charged particle
        with a given pitch angle will "mirror" and reverse its direction.

        Args:
            time (datetime | str | pd.Timestamp): The time of the calculation.
            position (Mapping): The starting position for the field line tracing.
            maginput (Mapping): Magnetic field model inputs.
            alpha (float): The local pitch angle in degrees.

        Returns:
            FindMirrorPointOutput: An object containing the location and magnetic field
                                   values at the mirror point.
        """
        c_iyear, c_idoy, c_ut, c_x1, c_x2, c_x3 = self._prep_time_pos(time, position)
        c_maginput = self._prep_maginput(maginput)

        c_alpha = ctypes.c_double(alpha)

        logger.debug("Running IRBEM-LIB find_mirror_point for multiple time steps and pitch angles")

        c_blocal = ctypes.c_double(BADDATA)
        c_bmirr = ctypes.c_double(BADDATA)
        c_posit = (3 * ctypes.c_double)(BADDATA, BADDATA, BADDATA)

        self._irbem_obj.find_mirror_point1_(
            ctypes.byref(self.kext),
            ctypes.byref(self.options),
            ctypes.byref(self.sysaxes),
            ctypes.byref(c_iyear),
            ctypes.byref(c_idoy),
            ctypes.byref(c_ut),
            ctypes.byref(c_x1),
            ctypes.byref(c_x2),
            ctypes.byref(c_x3),
            ctypes.byref(c_alpha),
            ctypes.byref(c_maginput),
            ctypes.byref(c_blocal),
            ctypes.byref(c_bmirr),
            ctypes.byref(c_posit),
        )

        return FindMirrorPointOutput(
            blocal=_scalar_baddata_to_nan(c_blocal.value),
            bmirr=_scalar_baddata_to_nan(c_bmirr.value),
            posit=_baddata_to_nan(c_posit),
        )

    def find_foot_point(
        self,
        time: datetime | str | pd.Timestamp,
        position: Mapping[Literal["x1", "x2", "x3"], np.floating],
        maginput: Mapping[MagInputKeys, NDArray[np.number] | list[np.number] | np.number],
        stop_alt: float,
        hemi_flag: int,
    ) -> FindFootPointOutput:
        """Finds the magnetic field line footprint at a specified altitude.

        This method traces a field line from a given starting position to a `stop_alt` in one hemisphere.

        Args:
            time (datetime | str | pd.Timestamp): The time of the calculation.
            position (Mapping): The starting position for the field line tracing.
            maginput (Mapping): Magnetic field model inputs.
            stop_alt (float): The desired altitude (in km) for the footprint.
            hemi_flag (int): The hemisphere to trace to (e.g., +1 for northern, -1 for southern).

        Returns:
            FindFootPointOutput: An object containing the footprint's position, magnetic field vector, and magnitude.
        """
        c_iyear, c_idoy, c_ut, c_x1, c_x2, c_x3 = self._prep_time_pos(time, position)
        c_maginput = self._prep_maginput(maginput)

        c_stop_alt = ctypes.c_double(stop_alt)
        c_hemi_flag = ctypes.c_int(hemi_flag)

        logger.debug("Running IRBEM-LIB find_foot_point")

        c_xfoot = (ctypes.c_double * 3)()
        c_bfoot = (ctypes.c_double * 3)()
        c_bfootmag = ctypes.c_double()

        self._irbem_obj.find_foot_point1_(
            ctypes.byref(self.kext),
            ctypes.byref(self.options),
            ctypes.byref(self.sysaxes),
            ctypes.byref(c_iyear),
            ctypes.byref(c_idoy),
            ctypes.byref(c_ut),
            ctypes.byref(c_x1),
            ctypes.byref(c_x2),
            ctypes.byref(c_x3),
            ctypes.byref(c_stop_alt),
            ctypes.byref(c_hemi_flag),
            ctypes.byref(c_maginput),
            ctypes.byref(c_xfoot),
            ctypes.byref(c_bfoot),
            ctypes.byref(c_bfootmag),
        )

        # Stack the results into a single NumPy array, adding a new dimension for time
        return FindFootPointOutput(
            x_foot=_baddata_to_nan(c_xfoot),
            b_foot=_baddata_to_nan(c_bfoot),
            b_foot_mag=_scalar_baddata_to_nan(c_bfootmag.value),
        )

    def trace_field_line(
        self,
        time: datetime | str | pd.Timestamp,
        position: Mapping[Literal["x1", "x2", "x3"], np.floating],
        maginput: Mapping[MagInputKeys, NDArray[np.number] | list[np.number] | np.number],
        r0: float = 1,
    ) -> TraceFieldLineOutput:
        """Traces a magnetic field line from a starting position.

        Args:
            time (datetime | str | pd.Timestamp): The time of the calculation.
            position (Mapping): The starting position for the field line tracing.
            maginput (Mapping): Magnetic field model inputs.
            r0 (float, optional): The stopping altitude of the field line trace in Earth radii (Re). Defaults to 1.

        Returns:
            TraceFieldLineOutput: An object containing the traced field line coordinates and associated parameters.
        """
        c_iyear, c_idoy, c_ut, c_x1, c_x2, c_x3 = self._prep_time_pos(time, position)
        c_maginput = self._prep_maginput(maginput)

        c_r0 = ctypes.c_double(r0)

        logger.debug("Running IRBEM-LIB trace_field_line for multiple time steps")

        c_posit = ((ctypes.c_double * 3) * 3000)()
        # IRBEM leaves Nposit unset when the trace fails, so start from no points at all
        c_n_posit = ctypes.c_int(0)
        c_lm, c_bmin, c_xj = [ctypes.c_double(BADDATA) for _ in range(3)]
        c_blocal = (ctypes.c_double * 3000)()

        self._irbem_obj.trace_field_line2_1_(
            ctypes.byref(self.kext),
            ctypes.byref(self.options),
            ctypes.byref(self.sysaxes),
            ctypes.byref(c_iyear),
            ctypes.byref(c_idoy),
            ctypes.byref(c_ut),
            ctypes.byref(c_x1),
            ctypes.byref(c_x2),
            ctypes.byref(c_x3),
            ctypes.byref(c_maginput),
            ctypes.byref(c_r0),
            ctypes.byref(c_lm),
            ctypes.byref(c_blocal),
            ctypes.byref(c_bmin),
            ctypes.byref(c_xj),
            ctypes.byref(c_posit),
            ctypes.byref(c_n_posit),
        )

        n_posit = max(c_n_posit.value, 0)

        return TraceFieldLineOutput(
            posit=_baddata_to_nan(c_posit[:n_posit]).reshape(n_posit, 3),
            n_posit=n_posit,
            lm=_scalar_baddata_to_nan(c_lm.value),
            blocal=_baddata_to_nan(c_blocal[:n_posit]),
            bmin=_scalar_baddata_to_nan(c_bmin.value),
            xj=_scalar_baddata_to_nan(c_xj.value),
        )

    def find_magequator(
        self,
        time: datetime | str | pd.Timestamp,
        position: Mapping[Literal["x1", "x2", "x3"], np.floating],
        maginput: Mapping[MagInputKeys, NDArray[np.number] | list[np.number] | np.number],
    ) -> FindMagEquatorOutput:
        """Finds the magnetic equator for a given magnetic field line.

        Args:
            time (datetime | str | pd.Timestamp): The time of the calculation.
            position (Mapping): The starting position for the field line tracing.
            maginput (Mapping): Magnetic field model inputs.

        Returns:
            FindMagEquatorOutput: An object containing the magnetic equator's
            position and the minimum magnetic field magnitude.
        """
        c_iyear, c_idoy, c_ut, c_x1, c_x2, c_x3 = self._prep_time_pos(time, position)
        c_maginput = self._prep_maginput(maginput)

        logger.debug("Running IRBEM-LIB find_magequator for multiple time steps")

        c_xgeo = (ctypes.c_double * 3)(BADDATA, BADDATA, BADDATA)
        c_bmin = ctypes.c_double(BADDATA)

        self._irbem_obj.find_magequator1_(
            ctypes.byref(self.kext),
            ctypes.byref(self.options),
            ctypes.byref(self.sysaxes),
            ctypes.byref(c_iyear),
            ctypes.byref(c_idoy),
            ctypes.byref(c_ut),
            ctypes.byref(c_x1),
            ctypes.byref(c_x2),
            ctypes.byref(c_x3),
            ctypes.byref(c_maginput),
            ctypes.byref(c_bmin),
            ctypes.byref(c_xgeo),
        )

        return FindMagEquatorOutput(xgeo=_baddata_to_nan(c_xgeo), bmin=_scalar_baddata_to_nan(c_bmin.value))

    def get_field_multi(
        self,
        time: Sequence[datetime | str] | datetime | str,
        position: Mapping[Literal["x1", "x2", "x3"], Sequence[np.floating] | NDArray[np.floating] | np.floating],
        maginput: Mapping[MagInputKeys, NDArray[np.number] | list[np.number] | np.number],
    ) -> GetFieldMultiOutput:
        """Calculates the magnetic field vector and magnitude for a series of time steps.

        Args:
            time (Sequence[datetime | str] | datetime | str): A single datetime object or a sequence
                                                              of datetime objects or ISO-formatted strings.
            position (Mapping): A dictionary containing 'x1', 'x2', and 'x3' keys with single values
                                or sequences of coordinates corresponding to each time step.
            maginput (Mapping): Magnetic field model inputs.

        Returns:
            GetFieldMultiOutput: An object containing NumPy arrays of the magnetic field vector and magnitude.
        """
        # Prep the time and position variables.
        c_ntime, c_iyear, c_idoy, c_ut, c_x1, c_x2, c_x3 = self._prep_time_pos_array(time, position)

        # Prep magnetic field model inputs
        c_maginput = self._prep_maginput(maginput)

        # Model output arrays
        c_bmag = (ctypes.c_double * c_ntime.value)()
        c_bgeo = ((ctypes.c_double * 3) * c_ntime.value)()

        logger.debug("Running IRBEM-LIB get_field_multi")

        self._irbem_obj.get_field_multi_(
            ctypes.byref(c_ntime),
            ctypes.byref(self.kext),
            ctypes.byref(self.options),
            ctypes.byref(self.sysaxes),
            ctypes.byref(c_iyear),
            ctypes.byref(c_idoy),
            ctypes.byref(c_ut),
            ctypes.byref(c_x1),
            ctypes.byref(c_x2),
            ctypes.byref(c_x3),
            ctypes.byref(c_maginput),
            ctypes.byref(c_bgeo),
            ctypes.byref(c_bmag),
        )

        return GetFieldMultiOutput(_baddata_to_nan(c_bgeo), _baddata_to_nan(c_bmag))

    def get_mlt(
        self,
        time: datetime | str | pd.Timestamp,
        position: Mapping[Literal["x1", "x2", "x3"], np.floating],
    ) -> float:
        """Calculates the Magnetic Local Time (MLT) for a single time and position.

        Args:
            time (datetime | str | pd.Timestamp): The time of the calculation.
            position (Mapping): The position for which to calculate MLT.

        Returns:
            float: The calculated Magnetic Local Time in hours.
        """
        c_iyear, c_idoy, c_ut, c_x1, c_x2, c_x3 = self._prep_time_pos(time, position)

        logger.debug("Running IRBEM-LIB get_mlt in a time loop")

        c_mlt = ctypes.c_double(BADDATA)
        c_position = (ctypes.c_double * 3)(c_x1, c_x2, c_x3)

        self._irbem_obj.get_mlt1_(
            ctypes.byref(c_iyear),
            ctypes.byref(c_idoy),
            ctypes.byref(c_ut),
            ctypes.byref(c_position),
            ctypes.byref(c_mlt),
        )

        return _scalar_baddata_to_nan(c_mlt.value)

    def _prep_time_pos(
        self,
        time: datetime | str | pd.Timestamp,
        position: Mapping[Literal["x1", "x2", "x3"], np.floating],
    ) -> tuple[ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]:
        logger.debug("Prepping time and space input variables")

        # Deep copy X so if the single inputs get encapsulated in
        # an array, it wont be propaged back to the user.
        position = copy.deepcopy(position)
        time = copy.deepcopy(time)

        if isinstance(time, datetime):
            time_dt = time
        elif isinstance(time, pd.Timestamp):
            time_dt = time.to_pydatetime()
        else:
            time_dt = dateutil.parser.parse(time)

        c_iyear = ctypes.c_int(time_dt.year)
        c_idoy = ctypes.c_int(time_dt.timetuple().tm_yday)
        c_ut = ctypes.c_double(3600 * time_dt.hour + 60 * time_dt.minute + time_dt.second)  # Seconds of day
        c_x1 = ctypes.c_double(float(position["x1"]))
        c_x2 = ctypes.c_double(float(position["x2"]))
        c_x3 = ctypes.c_double(float(position["x3"]))

        logger.debug("Done prepping time and space input variables")

        return c_iyear, c_idoy, c_ut, c_x1, c_x2, c_x3

    def _prep_time_pos_array(
        self,
        time: Sequence[datetime | str | pd.Timestamp] | NDArray[np.generic] | datetime | str | pd.Timestamp,
        position: Mapping[Literal["x1", "x2", "x3"], Sequence[np.floating] | NDArray[np.floating] | np.floating],
    ) -> tuple[
        ctypes.c_int,
        ctypes.Array[ctypes.c_int],
        ctypes.Array[ctypes.c_int],
        ctypes.Array[ctypes.c_double],
        ctypes.Array[ctypes.c_double],
        ctypes.Array[ctypes.c_double],
        ctypes.Array[ctypes.c_double],
    ]:
        time = copy.deepcopy(time)
        time = np.atleast_1d(np.asarray(time))

        position = dict(copy.deepcopy(position))
        position["x1"] = np.atleast_1d(np.asarray(position["x1"]))
        position["x2"] = np.atleast_1d(np.asarray(position["x2"]))
        position["x3"] = np.atleast_1d(np.asarray(position["x3"]))

        # Check that the input array length does not exceed NTIME_MAX.
        if len(time) > self.ntime_max.value:
            msg = (
                f"Input array length {len(time)} is longer "
                f"than IRBEM's NTIME_MAX = {self.ntime_max.value}. "
                f"Use a for loop."
            )
            raise ValueError(msg)
        ntime = ctypes.c_int(len(time))

        # Check that the times are datetime objects, and convert otherwise.
        if isinstance(time[0], datetime):
            time = typing.cast("Sequence[datetime]", time)
            time_dt = time
        elif isinstance(time[0], pd.Timestamp):
            time = typing.cast("Sequence[pd.Timestamp]", time)
            time_dt = [t.to_pydatetime() for t in time]
        else:
            time = typing.cast("Sequence[str]", time)
            time_dt = [dateutil.parser.parse(t) for t in time]

        # C arrays are statically defined with the following procedure.
        # There are a few ways of doing this...
        iyear = (ctypes.c_int * len(time_dt))()
        idoy = (ctypes.c_int * len(time_dt))()

        ut, x1, x2, x3 = [(ctypes.c_double * len(time_dt))() for _ in range(4)]

        # Now fill the input time and model sampling (s/c location) parameters.
        for dt in range(len(time_dt)):
            iyear[dt] = time_dt[dt].year
            idoy[dt] = time_dt[dt].timetuple().tm_yday
            ut[dt] = 3600 * time_dt[dt].hour + 60 * time_dt[dt].minute + time_dt[dt].second
            x1[dt] = position["x1"][dt]
            x2[dt] = position["x2"][dt]
            x3[dt] = position["x3"][dt]

        return ntime, iyear, idoy, ut, x1, x2, x3

    def _prep_maginput(
        self, input_dict: Mapping[MagInputKeys, list[np.number] | NDArray[np.number] | np.number] | None = None
    ) -> ctypes.Array[ctypes.c_double] | ctypes.Array[ctypes.Array[ctypes.c_double]]:
        logger.debug("Prepping magnetic field inputs.")

        # If no model inputs (statis magnetic field model)
        if (input_dict is None) or (input_dict == {}):
            maginput = (ctypes.c_double * 25)()
            for i in range(25):
                maginput[i] = -9999
            return maginput

        ordered_keys = (
            "Kp",
            "Dst",
            "dens",
            "velo",
            "Pdyn",
            "ByIMF",
            "BzIMF",
            "G1",
            "G2",
            "G3",
            "W1",
            "W2",
            "W3",
            "W4",
            "W5",
            "W6",
            "AL",
        )

        # If the model inputs are arrays
        if isinstance(next(iter(input_dict.values())), list | np.ndarray):
            input_dict = typing.cast("dict[MagInputKeys, list[np.number] | NDArray[np.number]]", input_dict)
            ntime = len(next(iter(input_dict.values())))

            maginput = ((ctypes.c_double * 25) * ntime)()

            for i, key in enumerate(ordered_keys):
                for dt in range(ntime):
                    if key in input_dict:
                        maginput[dt][i] = input_dict[key][dt]
                    else:
                        maginput[dt][i] = ctypes.c_double(-9999)

        else:
            maginput = (ctypes.c_double * 25)()

            for i, key in enumerate(ordered_keys):
                if key in input_dict:
                    maginput[i] = input_dict[key]  # ty:ignore[invalid-argument-type]
                else:
                    maginput[i] = ctypes.c_double(-9999)

        logger.debug("Done prepping magnetic field inputs.")

        return maginput


SYSAXES_STR_TO_INT = {"GDZ": 0, "GEO": 1, "GSM": 2, "GSE": 3, "SM": 4, "GEI": 5, "MAG": 6, "SPH": 7, "RLL": 8}


class Coords:
    """A class to handle coordinate transformations using the IRBEM library.

    This class provides a Pythonic interface for converting coordinates between
    different systems (e.g., GEO, GSE, GSM) by calling the underlying
    Fortran routines of the IRBEM-LIB.
    """

    def __init__(self, *, lib_path: Optional[str | Path] = DEFAULT_LIBIRBEM_PATH) -> None:
        """Initializes the Coords object and loads the IRBEM shared library.

        Args:
            lib_path (Optional[str | Path]): The path to the IRBEM shared library file.
                                                    Defaults to DEFAULT_LIBIRBEM_PATH.
        """
        if isinstance(lib_path, str):
            lib_path = Path(lib_path)

        self.irbem_obj_path = lib_path

        self.irbem_obj_path, self._irbem_obj = _load_shared_object(self.irbem_obj_path)

    def transform(
        self,
        time: list[datetime] | list[str] | datetime | str,
        pos: NDArray[np.floating],
        sysaxes_in: int | str,
        sysaxes_out: int | str,
    ) -> NDArray[np.float64]:
        """Transforms coordinates from one system to another.

        This method is a vectorized wrapper around the `coord_trans_vec1_` Fortran subroutine,
        which handles multiple input positions and times efficiently.

        Args:
            time (list[datetime] | list[str] | datetime | str): A single datetime object or a list of
                                                                datetime objects or ISO-formatted strings.
            pos (NDArray[np.floating]): An array of input positions. The shape should be (N, 3), where
                                        N is the number of time steps.
            sysaxes_in (int | str): The integer code or string name of the input coordinate system
                                    (e.g., 0 or 'GDZ' for GEI/GEO).
            sysaxes_out (int | str): The integer code or string name of the output coordinate system.

        Returns:
            NDArray[np.float64]: A NumPy array of the transformed positions with shape (N, 3).
        """
        if isinstance(time, (datetime)):
            time = [time]
        if isinstance(time, (str)):
            time = [time]

        pos = np.atleast_2d(pos)

        c_pos_in = ((ctypes.c_double * 3) * len(time))()
        c_pos_out = ((ctypes.c_double * 3) * len(time))()
        c_iyear, c_idoy, c_ut = self._convert_to_c_times(time)
        c_sys_in = self._get_c_sysaxes(sysaxes_in)
        c_sys_out = self._get_c_sysaxes(sysaxes_out)
        c_ntime = ctypes.c_int(len(time))

        for it, ix in itertools.product(range(pos.shape[0]), range(3)):
            c_pos_in[it][ix] = ctypes.c_double(pos[it, ix])

        self._irbem_obj.coord_trans_vec1_(
            ctypes.byref(c_ntime),
            ctypes.byref(c_sys_in),
            ctypes.byref(c_sys_out),
            ctypes.byref(c_iyear),
            ctypes.byref(c_idoy),
            ctypes.byref(c_ut),
            ctypes.byref(c_pos_in),
            ctypes.byref(c_pos_out),
        )
        return _baddata_to_nan(c_pos_out)

    def _convert_to_c_times(
        self, time: list[datetime] | list[str] | datetime | str
    ) -> tuple[ctypes.Array[ctypes.c_int], ctypes.Array[ctypes.c_int], ctypes.Array[ctypes.c_double]]:
        if isinstance(time, (datetime)):
            time = [time]
        if isinstance(time, (str)):
            time = [time]

        iyear = (ctypes.c_int * len(time))()
        idoy = (ctypes.c_int * len(time))()
        ut = (ctypes.c_double * len(time))()

        if isinstance(time[0], str):
            time = typing.cast("list[str]", time)
            time = [dateutil.parser.parse(t) for t in time]

        assert isinstance(time[0], datetime)
        time = typing.cast("list[datetime]", time)

        for it, t in enumerate(time):
            iyear[it] = ctypes.c_int(t.year)
            idoy[it] = ctypes.c_int(t.timetuple().tm_yday)
            ut[it] = ctypes.c_double(3600 * t.hour + 60 * t.minute + t.second)

        return iyear, idoy, ut

    def _get_c_sysaxes(self, sysaxes: int | str) -> ctypes.c_int:
        if isinstance(sysaxes, str):
            assert sysaxes.upper() in SYSAXES_STR_TO_INT, (
                "ERROR: Unknown coordinate system! Choose from GDZ, GEO, GSM, GSE, SM, GEI, MAG, SPH, RLL."
            )
            return ctypes.c_int(SYSAXES_STR_TO_INT[sysaxes])
        if isinstance(sysaxes, int):
            return ctypes.c_int(sysaxes)
        msg = "Error, coordinate axis can only be a string or int!"
        raise ValueError(msg)


def _load_shared_object(path: Path | None = None) -> tuple[Path, ctypes.CDLL]:
    """Searches for and loads a shared object (.so or .dll file).

    If path is specified it doesn't search for the file.
    """
    if path is None:
        if (sys.platform == "win32") or (sys.platform == "cygwin"):
            obj_name = "libirbem.dll"
        else:
            obj_name = "libirbem.so"
        matched_object_files = list(Path(__file__).parents[2].rglob(obj_name))
        if len(matched_object_files) != 1:
            msg = (
                f"{len(matched_object_files)} .so or .dll shared object files found in "
                f"{Path(__file__).parents[2]} folder: {matched_object_files}."
            )
            raise ValueError(msg)

        path = matched_object_files[0]

    # Open the shared object file.
    try:
        if (sys.platform == "win32") or (sys.platform == "cygwin"):
            # Some versions of ctypes (Python) need to know where msys64 binary
            # files are located, or ctypes is unable to load the IREBM dll.
            gfortran_path = pathlib.Path(shutil.which("gfortran.exe"))
            os.add_dll_directory(gfortran_path.parent)  # e.g. C:\msys64\mingw64\bin
            irbem_obj = ctypes.WinDLL(str(path))
        else:
            irbem_obj = ctypes.CDLL(str(path))
    except OSError as err:
        if "Try using the full path with constructor syntax." in str(err):
            msg = f"Could not load the IRBEM shared object file in {path}"
            raise OSError(msg) from err
        raise

    return path, irbem_obj
