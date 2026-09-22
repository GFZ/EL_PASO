# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, NamedTuple, TypeVar

import numpy as np
from astropy import units as u
from numpy.typing import NDArray
from richpool import MultiPool

import el_paso as ep
from el_paso.processing.magnetic_field_utils import IrbemOptions
from el_paso.processing.magnetic_field_utils.irbem import (
    FORTRAN_BAD_VALUE,
    Coords,
    LCDSSearchParams,
    LstarQuantity,
    MagFields,
)
from el_paso.processing.magnetic_field_utils.mag_field_enum import MagneticField
from el_paso.typing import MagFieldVarTypes, MagInputKeys
from el_paso.utils import timed_function

logger = logging.getLogger(__name__)


def create_var_name(var_type: MagFieldVarTypes, mag_field: MagneticField) -> str:
    """Creates a standardized variable name combining the variable type and magnetic field model.

    This function is used internally to generate consistent variable names
    for all output quantities based on the magnetic field model used.

    Args:
        var_type (MagFieldVarTypes): The type of the magnetic field-related variable
                                     (e.g., "B_eq", "MLT").
        mag_field (MagneticField): The specific magnetic field model used for the calculation.

    Returns:
        str: The concatenated and standardized variable name.
    """
    return var_type + "_" + mag_field.value


@dataclass
class IrbemInput:
    """A data class to hold all necessary input parameters for IRBEM calculations.

    Attributes:
        magnetic_field (MagneticField): The magnetic field model to be used.
        maginput (dict[MagInputKeys, NDArray[np.float64]]): A dictionary of
            magnetic field input parameters required by IRBEM (e.g., Kp, Dst).
        irbem_options (IrbemOptions): The IRBEM-LIB options to configure the
            library's behavior.
        num_cores (int): The number of CPU cores to use for parallel processing.
            Defaults to 4.
        irbem_lib_path (str|Path): The file path to the compiled IRBEM library.
            Defaults to the 'libirbem.so' located in the same directory as the el_paso package.
    """

    magnetic_field: MagneticField
    maginput: dict[MagInputKeys, NDArray[np.float64]]
    irbem_options: IrbemOptions
    num_cores: int = 4
    irbem_lib_path: str | Path = str(Path(ep.__file__).parent / "libirbem.so")


class IrbemOutput(NamedTuple):
    """A named tuple to represent the output of a single IRBEM calculation.

    This is used for intermediate results in parallel processing.

    Attributes:
        arr (NDArray[np.float64]): The calculated data as a NumPy array.
        unit (u.UnitBase): The unit of the calculated data.
    """

    arr: NDArray[np.float64]
    unit: u.UnitBase


@dataclass
class _IrbemWorkerContext:
    """Holds the per-process state of a parallel IRBEM calculation.

    The IRBEM model wraps a shared library handle and can therefore not be pickled into a worker
    process. It is instead constructed once per worker by `_init_irbem_worker`, together with the
    read-only input arrays which would otherwise be shipped to the workers with every chunk.

    Attributes:
        model (MagFields): The IRBEM model used by this worker process.
        datetimes (list[datetime]): Timestamps of the satellite positions.
        maginput (dict[MagInputKeys, NDArray[np.float64]]): Magnetic field input parameters for IRBEM.
        x_geo (NDArray[np.float64] | None): Satellite positions in GEO coordinates, if the calculation
            starts from known positions. Searches like the LCDS one do not use any.
        pitch_angles (NDArray[np.float64] | None): Pitch angles of every time step, if the calculation
            requires them. Local pitch angles for the mirror point and L* searches, equatorial ones
            for the LCDS search.
        search_params (LCDSSearchParams | None): Radial search settings, if the calculation is an LCDS
            search. Each worker owns its copy and updates it in place to warm-start the next time step.
    """

    model: MagFields
    datetimes: list[datetime]
    maginput: dict[MagInputKeys, NDArray[np.float64]]
    x_geo: NDArray[np.float64] | None = None
    pitch_angles: NDArray[np.float64] | None = None
    search_params: LCDSSearchParams | None = None

    def position_at(self, it: int) -> dict[Literal["x1", "x2", "x3"], np.float64]:
        """Returns the satellite position of time step `it` in the dict format expected by IRBEM.

        Raises:
            RuntimeError: If the worker was initialized without satellite positions.
        """
        if self.x_geo is None:
            msg = "The IRBEM worker was initialized without satellite positions!"
            raise RuntimeError(msg)
        return {
            "x1": self.x_geo[it, 0],
            "x2": self.x_geo[it, 1],
            "x3": self.x_geo[it, 2],
        }

    def maginput_at(self, it: int) -> dict[MagInputKeys, np.float64]:
        """Returns the magnetic field input parameters of time step `it`."""
        return {key: arr[it] for key, arr in self.maginput.items()}

    def pitch_angles_at(self, it: int) -> NDArray[np.float64]:
        """Returns the pitch angles of time step `it`.

        Raises:
            RuntimeError: If the worker was initialized without pitch angles.
        """
        if self.pitch_angles is None:
            msg = "The IRBEM worker was initialized without pitch angles!"
            raise RuntimeError(msg)
        return self.pitch_angles[it, :]

    def lcds_search_params(self) -> LCDSSearchParams:
        """Returns this worker's LCDS search settings.

        Raises:
            RuntimeError: If the worker was initialized without LCDS search settings.
        """
        if self.search_params is None:
            msg = "The IRBEM worker was initialized without LCDS search parameters!"
            raise RuntimeError(msg)
        return self.search_params


_worker_context: _IrbemWorkerContext | None = None


def _init_irbem_worker(
    irbem_args: tuple[str | Path, IrbemOptions, int, int],
    datetimes: list[datetime],
    maginput: dict[MagInputKeys, NDArray[np.float64]],
    x_geo: NDArray[np.float64] | None = None,
    pitch_angles: NDArray[np.float64] | None = None,
    search_params: LCDSSearchParams | None = None,
) -> None:
    """Initializes one worker process of a parallel IRBEM calculation.

    This runs once per worker process instead of once per time step, so that the IRBEM shared
    library is loaded only once and the input arrays are transferred only once per worker.
    """
    global _worker_context

    _worker_context = _IrbemWorkerContext(
        model=MagFields(
            lib_path=irbem_args[0],
            options=irbem_args[1],
            kext=irbem_args[2],
            sysaxes=irbem_args[3],
        ),
        datetimes=datetimes,
        maginput=maginput,
        x_geo=x_geo,
        pitch_angles=pitch_angles,
        search_params=search_params,
    )


def _get_worker_context() -> _IrbemWorkerContext:
    """Returns the context of the current worker process.

    Raises:
        RuntimeError: If the worker process was not initialized by `_init_irbem_worker`.
    """
    if _worker_context is None:
        msg = "The IRBEM worker context is not initialized! _init_irbem_worker must run in every worker process."
        raise RuntimeError(msg)
    return _worker_context


_T = TypeVar("_T")


def _run_irbem_parallel(
    worker_func: Callable[[int], _T],
    irbem_input: IrbemInput,
    datetimes: list[datetime],
    *,
    sysaxes: int,
    desc: str,
    x_geo: NDArray[np.float64] | None = None,
    pitch_angles: NDArray[np.float64] | None = None,
    search_params: LCDSSearchParams | None = None,
) -> list[_T]:
    """Maps `worker_func` over all time steps in a process pool with initialized IRBEM workers."""
    irbem_args = (
        irbem_input.irbem_lib_path,
        irbem_input.irbem_options,
        irbem_input.magnetic_field.get_kext(),
        sysaxes,
    )

    # try to build the MagFields object to see if any errors occur
    MagFields(lib_path=irbem_args[0], options=irbem_args[1], kext=irbem_args[2], sysaxes=irbem_args[3])

    chunksize = max(1, len(datetimes) // irbem_input.num_cores // 4)  # same as default

    with MultiPool(
        processes=irbem_input.num_cores,
        initializer=_init_irbem_worker,
        initargs=(irbem_args, datetimes, irbem_input.maginput, x_geo, pitch_angles, search_params),
    ) as pool:
        return pool.map(worker_func, range(len(datetimes)), chunksize=chunksize, desc=desc)


def _get_magequator_parallel(it: int) -> tuple[float, NDArray[np.float64]]:
    context = _get_worker_context()

    magequator_output = context.model.find_magequator(
        context.datetimes[it], context.position_at(it), context.maginput_at(it)
    )
    bmin = magequator_output.bmin
    xgeo = magequator_output.xgeo

    assert isinstance(bmin, float)
    assert isinstance(xgeo, np.ndarray)

    return bmin, xgeo


@timed_function()
def get_magequator(xgeo_var: ep.Variable, time_var: ep.Variable, irbem_input: IrbemInput) -> dict[str, ep.Variable]:
    """Calculates the magnetic field strength and radial distance at the magnetic equator.

    This function uses parallel processing to efficiently compute magnetic field
    and position properties at the magnetic equator for a given set of satellite
    positions over time. It returns the results as a dictionary of `el_paso.Variable` objects.

    Args:
        xgeo_var (ep.Variable): The variable containing satellite position data in GEO coordinates.
        time_var (ep.Variable): The variable containing the timestamps for the data.
        irbem_input (IrbemInput): A data class with all required IRBEM input parameters.

    Returns:
        dict[str, ep.Variable]: A dictionary containing the calculated `B_eq`, `R_eq`, and
                                `xGEO_eq` variables.
    """
    logger.info("\tCalculating magnetic field and radial distance at the equator ...")

    timestamps = time_var.get_data(ep.units.posixtime)
    x_geo = xgeo_var.get_data(ep.units.RE)

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
    sysaxes = ep.IRBEM_SYSAXIS_GEO

    x_geo = x_geo.astype(np.float64)

    if len(datetimes) != len(x_geo):
        msg = f"Encountered size mismatch for x_geo: len of x_geo data: {len(x_geo)}, requested len: {len(datetimes)}"
        raise ValueError(msg)

    results = _run_irbem_parallel(
        _get_magequator_parallel,
        irbem_input,
        datetimes,
        sysaxes=sysaxes,
        desc="Calculating magnetic equator",
        x_geo=x_geo,
    )

    # write results into one array
    B_eq = np.empty_like(datetimes)
    x_geo_min = np.empty_like(x_geo)

    for i in range(len(datetimes)):
        B_eq[i] = results[i][0]
        x_geo_min[i] = results[i][1]

    B_eq[B_eq == FORTRAN_BAD_VALUE] = np.nan
    x_geo_min[x_geo_min == FORTRAN_BAD_VALUE] = np.nan

    B_eq_var = ep.Variable(data=B_eq.astype(np.float64), original_unit=u.nT)
    B_eq_var.metadata.add_processing_note(
        f"Calculated magnetic field at the equator using IRBEM model {irbem_input.magnetic_field} "
        f"with options {irbem_input.irbem_options}."
    )

    x_geo_var = ep.Variable(data=x_geo_min.astype(np.float64), original_unit=ep.units.RE)
    x_geo_var.metadata.add_processing_note(
        f"Calculated radial distance at the equator using IRBEM model {irbem_input.magnetic_field} "
        f"with options {irbem_input.irbem_options}."
    )

    # add radial distance field in SM coordinates
    x_gsm = Coords(lib_path=irbem_input.irbem_lib_path).transform(
        datetimes,
        x_geo_min,
        ep.IRBEM_SYSAXIS_GEO,
        ep.IRBEM_SYSAXIS_GSM,
    )

    R_eq_var = ep.Variable(
        data=np.linalg.norm(x_gsm, ord=2, axis=1).astype(np.float64),
        original_unit=ep.units.RE,
    )
    R_eq_var.metadata.add_processing_note(
        f"Calculated radial distance at the equator in GSM coordinates using IRBEM model {irbem_input.magnetic_field} "
        f"with options {irbem_input.irbem_options}."
    )

    p_gsm = np.arctan2(x_gsm[:, 1], x_gsm[:, 0])
    mlt_gsm = ((p_gsm * 12 / np.pi) + 12) % 24

    mlt_eq_var = ep.Variable(
        data=mlt_gsm.astype(np.float64),
        original_unit=u.hour,
    )
    mlt_eq_var.metadata.add_processing_note(
        "Calculated magnetic local time at the equator in GSM coordinates using "
        f"IRBEM model {irbem_input.magnetic_field} with options {irbem_input.irbem_options}."
    )

    return {
        create_var_name("B_Eq", irbem_input.magnetic_field): B_eq_var,
        create_var_name("R_Eq", irbem_input.magnetic_field): R_eq_var,
        create_var_name("MLT_Eq", irbem_input.magnetic_field): mlt_eq_var,
        create_var_name("xGEO_Eq", irbem_input.magnetic_field): x_geo_var,
    }


def _get_footpoint_atmosphere_parallel(it: int) -> NDArray[np.float64]:
    context = _get_worker_context()

    footpoint_output = context.model.find_foot_point(
        context.datetimes[it], context.position_at(it), context.maginput_at(it), stop_alt=100, hemi_flag=0
    )

    return np.asarray(footpoint_output.b_foot_mag)


@timed_function()
def get_footpoint_atmosphere(
    xgeo_var: ep.Variable, time_var: ep.Variable, irbem_input: IrbemInput
) -> dict[str, ep.Variable]:
    """Calculates the magnetic field strength at the atmospheric foot point.

    This function uses parallel processing to calculate the magnetic field strength
    at the atmospheric foot point (100 km altitude) for each satellite position.
    It returns the result as a dictionary of `el_paso.Variable`.

    Args:
        xgeo_var (ep.Variable): The variable containing satellite position data in GEO coordinates.
        time_var (ep.Variable): The variable containing the timestamps.
        irbem_input (IrbemInput): A data class with all required IRBEM input parameters.

    Returns:
        dict[str, ep.Variable]: A dictionary containing the calculated `B_fofl` variable.
    """
    logger.info("\tCalculating magnetic foot point at the atmosphere ...")

    timestamps = time_var.get_data(ep.units.posixtime)
    x_geo = xgeo_var.get_data(ep.units.RE)

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
    sysaxes = ep.IRBEM_SYSAXIS_GEO

    x_geo = x_geo.astype(np.float64)

    if len(datetimes) != len(x_geo):
        msg = f"Encountered size mismatch for x_geo: len of x_geo data: {len(x_geo)}, requested len: {len(datetimes)}"
        raise ValueError(msg)

    results = _run_irbem_parallel(
        _get_footpoint_atmosphere_parallel,
        irbem_input,
        datetimes,
        sysaxes=sysaxes,
        desc="Calculating foot point",
        x_geo=x_geo,
    )

    # write results into one array
    B_foot = np.empty_like(datetimes)

    for i in range(len(datetimes)):
        B_foot[i] = results[i]

    B_foot[B_foot == FORTRAN_BAD_VALUE] = np.nan

    var = ep.Variable(data=B_foot.astype(np.float64), original_unit=u.nT)
    var.metadata.add_processing_note(
        f"Calculated foot point at the atmosphere using IRBEM model {irbem_input.magnetic_field} "
        f"with options {irbem_input.irbem_options}."
    )

    return {create_var_name("B_fofl", irbem_input.magnetic_field): var}


@timed_function()
def get_MLT(xgeo_var: ep.Variable, time_var: ep.Variable, irbem_input: IrbemInput) -> dict[str, ep.Variable]:
    """Calculates the magnetic local time (MLT).

    Args:
        xgeo_var (ep.Variable): The variable containing satellite position data in GEO coordinates.
        time_var (ep.Variable): The variable containing the timestamps.
        irbem_input (IrbemInput): A data class with all required IRBEM input parameters.

    Returns:
        dict[str, ep.Variable]: A dictionary containing the calculated `MLT` variable.
    """
    logger.info("\tCalculating magnetic local time ...")

    timestamps = time_var.get_data(ep.units.posixtime)
    x_geo = xgeo_var.get_data(ep.units.RE)

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
    sysaxes = ep.IRBEM_SYSAXIS_GEO

    # Ensure xGEO and maginput are floating-point arrays
    x_geo = x_geo.astype(np.float64)

    if len(datetimes) != len(x_geo):
        msg = f"Encountered size mismatch for x_geo: len of x_geo data: {len(x_geo)}, requested len: {len(datetimes)}"
        raise ValueError(msg)

    kext = irbem_input.magnetic_field.get_kext()

    model = MagFields(
        lib_path=irbem_input.irbem_lib_path,
        options=irbem_input.irbem_options,
        kext=kext,
        sysaxes=sysaxes,
    )

    mlt_output = np.empty_like(datetimes)

    for i in range(len(datetimes)):
        x_dict: dict[Literal["x1", "x2", "x3"], np.floating] = {
            "x1": x_geo[i, 0],
            "x2": x_geo[i, 1],
            "x3": x_geo[i, 2],
        }
        mlt_output[i] = model.get_mlt(datetimes[i], x_dict)

    mlt_output = mlt_output.astype(np.float64)

    var = ep.Variable(data=mlt_output, original_unit=u.hour)
    var.metadata.add_processing_note(
        f"Calculated MLT using IRBEM model {irbem_input.magnetic_field} with options {irbem_input.irbem_options}."
    )

    return {create_var_name("MLT", irbem_input.magnetic_field): var}


@timed_function()
def get_local_B_field(xgeo_var: ep.Variable, time_var: ep.Variable, irbem_input: IrbemInput) -> dict[str, ep.Variable]:
    """Calculates the local magnetic field magnitude.

    Args:
        xgeo_var (ep.Variable): The variable containing satellite position data in GEO coordinates.
        time_var (ep.Variable): The variable containing the timestamps.
        irbem_input (IrbemInput): A data class with all required IRBEM input parameters.

    Returns:
        dict[str, ep.Variable]: A dictionary containing the calculated `B_local` variable.
    """
    logger.info("\tCalculating local magnetic field values ...")

    timestamps = time_var.get_data(ep.units.posixtime)
    x_geo = xgeo_var.get_data(ep.units.RE)

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
    sysaxes = ep.IRBEM_SYSAXIS_GEO

    # Define Fortran bad value as a float
    fortran_bad_value = np.float64(-1.0e31)
    # Ensure x_geo and maginput are floating-point arrays
    x_geo = x_geo.astype(np.float64)
    for key in irbem_input.maginput:
        irbem_input.maginput[key] = np.array(irbem_input.maginput[key], dtype=np.float64)

    if len(datetimes) != len(irbem_input.maginput["Kp"]):
        msg = (
            f"Encountered size mismatch for Kp: len of Kp data: {len(irbem_input.maginput['Kp'])}, "
            f"requested len: {len(datetimes)}"
        )
        raise ValueError(msg)
    if len(datetimes) != len(x_geo):
        msg = f"Encountered size mismatch for x_geo: len of x_geo data: {len(x_geo)}, requested len: {len(datetimes)}"
        raise ValueError(msg)

    x_dict: dict[Literal["x1", "x2", "x3"], NDArray[np.floating]] = {
        "x1": x_geo[:, 0],
        "x2": x_geo[:, 1],
        "x3": x_geo[:, 2],
    }
    kext = irbem_input.magnetic_field.get_kext()

    model = MagFields(
        lib_path=irbem_input.irbem_lib_path,
        options=irbem_input.irbem_options,
        kext=kext,
        sysaxes=sysaxes,
    )

    field_multi_output = model.get_field_multi(datetimes, x_dict, irbem_input.maginput)

    # replace bad values with nan
    field_multi_output.bgeo[field_multi_output.bgeo == fortran_bad_value] = np.nan
    field_multi_output.blocal[field_multi_output.blocal == fortran_bad_value] = np.nan

    b_local_var = ep.Variable(data=field_multi_output.blocal, original_unit=u.nT)
    return {create_var_name("B_Calc", irbem_input.magnetic_field): b_local_var}


def _get_mirror_point_parallel(it: int) -> NDArray[np.float64]:
    context = _get_worker_context()

    x_dict_single = context.position_at(it)
    maginput = context.maginput_at(it)
    pitch_angles = context.pitch_angles_at(it)

    bmin_output = np.empty_like(pitch_angles)

    for i, pa in enumerate(pitch_angles):
        bmin_output[i] = context.model.find_mirror_point(
            context.datetimes[it], x_dict_single, maginput, float(pa)
        ).bmirr

    return bmin_output.astype(np.float64)


@timed_function()
def get_mirror_point(
    xgeo_var: ep.Variable, time_var: ep.Variable, pa_local_var: ep.Variable, irbem_input: IrbemInput
) -> dict[str, ep.Variable]:
    """Calculates the magnetic field strength at the mirror point.

    This function computes the magnetic field strength at the mirror point for a
    given set of local pitch angles.

    Args:
        xgeo_var (ep.Variable): The variable containing satellite position data in GEO coordinates.
        time_var (ep.Variable): The variable containing the timestamps.
        pa_local_var (ep.Variable): The variable containing the local pitch angle data.
        irbem_input (IrbemInput): A data class with all required IRBEM input parameters.

    Returns:
        dict[str, ep.Variable]: A dictionary containing the calculated `B_mirr` variable.
    """
    logger.info("\tCalculating mirror points ...")

    timestamps = time_var.get_data(ep.units.posixtime)
    x_geo = xgeo_var.get_data(ep.units.RE)
    pa_local = pa_local_var.get_data(u.deg)

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
    sysaxes = ep.IRBEM_SYSAXIS_GEO

    x_geo = x_geo.astype(np.float64)
    pa_local = pa_local.astype(np.float64)
    irbem_input.maginput = {key: arr.astype(np.float64) for key, arr in irbem_input.maginput.items()}

    if len(datetimes) != len(irbem_input.maginput["Kp"]):
        msg = (
            f"Encountered size mismatch for Kp: len of Kp data: {len(irbem_input.maginput['Kp'])}, "
            f"requested len: {len(datetimes)}"
        )
        raise ValueError(msg)
    if len(datetimes) != len(x_geo):
        msg = f"Encountered size mismatch for x_geo: len of x_geo data: {len(x_geo)}, requested len: {len(datetimes)}"
        raise ValueError(msg)
    if len(datetimes) != len(pa_local):
        msg = (
            f"Encountered size mismatch for pa_local: len of pa_local data: {len(pa_local)}, "
            f"requested len: {len(datetimes)}"
        )
        raise ValueError(msg)

    results = _run_irbem_parallel(
        _get_mirror_point_parallel,
        irbem_input,
        datetimes,
        sysaxes=sysaxes,
        desc="Calculating mirror points",
        x_geo=x_geo,
        pitch_angles=pa_local,
    )

    # write results into one array
    mirror_point_output = np.empty_like(pa_local)

    for i in range(len(datetimes)):
        mirror_point_output[i, :] = results[i]

    # replace bad values with nan
    mirror_point_output[mirror_point_output < 0] = np.nan

    var = ep.Variable(data=mirror_point_output.astype(np.float64), original_unit=u.nT)
    var.metadata.add_processing_note(
        f"Calculated mirror points using IRBEM model {irbem_input.magnetic_field} "
        f"with options {irbem_input.irbem_options}."
    )

    return {create_var_name("B_mirr", irbem_input.magnetic_field): var}


def _make_lstar_shell_splitting_parallel(
    it: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    context = _get_worker_context()

    x_dict_single = context.position_at(it)
    maginput = context.maginput_at(it)
    pitch_angles = context.pitch_angles_at(it)

    Lm = np.empty_like(pitch_angles)
    Lstar = np.empty_like(pitch_angles)
    xj = np.empty_like(pitch_angles)

    for i, pa in enumerate(pitch_angles):
        Lstar_output_single = context.model.make_lstar_shell_splitting(
            context.datetimes[it], x_dict_single, maginput, pa
        )

        Lm[i] = np.squeeze(Lstar_output_single.lm)
        Lstar[i] = np.squeeze(Lstar_output_single.lstar)
        xj[i] = np.squeeze(Lstar_output_single.xj)

    return (Lm.astype(np.float64), Lstar.astype(np.float64), xj.astype(np.float64))


@timed_function()
def get_Lstar(
    xgeo_var: ep.Variable, time_var: ep.Variable, pa_local_var: ep.Variable, irbem_input: IrbemInput
) -> dict[str, ep.Variable]:
    """Calculates Lm, Lstar, and the third adiabatic invariant (J).

    This function computes Lm and Lstar for each satellite position and local
    pitch angle. These are crucial parameters for characterizing particle drift shells
    in the magnetosphere. The calculation is parallelized for performance.

    Args:
        xgeo_var (ep.Variable): The variable containing satellite position data in GEO coordinates.
        time_var (ep.Variable): The variable containing the timestamps.
        pa_local_var (ep.Variable): The variable containing the local pitch angle data.
        irbem_input (IrbemInput): A data class with all required IRBEM input parameters.

    Returns:
        dict[str, ep.Variable]: A dictionary containing the calculated `Lm`, `Lstar`, and `XJ` variables.
    """
    logger.info("\tCalculating Lstar and J ...")

    timestamps = time_var.get_data(ep.units.posixtime)
    x_geo = xgeo_var.get_data(ep.units.RE)
    pa_local = pa_local_var.get_data(u.deg)

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
    sysaxes = ep.IRBEM_SYSAXIS_GEO

    x_geo = x_geo.astype(np.float64)
    pa_local = pa_local.astype(np.float64)
    irbem_input.maginput = {key: arr.astype(np.float64) for key, arr in irbem_input.maginput.items()}

    if len(datetimes) != len(irbem_input.maginput["Kp"]):
        msg = (
            f"Encountered size mismatch for Kp: len of Kp data: {len(irbem_input.maginput['Kp'])}, "
            f"requested len: {len(datetimes)}"
        )
        raise ValueError(msg)
    if len(datetimes) != len(x_geo):
        msg = f"Encountered size mismatch for x_geo: len of x_geo data: {len(x_geo)}, requested len: {len(datetimes)}"
        raise ValueError(msg)
    if len(datetimes) != len(pa_local):
        msg = (
            f"Encountered size mismatch for pa_local: len of pa_local data: {len(pa_local)}, "
            f"requested len: {len(datetimes)}"
        )
        raise ValueError(msg)

    results = _run_irbem_parallel(
        _make_lstar_shell_splitting_parallel,
        irbem_input,
        datetimes,
        sysaxes=sysaxes,
        desc="Calculating Lstar",
        x_geo=x_geo,
        pitch_angles=pa_local,
    )

    # write results into one array
    Lm = np.empty_like(pa_local)
    Lstar = np.empty_like(pa_local)
    xj = np.empty_like(pa_local)

    for i in range(len(datetimes)):
        Lm[i, :] = results[i][0]
        Lstar[i, :] = results[i][1]
        xj[i, :] = results[i][2]

    # replace bad values with nan
    for arr in [Lm, Lstar, xj]:
        arr[arr < 0] = np.nan
        if not np.any(np.isfinite(arr)) and irbem_input.irbem_options.lstar_quantity != LstarQuantity.NONE:
            msg = (
                "Lstar calculation failed! All points are NaNs! Hints for debugging:\n"
                "1) The calculation can fail for very low pitch-angles, where particles"
                "are not actually trapped\n"
                "2) Make sure your equatorial pitch angles and xGEO positions are correct\n"
                "3) Check other magnetic field outputs like equatorial magnetic fields."
                "If they are also NaN, the maginput to IRBEM might be wrong and needs debugging."
            )
            raise ValueError(msg)

    Lm_var = ep.Variable(data=Lm.astype(np.float64), original_unit=u.dimensionless_unscaled)
    Lm_var.metadata.add_processing_note(
        f"Calculated Lm using IRBEM model {irbem_input.magnetic_field} with options {irbem_input.irbem_options}."
    )

    Lstar_var = ep.Variable(data=Lstar.astype(np.float64), original_unit=u.dimensionless_unscaled)
    Lstar_var.metadata.add_processing_note(
        f"Calculated Lstar using IRBEM model {irbem_input.magnetic_field} with options {irbem_input.irbem_options}."
    )

    XJ_var = ep.Variable(data=xj.astype(np.float64), original_unit=ep.units.RE)
    XJ_var.metadata.add_processing_note(
        f"Calculated XJ using IRBEM model {irbem_input.magnetic_field} with options {irbem_input.irbem_options}."
    )

    return {
        create_var_name("L_m", irbem_input.magnetic_field): Lm_var,
        create_var_name("L_star", irbem_input.magnetic_field): Lstar_var,
        create_var_name("I", irbem_input.magnetic_field): XJ_var,
    }


def _get_LCDS_parallel(it: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    context = _get_worker_context()
    search_params = context.lcds_search_params()

    pitch_angles = context.pitch_angles_at(it)

    lcds = np.full(pitch_angles.shape, np.inf)
    inv_k = np.full(pitch_angles.shape, np.inf)

    result = context.model.get_lcds(context.datetimes[it], pitch_angles, context.maginput_at(it), search_params)

    # Pitch angles whose shell is still closed at the ceiling keep the censored np.inf.
    keep = ~result.at_ceiling
    lcds[keep] = result.lcds[keep]
    inv_k[keep] = result.inv_k[keep]

    # The LCDS moves smoothly in time and every worker walks a contiguous block of time steps,
    # so the next step starts its radial march at this step's boundary instead of at max_r.
    search_params.start_r = result.x_sm if np.isfinite(result.x_sm) else search_params.max_r

    return lcds.astype(np.float64), inv_k.astype(np.float64)


@timed_function()
def get_LCDS(
    time_var: ep.Variable,
    pa_eq_var: ep.Variable,
    irbem_input: IrbemInput,
    search_params: LCDSSearchParams | None = None,
) -> tuple[ep.Variable, ep.Variable]:
    """Calculates the last closed drift shell (LCDS) and its second adiabatic invariant K.

    For every time step, a radial search in the SM equatorial plane locates the outermost
    drift shell that is still closed, separately for each equatorial pitch angle. The time
    steps are distributed over `irbem_input.num_cores` worker processes; each worker walks a
    contiguous block of time steps and warm-starts every step from the boundary found at the
    previous one, which is much cheaper than restarting the search at `search_params.max_r`.

    Args:
        time_var (ep.Variable): The variable containing the timestamps.
        pa_eq_var (ep.Variable): The variable containing the equatorial pitch angles, one row
                                 per time step.
        irbem_input (IrbemInput): A data class with all required IRBEM input parameters.
        search_params (LCDSSearchParams | None): The settings of the radial search. Defaults to
                                                 the settings of Kellerman's LCDS2 routine.

    Returns:
        tuple[ep.Variable, ep.Variable]: The LCDS L* and the corresponding invariant K, both of
        shape (number of time steps, number of pitch angles). Pitch angles whose drift shell is
        still closed at `search_params.max_r` are set to infinity, since their LCDS only has a
        lower bound; failed searches are NaN.

    Raises:
        ValueError: If the maginput or pitch angle arrays do not match the number of time steps.
    """
    logger.info("\tCalculating LCDS ...")

    if search_params is None:
        search_params = LCDSSearchParams(
            max_r=10,
            coarse_step=1,
            medium_step=0.5,
            fine_step=0.1,
            trace_r0=0.8,
            start_r=10,
        )

    timestamps = time_var.get_data(ep.units.posixtime)
    pa_eq = pa_eq_var.get_data(u.deg)

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]

    pa_eq = pa_eq.astype(np.float64)
    if pa_eq.ndim == 1:  # a single pitch angle per time step
        pa_eq = pa_eq[:, np.newaxis]
    if pa_eq.ndim != 2:
        msg = f"pa_eq must have one row of pitch angles per time step, but has shape {pa_eq.shape}."
        raise ValueError(msg)

    irbem_input.maginput = {key: arr.astype(np.float64) for key, arr in irbem_input.maginput.items()}

    n_times = len(datetimes)

    for key, arr in irbem_input.maginput.items():
        if len(arr) != n_times:
            msg = f"maginput['{key}'] has length {len(arr)} but {n_times} times were given."
            raise ValueError(msg)

    if n_times != len(pa_eq):
        msg = f"Encountered size mismatch for pa_eq: len of pa_eq data: {len(pa_eq)}, requested len: {n_times}"
        raise ValueError(msg)

    results = _run_irbem_parallel(
        _get_LCDS_parallel,
        irbem_input,
        datetimes,
        sysaxes=ep.IRBEM_SYSAXIS_SM,
        desc="Calculating LCDS",
        pitch_angles=pa_eq,
        search_params=search_params,
    )

    lcds = np.full(pa_eq.shape, np.inf, dtype=np.float64)
    inv_K = np.full(pa_eq.shape, np.inf, dtype=np.float64)

    for i in range(n_times):
        lcds[i, :] = results[i][0]
        inv_K[i, :] = results[i][1]

    lcds[lcds == FORTRAN_BAD_VALUE] = np.nan
    inv_K[inv_K == FORTRAN_BAD_VALUE] = np.nan

    lcds_var = ep.Variable(data=lcds, original_unit=u.dimensionless_unscaled)
    lcds_var.metadata.add_processing_note(
        f"Calculated LCDS using IRBEM model {irbem_input.magnetic_field} with options "
        f"{irbem_input.irbem_options} and {search_params}."
    )

    inv_K_var = ep.Variable(data=inv_K, original_unit=ep.units.RE * u.G**0.5)
    inv_K_var.metadata.add_processing_note(
        f"Calculated inv_K from LCDS using IRBEM model {irbem_input.magnetic_field} with options "
        f"{irbem_input.irbem_options} and {search_params}."
    )

    return lcds_var, inv_K_var
