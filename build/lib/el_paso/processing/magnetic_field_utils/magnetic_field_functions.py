# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: N802, N806

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from multiprocessing import Pool
from typing import Literal, NamedTuple

import numpy as np
from astropy import units as u  # type: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray

import el_paso as ep
from el_paso.processing.magnetic_field_utils.construct_maginput import MagInputKeys
from el_paso.processing.magnetic_field_utils.irbem import Coords, MagFields
from el_paso.processing.magnetic_field_utils.mag_field_enum import MagneticField
from el_paso.utils import show_process_bar_for_map_async, timed_function

logger = logging.getLogger(__name__)

FORTRAN_BAD_VALUE = np.float64(-1.0e31)

MagFieldVarTypes = Literal[
    "B_local", "B_fofl", "B_eq", "B_mirr", "xGEO_eq", "MLT", "R_eq", "Lstar", "Lm", "PA_eq", "invMu", "invK", "XJ"
]


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
        irbem_lib_path (str): The file path to the compiled IRBEM library.
        magnetic_field (MagneticField): The magnetic field model to be used.
        maginput (dict[MagInputKeys, NDArray[np.float64]]): A dictionary of
            magnetic field input parameters required by IRBEM (e.g., Kp, Dst).
        irbem_options (list[int]): A list of integer options to configure the
            IRBEM library's behavior.
        num_cores (int): The number of CPU cores to use for parallel processing.
                         Defaults to 4.
    """

    irbem_lib_path: str
    magnetic_field: MagneticField
    maginput: dict[MagInputKeys, NDArray[np.float64]]
    irbem_options: list[int]
    num_cores: int = 4


class IrbemOutput(NamedTuple):
    """A named tuple to represent the output of a single IRBEM calculation.

    This is used for intermediate results in parallel processing.

    Attributes:
        arr (NDArray[np.float64]): The calculated data as a NumPy array.
        unit (u.UnitBase): The unit of the calculated data.
    """

    arr: NDArray[np.float64]
    unit: u.UnitBase


def _get_magequator_parallel(
    irbem_args: tuple[str, list[int], int, int],
    x_geo: NDArray[np.float64],
    datetimes: list[datetime],
    maginput: dict[MagInputKeys, NDArray[np.float64]],
    it: int,
) -> tuple[float, NDArray[np.float64]]:
    model = MagFields(
        lib_path=irbem_args[0],
        options=irbem_args[1],
        kext=irbem_args[2],
        sysaxes=irbem_args[3],
    )

    x_dict_single: dict[Literal["x1", "x2", "x3"], np.float64] = {
        "x1": x_geo[it, 0],
        "x2": x_geo[it, 1],
        "x3": x_geo[it, 2],
    }
    maginput = {key: maginput[key][it] for key in maginput}

    magequator_output = model.find_magequator(datetimes[it], x_dict_single, maginput)
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

    kext = irbem_input.magnetic_field.kext()

    irbem_args = (irbem_input.irbem_lib_path, irbem_input.irbem_options, kext, sysaxes)

    parallel_func = partial(_get_magequator_parallel, irbem_args, x_geo, datetimes, irbem_input.maginput)

    with Pool(processes=irbem_input.num_cores) as pool:
        chunksize = max(1, len(datetimes) // irbem_input.num_cores // 4)  # same as default
        rs = pool.map_async(parallel_func, range(len(datetimes)), chunksize=chunksize)
        show_process_bar_for_map_async(rs, chunksize)

    # write async results into one array
    B_eq = np.empty_like(datetimes)
    x_geo_min = np.empty_like(x_geo)

    results = rs.get()

    for i in range(len(datetimes)):
        B_eq[i] = results[i][0]
        x_geo_min[i] = results[i][1]

    B_eq[B_eq == FORTRAN_BAD_VALUE] = np.nan
    x_geo_min[x_geo_min == FORTRAN_BAD_VALUE] = np.nan

    B_eq_var = ep.Variable(data=B_eq.astype(np.float64), original_unit=u.nT)  # type:ignore [reportAttributeAccessIssue]
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
    x_sm = Coords(lib_path=irbem_input.irbem_lib_path).transform(
        datetimes,
        x_geo_min,
        ep.IRBEM_SYSAXIS_GEO,
        ep.IRBEM_SYSAXIS_SM,
    )

    R_eq_var = ep.Variable(
        data=np.linalg.norm(x_sm, ord=2, axis=1).astype(np.float64),
        original_unit=ep.units.RE,
    )
    R_eq_var.metadata.add_processing_note(
        f"Calculated radial distance at the equator in SM coordinates using IRBEM model {irbem_input.magnetic_field} "
        f"with options {irbem_input.irbem_options}."
    )

    return {
        create_var_name("B_eq", irbem_input.magnetic_field): B_eq_var,
        create_var_name("R_eq", irbem_input.magnetic_field): R_eq_var,
        create_var_name("xGEO_eq", irbem_input.magnetic_field): x_geo_var,
    }


def _get_footpoint_atmosphere_parallel(
    irbem_args: tuple[str, list[int], int, int],
    x_geo: NDArray[np.float64],
    datetimes: list[datetime],
    maginput: dict[MagInputKeys, NDArray[np.float64]],
    it: int,
) -> NDArray[np.float64]:
    model = MagFields(
        lib_path=irbem_args[0],
        options=irbem_args[1],
        kext=irbem_args[2],
        sysaxes=irbem_args[3],
    )

    x_dict_single: dict[Literal["x1", "x2", "x3"], np.float64] = {
        "x1": x_geo[it, 0],
        "x2": x_geo[it, 1],
        "x3": x_geo[it, 2],
    }
    maginput = {key: maginput[key][it] for key in maginput}

    footpoint_output = model.find_foot_point(datetimes[it], x_dict_single, maginput, stop_alt=100, hemi_flag=0)

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

    kext = irbem_input.magnetic_field.kext()

    irbem_args = (irbem_input.irbem_lib_path, irbem_input.irbem_options, kext, sysaxes)

    parallel_func = partial(_get_footpoint_atmosphere_parallel, irbem_args, x_geo, datetimes, irbem_input.maginput)

    with Pool(processes=irbem_input.num_cores) as pool:
        chunksize = max(1, len(datetimes) // irbem_input.num_cores // 4)  # same as default
        rs = pool.map_async(parallel_func, range(len(datetimes)), chunksize=chunksize)
        show_process_bar_for_map_async(rs, chunksize)

    # write async results into one array
    B_foot = np.empty_like(datetimes)

    results = rs.get()

    for i in range(len(datetimes)):
        B_foot[i] = results[i][0]

    B_foot[B_foot == FORTRAN_BAD_VALUE] = np.nan

    var = ep.Variable(data=B_foot.astype(np.float64), original_unit=u.nT)  # type:ignore [reportAttributeAccessIssue]
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

    kext = irbem_input.magnetic_field.kext()

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

    var = ep.Variable(data=mlt_output, original_unit=u.hour)  # type:ignore [reportAttributeAccessIssue]
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
    kext = irbem_input.magnetic_field.kext()

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

    b_local_var = ep.Variable(data=field_multi_output.blocal, original_unit=u.nT)  # type:ignore [reportAttributeAccessIssue]
    return {create_var_name("B_local", irbem_input.magnetic_field): b_local_var}


def _get_mirror_point_parallel(
    irbem_args: tuple[str, list[int], int, int],
    x_geo: NDArray[np.float64],
    datetimes: list[datetime],
    maginput: dict[MagInputKeys, NDArray[np.float64]],
    pa_local: NDArray[np.float64],
    it: int,
) -> NDArray[np.float64]:
    model = MagFields(
        lib_path=irbem_args[0],
        options=irbem_args[1],
        kext=irbem_args[2],
        sysaxes=irbem_args[3],
    )

    x_dict_single: dict[Literal["x1", "x2", "x3"], np.floating] = {
        "x1": x_geo[it, 0],
        "x2": x_geo[it, 1],
        "x3": x_geo[it, 2],
    }
    maginput = {key: maginput[key][it] for key in maginput}

    bmin_output = np.empty_like(pa_local[it, :])

    for i, pa in enumerate(pa_local[it, :]):
        bmin_output[i] = model.find_mirror_point(datetimes[it], x_dict_single, maginput, pa).bmin

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
    pa_local = pa_local_var.get_data(u.deg)  # type:ignore [reportAttributeAccessIssue]

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

    kext = irbem_input.magnetic_field.kext()

    irbem_args = (irbem_input.irbem_lib_path, irbem_input.irbem_options, kext, sysaxes)

    parallel_func = partial(_get_mirror_point_parallel, irbem_args, x_geo, datetimes, irbem_input.maginput, pa_local)

    with Pool(processes=irbem_input.num_cores) as pool:
        chunksize = max(1, len(datetimes) // irbem_input.num_cores // 4)  # same as default
        rs = pool.map_async(parallel_func, range(len(datetimes)), chunksize=chunksize)
        show_process_bar_for_map_async(rs, chunksize)

    # write async results into one array
    mirror_point_output = np.empty_like(pa_local)

    results = rs.get()

    for i in range(len(datetimes)):
        mirror_point_output[i, :] = results[i]

    # replace bad values with nan
    mirror_point_output[mirror_point_output < 0] = np.nan

    var = ep.Variable(data=mirror_point_output.astype(np.float64), original_unit=u.nT)  # type:ignore [reportAttributeAccessIssue]
    var.metadata.add_processing_note(
        f"Calculated mirror points using IRBEM model {irbem_input.magnetic_field} "
        f"with options {irbem_input.irbem_options}."
    )

    return {create_var_name("B_mirr", irbem_input.magnetic_field): var}


def _make_lstar_shell_splitting_parallel(
    irbem_args: tuple[str, list[int], int, int],
    x_geo: NDArray[np.float64],
    datetimes: list[datetime],
    maginput: dict[MagInputKeys, NDArray[np.float64]],
    pa_local: NDArray[np.float64],
    it: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    model = MagFields(
        lib_path=irbem_args[0],
        options=irbem_args[1],
        kext=irbem_args[2],
        sysaxes=irbem_args[3],
    )

    x_dict_single: dict[Literal["x1", "x2", "x3"], np.floating] = {
        "x1": x_geo[it, 0],
        "x2": x_geo[it, 1],
        "x3": x_geo[it, 2],
    }
    maginput = {key: maginput[key][it] for key in maginput}

    Lm = np.empty_like(pa_local[it, :])
    Lstar = np.empty_like(pa_local[it, :])
    xj = np.empty_like(pa_local[it, :])

    for i, pa in enumerate(pa_local[it, :]):
        Lstar_output_single = model.make_lstar_shell_splitting(datetimes[it], x_dict_single, maginput, pa)

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
    pa_local = pa_local_var.get_data(u.deg)  # type:ignore [reportAttributeAccessIssue]

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

    kext = irbem_input.magnetic_field.kext()

    irbem_args = (irbem_input.irbem_lib_path, irbem_input.irbem_options, kext, sysaxes)

    parallel_func = partial(
        _make_lstar_shell_splitting_parallel,
        irbem_args,
        x_geo,
        datetimes,
        irbem_input.maginput,
        pa_local,
    )

    with Pool(processes=irbem_input.num_cores) as pool:
        chunksize = max(1, len(datetimes) // irbem_input.num_cores // 4)  # same as default
        rs = pool.map_async(parallel_func, range(len(datetimes)), chunksize=chunksize)
        show_process_bar_for_map_async(rs, chunksize)

    # write async results into one array
    Lm = np.empty_like(pa_local)
    Lstar = np.empty_like(pa_local)
    xj = np.empty_like(pa_local)

    results = rs.get()

    for i in range(len(datetimes)):
        Lm[i, :] = results[i][0]
        Lstar[i, :] = results[i][1]
        xj[i, :] = results[i][2]

    # replace bad values with nan
    for arr in [Lm, Lstar, xj]:
        arr[arr < 0] = np.nan
        if not np.any(np.isfinite(arr)):
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
        create_var_name("Lm", irbem_input.magnetic_field): Lm_var,
        create_var_name("Lstar", irbem_input.magnetic_field): Lstar_var,
        create_var_name("XJ", irbem_input.magnetic_field): XJ_var,
    }
