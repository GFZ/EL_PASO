# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0


import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
from astropy import units as u
from numpy.typing import NDArray

import el_paso as ep
from el_paso.processing.magnetic_field_utils import IrbemOptions
from el_paso.processing.magnetic_field_utils.irbem import (
    FORTRAN_BAD_VALUE,
    SYSAXES_STR_TO_INT,
    Coords,
    LCDSSearchParams,
    LstarQuantity,
    MagFields,
)
from el_paso.processing.magnetic_field_utils.mag_field_enum import MagneticField
from el_paso.typing import MagFieldVarTypes, MagInputKeys
from el_paso.utils import show_process_bar_for_map_async, timed_function

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


def _get_magequator_parallel(
    irbem_args: tuple[Path | str, IrbemOptions, int, int],
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


def _get_footpoint_atmosphere_parallel(
    irbem_args: tuple[str | Path, IrbemOptions, int, int],
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

    b_local_var = ep.Variable(data=field_multi_output.blocal, original_unit=u.nT)
    return {create_var_name("B_Calc", irbem_input.magnetic_field): b_local_var}


def _get_mirror_point_parallel(
    irbem_args: tuple[str | Path, IrbemOptions, int, int],
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

    var = ep.Variable(data=mirror_point_output.astype(np.float64), original_unit=u.nT)
    var.metadata.add_processing_note(
        f"Calculated mirror points using IRBEM model {irbem_input.magnetic_field} "
        f"with options {irbem_input.irbem_options}."
    )

    return {create_var_name("B_mirr", irbem_input.magnetic_field): var}


def _make_lstar_shell_splitting_parallel(
    irbem_args: tuple[str | Path, IrbemOptions, int, int],
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

    x_dict_single: dict[Literal["x1", "x2", "x3"], np.float64] = {
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


def _get_LCDS_parallel(
    irbem_args: tuple[str | Path, IrbemOptions, int, int],
    datetimes: list[datetime],
    maginput: Mapping[MagInputKeys, NDArray[np.floating]],
    pa_eq: NDArray[np.floating],
    search_params: LCDSSearchParams,
    index_chunk: Sequence[int],
) -> NDArray[np.floating]:

    model = MagFields(
        lib_path=irbem_args[0],
        options=irbem_args[1],
        kext=irbem_args[2],
        sysaxes=irbem_args[3],
    )

    lcds_result = np.full((len(index_chunk), pa_eq.shape[1]), np.inf)

    import time

    for ic, it in enumerate(index_chunk):
        for ipa, pa in enumerate(pa_eq[it, :]):
            it = int(it)
            maginput_single = {key: maginput[key][it] for key in maginput}

            start = time.perf_counter()
            res = model.get_lcds(datetimes[it], pa, maginput_single, search_params)
            end = time.perf_counter()
            print(f"ic = {ic}, pa = {pa}, time = {end - start}")

            if not res.at_ceiling:
                lcds_result[ic, ipa] = res.lcds

            # carry the solution forward as the next iteration's starting radius
            search_params.start_r = res.x_sm if res.found else search_params.max_r


    return lcds_result


@timed_function()
def get_LCDS(
    time_var: ep.Variable,
    pa_eq_var: ep.Variable,
    irbem_input: IrbemInput,
    search_params: LCDSSearchParams | None = None,
) -> ep.Variable:
    """Compute the LCDS L* over a time series for one equatorial pitch angle.

    The N time steps are split into ``num_cores`` contiguous chunks; each chunk is
    processed by one worker that reuses its IRBEM handle and warm-starts each step from
    the previous one. (The first step of each chunk cold-starts at ``max_r``.)

    Args:
        times: Sequence of epochs (length N).
        maginput: Magnetic-field-model inputs as arrays of length N keyed by IRBEM name.
        alpha_eq_deg: Equatorial pitch angle in degrees.
        num_cores: Number of worker processes / chunks (1 => serial).
        warm_start_buffer: RE added above the previous solution when warm-starting the
            coarse march. Defaults to ``coarse_step``; set 0 to start exactly at the
            previous boundary.
        censored_value: Value written for steps where the shell was still closed at
            ``max_r`` (the boundary lies beyond the window, so L* is only a lower bound).
            Default ``np.inf`` so a downstream "is L < LCDS?" test treats everything in
            the window as trapped. Pass ``None`` to keep the lower-bound L*(max_r) value
            instead, or ``np.nan`` to drop these steps.
        (remaining args as in ``compute_lcds``.)

    Returns:
        float64 array of length N with the LCDS L* per time step. Misses are NaN;
        ceiling-censored steps are set to ``censored_value`` (unless it is ``None``).
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
    irbem_input.maginput = {key: arr.astype(np.float64) for key, arr in irbem_input.maginput.items()}

    n_times = len(datetimes)

    for key, arr in irbem_input.maginput.items():
        if len(arr) != n_times:
            msg = f"maginput['{key}'] has length {len(arr)} but {n_times} times were given."
            raise ValueError(msg)

    if n_times != len(pa_eq):
        msg = f"Encountered size mismatch for pa_eq: len of pa_eq data: {len(pa_eq)}, requested len: {n_times}"
        raise ValueError(msg)

    kext = irbem_input.magnetic_field.kext()
    irbem_args = (irbem_input.irbem_lib_path, irbem_input.irbem_options, kext, ep.IRBEM_SYSAXIS_SM)

    n_chunks = max(1, min(irbem_input.num_cores, n_times))
    index_chunks = [[int(i) for i in chunk] for chunk in np.array_split(np.arange(n_times), n_chunks)]
    parallel_func = partial(_get_LCDS_parallel, irbem_args, datetimes, irbem_input.maginput, pa_eq, search_params)

    if irbem_input.num_cores > 1:
        logger.info("Computing LCDS for %d steps across %d chunks ...", n_times, n_chunks)
        with Pool(processes=irbem_input.num_cores) as pool:
            rs = pool.map_async(parallel_func, index_chunks)
            show_process_bar_for_map_async(rs, len(index_chunks[0]))

        results = rs.get()
    else:
        logger.info("Computing LCDS for %d steps serially ...", n_times)
        results = [parallel_func(chunk) for chunk in index_chunks]

    lcds = np.full((n_times, pa_eq.shape[1]), np.inf, dtype=np.float64)
    for chunk_indices, chunk_result in zip(index_chunks, results, strict=True):
        lcds[chunk_indices, :] = chunk_result

    lcds[lcds == FORTRAN_BAD_VALUE] = np.nan

    lcds_var = ep.Variable(data=lcds, original_unit=u.dimensionless_unscaled)
    lcds_var.metadata.add_processing_note(
        f"Calculated LCDS using IRBEM model {irbem_input.magnetic_field} with options "
        f"{irbem_input.irbem_options} and {search_params}."
    )

    return lcds_var
