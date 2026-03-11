# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import logging
from datetime import datetime, timezone
from functools import cache
from typing import Literal

import numpy as np
from numpy.typing import NDArray

import el_paso as ep
from el_paso.load_indices_solar_wind_parameters import SW_Index

from .mag_field_enum import MagneticField, kext

logger = logging.getLogger(__name__)

FORTRAN_BAD_VALUE = np.float64(-1.0e31)

MAGINPUT_CLIP_RANGES: dict[kext, dict[SW_Index, tuple[float, float]]] = {
    MagneticField.T01.kext(): {
        "Dst": (-50, 20),
        "Pdyn": (0.5, 5),
        "IMF_By": (-5, 5),
        "IMF_Bz": (-5, 5),
        "G1": (0, 10),
        "G2": (0, 10),
    },
    MagneticField.T01s.kext(): {},
    MagneticField.T96.kext(): {
        "Dst": (-100, 20),
        "Pdyn": (0.5, 10),
        "IMF_By": (-10, 10),
        "IMF_Bz": (-10, 10),
    },
    MagneticField.T89.kext(): {},
    MagneticField.OP77Q.kext(): {},
    MagneticField.T04s.kext(): {},
}

MAGINPUT_REQUIRED_INPUTS: dict[kext, list[SW_Index]] = {
    MagneticField.T89.kext(): ["Kp"],
    MagneticField.T96.kext(): ["Kp", "Dst", "Pdyn", "IMF_By", "IMF_Bz"],
    MagneticField.T01.kext(): ["Kp", "Dst", "Pdyn", "IMF_By", "IMF_Bz", "SW_speed", "SW_density", "G1", "G2"],
    MagneticField.T01s.kext(): ["Kp", "Dst", "Pdyn", "IMF_By", "IMF_Bz", "SW_speed", "SW_density", "G2", "G3"],
    MagneticField.T04s.kext(): ["Kp", "Dst", "Pdyn", "IMF_By", "IMF_Bz", "W_params"],
    MagneticField.OP77Q.kext(): [],
}

MAGINPUT_TO_INDEX: dict[SW_Index, int | list[int]] = {
    "Kp": 0,
    "Dst": 1,
    "SW_density": 2,
    "SW_speed": 3,
    "Pdyn": 4,
    "IMF_By": 5,
    "IMF_Bz": 6,
    "G1": 7,
    "G2": 8,
    "G3": 9,
    "W_params": list(range(10, 16)),
}

MagInputKeys = Literal[
    "Kp", "Dst", "dens", "velo", "Pdyn", "ByIMF", "BzIMF", "G1", "G2", "G3", "W1", "W2", "W3", "W4", "W5", "W6", "AL"
]


@cache
def construct_maginput(
    time_var: ep.Variable, magnetic_field: MagneticField, indices_solar_wind: dict[str, ep.Variable] | None = None
) -> dict[MagInputKeys, NDArray[np.float64]]:
    """Construct the basic magnetospheric input parameters array.

    This function retrieves all solar wind data from the ACE dataset on CDAWeb, as well as the Kp and Dst indices,
    interpolates them to the cadence of `newtime`, and returns an array with the columns as follows:
    1: Kp, value of Kp as in OMNI2 files but as double instead of integer type.
    (NOTE: consistent with OMNI2, this is Kp*10, and it is in the range 0 to 90)
    2: Dst, Dst index (nT)
    3: Dsw, solar wind density (cm-3)
    4: Vsw, solar wind velocity (km/s)
    5: Pdyn, solar wind dynamic pressure (nPa)
    6: By, GSM y component of interplanetary magnetic field (nT)
    7: Bz, GSM z component of interplanetary magnetic field (nT), from ACE
    8-16: Qin-Denton parameters, implement this!
    17: AL auroral index (if not in ACE, fill with NaN)
    18-25: fill with NaN

    Args:
        time_var (ep.Variable): Array of new time points for interpolation.
        magnetic_field (MagneticField): The magnetic field model used to determine the required inputs.
        indices_solar_wind (dict[str, ep.Variable] | None, optional): A dictionary of pre-loaded solar
                                                                    wind variables. Defaults to None.

    Returns:
        dict[MagInputKeys, NDArray[np.float64]]: A dictionary containing the interpolated magnetospheric
                                                input parameters.
    """
    time = time_var.get_data(ep.units.posixtime).astype(np.float64)
    start_time = datetime.fromtimestamp(time[0], tz=timezone.utc)
    end_time = datetime.fromtimestamp(time[-1], tz=timezone.utc)

    if indices_solar_wind is None:
        indices_solar_wind = {}

    kext = magnetic_field.kext()

    required_inputs = MAGINPUT_REQUIRED_INPUTS[kext]
    clip_ranges = MAGINPUT_CLIP_RANGES[kext]

    maginput = np.full((len(time), 25), np.nan).astype(np.float64)

    for req_input in required_inputs:
        if req_input not in indices_solar_wind:
            logger.debug(f"Required input '{req_input}' not found in indices_solar_wind!")
            indices_solar_wind |= ep.load_indices_solar_wind_parameters(start_time, end_time, [req_input], time_var)

        req_input_data = indices_solar_wind[req_input].get_data().astype(np.float64)

        if len(req_input_data) != len(time):
            msg = (
                f"Encountered size missmatch for {req_input}: len of {req_input} data: "
                f"{len(req_input_data)}, requested len: {len(time)}"
            )
            raise ValueError(msg)

        if req_input in clip_ranges:
            clip_range = clip_ranges[req_input]
            req_input_data = req_input_data.clip(clip_range[0], clip_range[1])

        if req_input == "Kp":
            req_input_data = np.round(req_input_data * 10)

        maginput[:, MAGINPUT_TO_INDEX[req_input]] = np.asarray(req_input_data, dtype=np.float64)

    maginput_dict: dict[MagInputKeys, NDArray[np.float64]] = {
        "Kp": maginput[:, 0],
        "Dst": maginput[:, 1],
        "dens": maginput[:, 2],
        "velo": maginput[:, 3],
        "Pdyn": maginput[:, 4],
        "ByIMF": maginput[:, 5],
        "BzIMF": maginput[:, 6],
        "G1": maginput[:, 7],
        "G2": maginput[:, 8],
        "G3": maginput[:, 9],
        "W1": maginput[:, 10],
        "W2": maginput[:, 11],
        "W3": maginput[:, 12],
        "W4": maginput[:, 13],
        "W5": maginput[:, 14],
        "W6": maginput[:, 15],
        "AL": maginput[:, 16],
    }

    return maginput_dict
