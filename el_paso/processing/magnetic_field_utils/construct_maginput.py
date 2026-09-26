# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import cache
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

import el_paso as ep

from .mag_field_enum import MagneticField, kext

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from el_paso.data_standard import DataStandard
    from el_paso.load_indices_solar_wind_parameters import SW_Index
    from el_paso.typing import MagInputKeys, MagneticFieldLiteral, StandardName, VariablesDict

logger = logging.getLogger(__name__)

FORTRAN_BAD_VALUE = np.float64(-1.0e31)

MAGINPUT_CLIP_RANGES: dict[kext, dict[SW_Index, tuple[float, float]]] = {
    MagneticField.T01.get_kext(): {
        "Dst": (-50, 20),
        "Pdyn": (0.5, 5),
        "ByIMF": (-5, 5),
        "BzIMF": (-5, 5),
        "G1": (0, 10),
        "G2": (0, 10),
    },
    MagneticField.T01s.get_kext(): {},
    MagneticField.T96.get_kext(): {
        "Dst": (-100, 20),
        "Pdyn": (0.5, 10),
        "ByIMF": (-10, 10),
        "BzIMF": (-10, 10),
    },
    MagneticField.T89.get_kext(): {},
    MagneticField.OP77Q.get_kext(): {},
    MagneticField.T04s.get_kext(): {},
    MagneticField.Dip.get_kext(): {},
}

MAGINPUT_REQUIRED_INPUTS: dict[kext, list[SW_Index]] = {
    MagneticField.T89.get_kext(): ["Kp"],
    MagneticField.T96.get_kext(): ["Kp", "Dst", "Pdyn", "ByIMF", "BzIMF"],
    MagneticField.T01.get_kext(): ["Kp", "Dst", "Pdyn", "ByIMF", "BzIMF", "Vsw", "Nsw", "G1", "G2"],
    MagneticField.T01s.get_kext(): ["Kp", "Dst", "Pdyn", "ByIMF", "BzIMF", "Vsw", "Nsw", "G2", "G3"],
    MagneticField.T04s.get_kext(): ["Kp", "Dst", "Pdyn", "ByIMF", "BzIMF", "W_params"],
    MagneticField.OP77Q.get_kext(): [],
    MagneticField.Dip.get_kext(): [],
}

MAGINPUT_TO_INDEX: dict[SW_Index, int | list[int]] = {
    "Kp": 0,
    "Dst": 1,
    "Nsw": 2,
    "Vsw": 3,
    "Pdyn": 4,
    "ByIMF": 5,
    "BzIMF": 6,
    "G1": 7,
    "G2": 8,
    "G3": 9,
    "W_params": list(range(10, 16)),
}


def get_saveable_sw_indices(
    mag_field: MagneticFieldLiteral | MagneticField, data_standard: DataStandard[StandardName]
) -> list[SW_Index]:
    """Returns which SW indices a magnetic field model requires that can also be saved as output.

    Intersects `MAGINPUT_REQUIRED_INPUTS` for `mag_field`'s kext with the names `data_standard`
    actually registers, so saving strategies that want to persist the solar wind/geomagnetic
    indices used to compute their magnetic-field output alongside it don't each have to duplicate
    this filtering logic.

    Args:
        mag_field (MagneticFieldLiteral | MagneticField): The magnetic field model (or its
            literal name) to look up required inputs for.
        data_standard (DataStandard[StandardName]): The data standard whose registered
            `InternalName`s constrain which of the required indices can actually be saved.

    Returns:
        list[SW_Index]: The saveable indices required by `mag_field`, in a stable order.
    """
    if isinstance(mag_field, str):
        mag_field = MagneticField(mag_field)

    kext_value = mag_field.get_kext()

    return [idx for idx in MAGINPUT_REQUIRED_INPUTS.get(kext_value, []) if idx in data_standard.variable_infos]


class MagInputResult(NamedTuple):
    """The result of :func:`construct_maginput`.

    Attributes:
        maginput (dict[MagInputKeys, NDArray[np.float64]]): The IRBEM-facing magnetospheric input
            parameters, interpolated to the cadence of `time_var`.
        indices_solar_wind (VariablesDict): The actual `ep.Variable`s (with
            their original units and metadata, e.g. `source_files`) used to build `maginput`,
            keyed by the `SW_Index` they were loaded for (every `SW_Index` is also a valid
            `InternalName`, so callers can merge this directly into a `variables_to_save` dict).
            Restricted to the indices the requested `magnetic_field` model actually needs.
    """

    maginput: dict[MagInputKeys, NDArray[np.float64]]
    indices_solar_wind: VariablesDict


@cache
def construct_maginput(
    time_var: ep.Variable, magnetic_field: MagneticField, indices_solar_wind: dict[SW_Index, ep.Variable] | None = None
) -> MagInputResult:
    """Construct the magnetospheric input parameters required by IRBEM magnetic field models.

    This function gathers the geomagnetic indices and solar wind parameters required by the
    given `magnetic_field` model (loading any that are not already present in
    `indices_solar_wind` via `ep.load_indices_solar_wind_parameters`), interpolates them to the
    cadence of `time_var`, clips them to their valid ranges where applicable, and returns them
    as a dictionary keyed by `MagInputKeys`:

    - "Kp": Kp index * 10 (as in OMNI2 files), in the range 0 to 90.
    - "Dst": Dst index (nT).
    - "Nsw": Solar wind density (cm^-3).
    - "Vsw": Solar wind velocity (km/s).
    - "Pdyn": Solar wind dynamic pressure (nPa).
    - "ByIMF" / "BzIMF": GSM y/z components of the interplanetary magnetic field (nT).
    - "G1", "G2", "G3": Tsyganenko G parameters.
    - "W1"-"W6": Tsyganenko-Sitnov W parameters.
    - "AL": AL auroral index (NaN if not available).

    Args:
        time_var (ep.Variable): Array of new time points for interpolation.
        magnetic_field (MagneticField): The magnetic field model used to determine the required inputs.
        indices_solar_wind (dict[SW_Index, ep.Variable] | None, optional): A dictionary of pre-loaded solar
                                                                    wind variables. Defaults to None.

    Returns:
        MagInputResult: The interpolated magnetospheric input parameters, plus the underlying
            `ep.Variable`s (with metadata) they were built from.
    """
    time = time_var.get_data(ep.units.posixtime).astype(np.float64)
    start_time = datetime.fromtimestamp(time[0], tz=timezone.utc)
    end_time = datetime.fromtimestamp(time[-1], tz=timezone.utc)

    if indices_solar_wind is None:
        indices_solar_wind = {}

    kext = magnetic_field.get_kext()

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
        "Nsw": maginput[:, 2],
        "Vsw": maginput[:, 3],
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

    used_indices_solar_wind: VariablesDict = {req_input: indices_solar_wind[req_input] for req_input in required_inputs}

    return MagInputResult(maginput=maginput_dict, indices_solar_wind=used_indices_solar_wind)
