# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from el_paso.processing.magnetic_field_utils.construct_maginput import construct_maginput
from el_paso.processing.magnetic_field_utils.irbem import Coords
from el_paso.processing.magnetic_field_utils.mag_field_enum import MagneticField, MagneticFieldLiteral, kext
from el_paso.processing.magnetic_field_utils.magnetic_field_functions import (
    IrbemInput,
    IrbemOutput,
    MagFieldVarTypes,
    create_var_name,
    get_footpoint_atmosphere,
    get_local_B_field,
    get_Lstar,
    get_magequator,
    get_mirror_point,
    get_MLT,
)

__all__ = [
    "Coords",
    "IrbemInput",
    "IrbemOutput",
    "MagFieldVarTypes",
    "MagneticField",
    "MagneticFieldLiteral",
    "construct_maginput",
    "create_var_name",
    "get_Lstar",
    "get_MLT",
    "get_footpoint_atmosphere",
    "get_local_B_field",
    "get_magequator",
    "get_mirror_point",
    "kext",
]
