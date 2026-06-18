# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from el_paso.processing.magnetic_field_utils.construct_maginput import construct_maginput
from el_paso.processing.magnetic_field_utils.irbem import Coords, InternalFieldModel, IrbemOptions, LstarQuantity
from el_paso.processing.magnetic_field_utils.mag_field_enum import MagneticField, kext
from el_paso.processing.magnetic_field_utils.dipole_field import (
    dipole_get_footpoint_atmosphere,
    dipole_get_local_B_field,
    dipole_get_Lstar,
    dipole_get_magequator,
    dipole_get_mirror_point,
    dipole_get_MLT,
)
from el_paso.processing.magnetic_field_utils.magnetic_field_functions import (
    IrbemInput,
    IrbemOutput,
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
    "InternalFieldModel",
    "IrbemInput",
    "IrbemOptions",
    "IrbemOutput",
    "LstarQuantity",
    "MagneticField",
    "construct_maginput",
    "create_var_name",
    "dipole_get_Lstar",
    "dipole_get_MLT",
    "dipole_get_footpoint_atmosphere",
    "dipole_get_local_B_field",
    "dipole_get_magequator",
    "dipole_get_mirror_point",
    "get_Lstar",
    "get_MLT",
    "get_footpoint_atmosphere",
    "get_local_B_field",
    "get_magequator",
    "get_mirror_point",
    "kext",
]
