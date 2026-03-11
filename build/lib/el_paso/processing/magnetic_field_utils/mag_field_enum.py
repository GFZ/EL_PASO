# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Literal, NewType

kext = NewType("kext", int)


def _magnetic_field_str_to_kext(magnetic_field_str: str) -> kext:
    match magnetic_field_str:
        case "T89":
            mag_kext = kext(4)
        case "T01":
            mag_kext = kext(9)
        case "T01s":
            mag_kext = kext(10)
        case "TS04" | "TS05" | "T04s":
            mag_kext = kext(11)
        case "T96":
            mag_kext = kext(7)
        case "OP77Q":
            mag_kext = kext(5)
        case "OP77":
            mag_kext = kext(5)
        case _:
            msg = "Invalid magnetic field model!"
            raise ValueError(msg)

    return mag_kext


MagneticFieldLiteral = Literal["T89", "T01", "T01s", "TS04", "TS05", "T04s", "T96", "OP77Q", "OP77"]


class MagneticField(Enum):
    """Enum for magnetic field models."""

    T89 = "T89"
    T01 = "T01"
    T01s = "T01s"
    TS04 = "TS04"
    TS05 = "TS05"
    T04s = "T04s"
    T96 = "T96"
    OP77Q = "OP77Q"
    OP77 = "OP77"

    def kext(self) -> kext:
        """Returns the kext value for the magnetic field model."""
        return _magnetic_field_str_to_kext(self.value)

    @classmethod
    def _missing_(cls, value: object) -> None:
        msg = "{!r} is not a valid {}.  Valid types: {}".format(
            value,
            cls.__name__,
            ", ".join([repr(m.value) for m in cls]),
        )
        raise ValueError(msg)
