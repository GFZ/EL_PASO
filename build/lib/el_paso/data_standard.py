# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import el_paso as ep


class DataStandard(ABC):
    """Abstract base class for data standardization."""

    @abstractmethod
    def standardize_variable(
        self, standard_name: str, variable: ep.Variable, *, reset_consistency_check: bool
    ) -> ep.Variable:
        """Standardizes a variable according to the data standard's rules.

        This abstract method takes avariable and a standard name,
        and returns a new `el_paso.Variable` that conforms to the specified standard.

        Args:
            standard_name (str): The name of the standard to apply to the variable.
            variable (ep.Variable): The variable to be standardized.
            reset_consistency_check (bool): If set to true, the consistency check will be reseted.

        Returns:
            ep.Variable: The standardized variable.
        """


class _SizeAttr(NamedTuple):
    """A named tuple to store the name and size of a data dimension."""

    name: str = ""
    size: int = 0


@dataclass
class ConsistencyCheck:
    """A utility class for checking the consistency of data dimensions.

    This class helps verify that multiple variables saved to a file have
    the same length for shared dimensions (e.g., time, pitch angle, energy).

    Attributes:
        len_time (_SizeAttr | None): Stores the size of the time dimension from
                                     the first variable checked.
        len_pitch_angle (_SizeAttr | None): Stores the size of the pitch angle
                                            dimension from the first variable checked.
        len_energy (_SizeAttr | None): Stores the size of the energy dimension
                                       from the first variable checked.
    """

    len_time: _SizeAttr | None = None
    len_pitch_angle: _SizeAttr | None = None
    len_energy: _SizeAttr | None = None

    def check_time_size(self, provided_len_time: int, name_in_file: str) -> None:
        """Checks for consistency in the time dimension's length.

        The first time this method is called, it stores the provided length.
        Subsequent calls will raise a `ValueError` if the new length does not
        match the stored length.

        Args:
            provided_len_time (int): The length of the time dimension for the current variable.
            name_in_file (str): The name of the variable being checked.

        Raises:
            ValueError: If `provided_len_time` does not match the previously stored time length.
        """
        if self.len_time is None:
            self.len_time = _SizeAttr(name_in_file, provided_len_time)
        elif self.len_time.size != provided_len_time:
            msg = (
                f"Time length mismatch! Time length of variable {self.len_time.name}: {self.len_time.size}",
                f"and of variable {name_in_file}: {provided_len_time}",
            )
            raise ValueError(msg)

    def check_pitch_angle_size(self, provided_len_pitch_angle: int, name_in_file: str) -> None:
        """Checks for consistency in the pitch angle dimension's length.

        Args:
            provided_len_pitch_angle (int): The length of the pitch angle dimension.
            name_in_file (str): The name of the variable being checked.

        Raises:
            ValueError: If `provided_len_pitch_angle` does not match the previously stored
                        pitch angle length.
        """
        if self.len_pitch_angle is None:
            self.len_pitch_angle = _SizeAttr(name_in_file, provided_len_pitch_angle)
        elif self.len_pitch_angle.size != provided_len_pitch_angle:
            msg = (
                f"Pitch angle length mismatch! Pitch angle length of variable {self.len_pitch_angle.name}:"
                f"{self.len_pitch_angle.size} and of variable {name_in_file}: {provided_len_pitch_angle}"
            )
            raise ValueError(msg)

    def check_energy_size(self, provided_len_energy: int, name_in_file: str) -> None:
        """Checks for consistency in the energy dimension's length.

        Args:
            provided_len_energy (int): The length of the energy dimension.
            name_in_file (str): The name of the variable being checked.

        Raises:
            ValueError: If `provided_len_energy` does not match the previously stored
                        energy length.
        """
        if self.len_energy is None:
            self.len_energy = _SizeAttr(name_in_file, provided_len_energy)
        elif self.len_energy.size != provided_len_energy:
            msg = (
                f"Energy length mismatch! Energy length of variable {self.len_energy.name}:"
                f"{self.len_energy.size} and of variable {name_in_file}: {provided_len_energy}"
            )
            raise ValueError(msg)
