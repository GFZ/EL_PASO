# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, NamedTuple, TypeVar

from el_paso.utils import assert_n_dim

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astropy import units as u

    import el_paso as ep
    from el_paso.typing import InternalName, Variable


logger = logging.getLogger("__name__")

T = TypeVar("T", bound=str, covariant=True)


class VariableInfo(NamedTuple, Generic[T]):
    standard_name: T
    description: str
    unit: u.UnitBase
    dependencies: list[InternalName | str]


class DataStandard(ABC, Generic[T]):
    """Abstract base class for data standardization."""

    variable_infos: dict[InternalName, VariableInfo[T]]

    def get_internal_name(self, standard_name: T) -> InternalName | None:
        for internal_name, var_info in self.variable_infos.items():
            if var_info.standard_name == standard_name:
                return internal_name

        return None

    def get_full_var_name(self, internal_name: InternalName) -> T:
        return self.variable_infos[internal_name].standard_name

    def get_dependencies(self, internal_name: InternalName) -> list[InternalName | str]:
        return self.variable_infos[internal_name].dependencies

    def standardize_variable(
        self, internal_name: InternalName, variable: Variable, *, reset_consistency_check: bool
    ) -> Variable:
        """Standardizes a variable according to the data standard's rules.

        This abstract method takes avariable and a standard name,
        and returns a new `el_paso.Variable` that conforms to the specified standard.

        Args:
            internal_name (str): The name of the standard to apply to the variable.
            variable (Variable): The variable to be standardized.
            reset_consistency_check (bool): If set to true, the consistency check will be reseted.

        Returns:
            Variable: The standardized variable.
        """
        if reset_consistency_check:
            self.consistency_check = ConsistencyCheck()

        if internal_name not in self.variable_infos:
            logger.warning(f"Encountered custom variable which cannot be standardized: {internal_name}")
            return variable

        variable_info = self.variable_infos[internal_name]

        variable.convert_to_unit(variable_info.unit)
        if len(variable.metadata.description) == 0:
            variable.metadata.description = variable_info.description
        assert_n_dim(variable, len(variable_info.dependencies), internal_name)
        self.consistency_check.check(variable.get_data().shape, variable_info.dependencies, internal_name)

        return variable


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

    lengths: dict[str | int, _SizeAttr] = field(default_factory=dict[str | int, _SizeAttr])

    # len_time: _SizeAttr | None = None
    # len_pitch_angle: _SizeAttr | None = None
    # len_energy: _SizeAttr | None = None

    def check(self, data_shape: tuple[int, ...], dim_names_or_sizes: Sequence[str | int], var_name: str) -> None:
        if len(data_shape) != len(dim_names_or_sizes):
            msg = "Encountered size missmatch!"
            raise ValueError(msg)

        for i, dim_name_or_size in enumerate(dim_names_or_sizes):
            self.check_size(data_shape[i], dim_name_or_size, var_name)

    def check_size(self, provided_len: int, dim_name_or_size: str | int, var_name: str) -> None:
        if isinstance(dim_name_or_size, int) and dim_name_or_size != provided_len:
            msg = (
                f"Length mismatch! Variable {var_name} should have length {dim_name_or_size},"
                f"but encountered {provided_len}!",
            )
            raise ValueError(msg)

        if dim_name_or_size in self.lengths and self.lengths[dim_name_or_size].size != provided_len:
            msg = (
                f"Length mismatch! {dim_name_or_size} length of variable"
                f"{self.lengths[dim_name_or_size].name}: {self.lengths[dim_name_or_size].size}",
                f"and of variable {var_name}: {provided_len}",
            )
            raise ValueError(msg)

    # def check_time_size(self, provided_len_time: int, name_in_file: str) -> None:
    #     """Checks for consistency in the time dimension's length.

    #     The first time this method is called, it stores the provided length.
    #     Subsequent calls will raise a `ValueError` if the new length does not
    #     match the stored length.

    #     Args:
    #         provided_len_time (int): The length of the time dimension for the current variable.
    #         name_in_file (str): The name of the variable being checked.

    #     Raises:
    #         ValueError: If `provided_len_time` does not match the previously stored time length.
    #     """
    #     if self.len_time is None:
    #         self.len_time = _SizeAttr(name_in_file, provided_len_time)
    #     elif self.len_time.size != provided_len_time:
    #         msg = (
    #             f"Time length mismatch! Time length of variable {self.len_time.name}: {self.len_time.size}",
    #             f"and of variable {name_in_file}: {provided_len_time}",
    #         )
    #         raise ValueError(msg)

    # def check_pitch_angle_size(self, provided_len_pitch_angle: int, name_in_file: str) -> None:
    #     """Checks for consistency in the pitch angle dimension's length.

    #     Args:
    #         provided_len_pitch_angle (int): The length of the pitch angle dimension.
    #         name_in_file (str): The name of the variable being checked.

    #     Raises:
    #         ValueError: If `provided_len_pitch_angle` does not match the previously stored
    #                     pitch angle length.
    #     """
    #     if self.len_pitch_angle is None:
    #         self.len_pitch_angle = _SizeAttr(name_in_file, provided_len_pitch_angle)
    #     elif self.len_pitch_angle.size != provided_len_pitch_angle:
    #         msg = (
    #             f"Pitch angle length mismatch! Pitch angle length of variable {self.len_pitch_angle.name}:"
    #             f"{self.len_pitch_angle.size} and of variable {name_in_file}: {provided_len_pitch_angle}"
    #         )
    #         raise ValueError(msg)

    # def check_energy_size(self, provided_len_energy: int, name_in_file: str) -> None:
    #     """Checks for consistency in the energy dimension's length.

    #     Args:
    #         provided_len_energy (int): The length of the energy dimension.
    #         name_in_file (str): The name of the variable being checked.

    #     Raises:
    #         ValueError: If `provided_len_energy` does not match the previously stored
    #                     energy length.
    #     """
    #     if self.len_energy is None:
    #         self.len_energy = _SizeAttr(name_in_file, provided_len_energy)
    #     elif self.len_energy.size != provided_len_energy:
    #         msg = (
    #             f"Energy length mismatch! Energy length of variable {self.len_energy.name}:"
    #             f"{self.len_energy.size} and of variable {name_in_file}: {provided_len_energy}"
    #         )
    #         raise ValueError(msg)
