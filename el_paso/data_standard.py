# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import inspect

import logging
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, NamedTuple, TypeVar

from el_paso.utils import assert_n_dim

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astropy import units as u

    from el_paso.typing import InternalName, StandardName, Variable


logger = logging.getLogger("__name__")

T_co = TypeVar("T_co", bound=str, covariant=True)


class VariableInfo(NamedTuple, Generic[T_co]):  # noqa: D101
    standard_name: T_co
    description: str
    unit: u.UnitBase
    dependencies: list[InternalName | str]


class DataStandard(ABC, Generic[T_co]):
    """Abstract base class for data standardization."""

    variable_infos: dict[InternalName, VariableInfo[T_co]]

    def __repr__(self) -> str:
        cls = type(self)

        constructor_params = inspect.signature(cls.__init__).parameters
        args = []

        for name in constructor_params:
            if name == "self":
                continue

            if hasattr(self, name):
                value = getattr(self, name)
                args.append(f"{name}={value!r}")

        return f"{cls.__name__}({', '.join(args)})"

    def __str__(self) -> str:
        return self.__repr__()

    def get_internal_name(self, standard_name: StandardName) -> InternalName | None:
        for internal_name, var_info in self.variable_infos.items():
            if var_info.standard_name == standard_name:
                return internal_name

        return None

    def get_standard_name(self, internal_name: InternalName) -> T_co:
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

    def check(self, data_shape: tuple[int, ...], dim_names_or_sizes: Sequence[str | int], var_name: str) -> None:
        if len(data_shape) != len(dim_names_or_sizes):
            msg = "Encountered size missmatch!"
            raise ValueError(msg)

        for i, dim_name_or_size in enumerate(dim_names_or_sizes):
            self.check_size(data_shape[i], dim_name_or_size, var_name)

    def check_size(self, provided_len: int, dim_name_or_size: str | int, var_name: str) -> None:
        if isinstance(dim_name_or_size, int):
            if dim_name_or_size != provided_len:
                msg = (
                    f"Length mismatch! Variable {var_name} should have length {dim_name_or_size}, "
                    f"but encountered {provided_len}!",
                )
                raise ValueError(msg)
            return

        if dim_name_or_size in self.lengths:
            if self.lengths[dim_name_or_size].size != provided_len:
                msg = (
                    f"Length mismatch! {dim_name_or_size} length of variable "
                    f"{self.lengths[dim_name_or_size].name}: {self.lengths[dim_name_or_size].size}",
                    f"and of variable {var_name}: {provided_len}",
                )
                raise ValueError(msg)
        else:
            self.lengths[dim_name_or_size] = _SizeAttr(var_name, provided_len)
