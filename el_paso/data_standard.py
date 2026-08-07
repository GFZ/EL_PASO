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

    from el_paso.typing import FixedDimensionName, InternalName, StandardName, Variable


logger = logging.getLogger(__name__)

T_co = TypeVar("T_co", bound=str, covariant=True)


class VariableInfo(NamedTuple, Generic[T_co]):
    """A named tuple to store information about a variable in a data standard.

    A dependency entry can either be a single dimension name, or a tuple of
    alternative dimension names (e.g. `("Energy_FEDU", "Energy_FPDU")`).
    """

    standard_name: T_co
    description: str
    unit: u.UnitBase
    dependencies: list[InternalName | FixedDimensionName | tuple[InternalName, ...]]


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

        if internal_name not in self.variable_infos:
            msg = f"Internal name {internal_name} is not part of the {type(self)}!"
            raise ValueError(msg)

        return self.variable_infos[internal_name].standard_name

    def get_dependencies(
        self, internal_name: InternalName
    ) -> list[InternalName | FixedDimensionName | tuple[InternalName, ...]]:
        return self.variable_infos[internal_name].dependencies

    @staticmethod
    def resolve_dependencies(
        dependencies: list[InternalName | FixedDimensionName | tuple[InternalName, ...]],
        available_keys: set[InternalName] | None,
    ) -> list[InternalName | FixedDimensionName]:
        """Resolves alternative-name dependency entries to a single concrete name.

        For each dependency entry that is a tuple of alternatives, picks whichever
        alternative is present in `available_keys`. If `available_keys` is not
        provided, or none of the alternatives are present, falls back to the first
        alternative so downstream shape/consistency checks still have a name to use.
        """
        resolved: list[InternalName | FixedDimensionName] = []
        for dep in dependencies:
            if isinstance(dep, tuple):
                match = next((alt for alt in dep if available_keys and alt in available_keys), None)
                resolved.append(match if match is not None else dep[0])
            else:
                resolved.append(dep)
        return resolved

    def standardize_variable(
        self,
        internal_name: InternalName,
        variable: Variable,
        *,
        reset_consistency_check: bool,
        available_keys: set[InternalName] | None = None,
    ) -> Variable:
        """Standardizes a variable according to the data standard's rules.

        Args:
            internal_name (str): The internal name of the variable to be standardized.
            variable (Variable): The variable to be standardized.
            reset_consistency_check (bool): If set to true, the consistency check will be reseted.
            available_keys (set[InternalName] | None): The full set of internal names being
                saved together in this call.

        Returns:
            Variable: The standardized variable.
        """
        if reset_consistency_check:
            self.consistency_check = ConsistencyCheck()

        if internal_name not in self.variable_infos:
            logger.warning(f"Encountered custom variable which cannot be standardized: {internal_name}")
            return variable

        variable_info = self.variable_infos[internal_name]
        resolved_dependencies = self.resolve_dependencies(variable_info.dependencies, available_keys)

        variable.convert_to_unit(variable_info.unit)
        if len(variable.metadata.description) == 0:
            variable.metadata.description = variable_info.description
        assert_n_dim(variable, len(resolved_dependencies), internal_name)
        self.consistency_check.check(variable.get_data().shape, resolved_dependencies, internal_name)

        return variable

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataStandard):
            return NotImplemented
        return type(self) is type(other) and self.variable_infos == other.variable_infos


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
        lengths (dict[str | int, _SizeAttr]): Maps each named dimension (e.g. "time",
            "pitch_angle", "energy") to the variable name and size that were first
            observed for that dimension.
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
                    f"Length mismatch! Variable {var_name} should have length {dim_name_or_size}, but encountered {provided_len}!",  # noqa: E501
                )
                raise ValueError(msg)
            return

        if dim_name_or_size in self.lengths:
            if self.lengths[dim_name_or_size].size != provided_len:
                msg = (
                    f"Length mismatch! {dim_name_or_size} length of variable "
                    f"{self.lengths[dim_name_or_size].name}: {self.lengths[dim_name_or_size].size} "
                    f"and of variable {var_name}: {provided_len}"
                )
                raise ValueError(msg)
        else:
            self.lengths[dim_name_or_size] = _SizeAttr(var_name, provided_len)
