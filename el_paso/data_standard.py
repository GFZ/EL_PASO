# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import logging
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, Literal, NamedTuple, TypeVar

import numpy as np

from el_paso.typing import InternalName
from el_paso.utils import assert_n_dim

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astropy import units as u

    from el_paso.typing import FixedDimensionName, StandardName, Variable


logger = logging.getLogger(__name__)

T_co = TypeVar("T_co", bound=str, covariant=True)


SortOrder = tuple[InternalName, Literal["ascending", "descending"]]


class VariableInfo(NamedTuple, Generic[T_co]):
    """A named tuple to store information about a variable in a data standard.

    Attributes:
        standard_name: The canonical name used by this data standard.
        description: Human-readable description of the variable.
        unit: The physical unit the data must be stored in.
        dependencies: Ordered list of dimension names for this variable.
        sorted_along: Optional ``(dimension_name, "ascending" | "descending")``
            tuple.  When set, :meth:`DataStandard.standardize_variable` will
            verify that the data is monotonically sorted along the given
            dimension.
    """

    standard_name: T_co
    description: str
    unit: u.UnitBase
    dependencies: list[InternalName | FixedDimensionName]
    sorted_along: SortOrder | None = None


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
        """Looks up the internal name registered for a given standard name.

        Args:
            standard_name (StandardName): The standard name to look up.

        Returns:
            InternalName | None: The internal name associated with ``standard_name``,
            or ``None`` if no variable in this data standard has that standard name.
        """
        for internal_name, var_info in self.variable_infos.items():
            if var_info.standard_name == standard_name:
                return internal_name

        return None

    def get_standard_name(self, internal_name: InternalName) -> T_co:
        """Looks up the standard name registered for a given internal name.

        Args:
            internal_name (InternalName): The internal name to look up.

        Returns:
            T_co: The standard name associated with ``internal_name``.

        Raises:
            ValueError: If ``internal_name`` is not part of this data standard.
        """
        if internal_name not in self.variable_infos:
            msg = f"Internal name {internal_name} is not part of the {type(self)}!"
            raise ValueError(msg)

        return self.variable_infos[internal_name].standard_name

    def get_dependencies(self, internal_name: InternalName) -> list[InternalName | FixedDimensionName]:
        """Returns the ordered dimension names that a variable depends on.

        Args:
            internal_name (InternalName): The internal name of the variable.

        Returns:
            list[InternalName | FixedDimensionName]: The ordered list of dimension names.
        """
        return self.variable_infos[internal_name].dependencies

    def standardize_variable(
        self, internal_name: InternalName, variable: Variable, *, reset_consistency_check: bool
    ) -> Variable:
        """Standardizes a variable according to the data standard's rules.

        This abstract method takes avariable and a standard name,
        and returns a new `el_paso.Variable` that conforms to the specified standard.

        Args:
            internal_name (str): The internal name of the variable to be standardized.
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

        if variable_info.sorted_along is not None:
            _assert_sorted(variable.get_data(), variable_info, internal_name)

        return variable

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataStandard):
            return NotImplemented
        return type(self) is type(other) and self.variable_infos == other.variable_infos


def _assert_sorted(
    data: np.ndarray,
    variable_info: VariableInfo[str],
    internal_name: str,
) -> None:
    assert variable_info.sorted_along is not None
    dim_name, order = variable_info.sorted_along

    axis = list(variable_info.dependencies).index(dim_name)
    diffs = np.diff(data, axis=axis)

    violating = np.any(diffs < 0) if order == "ascending" else np.any(diffs > 0)

    if violating:
        msg = f"Variable '{internal_name}' must be sorted {order} along dimension '{dim_name}' (axis {axis})."
        raise ValueError(msg)


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
        """Checks a variable's data shape against previously observed dimension sizes.

        Args:
            data_shape (tuple[int, ...]): The shape of the variable's data.
            dim_names_or_sizes (Sequence[str | int]): For each axis in ``data_shape``,
                either the name of the shared dimension or a fixed expected size.
            var_name (str): Name of the variable being checked, used in error messages.

        Raises:
            ValueError: If ``data_shape`` and ``dim_names_or_sizes`` have different
                lengths, or if a checked size is inconsistent with a previous observation.
        """
        if len(data_shape) != len(dim_names_or_sizes):
            msg = "Encountered size missmatch!"
            raise ValueError(msg)

        for i, dim_name_or_size in enumerate(dim_names_or_sizes):
            self.check_size(data_shape[i], dim_name_or_size, var_name)

    def check_size(self, provided_len: int, dim_name_or_size: str | int, var_name: str) -> None:
        """Checks a single dimension's size against previously observed sizes.

        If ``dim_name_or_size`` is a named dimension seen for the first time, its
        size is recorded for future comparisons.

        Args:
            provided_len (int): The size to check.
            dim_name_or_size (str | int): Either the name of a shared dimension, or a
                fixed integer size that ``provided_len`` must exactly match.
            var_name (str): Name of the variable being checked, used in error messages.

        Raises:
            ValueError: If ``provided_len`` does not match the fixed size given, or
                does not match the previously recorded size for the named dimension.
        """
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
