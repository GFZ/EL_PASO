# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, overload

import numpy as np
from astropy import units as u  # type: ignore[reportMissingTypeStubs]

import el_paso as ep
from el_paso.utils import enforce_utc_timezone

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class VariableMetadata:
    """A class holding the metadata of a variable.

    Attributes:
        unit (u.UnitBase): The unit of the variable. Defaults to
            `u.dimensionless_unscaled`.
        original_cadence_seconds (float): The original cadence of the data in seconds.
            Defaults to 0.
        source_files (list[str]): The list of SourceFiles, which variable contains
            data from. Defaults to an empty list.
        description (str): The description of the variable explaining what kind of data
            this variable contains. Defaults to "".
        processing_notes (str): The processing notes of the variable explaining all
            steps done to achieve the final result. Defaults to "".
        standard_name (str): The name of the standard variable this variable complies
            to. Defaults to "".
    """

    unit: u.UnitBase = u.dimensionless_unscaled
    original_cadence_seconds: float = 0
    source_files: list[str] = field(default_factory=list[str])
    description: str = ""
    processing_notes: str = ""
    standard_name: str = ""

    def __post_init__(self) -> None:
        """Initializes the processing_steps_counter attribute to 1 after the dataclass has been instantiated.

        This method is automatically called by the dataclass after the __init__ method.
        """
        self.processing_steps_counter = 1

        if ep.is_in_release_mode():
            self.processing_notes += ep.get_release_msg() + "\n"


    def add_processing_note(self, processing_note: str) -> None:
        """Adds a processing note to the metadata.

        The note is prefixed with the current processing steps counter and a newline
        character is appended. The processing steps counter is then incremented.

        Args:
            processing_note (str): The note to be added to the processing notes.
        """
        processing_note = f"{self.processing_steps_counter}) {processing_note}\n"

        self.processing_notes += processing_note
        self.processing_steps_counter += 1


class Variable:
    """Variable class holding data and metadata.

    Attributes:
        _data (NDArray[np.generic]): The numerical data of the variable.
        metadata (VariableMetadata): An instance of `VariableMetadata` holding
            information about the variable.
    """

    __slots__ = "_data", "metadata"

    _data: NDArray[np.generic]
    metadata: VariableMetadata

    def __init__(
        self,
        original_unit: u.UnitBase,
        data: NDArray[np.generic] | None = None,
        description: str = "",
        processing_notes: str = "",
    ) -> None:
        """Initializes a Variable instance.

        Args:
            original_unit (u.UnitBase): The original unit of the data.
            data (NDArray[np.generic] | None): The numerical data. Defaults to an empty
                numpy array if None.
            description (str): A description of the variable. Defaults to "".
            processing_notes (str): Notes on how the data was processed. Defaults to "".
        """
        self._data = np.array([]) if data is None else data

        self.metadata = VariableMetadata(
            unit=original_unit,
            description=description,
            processing_notes=processing_notes,
        )

    def __repr__(self) -> str:
        """Returns a string representation of the Variable object."""
        return f"Variable holding {self._data.shape} data points with metadata: {self.metadata}"

    def convert_to_unit(self, target_unit: u.UnitBase | str) -> None:
        """Converts the data to a given unit.

        Args:
            target_unit (u.UnitBase | str): The unit the data should be converted to.
        """
        if isinstance(target_unit, str):
            target_unit = u.Unit(target_unit)

        if self.metadata.unit != target_unit:
            data_with_unit = u.Quantity(self._data, self.metadata.unit)
            self._data = typing.cast("NDArray[np.generic]", data_with_unit.to_value(target_unit))  # type: ignore[reportUnknownMemberType]

            self.metadata.unit = target_unit

    @overload
    def get_data(self, target_unit: u.UnitBase | str) -> NDArray[np.floating | np.integer]: ...

    @overload
    def get_data(self, target_unit: None = None) -> NDArray[np.generic]: ...

    def get_data(self, target_unit: u.UnitBase | str | None = None) -> NDArray[np.generic]:
        """Gets the data of the variable.

        Args:
            target_unit (u.UnitBase | str | None): The unit to convert the data to
                before returning. If None, the data is returned in its current unit.
                Defaults to None.

        Returns:
            NDArray[np.generic]: The data of the variable.

        Raises:
            TypeError: If `target_unit` is provided and the data is not numeric.
        """
        if target_unit is None:
            return self._data

        if isinstance(target_unit, str):
            target_unit = u.Unit(target_unit)

        if not np.issubdtype(self._data.dtype, np.number):
            msg = f"Unit conversion is only supported for numeric types! Encountered for variable {self}."
            raise TypeError(msg)

        return typing.cast("NDArray[np.generic]", u.Quantity(self._data, self.metadata.unit).to_value(target_unit))  # type: ignore[reportUnknownMemberType]

    def set_data(self, data: NDArray[np.generic], unit: Literal["same"] | str | u.UnitBase) -> None:  # noqa: PYI051
        """Sets the data and optionally updates the unit of the variable.

        Args:
            data (NDArray[np.generic]): The new data array.
            unit (Literal["same"] | str | u.UnitBase): The unit of the new data.
                If "same", the existing unit is kept. Can be a string representation
                of a unit or an `astropy.units.UnitBase` object.

        Raises:
            TypeError: If `unit` is not "same", a string, or an `astropy.units.UnitBase` object.
        """
        self._data = data

        if isinstance(unit, str):
            if unit != "same":
                self.metadata.unit = u.Unit(unit)
        elif isinstance(unit, u.UnitBase):  # type: ignore[reportUnknownMemberType]
            self.metadata.unit = unit
        else:
            msg = "unit must be either a str or a astropy unit!"
            raise TypeError(msg)

    def transpose_data(self, seq: list[int] | tuple[int, ...]) -> None:
        """Transposes the internal data array.

        Args:
            seq (list[int] | tuple[int, ...]): The axes to transpose to. See
                `numpy.transpose` for details.
        """
        self._data = np.transpose(self._data, axes=seq)

    def apply_thresholds_on_data(self, lower_threshold: float = -np.inf, upper_threshold: float = np.inf) -> None:
        """Applies lower and upper thresholds to the data.

        Values outside the thresholds (exclusive) are set to NaN.

        Args:
            lower_threshold (float): The lower bound for the data. Defaults to
                negative infinity.
            upper_threshold (float): The upper bound for the data. Defaults to
                positive infinity.

        Raises:
            TypeError: If the data is not numeric.
        """
        if not np.issubdtype(self._data.dtype, np.number):
            msg = f"Thresholds are only supported for numeric types! Encountered for variable {self}."
            raise TypeError(msg)
        self._data = typing.cast("NDArray[np.number]", self._data)

        self._data = np.where((self._data > lower_threshold) & (self._data < upper_threshold), self._data, np.nan)

    def truncate(self, time_variable: Variable, start_time: float | datetime, end_time: float | datetime) -> None:
        """Truncates the variable's data based on a time variable and a time range.

        Args:
            time_variable (Variable): A `Variable` object containing the time data.
            start_time (float | datetime): The start time for truncation. Can be a
                Unix timestamp (float) or a `datetime` object.
            end_time (float | datetime): The end time for truncation. Can be a
                Unix timestamp (float) or a `datetime` object.

        Raises:
            ValueError: If the length of the variable's data does not match the
                length of the `time_variable`'s data.
        """
        if isinstance(start_time, datetime):
            start_time = enforce_utc_timezone(start_time).timestamp()
        if isinstance(end_time, datetime):
            end_time = enforce_utc_timezone(end_time).timestamp()

        if self._data.shape[0] != time_variable.get_data().shape[0]:
            msg = f"Encountered length missmatch between variable and time variable! Variable: {self}"
            raise ValueError(msg)

        time_var_data = time_variable.get_data(ep.units.posixtime)

        self._data = self._data[(time_var_data >= start_time) & (time_var_data <= end_time)]

    def __hash__(self) -> int:
        """Computes a hash value for the variable based on its holding data.

        Returns:
            int: The integer hash value.
        """
        return hash(self._data.tobytes())
