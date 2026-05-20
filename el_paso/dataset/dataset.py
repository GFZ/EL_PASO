# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""Combined RBM Dataset class supporting .mat, .pickle, and .nc file formats."""

from __future__ import annotations

import inspect
import logging
from datetime import timezone
from typing import TYPE_CHECKING, Any, cast

import distance
import numpy as np
from astropy.time import Time
from swvo.io.utils import enforce_utc_timezone

import el_paso as ep
from el_paso.dataset.utils import (
    join_var,
    matlab2python,
)

if TYPE_CHECKING:
    from datetime import datetime

    from numpy.typing import NDArray

    from el_paso.typing import (
        FileLoader,
        InternalName,
        MFSFormats,
        SavedDataDict,
        SavingStrategy,
    )

    DataDict = SavedDataDict
    FormatLoader = FileLoader

logger = logging.getLogger(__name__)


class DataSet:
    """DataSet class supporting .mat, and .nc file formats.

    This unified class handles loading data from multiple file formats and
    dictionary-backed sources. Dictionary updates are always allowed and replace
    existing values for standard variables.
    """

    _internal_attrs = frozenset(
        {
            "_verbose",
            "_preferred_ext",
            "saving_strategy",
            "possible_variables",
            "_start_time",
            "_end_time",
            "_date_list",
            "_loaders",
        }
    )

    def __init__(
        self,
        saving_strategy: SavingStrategy,
        start_time: datetime,
        end_time: datetime,
        preferred_extension: MFSFormats = "nc",
        *,
        verbose: bool = True,
    ) -> None:
        """Initializes a DataSet instance.

        Constructs the saving strategy, invokes the parent DataSet initializer,
        and populates the list of possible variables from class annotations.

        Args:
            saving_strategy (SavingStrategy): Instance of the saving strategy used to save the data.
            start_time (datetime): Beginning of the time range to load.
            end_time (datetime): End of the time range to load.
            preferred_extension (MFSFormats): File format to prefer when reading
                and writing data. Defaults to ``"nc"`` (NetCDF).
            verbose (bool): If ``True``, print progress and diagnostic messages.
                Defaults to ``True``.
        """
        self._verbose = verbose
        self._preferred_ext = preferred_extension

        self.saving_strategy = saving_strategy
        self.possible_variables = [
            *set(self.saving_strategy.get_all_standard_names()),
            "datetime",
            "P",
            "InvV",
        ]  # add computed properties and datetime
        assert end_time > start_time, "end_time must be after start_time"

        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time)

        self._start_time = start_time
        self._end_time = end_time
        self._date_list = ep.utils.get_monthly_datetime_intervals(start_time, end_time)

        self._loaders: dict[str, FormatLoader] = {
            ".mat": ep.utils.load_mat_data,
            ".h5": ep.utils.load_h5_data,
            ".nc": ep.utils.load_netcdf_data,
            ".cdf": ep.utils.load_cdf_data,
        }

    def __repr__(self) -> str:  # noqa: D105
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

    def __setattr__(self, name: str, value: object) -> None:  # noqa: D105
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        if name in self._internal_attrs:
            object.__setattr__(self, name, value)
            return

        possible_variables = getattr(self, "possible_variables", None)

        if possible_variables is not None and name in possible_variables:
            object.__setattr__(self, name, value)
            return

        _, levenstein_info = self.find_similar_variable(name)
        if levenstein_info["min_distance"] <= 2:
            msg = f"Cannot set attribute '{name}'. Maybe you meant '{levenstein_info['var_name']}'?"
        else:
            msg = f"Cannot set attribute '{name}'. It is not part of {self.saving_strategy.data_standard}."

        raise AttributeError(msg)

    def __str__(self) -> str:  # noqa: D105
        return self.__repr__()

    def __getattr__(self, name: str) -> NDArray[np.float64]:  # noqa: D105
        # Avoid recursion for internal attributes
        if name.startswith("_"):
            msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
            raise AttributeError(msg)

        # Handle computed properties for both modes
        if name == "P":
            if len(self.MLT) == 0:  # MLT not found
                self.P = np.asarray([])
            else:
                self.P = ((self.MLT + 12) / 12 * np.pi) % (2 * np.pi)
            return self.P

        if name == "InvV":
            if len(self.InvK) == 0 or len(self.InvMu) == 0:  # invariants not found
                self.InvV = np.asarray([])
            else:
                inv_K_repeated = np.repeat(self.InvK[:, np.newaxis, :], self.InvMu.shape[1], axis=1)
                self.InvV = self.InvMu * (inv_K_repeated + 0.5) ** 2
            return self.InvV

        # check if a sat variable is requested
        # if we find a similar word, suggest that to the user
        sat_variable, levenstein_info = self.find_similar_variable(name)
        if sat_variable is not None and hasattr(self, "_date_list"):
            self._load_variable(sat_variable)
            return getattr(self, name)
        if name in self.possible_variables:
            msg = f"Attribute '{name}' exists in {self.saving_strategy.data_standard} but has not been set. "
            raise AttributeError(msg)
        if levenstein_info["min_distance"] <= 2:
            msg = f"{self.__class__.__name__} object has no attribute {name}. Maybe you meant {levenstein_info['var_name']}?"  # noqa: E501
        else:
            msg = f"{self.__class__.__name__} object has no attribute {name}"

        raise AttributeError(msg)

    def load(self, name_or_var: str) -> None:
        """Load data into memory."""
        getattr(self, name_or_var)

    def get_var_by_internal_name(self, internal_name: InternalName):  # noqa: ANN201
        standard_name = self.saving_strategy.data_standard.get_standard_name(internal_name)
        return getattr(self, standard_name)

    def find_similar_variable(self, name: str) -> tuple[None | str, dict[str, Any]]:
        levenstein_info: dict[str, Any] = {"min_distance": 10, "var_name": ""}
        sat_variable = None
        for var in self.possible_variables:
            if name == var:
                sat_variable = var
                break
            dist = distance.levenshtein(name, var)
            if not var:
                continue
            if name.lower() in var.lower():
                dist = 1

            if dist < levenstein_info["min_distance"]:
                levenstein_info["min_distance"] = dist
                levenstein_info["var_name"] = var

        return sat_variable, levenstein_info

    def get_satellite_name(self) -> str:
        return self.saving_strategy.satellite

    def get_satellite_and_instrument_name(self) -> str:
        return self.saving_strategy.satellite + "_" + self.saving_strategy.instrument

    def get_print_name(self) -> str:
        return self.saving_strategy.satellite + " " + self.saving_strategy.instrument

    def _load_variable(self, requested_name: str) -> None:
        """Load variable from .mat, or .nc files."""
        loaded_var_arrs: dict[str, NDArray[np.number]] = {}
        var_names_stored: list[str] = []

        # 1. Handle Computed Values
        if requested_name == "InvV":
            inv_K_repeated = np.repeat(self.InvK[:, np.newaxis, :], self.InvMu.shape[1], axis=1)
            self.InvV = self.InvMu * (inv_K_repeated + 0.5) ** 2
            return
        if requested_name == "P":
            self.P = ((self.MLT + 12) / 12 * np.pi) % (2 * np.pi)
            return

        if requested_name == "datetime":
            requested_name = self.saving_strategy.data_standard.get_standard_name("Epoch")

        output_file = self.saving_strategy.get_output_file(standard_name=requested_name)  # ty:ignore[invalid-argument-type]

        # useful when using GFZStrategy, when multiple files are there.
        if requested_name == "datetime" and output_file is None:
            time_key = self.saving_strategy.data_standard.get_standard_name("Epoch")
            output_file = next(
                (
                    candidate_output_file
                    for candidate_output_file in self.saving_strategy.output_files
                    if time_key in candidate_output_file.names_to_save
                ),
                None,
            )

        if output_file is None:
            msg = "This var name is not part of the chosen saving strategy!"
            raise ValueError(msg)

        # 2. Iterate through date ranges
        for time_start, time_end in self._date_list:
            full_file_path = self.saving_strategy.get_file_path(time_start, time_end, output_file)

            if full_file_path.exists():
                file_content = self._loaders[(ep.utils.normalize_file_format(full_file_path.suffix))](full_file_path)

                if self._verbose:
                    logger.info(f"Loading {full_file_path}")
            elif self._verbose:
                logger.warning(f"Tried to load {full_file_path}, but it does not exist")
                continue

            time_key = self.saving_strategy.data_standard.get_standard_name("Epoch")

            # 4. Process Datetimes
            raw_times = file_content[time_key]

            time_unit = self.saving_strategy.data_standard.variable_infos["Epoch"].unit

            posix_times = (raw_times * time_unit).to_value(ep.units.posixtime)
            times = Time(posix_times, format="unix", scale="utc")
            datetimes = times.to_datetime(timezone=timezone.utc)

            file_content["datetime"] = datetimes  # ty:ignore[invalid-assignment]
            correct_time_idx = (datetimes >= self._start_time) & (datetimes <= self._end_time)

            # 5. Filter and Join Arrays
            for key, var_arr in file_content.items():
                # Skip non-numeric metadata (excluding our new datetime)
                if key != "datetime" and (
                    not isinstance(var_arr, np.ndarray) or not np.issubdtype(var_arr.dtype, np.number)
                ):
                    continue

                var_arr = cast("NDArray[np.number]", var_arr)

                # Time-dependent trimming
                if var_arr.shape[0] == correct_time_idx.shape[0]:
                    var_arr = var_arr[correct_time_idx.reshape(-1), ...]
                    joined_value = join_var(loaded_var_arrs[key], var_arr) if key in loaded_var_arrs else var_arr
                else:
                    joined_value = var_arr

                loaded_var_arrs[key] = joined_value  # ty:ignore[invalid-assignment]
                if key not in var_names_stored:
                    var_names_stored.append(key)

        # 6. Final Assignment to Self
        if requested_name not in var_names_stored:
            setattr(self, requested_name, np.asarray([]))

        for var_name in var_names_stored:
            val = list(loaded_var_arrs[var_name]) if var_name == "datetime" else loaded_var_arrs[var_name]
            setattr(self, var_name, val)  # set standard name

    def get_loaded_variables(self) -> list[str]:
        """Get a list of currently loaded variable names."""
        return [var for var in self.possible_variables if var in self.__dict__]

    def assert_equal(self, other: DataSet) -> None:
        """Assert that two DataSet objects are equal."""
        assert self.saving_strategy.data_standard == other.saving_strategy.data_standard, (
            "Data standards are different:\n"
            f"{self.saving_strategy.data_standard!r} != {other.saving_strategy.data_standard!r}"
        )

        self_standard_names = sorted(self.saving_strategy.get_all_standard_names())
        other_standard_names = sorted(other.saving_strategy.get_all_standard_names())

        assert self_standard_names == other_standard_names, (
            f"Standard names in saving strategies are different:\n{self_standard_names!r}!={other_standard_names!r}"
        )

        different_vars = self.get_different_variables(other)

        assert len(different_vars) == 0, "Variables differ between datasets:\n" + "\n".join(
            str(var) for var in different_vars
        )

    def __eq__(self, other: object) -> bool:  # noqa: D105
        if not isinstance(other, DataSet):
            msg = f"Cannot compare DataSet with object of type {type(other)}"
            logger.error(msg)
            raise TypeError(msg)

        try:
            self.assert_equal(other)
        except AssertionError:
            return False

        return True

    def get_different_variables(self, other: DataSet) -> list[str]:
        """Compare the currently loaded variables in this DataSet with another DataSet and return a list of variable names that are different.

        Args:
            other (DataSet): Another DataSet instance to compare against.

        Returns:
            list[str]: A list of variable names that are different between the two DataSet instances.
            This includes variables that are present in one instance but not the other, as well as variables that are
            present in both instances but have different values or shapes.
        """  # noqa: E501
        different_vars: list[str] = []

        for var_name in self.possible_variables:
            self_var = getattr(self, var_name)
            other_var = getattr(other, var_name)

            if not isinstance(other_var, type(self_var)):
                different_vars.append(var_name)
                continue

            if isinstance(self_var, list):
                if len(self_var) != len(other_var) or any(a != b for a, b in zip(self_var, other_var, strict=True)):
                    different_vars.append(var_name)
                    continue
            elif isinstance(self_var, np.ndarray):
                if self_var.shape != other_var.shape or not np.allclose(self_var, other_var, equal_nan=True):
                    different_vars.append(var_name)
                    continue
            elif self_var != other_var:
                different_vars.append(var_name)
                continue

        return different_vars

    from .bin_and_interpolate_to_model_grid import bin_and_interpolate_to_model_grid  # noqa: PLC0415
    from .identify_orbits import identify_orbits  # noqa: PLC0415
    from .interp_functions import interp_flux, interp_psd  # noqa: PLC0415
    from .linearize_trajectories import linearize_trajectories  # noqa: PLC0415
