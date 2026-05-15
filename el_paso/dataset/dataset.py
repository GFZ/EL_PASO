# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""Combined RBM Dataset class supporting .mat, .pickle, and .nc file formats."""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import distance
import numpy as np
from swvo.io.utils import enforce_utc_timezone

import el_paso as ep
from el_paso.dataset.utils import (
    join_var,
    matlab2python,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from el_paso.typing import (
        FileLoader,
        InternalName,
        MFSFormats,
        SavedDataDict,
        SavingStrategy,
        StandardName,
    )

    DataDict = SavedDataDict
    FormatLoader = FileLoader

logger = logging.getLogger(__name__)


class DataSet:
    """DataSet class supporting .mat, and .nc file formats.

    This unified class handles loading RBM (Radiation Belt Model) data from multiple
    file formats. It can load data either from files or from a dictionary.

    For file-based loading, provide `start_time`, `end_time`, and `folder_path`.
    For dictionary-based loading, initialize without these parameters and use `update_from_dict()`.

    Parameters
    ----------
    satellite : Union[:class:`SatelliteLike`, :class:`DummyLike`]
        Satellite identifier as enum or string.
    instrument : Union[:class:`InstrumentLike`, :class:`DummyLike`]
        Instrument enumeration or string.
    mfm : Union[:class:`MfmLike`, :class:`DummyLike`]
        Magnetic field model enum or string.
    start_time : dt.datetime, optional
        Start time for file-based loading.
    end_time : dt.datetime, optional
        End time for file-based loading.
    folder_path : Path, optional
        Base folder path for file-based loading.
    preferred_extension : Literal["mat", "nc"], optional
        Preferred file extension for file-based loading. Default is "nc".
    verbose : bool, optional
        Whether to print verbose output. Default is True.
    enable_dict_loading : bool, optional
        Enable dictionary-based loading even in file mode. Default is False.

    Attributes:
    ----------
    datetime : list[dt.datetime]
    FEDU: NDArray[np.float64]
    FEDO: NDArray[np.float64]
    FEIU: NDArray[np.float64]
    Energy_FEDU: NDArray[np.float64]
    Epoch: NDArray[np.float64]
    Alpha: NDArray[np.float64]
    Alpha_Eq: NDArray[np.float64]
    Position: NDArray[np.float64]
    B_Calc: NDArray[np.float64]
    B_Eq: NDArray[np.float64]
    L_star: NDArray[np.float64]
    I: NDArray[np.float64]
    MLT: NDArray[np.float64]
    L_m: NDArray[np.float64]
    PSD: NDArray[np.float64]
    R_Eq: NDArray[np.float64]
    InvMu: NDArray[np.float64]
    InvK: NDArray[np.float64]

    """

    def __init__(  # noqa: D107
        self,
        saving_strategy: SavingStrategy,
        start_time: dt.datetime | None = None,
        end_time: dt.datetime | None = None,
        preferred_extension: MFSFormats = "nc",
        *,
        verbose: bool = True,
        enable_dict_loading: bool = False,
    ) -> None:

        # Store the original satellite enum for properties and other attributes
        self._verbose = verbose
        self._preferred_ext = preferred_extension

        self.saving_strategy = saving_strategy

        # For dict-based loading, modify satellite properties
        self._file_loading_mode = True
        if start_time is None and end_time is None:
            self._file_loading_mode = False
        else:
            # File loading mode: need all parameters
            if start_time is None or end_time is None:
                msg = "For file-based loading, start_time and end_time must be provided"
                raise ValueError(msg)

            start_time = enforce_utc_timezone(start_time)
            end_time = enforce_utc_timezone(end_time)

            self._start_time = start_time
            self._end_time = end_time
            self._date_list = self.saving_strategy.get_time_intervals_to_save(start_time, end_time)
            self._enable_dict_loading = enable_dict_loading
            self._dataset_cache: dict[Path, dict[StandardName, Any]] = {}
            self._is_nc_dataset: bool = False

            self._loaders: dict[str, FormatLoader] = {
                ".mat": ep.utils.load_mat_data,
                ".h5": ep.utils.load_h5_data,
                ".nc": ep.utils.load_netcdf_data,
                ".cdf": ep.utils.load_cdf_data,
            }

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
                inv_K_repeated = np.repeat(self.InvK[:, np.newaxis, :], self.InvMu.shape[1], axis=1)  # noqa: N806
                self.InvV = self.InvMu * (inv_K_repeated + 0.5) ** 2
            return self.InvV

        # check if a sat variable is requested
        # if we find a similar word, suggest that to the user
        sat_variable, levenstein_info = self.find_similar_variable(name)
        if sat_variable is not None and self._file_loading_mode:
            self._load_variable(sat_variable)
            return getattr(self, name)
        if not self._file_loading_mode and name in self.possible_variables:
            msg = (
                f"Attribute '{name}' exists in `VariableLiteral` but has not been set. "
                "Call `update_from_dict()` before accessing it."
            )
            raise AttributeError(msg)

        if levenstein_info["min_distance"] <= 2:  # noqa: PLR2004
            msg = f"{self.__class__.__name__} object has no attribute {name}. Maybe you meant {levenstein_info['var_name']}?"  # noqa: E501
        else:
            msg = f"{self.__class__.__name__} object has no attribute {name}"

        raise AttributeError(msg)

    def load(self, name_or_var: str) -> None:
        """Load data into memory."""
        getattr(self, name_or_var)

    def get_var_by_internal_name(self, internal_name: InternalName):  # noqa: ANN201
        standard_name = self.saving_strategy.data_standard.get_full_var_name(internal_name)
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

    # ruff: disable[ERA001, E501]
    # def update_from_dict(self, source_dict: dict[str, NDArray[np.floating] | list[dt.datetime]]) -> DataSet:
    #     """Get data from data dictionary and update the object.

    #     Parameters
    #     ----------
    #     source_dict : dict[str, VariableLiteral]
    #         Dictionary containing the data to be loaded into the object.

    #     Returns:
    #     -------
    #     DataSet
    #         The updated DataSet object.

    #     Raises:
    #     ------
    #     VariableNotFoundError
    #         If a key in the `source_dict` is not a valid `VariableLiteral`.
    #     RuntimeError
    #         If the DataSet is in file loading mode and dictionary loading is not enabled.

    #     """
    #     if self._file_loading_mode and not self._enable_dict_loading:
    #         msg = "DataSet is in file loading mode. Cannot update from dictionary. To use dictionary-based loading, set `enable_dict_loading=True` during initialization."
    #         raise RuntimeError(msg)
    #     for key, value in source_dict.items():
    #         _, levenstein_info = self.find_similar_variable(key)
    #         if key in self.possible_variables:
    #             setattr(self, key, value)
    #         elif levenstein_info["min_distance"] <= 2:
    #             msg = f"Key '{key}' is not a valid `VariableLiteral`. Maybe you meant '{levenstein_info['var_name']}'?"
    #             raise VariableNotFoundError(msg)
    #         else:
    #             msg = f"Key '{key}' is not a valid `VariableLiteral`."
    #             raise VariableNotFoundError(msg)
    #     return self
    # ruff:enable[ERA001, E501]

    def get_satellite_name(self) -> str:
        return self.saving_strategy.satellite

    def get_satellite_and_instrument_name(self) -> str:
        return self.saving_strategy.satellite + "_" + self.saving_strategy.instrument

    def get_print_name(self) -> str:
        return self.saving_strategy.satellite + " " + self.saving_strategy.instrument

    def _load_variable(self, requested_name: str) -> None:  # noqa: C901, PLR0912
        """Load variable from .mat, or .nc files."""
        loaded_var_arrs: dict[str, NDArray[np.number]] = {}
        var_names_stored: list[str] = []

        # 1. Handle Computed Values
        if requested_name == "InvV":
            inv_K_repeated = np.repeat(self.InvK[:, np.newaxis, :], self.InvMu.shape[1], axis=1)  # noqa: N806
            self.InvV = self.InvMu * (inv_K_repeated + 0.5) ** 2
            return
        if requested_name == "P":
            self.P = ((self.MLT + 12) / 12 * np.pi) % (2 * np.pi)
            return

        if requested_name == "datetime":
            requested_name = self.saving_strategy.data_standard.get_full_var_name("Epoch")

        output_file = self.saving_strategy.get_output_file(standard_name=requested_name)
        if requested_name == "datetime" and output_file is None:
            time_key = self.saving_strategy.data_standard.get_full_var_name("Epoch")
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
            file_content = self._get_cached_datasets(full_file_path)

            if not file_content:
                continue

            time_key = self.saving_strategy.data_standard.get_full_var_name("Epoch")

            # 4. Process Datetimes
            # ruff: disable[ERA001]
            # raw_times = file_content[time_key]
            # time_unit = self.saving_strategy.data_standard.variable_infos["Epoch"].unit

            # posix_times = (raw_times * time_unit).to_value(ep.units.posixtime)
            # datetimes = np.asarray(
            #     [dt.datetime.fromtimestamp(t.astype(np.int64), tz=dt.timezone.utc) for t in posix_times]
            # )
            # ruff: enable[ERA001]

            raw_times = file_content[time_key]

            if self.saving_strategy.data_standard.variable_infos["Epoch"].unit == ep.units.posixtime:
                datetimes = np.asarray(
                    [dt.datetime.fromtimestamp(t.astype(np.int64), tz=dt.timezone.utc) for t in raw_times]
                )
            elif self.saving_strategy.data_standard.variable_infos["Epoch"].unit == ep.units.datenum:
                # Matlab logic
                datetimes = np.asarray([matlab2python(t) for t in raw_times])

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

    def _get_cached_datasets(self, file_path: Path) -> dict[StandardName, Any]:
        """Return cached parsed content for a monthly file."""
        file_path = Path(file_path)
        if file_path not in self._dataset_cache:
            if self._verbose:
                logger.info(f"Loading {file_path}")

            self._dataset_cache[file_path] = self._loaders[(ep.utils.normalize_file_format(file_path.suffix))](
                file_path
            )
        return self._dataset_cache[file_path]

    def get_loaded_variables(self) -> list[str]:
        """Get a list of currently loaded variable names."""
        return [var for var in self.possible_variables if var in self.__dict__]

    def __eq__(self, other: DataSet) -> bool:  # ty :ignore[invalid-method-override]  # noqa: D105

        if self.saving_strategy.data_standard != other.saving_strategy.data_standard:
            return False
        if (
            self._file_loading_mode != other._file_loading_mode
            or self._satellite != other._satellite
            or self._instrument != other._instrument
            or self._mfm != other._mfm
        ):
            return False

        different_vars = self.get_different_variables(other)

        return len(different_vars) == 0

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

        self_vars = self.get_loaded_variables()
        other_vars = other.get_loaded_variables()

        for var in set(self_vars + other_vars):
            if var not in other_vars or var not in self_vars:
                different_vars.append(var)
                continue

            self_var = getattr(self, var)
            other_var = getattr(other, var)

            if not isinstance(other_var, type(self_var)):
                different_vars.append(var)
                continue

            if isinstance(self_var, list):
                if len(self_var) != len(other_var) or any(a != b for a, b in zip(self_var, other_var, strict=True)):
                    different_vars.append(var)
                    continue
            elif isinstance(self_var, np.ndarray):
                if self_var.shape != other_var.shape or not np.allclose(self_var, other_var, equal_nan=True):
                    different_vars.append(var)
                    continue
            elif self_var != other_var:
                different_vars.append(var)
                continue

        return different_vars

    from .bin_and_interpolate_to_model_grid import bin_and_interpolate_to_model_grid  # noqa: PLC0415
    from .identify_orbits import identify_orbits  # noqa: PLC0415
    from .interp_functions import interp_flux, interp_psd  # noqa: PLC0415
    from .linearize_trajectories import linearize_trajectories  # noqa: PLC0415
