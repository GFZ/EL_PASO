# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""Combined RBM Dataset class supporting .mat, .pickle, and .nc file formats."""

from __future__ import annotations

import datetime as dt
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any, Literal, cast

import distance
import numpy as np
from dateutil.relativedelta import relativedelta
from numpy.typing import NDArray
from swvo.io.exceptions import VariableNotFoundError
from swvo.io.RBMDataSet.utils import (
    get_file_path_any_format,
    join_var,
    load_file_any_format,
    matlab2python,
    read_all_datasets_netcdf,
)
from swvo.io.utils import enforce_utc_timezone

from el_paso.saving_strategy import SavingStrategy


class DataSet:
    """RBMDataSet class supporting .mat, .pickle, and .nc file formats.

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
    preferred_extension : Literal["mat", "pickle", "nc"], optional
        Preferred file extension for file-based loading. Default is "pickle".
    verbose : bool, optional
        Whether to print verbose output. Default is True.
    enable_dict_loading : bool, optional
        Enable dictionary-based loading even in file mode. Default is False.

    Attributes
    ----------
    datetime : list[dt.datetime]
    time : NDArray[np.float64]
    energy_channels : NDArray[np.float64]
    alpha_local : NDArray[np.float64]
    alpha_eq_model : NDArray[np.float64]
    alpha_eq_real : NDArray[np.float64]
    InvMu : NDArray[np.float64]
    InvMu_real : NDArray[np.float64]
    InvK : NDArray[np.float64]
    InvV : NDArray[np.float64]
    Lstar : NDArray[np.float64]
    Flux : NDArray[np.float64]
    PSD : NDArray[np.float64]
    MLT : NDArray[np.float64]
    B_SM : NDArray[np.float64]
    B_total : NDArray[np.float64]
    B_sat : NDArray[np.float64]
    xGEO : NDArray[np.float64]
    P : NDArray[np.float64]
    R0 : NDArray[np.float64]
    density : NDArray[np.float64]

    """

    _preferred_ext: Literal["mat", "pickle", "nc"]

    # internal names
    datetime : list[dt.datetime]
    Epoch : NDArray[np.float64]
    Energy_FEDU : NDArray[np.float64]
    Alpha : NDArray[np.float64]
    Alpha_Eq : NDArray[np.float64]
    alpha_eq_real : NDArray[np.float64]
    InvMu : NDArray[np.float64]
    InvMu_real : NDArray[np.float64]
    InvK : NDArray[np.float64]
    InvV : NDArray[np.float64]
    Lstar : NDArray[np.float64]
    Flux : NDArray[np.float64]
    PSD : NDArray[np.float64]
    MLT : NDArray[np.float64]
    B_SM : NDArray[np.float64]
    B_total : NDArray[np.float64]
    B_sat : NDArray[np.float64]
    xGEO : NDArray[np.float64]
    P : NDArray[np.float64]
    R0 : NDArray[np.float64]
    density : NDArray[np.float64]

    def __init__(
        self,
        saving_strategy: SavingStrategy,
        start_time: dt.datetime | None = None,
        end_time: dt.datetime | None = None,
        preferred_extension: Literal["mat", "pickle", "nc"] = "nc",
        *,
        verbose: bool = True,
        enable_dict_loading: bool = False,
    ) -> None:

        # Store the original satellite enum for properties and other attributes
        self._verbose = verbose
        self._preferred_ext = preferred_extension

        self.saving_strategy = saving_strategy

        # For dict-based loading, modify satellite properties
        if start_time is None and end_time is None and folder_path is None:
            self._file_loading_mode = False
        else:
            # File loading mode: need all parameters
            if start_time is None or end_time is None or folder_path is None:
                msg = "For file-based loading, start_time, end_time, and folder_path must all be provided"
                raise ValueError(msg)

            start_time = enforce_utc_timezone(start_time)
            end_time = enforce_utc_timezone(end_time)

            self._start_time = start_time
            self._end_time = end_time
            self._date_list = self.saving_strategy.get_time_intervals_to_save(start_time, end_time)
            self._enable_dict_loading = enable_dict_loading
            self._netcdf_dataset_cache: dict[Path, dict[str, Any]] = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.saving_strategy.satellite}, {self.saving_strategy.instrument}, {self.saving_strategy.mag_field})"

    def __str__(self) -> str:
        return self.__repr__()

    def __dir__(self) -> list[str]:
        return list(super().__dir__()) + [var.var_name for var in VariableEnum]

    def __getattr__(self, name: str) -> NDArray[np.float64]:
        # Avoid recursion for internal attributes
        if name.startswith("_"):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

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
        sat_variable = None
        sat_variable, levenstein_info = self.find_similar_variable(name)

        if sat_variable is not None and self._file_loading_mode:
            self._load_variable(sat_variable)
            return getattr(self, name)

        if not self._file_loading_mode and name in self.possible_variables:
            raise AttributeError(
                f"Attribute '{name}' exists in `VariableLiteral` but has not been set. "
                "Call `update_from_dict()` before accessing it."
            )

        if levenstein_info["min_distance"] <= 2:
            msg = f"{self.__class__.__name__} object has no attribute {name}. Maybe you meant {levenstein_info['var_name']}?"
        else:
            msg = f"{self.__class__.__name__} object has no attribute {name}"

        raise AttributeError(msg)

    def load(self, name_or_var: str | VariableEnum) -> None:
        """Load data into memory"""

        if isinstance(name_or_var, VariableEnum):
            getattr(self, name_or_var.var_name)
        else:
            getattr(self, name_or_var)

    def find_similar_variable(self, name: str) -> tuple[None | VariableEnum, dict[str, Any]]:
        levenstein_info: dict[str, Any] = {"min_distance": 10, "var_name": ""}
        sat_variable = None
        for var in VariableEnum:
            if name == var.var_name:
                sat_variable = var
                break
            else:
                dist = distance.levenshtein(name, var.var_name)
                if name.lower() in var.var_name.lower():
                    dist = 1

                if dist < levenstein_info["min_distance"]:
                    levenstein_info["min_distance"] = dist
                    levenstein_info["var_name"] = var.var_name

        return sat_variable, levenstein_info

    def update_from_dict(
        self, source_dict: dict[VariableLiteral, NDArray[np.floating] | list[dt.datetime]]
    ) -> DataSet:
        """Get data from data dictionary and update the object.

        Parameters
        ----------
        source_dict : dict[str, VariableLiteral]
            Dictionary containing the data to be loaded into the object.

        Returns
        -------
        RBMDataSet
            The updated RBMDataSet object.

        Raises
        ------
        VariableNotFoundError
            If a key in the `source_dict` is not a valid `VariableLiteral`.
        RuntimeError
            If the RBMDataSet is in file loading mode and dictionary loading is not enabled.

        """
        if self._file_loading_mode and not self._enable_dict_loading:
            msg = "RBMDataSet is in file loading mode. Cannot update from dictionary. To use dictionary-based loading, set `enable_dict_loading=True` during initialization."
            raise RuntimeError(msg)
        for key, value in source_dict.items():
            _, levenstein_info = self.find_similar_variable(key)
            if key in self.possible_variables:
                setattr(self, key, value)
            elif levenstein_info["min_distance"] <= 2:
                msg = f"Key '{key}' is not a valid `VariableLiteral`. Maybe you meant '{levenstein_info['var_name']}'?"
                raise VariableNotFoundError(msg)
            else:
                msg = f"Key '{key}' is not a valid `VariableLiteral`."
                raise VariableNotFoundError(msg)
        return self

    def _check_if_nc_dataset(self) -> bool:
        does_processed_mat_files_folder_exist = (self._file_path_stem / "Processed_Mat_Files").exists()

        if does_processed_mat_files_folder_exist and self._preferred_ext in ["mat", "pickle"]:
            return False
        elif does_processed_mat_files_folder_exist and self._preferred_ext == "nc":
            # if any .nc files are stored in the file_path_stem, we switch to nc mode
            return next(self._file_path_stem.glob("*.nc"), None) is not None
        else:
            # if the Processed_Mat_Files folder does not exist, it is safe to assume nc mode
            return True

    def get_satellite_name(self) -> str:
        return self.saving_strategy.satellite

    def get_satellite_and_instrument_name(self) -> str:
        return self.saving_strategy.satellite + "_" + self.saving_strategy.instrument

    def get_print_name(self) -> str:
        return self.saving_strategy.satellite + " " + self.saving_strategy.instrument

    def _load_variable(self, requested_name: str) -> None:
        """Load variable from .mat, .pickle, or .nc files."""
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

        if requested_name in args(InternalName):
            internal_name = requested_name
        else:
            internal_name = self.saving_strategy.data_standard.get_internal_name(requested_name)

        output_file = self.saving_strategy.get_output_file(internal_name=internal_name)

        if output_file is None:
            msg = "This var name is not part of the chosen saving strategy!"
            raise ValueError(msg)

        # 2. Iterate through date ranges
        for time_start, time_end in self._date_list:
            # if self._folder_type != FolderTypeEnum.DataServer:
            #     raise NotImplementedError("Only DataServer folder type is currently supported.")

            full_file_path = self.saving_strategy.get_file_path(time_start, time_end, output_file)
            file_name_no_format = full_file_path.stem

            # 3. Handle File Pathing & Loading based on format
            if self._is_nc_dataset:
                # NOT IMPLEMENTED
                file_name = f"{self._file_name_stem}{date_str}_{self._mfm.mfm_name}.nc"
                full_file_path = self._file_path_stem / file_name
                file_content = self._get_cached_datasets_netcdf(full_file_path)
            else:
                # file_name_no_format = f"{self._file_name_stem}{date_str}_{var.mat_file_prefix}"
                # if var.mat_has_B:
                #     file_name_no_format += f"_n4_4_{self._mfm.mfm_name}"
                # file_name_no_format += "_ver4"

                full_file_path = get_file_path_any_format(
                    self.saving_strategy.base_data_path, file_name_no_format, self._preferred_ext, self._is_nc_dataset
                )
                if full_file_path is None:
                    print(f"File not found: {file_name_no_format}")
                    continue

                if self._verbose:
                    print(f"\tLoading {full_file_path}")
                file_content = load_file_any_format(full_file_path)

            if not file_content:
                continue

            time_key = self.saving_strategy.data_standard.get_full_var_name("Epoch")

            # 4. Process Datetimes
            raw_times = file_content[time_key]
            if self._is_nc_dataset:
                # NetCDF timestamp logic
                datetimes = np.asarray(
                    [dt.datetime.fromtimestamp(t.astype(np.int64), tz=dt.timezone.utc) for t in raw_times]
                )
            else:
                # Matlab logic
                datetimes = np.asarray([matlab2python(t) for t in raw_times])

            file_content["datetime"] = datetimes
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

                loaded_var_arrs[key] = joined_value  # type: ignore
                if key not in var_names_stored:
                    var_names_stored.append(key)

        # 6. Final Assignment to Self
        if requested_name not in var_names_stored:
            setattr(self, var.var_name, np.asarray([]))

        for var_name in var_names_stored:
            val = list(loaded_var_arrs[var_name]) if var_name == "datetime" else loaded_var_arrs[var_name]

            if self._is_nc_dataset:
                # NetCDF name mapping logic
                rbm_names = self._get_rbm_name_for_nc(var_name, self._mfm.mfm_name)  # type: ignore
                if rbm_names:
                    for name in rbm_names if isinstance(rbm_names, list) else [rbm_names]:
                        setattr(self, name, val)
            else:

                internal_name = self.saving_strategy.data_standard.get_internal_name(standard_name=var_name)

                setattr(self, var_name, val) # set standard name
                setattr(self, internal_name, val) # set standard name



    def _get_cached_datasets_netcdf(self, file_path: Path) -> dict[str, Any]:
        """Return cached parsed NetCDF content for a monthly file."""
        file_path = Path(file_path)
        if file_path not in self._netcdf_dataset_cache:
            if self._verbose:
                print(f"\tLoading {file_path}")

            self._netcdf_dataset_cache[file_path] = read_all_datasets_netcdf(file_path)
        return self._netcdf_dataset_cache[file_path]

    def get_loaded_variables(self) -> list[str]:
        """Get a list of currently loaded variable names."""
        loaded_vars = []
        for var in VariableEnum:
            if var.var_name in self.__dict__:
                loaded_vars.append(var.var_name)
        return loaded_vars

    def __eq__(self, other: RBMDataSet) -> bool:  # ty :ignore[invalid-method-override]
        if (
            self._file_loading_mode != other._file_loading_mode
            or self._satellite != other._satellite
            or self._instrument != other._instrument
            or self._mfm != other._mfm
        ):
            return False

        different_vars = self.get_different_variables(other)

        return len(different_vars) == 0

    def get_different_variables(self, rbm_other: RBMDataSet) -> list[str]:
        different_vars: list[str] = []

        self_vars = self.get_loaded_variables()
        other_vars = rbm_other.get_loaded_variables()

        for var in set(self_vars + other_vars):
            if var not in other_vars or var not in self_vars:
                different_vars.append(var)
                continue

            self_var = getattr(self, var)
            other_var = getattr(rbm_other, var)

            if not isinstance(other_var, type(self_var)):
                different_vars.append(var)
                continue

            if isinstance(self_var, list):
                if len(self_var) != len(other_var) or any(a != b for a, b in zip(self_var, other_var)):
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

    from .bin_and_interpolate_to_model_grid import bin_and_interpolate_to_model_grid  # noqa: I001
    from .identify_orbits import identify_orbits
    from .interp_functions import interp_flux, interp_psd
    from .linearize_trajectories import linearize_trajectories
