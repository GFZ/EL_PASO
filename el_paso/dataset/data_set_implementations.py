from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import el_paso as ep
from el_paso.dataset.dataset import DataSet

if TYPE_CHECKING:
    import datetime as dt

    import numpy as np
    from numpy.typing import NDArray

    from el_paso.typing import MFSFormats, SavingStrategy


class DataOrgDataSet(DataSet):
    datetime: list[dt.datetime]
    time: NDArray[np.float64]
    energy_channels: NDArray[np.float64]
    alpha_local: NDArray[np.float64]
    alpha_eq_model: NDArray[np.float64]
    alpha_eq_real: NDArray[np.float64]
    InvMu: NDArray[np.float64]
    InvMu_real: NDArray[np.float64]
    InvK: NDArray[np.float64]
    InvV: NDArray[np.float64]
    Lstar: NDArray[np.float64]
    Flux: NDArray[np.float64]
    PSD: NDArray[np.float64]
    MLT: NDArray[np.float64]
    B_SM: NDArray[np.float64]
    B_total: NDArray[np.float64]
    B_sat: NDArray[np.float64]
    xGEO: NDArray[np.float64]  # noqa: N815
    P: NDArray[np.float64]
    R0: NDArray[np.float64]
    density: NDArray[np.float64]

    def __init__(
        self,
        mission: str,
        satellite: str,
        instrument: str,
        mag_field: str,
        base_path: str,
        start_time: dt.datetime | None = None,
        end_time: dt.datetime | None = None,
        preferred_extension: MFSFormats = "nc",
        saving_strategy_type: type[SavingStrategy] = ep.saving_strategies.MonthlyFileStrategy,
        *,
        verbose: bool = True,
        enable_dict_loading: bool = False,
    ) -> None:

        self._mission = mission
        self._satellite = satellite
        self._instrument = instrument
        self._mag_field = mag_field
        self._base_path = base_path
        self._start_time = start_time
        self._end_time = end_time
        self._preferred_ext = preferred_extension
        self._verbose = verbose
        self._enable_dict_loading = enable_dict_loading
        self.saving_strategy = saving_strategy_type(
            self._base_path,  # ty:ignore[too-many-positional-arguments]
            self._mission,
            self._satellite,
            self._instrument,
            mag_field=self._mag_field,  # ty:ignore[unknown-argument]
            data_standard=ep.data_standards.DataOrgStandard,  # ty:ignore[unknown-argument]
            file_format=self._preferred_ext,  # ty:ignore[unknown-argument]
        )
        super().__init__(
            self.saving_strategy,
            self._start_time,
            self._end_time,
            self._preferred_ext,
            verbose=self._verbose,
            enable_dict_loading=self._enable_dict_loading,
        )

        possible_vars: list[str] = []
        for attr_name in getattr(DataOrgDataSet, "__annotations__", {}):
            if attr_name.startswith("_"):
                continue
            if attr_name not in possible_vars:
                possible_vars.append(attr_name)
        self.possible_variables = possible_vars

        self._is_nc_dataset = True

    def __repr__(self) -> str:  # noqa: D105
        saving_strategy_type_name = f"{self.saving_strategy.__module__}.{self.saving_strategy}"

        return (
            f"{self.__class__.__name__}("
            f"mission={self._mission!r}, "
            f"satellite={self._satellite!r}, "
            f"instrument={self._instrument!r}, "
            f"mag_field={self._mag_field!r}, "
            f"base_path={self._base_path!r}, "
            f"start_time={self._start_time!r}, "
            f"end_time={self._end_time!r}, "
            f"preferred_extension={self._preferred_ext!r}, "
            f"saving_strategy_type={saving_strategy_type_name}, "
            f"verbose={self._verbose!r}, "
            f"enable_dict_loading={self._enable_dict_loading!r}"
            f")"
        )

    def __str__(self) -> str:  # noqa: D105
        return self.__repr__()
