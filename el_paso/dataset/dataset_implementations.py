from __future__ import annotations

from typing import TYPE_CHECKING

from el_paso.dataset.dataset import DataSet

if TYPE_CHECKING:
    import datetime as dt

    import numpy as np
    from numpy.typing import NDArray

    from el_paso.typing import MFSFormats, SavingStrategy


class DataOrgDataSet(DataSet):
    """A concrete implementation of DataSet for the DataOrgStandard.

    Represents a dataset with variables defined by the DataOrgStandard,
    providing structured access to space physics measurements including
    particle fluxes, phase space densities, magnetic field data, and
    adiabatic invariants.

    Attributes:
        datetime (list[dt.datetime]): List of datetime objects corresponding to each time step.
        time (NDArray[np.float64]): Array of time values as floats (e.g. seconds since epoch).
        energy_channels (NDArray[np.float64]): Array of energy channel center values in keV.
        alpha_local (NDArray[np.float64]): Local pitch angle array in degrees.
        alpha_eq_model (NDArray[np.float64]): Model-derived equatorial pitch angle in degrees.
        alpha_eq_real (NDArray[np.float64]): Measured equatorial pitch angle in degrees.
        InvMu (NDArray[np.float64]): First adiabatic invariant (magnetic moment) array.
        InvMu_real (NDArray[np.float64]): Measured first adiabatic invariant array.
        InvK (NDArray[np.float64]): Second adiabatic invariant array.
        InvV (NDArray[np.float64]): Third adiabatic invariant (drift shell) array.
        Lstar (NDArray[np.float64]): Roederer L* (drift shell parameter) array.
        Flux (NDArray[np.float64]): Differential particle flux array.
        PSD (NDArray[np.float64]): Phase space density array.
        MLT (NDArray[np.float64]): Magnetic local time array in hours.
        B_SM (NDArray[np.float64]): Magnetic field vector in Solar Magnetic (SM) coordinates.
        B_total (NDArray[np.float64]): Total magnetic field magnitude array in nT.
        B_sat (NDArray[np.float64]): Magnetic field magnitude at the satellite location in nT.
        xGEO (NDArray[np.float64]): Position vector in Geocentric (GEO) coordinates.
        P (NDArray[np.float64]): Pressure array.
        R0 (NDArray[np.float64]): Equatorial crossing distance array in Earth radii.
        density (NDArray[np.float64]): Plasma density array.
    """

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
    Lm: NDArray[np.float64]
    Flux: NDArray[np.float64]
    PSD: NDArray[np.float64]
    MLT: NDArray[np.float64]
    B_eq: NDArray[np.float64]
    B_sat: NDArray[np.float64]
    xGEO: NDArray[np.float64]  # noqa: N815
    P: NDArray[np.float64]
    R0: NDArray[np.float64]
    density: NDArray[np.float64]

    def __init__(
        self,
        saving_strategy: SavingStrategy,
        start_time: dt.datetime | None = None,
        end_time: dt.datetime | None = None,
        preferred_extension: MFSFormats = "nc",
        *,
        verbose: bool = True,
        enable_dict_loading: bool = False,
    ) -> None:
        """Initializes a DataOrgDataSet instance.

        Constructs the saving strategy, invokes the parent DataSet initializer,
        and populates the list of possible variables from class annotations.

        Parameters:
            mission (str): Name of the mission (e.g. ``"RBSP"``).
            satellite (str): Satellite identifier within the mission (e.g. ``"A"``).
            instrument (str): Instrument name used to scope the data path.
            mag_field (str): Magnetic field model identifier (e.g. ``"T89"``).
            base_path (str): Root directory under which data files are stored.
            start_time (dt.datetime | None): Beginning of the time range to load.
                If ``None``, no lower bound is applied. Defaults to ``None``.
            end_time (dt.datetime | None): End of the time range to load.
                If ``None``, no upper bound is applied. Defaults to ``None``.
            preferred_extension (MFSFormats): File format to prefer when reading
                and writing data. Defaults to ``"nc"`` (NetCDF).
            saving_strategy_type (type[SavingStrategy]): Class (not instance) of
                the saving strategy used to resolve file paths. Defaults to
                ``ep.saving_strategies.MonthlyFileStrategy``.
            verbose (bool): If ``True``, print progress and diagnostic messages.
                Defaults to ``True``.
            enable_dict_loading (bool): If ``True``, allow loading data from
                dictionary-backed sources in addition to files. Defaults to
                ``False``.
        """
        self.saving_strategy = saving_strategy
        self._start_time = start_time
        self._end_time = end_time
        self._preferred_ext = preferred_extension
        self._verbose = verbose
        self._enable_dict_loading = enable_dict_loading

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

    def __repr__(self) -> str:
        """Returns an unambiguous string representation of the instance.

        Returns:
            str: A string of the form ``ClassName(param=value, ...)`` that could
            be used to reconstruct the object.
        """
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

    def __str__(self) -> str:
        """Returns a human-readable string representation of the instance.

        Delegates to :meth:`__repr__`.

        Returns:
            str: Same output as :meth:`__repr__`.
        """
        return self.__repr__()
