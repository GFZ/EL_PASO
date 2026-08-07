# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from el_paso.data_standards import GFZStandard, PRBEMStandard
from el_paso.dataset.dataset import DataSet
from el_paso.saving_strategies import GFZStrategy

if TYPE_CHECKING:
    import datetime as dt

    import numpy as np
    from numpy.typing import NDArray

    from el_paso.typing import GFZMetaData, MFSFormats, PRBEMMetaData, SavingStrategy


logger = logging.getLogger(__name__)


class GFZDataSet(DataSet):
    """A concrete implementation of DataSet for the GFZStandard.

    Represents a dataset with variables defined by the GFZStandard,
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
    B_total: NDArray[np.float64]
    xGEO: NDArray[np.float64]  # noqa: N815
    P: NDArray[np.float64]
    R0: NDArray[np.float64]
    density: NDArray[np.float64]
    metadata: GFZMetaData

    def __init__(
        self,
        saving_strategy: SavingStrategy,
        start_time: dt.datetime,
        end_time: dt.datetime,
        preferred_extension: MFSFormats = "nc",
        *,
        verbose: bool = True,
    ) -> None:
        """Initializes a GFZDataSet instance.

        Constructs the saving strategy, invokes the parent DataSet initializer,
        and populates the list of possible variables from class annotations.

        Args:
            saving_strategy (SavingStrategy): Instance of the saving strategy used to resolve file paths.
            start_time (dt.datetime): Beginning of the time range to load.
            end_time (dt.datetime): End of the time range to load.
            preferred_extension (MFSFormats): File format to prefer when reading
                and writing data. Defaults to ``"nc"`` (NetCDF).
            verbose (bool): If ``True``, print progress and diagnostic messages.
                Defaults to ``True``.
        """
        self.saving_strategy = saving_strategy
        self._start_time = start_time
        self._end_time = end_time
        self._preferred_ext = preferred_extension
        self._verbose = verbose

        if isinstance(self.saving_strategy, GFZStrategy):
            self._preferred_ext = "mat"
            logger.warning(
                "Overriding `preferred_extension` to 'mat' since `GFZStrategy` is used, which only supports .mat files."
                " Ignoring provided `preferred_extension` value."
            )

        if not isinstance(self.saving_strategy.data_standard, GFZStandard):
            msg = f"GFZDataSet requires a saving strategy with  `GFZStandard`, but got {type(self.saving_strategy.data_standard).__name__}"  # noqa: E501
            logger.error(msg)
            raise TypeError(msg)

        super().__init__(
            self.saving_strategy,
            self._start_time,
            self._end_time,
            self._preferred_ext,
            verbose=self._verbose,
        )


class PRBEMDataSet(DataSet):
    """A concrete implementation of DataSet for the PRBEMStandard.

    Represents a dataset with variables defined by the PRBEMStandard,
    providing structured access to space physics measurements including
    particle fluxes, phase space densities, magnetic field data, and
    adiabatic invariants.

    Attributes:
        datetime (list[dt.datetime]): List of datetime objects corresponding to each time step.
        Epoch (NDArray[np.float64]): Array of time values as POSIX timestamps (seconds since epoch).
        FEDU (NDArray[np.float64]): Processed unidirectional differential electron flux,
            in (cm^2 s sr keV)^-1.
        FPDU (NDArray[np.float64]): Processed unidirectional differential proton flux,
            in (cm^2 s sr keV)^-1.
        FEDO (NDArray[np.float64]): Processed omnidirectional differential electron flux.
        FEIU (NDArray[np.float64]): Processed unidirectional integral electron flux.
        Energy_FEDU (NDArray[np.float64]): Central energy of the FEDU channels, in MeV.
        Energy_FPDU (NDArray[np.float64]): Central energy of the FPDU channels, in MeV.
        Alpha (NDArray[np.float64]): Local pitch angle the instrument is looking at, in degrees.
        Alpha_Eq (NDArray[np.float64]): Computed equatorial pitch angle, in degrees.
        Position (NDArray[np.float64]): Spacecraft position in geographic cartesian coordinates, in km.
        B_Calc (NDArray[np.float64]): Calculated magnetic field strength at the spacecraft position, in nT.
        B_Eq (NDArray[np.float64]): Calculated magnetic field strength at the magnetic equator, in nT.
        L_star (NDArray[np.float64]): Calculated Roederer's L* parameter.
        I (NDArray[np.float64]): Second adiabatic invariant integral, PRBEM standard name "I".
        MLT (NDArray[np.float64]): Magnetic local time at the satellite location, in hours.
        L_m (NDArray[np.float64]): Calculated McIlwain's L parameter.
        PSD (NDArray[np.float64]): Calculated phase space density of particles.
        R_Eq (NDArray[np.float64]): Radial distance of the satellite location mapped to the equator, in Earth radii.
        InvMu (NDArray[np.float64]): Calculated first adiabatic invariant, in MeV/G.
        InvK (NDArray[np.float64]): Calculated modified second adiabatic invariant, in Earth radii * G^0.5.
        metadata (PRBEMMetaData): Metadata container for all loaded variables.
    """

    datetime: list[dt.datetime]
    Epoch: NDArray[np.float64]
    FEDU: NDArray[np.float64]
    FPDU: NDArray[np.float64]
    FEDO: NDArray[np.float64]
    FEIU: NDArray[np.float64]
    Energy_FEDU: NDArray[np.float64]
    Energy_FPDU: NDArray[np.float64]
    Alpha: NDArray[np.float64]
    Alpha_Eq: NDArray[np.float64]
    Position: NDArray[np.float64]
    B_Calc: NDArray[np.float64]
    B_Eq: NDArray[np.float64]
    L_star: NDArray[np.float64]
    I: NDArray[np.float64]  # noqa: E741
    MLT: NDArray[np.float64]
    L_m: NDArray[np.float64]
    PSD: NDArray[np.float64]
    R_Eq: NDArray[np.float64]
    InvMu: NDArray[np.float64]
    InvK: NDArray[np.float64]
    metadata: PRBEMMetaData

    def __init__(
        self,
        saving_strategy: SavingStrategy,
        start_time: dt.datetime,
        end_time: dt.datetime,
        preferred_extension: MFSFormats = "nc",
        *,
        verbose: bool = True,
    ) -> None:
        """Initializes a PRBEMDataSet instance.

        Constructs the saving strategy, invokes the parent DataSet initializer,
        and populates the list of possible variables from class annotations.

        Args:
            saving_strategy (SavingStrategy): Instance of the saving strategy used to resolve file paths.
            start_time (dt.datetime): Beginning of the time range to load.
            end_time (dt.datetime): End of the time range to load.
            preferred_extension (MFSFormats): File format to prefer when reading
                and writing data. Defaults to ``"nc"`` (NetCDF).
            verbose (bool): If ``True``, print progress and diagnostic messages.
                Defaults to ``True``.
        """
        self.saving_strategy = saving_strategy
        self._start_time = start_time
        self._end_time = end_time
        self._preferred_ext = preferred_extension
        self._verbose = verbose

        if not isinstance(self.saving_strategy.data_standard, PRBEMStandard):
            msg = f"PRBEMDataSet requires a saving strategy with  `PRBEMStandard`, but got {type(self.saving_strategy.data_standard).__name__}"  # noqa: E501
            logger.error(msg)
            raise TypeError(msg)

        super().__init__(
            self.saving_strategy,
            self._start_time,
            self._end_time,
            self._preferred_ext,
            verbose=self._verbose,
        )
