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

    Attribute names and descriptions (other than `datetime`, `P`, `InvV`, which are computed by
    `DataSet`) are generated from `GFZStandard().variable_infos` by
    `scripts/generate_metadata_stubs.py`.

    Attributes:
        datetime (list[dt.datetime]): List of datetime objects corresponding to each time step.
        P (NDArray[np.float64]): Computed phase angle, derived from MLT.
        InvV (NDArray[np.float64]): Computed third adiabatic invariant, derived from InvK and InvMu.
        # BEGIN GENERATED GFZ_DATASET_ATTRS DOCS
        BB (NDArray[np.float64]): Frequency of the power spectral density.
        B_eq (NDArray[np.float64]): Calculated magnetic field at the equator.
        B_sat (NDArray[np.float64]): Observered magnetic field at the satellite location.
        B_total (NDArray[np.float64]): Calculated magnetic field at the satellite location.
        FEDO (NDArray[np.float64]): Electron differential omnidirectional flux.
        FEIU (NDArray[np.float64]): Electron integral unidirectional flux.
        FPDU (NDArray[np.float64]): Proton differential unidirectional flux.
        Flux (NDArray[np.float64]): Electron differential unidirectional flux.
        InvK (NDArray[np.float64]): Calculated modified second adiabatic invariant.
        InvMu (NDArray[np.float64]): Calculated first adiabatic invariant.
        Lm (NDArray[np.float64]): Calculated Lm of the particles.
        Lstar (NDArray[np.float64]): Calculated Lstar of the particles.
        MLT (NDArray[np.float64]): Magnetic local time at the satellite location.
        MLT0 (NDArray[np.float64]): Magnetic local time at the mapped magnetic equator.
        MLat (NDArray[np.float64]): Frequency of the power spectral density.
        PSD (NDArray[np.float64]): Calculated phase space density of particles.
        R0 (NDArray[np.float64]): Radial distance of the satellite location mapped to the equator.
        alpha_eq_model (NDArray[np.float64]): Calculated equatorial pitch angles of the particles.
        alpha_eq_range (NDArray[np.float64]): Equatorial pitch angle ranges of the particles.
        alpha_lc (NDArray[np.float64]): Local loss cone size at the satellite location.
        alpha_lc_eq (NDArray[np.float64]): Local loss cone size at the satellite location mapped to the equator.
        alpha_local (NDArray[np.float64]): Local pitch angles of the particles.
        alpha_local_range (NDArray[np.float64]): Local pitch angle ranges of the particles.
        ellipticity (NDArray[np.float64]): Frequency of the power spectral density.
        energy_FEDO (NDArray[np.float64]): Central energy of measured omnidirecitonal flux.
        energy_FEIU (NDArray[np.float64]): Central energy of measured integral flux.
        energy_FPDU (NDArray[np.float64]): Central energy of measured proton differential flux.
        energy_channels (NDArray[np.float64]): Central energy of measured differential flux.
        freq (NDArray[np.float64]): Frequency of the power spectral density.
        freq_bw (NDArray[np.float64]): Frequency of the power spectral density.
        geo_alt (NDArray[np.float64]): Altitude in geographic cartesian coordinates.
        geo_lat (NDArray[np.float64]): Latitude in geographic cartesian coordinates.
        geo_lon (NDArray[np.float64]): Longitude in geographic cartesian coordinates.
        planarity (NDArray[np.float64]): Frequency of the power spectral density.
        time (NDArray[np.float64]): Time in MATLAB datenum format.
        wave_wna (NDArray[np.float64]): Frequency of the power spectral density.
        xGEO (NDArray[np.float64]): Position in geographic cartesian coordinates.
        # END GENERATED GFZ_DATASET_ATTRS DOCS
    """

    datetime: list[dt.datetime]
    P: NDArray[np.float64]
    InvV: NDArray[np.float64]
    # BEGIN GENERATED GFZ_DATASET_ATTRS
    BB: NDArray[np.float64]
    B_eq: NDArray[np.float64]
    B_sat: NDArray[np.float64]
    B_total: NDArray[np.float64]
    FEDO: NDArray[np.float64]
    FEIU: NDArray[np.float64]
    FPDU: NDArray[np.float64]
    Flux: NDArray[np.float64]
    InvK: NDArray[np.float64]
    InvMu: NDArray[np.float64]
    Lm: NDArray[np.float64]
    Lstar: NDArray[np.float64]
    MLT: NDArray[np.float64]
    MLT0: NDArray[np.float64]
    MLat: NDArray[np.float64]
    PSD: NDArray[np.float64]
    R0: NDArray[np.float64]
    alpha_eq_model: NDArray[np.float64]
    alpha_eq_range: NDArray[np.float64]
    alpha_lc: NDArray[np.float64]
    alpha_lc_eq: NDArray[np.float64]
    alpha_local: NDArray[np.float64]
    alpha_local_range: NDArray[np.float64]
    ellipticity: NDArray[np.float64]
    energy_FEDO: NDArray[np.float64]  # noqa: N815
    energy_FEIU: NDArray[np.float64]  # noqa: N815
    energy_FPDU: NDArray[np.float64]  # noqa: N815
    energy_channels: NDArray[np.float64]
    freq: NDArray[np.float64]
    freq_bw: NDArray[np.float64]
    geo_alt: NDArray[np.float64]
    geo_lat: NDArray[np.float64]
    geo_lon: NDArray[np.float64]
    planarity: NDArray[np.float64]
    time: NDArray[np.float64]
    wave_wna: NDArray[np.float64]
    xGEO: NDArray[np.float64]  # noqa: N815
    # END GENERATED GFZ_DATASET_ATTRS
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

    Attribute names and descriptions (other than `datetime`) are generated from
    `PRBEMStandard().variable_infos` by `scripts/generate_metadata_stubs.py`.

    Attributes:
        datetime (list[dt.datetime]): List of datetime objects corresponding to each time step.
        # BEGIN GENERATED PRBEM_DATASET_ATTRS DOCS
        Alpha (NDArray[np.float64]): Local pitch angle the instrument is looking at
        Alpha_Eq (NDArray[np.float64]): Computed equatorial pitch angle the instrument is looking from Alpha, B_Calc
            and B_Eq
        B_Calc (NDArray[np.float64]): Calculated magnetic field strength at the spacecraft position
        B_Eq (NDArray[np.float64]): Calculated magnetic field strength at magnetic equator
        Energy_FEDU (NDArray[np.float64]): Central energy of unidirectional differential electron flux
        Energy_FPDU (NDArray[np.float64]): Central energy of unidirectional differential proton flux
        Epoch (NDArray[np.float64]): Posix Time
        FEDU (NDArray[np.float64]): Processed unidirectional differential electron flux
        FPDU (NDArray[np.float64]): Processed unidirectional differential proton flux
        InvK (NDArray[np.float64]): Calculated modified second adiabatic invariant.
        InvMu (NDArray[np.float64]): Calculated first adiabatic invariant.
        L_m (NDArray[np.float64]): Calculated L McIlwain's L parameter
        L_star (NDArray[np.float64]): Calculated Roederer's L* parameter
        MLT (NDArray[np.float64]): Magnetic local time at the satellite location.
        PSD (NDArray[np.float64]): Calculated phase space density of particles.
        Position (NDArray[np.float64]): Spacecraft position in geographic cartesian coordinates
        R_Eq (NDArray[np.float64]): Radial distance of the satellite location mapped to the equator.
        # END GENERATED PRBEM_DATASET_ATTRS DOCS
        metadata (PRBEMMetaData): Metadata container for all loaded variables.
    """

    datetime: list[dt.datetime]
    # BEGIN GENERATED PRBEM_DATASET_ATTRS
    Alpha: NDArray[np.float64]
    Alpha_Eq: NDArray[np.float64]
    B_Calc: NDArray[np.float64]
    B_Eq: NDArray[np.float64]
    Energy_FEDU: NDArray[np.float64]
    Energy_FPDU: NDArray[np.float64]
    Epoch: NDArray[np.float64]
    FEDU: NDArray[np.float64]
    FPDU: NDArray[np.float64]
    InvK: NDArray[np.float64]
    InvMu: NDArray[np.float64]
    L_m: NDArray[np.float64]
    L_star: NDArray[np.float64]
    MLT: NDArray[np.float64]
    PSD: NDArray[np.float64]
    Position: NDArray[np.float64]
    R_Eq: NDArray[np.float64]
    # END GENERATED PRBEM_DATASET_ATTRS
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
