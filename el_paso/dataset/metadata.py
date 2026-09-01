# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""Metadata module for DataSet metadata management."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from astropy import units as u

from el_paso.variable import VariableMetadata

if TYPE_CHECKING:
    from el_paso.dataset.dataset import DataSet

logger = logging.getLogger(__name__)


class DatasetMetadata:
    """Attribute-accessible metadata container for a DataSet."""

    def __init__(self, dataset: DataSet | None = None) -> None:
        """Create a metadata namespace bound to a dataset."""
        self._dataset = dataset

    def _coerce_metadata(self, value: object) -> VariableMetadata | object:
        if isinstance(value, VariableMetadata):
            return value

        if isinstance(value, dict):
            metadata_dict = cast("dict[str, Any]", value)

            unit_value = metadata_dict.get("unit", u.dimensionless_unscaled)
            if isinstance(unit_value, str) and unit_value not in {"", "unknown"}:
                try:
                    unit_value = u.Unit(unit_value)
                except Exception:  # noqa: BLE001
                    unit_value = u.dimensionless_unscaled

            metadata = VariableMetadata(
                unit=unit_value if isinstance(unit_value, u.UnitBase) else u.dimensionless_unscaled,
                original_cadence_seconds=metadata_dict.get("original_cadence_seconds", 0),
                source_files=metadata_dict.get("source_files", []),
                description=metadata_dict.get("description", ""),
                processing_notes=metadata_dict.get("processing_notes", ""),
                standard_name=metadata_dict.get("standard_name", ""),
            )

            for key, attr_value in metadata_dict.items():
                if hasattr(metadata, key):
                    continue
                setattr(metadata, key, attr_value)

            return metadata

        return value

    def __getattr__(self, name: str) -> VariableMetadata | object:
        if name.startswith("_"):
            msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
            raise AttributeError(msg)

        if name in self.__dict__:
            return self.__dict__[name]

        dataset = self._dataset

        if dataset is not None:
            # Valid variable name
            if name in dataset.possible_variables:
                currently_loading = dataset.__dict__.get("_currently_loading", set())

                if name not in currently_loading:
                    dataset._load_variable(name)

                    if name in self.__dict__:
                        return self.__dict__[name]
                if name == "datetime":
                    epoch_standard = dataset.saving_strategy.data_standard.get_standard_name("Epoch")
                    epoch_meta = cast("VariableMetadata", self.__dict__.get(epoch_standard))
                    # Create a separate metadata object for `datetime` so we do not modify the original Epoch metadata
                    if epoch_meta is not None:
                        datetime_meta = VariableMetadata(
                            unit=epoch_meta.unit,
                            original_cadence_seconds=epoch_meta.original_cadence_seconds,
                            source_files=list(epoch_meta.source_files),
                            description=(
                                "Python datetime objects converted from Epoch variable. "
                                "This variable is not saved to disk but computed on the fly when requested."
                            ),
                            processing_notes=epoch_meta.processing_notes,
                            standard_name=epoch_meta.standard_name,
                        )

                        datetime_meta.add_processing_note("Computed datetime from Epoch.")

                        object.__setattr__(self, "datetime", datetime_meta)
                        return datetime_meta

                # Valid variable but no metadata found
                return {}

            # Invalid variable name
            _, levenstein_info = dataset.find_similar_variable(name)
            if levenstein_info["min_distance"] <= 2:
                msg = f"Cannot set attribute '{name}'. Maybe you meant '{levenstein_info['var_name']}'?"
            else:
                msg = f"Cannot set attribute '{name}'. It is not part of {dataset.saving_strategy.data_standard}."
        raise AttributeError(msg)

    def as_dict(self) -> dict[str, Any]:
        """Return metadata for all loaded variables as a dict."""
        return {name: value for name, value in self.__dict__.items() if not name.startswith("_")}

    def to_dict(self) -> dict[str, Any]:
        """Alias for `as_dict()` to provide a more intuitive method name for users."""
        return self.as_dict()

    def __repr__(self) -> str:
        return repr(self.to_dict())

    def __str__(self) -> str:
        return self.__repr__()

    def __setattr__(self, name: str, value: object) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        object.__setattr__(self, name, self._coerce_metadata(value))


class GFZMetaData(DatasetMetadata):
    """Metadata container for GFZStandard.

    Attribute names and descriptions are generated from `GFZStandard().variable_infos` by
    `scripts/generate_metadata_stubs.py`; `datetime` is a computed extra added by `DataSet`.

    Attributes:
        datetime (VariableMetadata): Metadata for the computed `datetime` variable.
        # BEGIN GENERATED GFZ_METADATA_ATTRS DOCS
        BB (VariableMetadata): Frequency of the power spectral density.
        B_eq (VariableMetadata): Calculated magnetic field at the equator.
        B_sat (VariableMetadata): Observered magnetic field at the satellite location.
        B_total (VariableMetadata): Calculated magnetic field at the satellite location.
        FEDO (VariableMetadata): Electron differential omnidirectional flux.
        FEIU (VariableMetadata): Electron integral unidirectional flux.
        FPDU (VariableMetadata): Proton differential unidirectional flux.
        Flux (VariableMetadata): Electron differential unidirectional flux.
        InvK (VariableMetadata): Calculated modified second adiabatic invariant.
        InvMu (VariableMetadata): Calculated first adiabatic invariant.
        Lm (VariableMetadata): Calculated Lm of the particles.
        Lstar (VariableMetadata): Calculated Lstar of the particles.
        MLT (VariableMetadata): Magnetic local time at the satellite location.
        MLT0 (VariableMetadata): Magnetic local time at the mapped magnetic equator.
        MLat (VariableMetadata): Frequency of the power spectral density.
        PSD (VariableMetadata): Calculated phase space density of particles.
        R0 (VariableMetadata): Radial distance of the satellite location mapped to the equator.
        alpha_eq_model (VariableMetadata): Calculated equatorial pitch angles of the particles.
        alpha_eq_range (VariableMetadata): Equatorial pitch angle ranges of the particles.
        alpha_lc (VariableMetadata): Local loss cone size at the satellite location.
        alpha_lc_eq (VariableMetadata): Local loss cone size at the satellite location mapped to the equator.
        alpha_local (VariableMetadata): Local pitch angles of the particles.
        alpha_local_range (VariableMetadata): Local pitch angle ranges of the particles.
        ellipticity (VariableMetadata): Frequency of the power spectral density.
        energy_FEDO (VariableMetadata): Central energy of measured omnidirecitonal flux.
        energy_FEIU (VariableMetadata): Central energy of measured integral flux.
        energy_FPDU (VariableMetadata): Central energy of measured proton differential flux.
        energy_channels (VariableMetadata): Central energy of measured differential flux.
        freq (VariableMetadata): Frequency of the power spectral density.
        freq_bw (VariableMetadata): Frequency of the power spectral density.
        geo_alt (VariableMetadata): Altitude in geographic cartesian coordinates.
        geo_lat (VariableMetadata): Latitude in geographic cartesian coordinates.
        geo_lon (VariableMetadata): Longitude in geographic cartesian coordinates.
        planarity (VariableMetadata): Frequency of the power spectral density.
        time (VariableMetadata): Time in MATLAB datenum format.
        wave_wna (VariableMetadata): Frequency of the power spectral density.
        xGEO (VariableMetadata): Position in geographic cartesian coordinates.
        # END GENERATED GFZ_METADATA_ATTRS DOCS
    """

    datetime: VariableMetadata
    # BEGIN GENERATED GFZ_METADATA_ATTRS
    BB: VariableMetadata
    B_eq: VariableMetadata
    B_sat: VariableMetadata
    B_total: VariableMetadata
    FEDO: VariableMetadata
    FEIU: VariableMetadata
    FPDU: VariableMetadata
    Flux: VariableMetadata
    InvK: VariableMetadata
    InvMu: VariableMetadata
    Lm: VariableMetadata
    Lstar: VariableMetadata
    MLT: VariableMetadata
    MLT0: VariableMetadata
    MLat: VariableMetadata
    PSD: VariableMetadata
    R0: VariableMetadata
    alpha_eq_model: VariableMetadata
    alpha_eq_range: VariableMetadata
    alpha_lc: VariableMetadata
    alpha_lc_eq: VariableMetadata
    alpha_local: VariableMetadata
    alpha_local_range: VariableMetadata
    ellipticity: VariableMetadata
    energy_FEDO: VariableMetadata  # noqa: N815
    energy_FEIU: VariableMetadata  # noqa: N815
    energy_FPDU: VariableMetadata  # noqa: N815
    energy_channels: VariableMetadata
    freq: VariableMetadata
    freq_bw: VariableMetadata
    geo_alt: VariableMetadata
    geo_lat: VariableMetadata
    geo_lon: VariableMetadata
    planarity: VariableMetadata
    time: VariableMetadata
    wave_wna: VariableMetadata
    xGEO: VariableMetadata  # noqa: N815
    # END GENERATED GFZ_METADATA_ATTRS


class PRBEMMetaData(DatasetMetadata):
    """Metadata container for PRBEMStandard.

    Attribute names and descriptions are generated from `PRBEMStandard().variable_infos` by
    `scripts/generate_metadata_stubs.py`; `datetime` is a computed extra added by `DataSet`.

    Attributes:
        datetime (VariableMetadata): Metadata for the computed `datetime` variable.
        # BEGIN GENERATED PRBEM_METADATA_ATTRS DOCS
        Alpha (VariableMetadata): Local pitch angle the instrument is looking at
        Alpha_Eq (VariableMetadata): Computed equatorial pitch angle the instrument is looking from Alpha, B_Calc
            and B_Eq
        B_Calc (VariableMetadata): Calculated magnetic field strength at the spacecraft position
        B_Eq (VariableMetadata): Calculated magnetic field strength at magnetic equator
        Energy_FEDU (VariableMetadata): Central energy of unidirectional differential electron flux
        Energy_FPDU (VariableMetadata): Central energy of unidirectional differential proton flux
        Epoch (VariableMetadata): Posix Time
        FEDU (VariableMetadata): Processed unidirectional differential electron flux
        FPDU (VariableMetadata): Processed unidirectional differential proton flux
        InvK (VariableMetadata): Calculated modified second adiabatic invariant.
        InvMu (VariableMetadata): Calculated first adiabatic invariant.
        L_m (VariableMetadata): Calculated L McIlwain's L parameter
        L_star (VariableMetadata): Calculated Roederer's L* parameter
        MLT (VariableMetadata): Magnetic local time at the satellite location.
        PSD (VariableMetadata): Calculated phase space density of particles.
        Position (VariableMetadata): Spacecraft position in geographic cartesian coordinates
        R_Eq (VariableMetadata): Radial distance of the satellite location mapped to the equator.
        # END GENERATED PRBEM_METADATA_ATTRS DOCS
    """

    datetime: VariableMetadata
    # BEGIN GENERATED PRBEM_METADATA_ATTRS
    Alpha: VariableMetadata
    Alpha_Eq: VariableMetadata
    B_Calc: VariableMetadata
    B_Eq: VariableMetadata
    Energy_FEDU: VariableMetadata
    Energy_FPDU: VariableMetadata
    Epoch: VariableMetadata
    FEDU: VariableMetadata
    FPDU: VariableMetadata
    InvK: VariableMetadata
    InvMu: VariableMetadata
    L_m: VariableMetadata
    L_star: VariableMetadata
    MLT: VariableMetadata
    PSD: VariableMetadata
    Position: VariableMetadata
    R_Eq: VariableMetadata
    # END GENERATED PRBEM_METADATA_ATTRS
