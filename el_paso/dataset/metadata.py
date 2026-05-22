# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""Combined RBM Dataset class supporting .mat, .pickle, and .nc file formats."""

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

    def __repr__(self) -> str:
        return repr(self.as_dict())

    def __setattr__(self, name: str, value: object) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        object.__setattr__(self, name, self._coerce_metadata(value))


class GFZMetaData(DatasetMetadata):  # noqa: D101
    datetime: VariableMetadata
    time: VariableMetadata
    energy_channels: VariableMetadata
    alpha_local: VariableMetadata
    alpha_eq_model: VariableMetadata
    alpha_eq_real: VariableMetadata
    InvMu: VariableMetadata
    InvMu_real: VariableMetadata
    InvK: VariableMetadata
    InvV: VariableMetadata
    Lstar: VariableMetadata
    Lm: VariableMetadata
    Flux: VariableMetadata
    PSD: VariableMetadata
    MLT: VariableMetadata
    B_eq: VariableMetadata
    B_total: VariableMetadata
    xGEO: VariableMetadata  # noqa: N815
    P: VariableMetadata
    R0: VariableMetadata
    density: VariableMetadata


class PRBEMMetaData(DatasetMetadata):  # noqa: D101
    datetime: VariableMetadata
    Epoch: VariableMetadata
    FEDU: VariableMetadata
    FEDO: VariableMetadata
    FEIU: VariableMetadata
    Energy_FEDU: VariableMetadata
    Alpha: VariableMetadata
    Alpha_Eq: VariableMetadata
    Position: VariableMetadata
    B_Calc: VariableMetadata
    B_Eq: VariableMetadata
    L_star: VariableMetadata
    I: VariableMetadata  # noqa: E741
    MLT: VariableMetadata
    L_m: VariableMetadata
    PSD: VariableMetadata
    R_Eq: VariableMetadata
    InvMu: VariableMetadata
    InvK: VariableMetadata
