# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""Import-safe typing helpers for data standards and saving strategies.

The aliases in this module are safe to import from inside EL-PASO
modules. Concrete EL-PASO classes are exposed lazily so user code can also import
them from here for annotations without creating import cycles during package
initialization.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias

if TYPE_CHECKING:
    from el_paso import ExtractionInfo
    from el_paso.data_standard import ConsistencyCheck, DataStandard, VariableInfo
    from el_paso.data_standards import GFZStandard, PRBEMStandard
    from el_paso.dataset.metadata import GFZMetaData, PRBEMMetaData
    from el_paso.processing.compute_magnetic_field_variables import VariableRequest
    from el_paso.saving_strategies.density_netcdf_strategy import DensityNetCDFStrategy
    from el_paso.saving_strategies.gfz_strategy import GFZStrategy
    from el_paso.saving_strategies.monthly_rb_strategy import MonthlyRBStrategy
    from el_paso.saving_strategies.single_file_strategy import SingleFileStrategy
    from el_paso.saving_strategy import OutputFile, SavingStrategy
    from el_paso.variable import Variable, VariableMetadata


InternalName: TypeAlias = Literal[
    "FEDU",
    "FEDO",
    "FEIU",
    "Energy_FEDU",
    "Epoch",
    "Alpha",
    "Alpha_Eq",
    "Position",
    "B_Calc",
    "B_Eq",
    "L_star",
    "I",
    "MLT",
    "L_m",
    "PSD",
    "R_Eq",
    "InvMu",
    "InvK",
]
"""Internal EL-PASO variable names used as canonical keys across standards."""

PRBEMName: TypeAlias = InternalName
"""PRBEM-standard variable names, which match EL-PASO internal names."""

GFZVarNames: TypeAlias = Literal[
    "time",
    "xGEO",
    "energy_channels",
    "Flux",
    "alpha_local",
    "alpha_eq_model",
    "PSD",
    "MLT",
    "Lstar",
    "Lm",
    "B_eq",
    "B_sat",
    "B_total",
    "R0",
    "InvMu",
    "InvK",
]
"""Variable names used by the GFZ output standard."""

StandardName: TypeAlias = PRBEMName | GFZVarNames | Literal["metadata"]
"""Any standard-facing variable name accepted by EL-PASO data standards."""

MagneticFieldLiteral: TypeAlias = Literal[
    "T89",
    "T01",
    "T01s",
    "TS04",
    "TS05",
    "T04s",
    "T96",
    "OP77Q",
    "OP77",
]
"""Supported magnetic-field model identifiers."""

MagInputKeys: TypeAlias = Literal[
    "Kp", "Dst", "dens", "velo", "Pdyn", "ByIMF", "BzIMF", "G1", "G2", "G3", "W1", "W2", "W3", "W4", "W5", "W6", "AL"
]
"""Supported magnetic-field model input parameter names."""

MagFieldVarTypes: TypeAlias = Literal[
    "B_local",
    "B_fofl",
    "B_eq",
    "B_mirr",
    "xGEO_eq",
    "MLT",
    "R_eq",
    "MLT_eq",
    "Lstar",
    "Lm",
    "PA_eq",
    "invMu",
    "invK",
    "XJ",
]
"""Magnetic-field variables that can be requested from the field-line service."""

MFSFormats: TypeAlias = Literal["nc", "cdf", "h5", "mat", ".nc", ".cdf", ".h5", ".mat"]
"""File formats supported by MonthlyRBStrategy."""

TimeInterval: TypeAlias = tuple[datetime, datetime]
"""Inclusive start and end datetimes for a processing or saving interval."""

SavedDataDict: TypeAlias = dict[InternalName | Literal["metadata"], Any]
"""Dictionary passed to saving backends, keyed by internal variable name or metadata."""

FileLoader: TypeAlias = Callable[[Path], dict[StandardName, Any]]
"""Callable that loads a data file into a dictionary keyed by standard variable names."""


class FileWriter(Protocol):
    """Callable interface for writing standardized EL-PASO data to disk."""

    def __call__(
        self,
        file_path: Path,
        data_dict: SavedDataDict,
        data_standard: DataStandard,
    ) -> None:
        """Write `data_dict` to `file_path` using `data_standard`."""
        ...


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ConsistencyCheck": ("el_paso.data_standard", "ConsistencyCheck"),
    "GFZStandard": ("el_paso.data_standards.gfz_standard", "GFZStandard"),
    "GFZStrategy": ("el_paso.saving_strategies.gfz_strategy", "GFZStrategy"),
    "DataStandard": ("el_paso.data_standard", "DataStandard"),
    "DensityNetCDFStrategy": ("el_paso.saving_strategies.density_netcdf_strategy", "DensityNetCDFStrategy"),
    "MonthlyRBStrategy": ("el_paso.saving_strategies.monthly_rb_strategy", "MonthlyRBStrategy"),
    "OutputFile": ("el_paso.saving_strategy", "OutputFile"),
    "PRBEMStandard": ("el_paso.data_standards.prbem_standard", "PRBEMStandard"),
    "VariableRequest": ("el_paso.processing.compute_magnetic_field_variables", "VariableRequest"),
    "SavingStrategy": ("el_paso.saving_strategy", "SavingStrategy"),
    "SingleFileStrategy": ("el_paso.saving_strategies.single_file_strategy", "SingleFileStrategy"),
    "Variable": ("el_paso.variable", "Variable"),
    "VariableInfo": ("el_paso.data_standard", "VariableInfo"),
    "VariableMetadata": ("el_paso.variable", "VariableMetadata"),
    "GFZMetaData": ("el_paso.dataset.metadata", "GFZMetaData"),
    "PRBEMMetaData": ("el_paso.dataset.metadata", "PRBEMMetaData"),
    "ExtractionInfo": ("el_paso", "ExtractionInfo"),
}


def __getattr__(name: str) -> object:
    """Lazily resolve concrete EL-PASO classes exported for user annotations."""
    if name not in _LAZY_EXPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    module_name, attribute_name = _LAZY_EXPORTS[name]
    attribute = getattr(import_module(module_name), attribute_name)
    globals()[name] = attribute
    return attribute


def __dir__() -> list[str]:
    """Return public names exposed by this import-safe typing module."""
    return __all__ + [k for k in globals() if k.startswith("__") and k.endswith("__")]


__all__ = [
    "ConsistencyCheck",
    "DataStandard",
    "DensityNetCDFStrategy",
    "ExtractionInfo",
    "FileLoader",
    "FileWriter",
    "GFZMetaData",
    "GFZStandard",
    "GFZStrategy",
    "GFZVarNames",
    "InternalName",
    "MFSFormats",
    "MagFieldVarTypes",
    "MagInputKeys",
    "MagneticFieldLiteral",
    "MonthlyRBStrategy",
    "OutputFile",
    "PRBEMMetaData",
    "PRBEMName",
    "PRBEMStandard",
    "SavedDataDict",
    "SavingStrategy",
    "SingleFileStrategy",
    "StandardName",
    "TimeInterval",
    "Variable",
    "VariableInfo",
    "VariableMetadata",
    "VariableRequest",
]
