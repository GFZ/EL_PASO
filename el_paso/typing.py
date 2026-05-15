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

from collections.abc import Callable, Mapping
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

if TYPE_CHECKING:
    from el_paso.data_standard import ConsistencyCheck, DataStandard, VariableInfo
    from el_paso.data_standards import DataOrgStandard, PRBEMStandard
    from el_paso.saving_strategies.data_org_strategy import DataOrgStrategy
    from el_paso.saving_strategies.density_netcdf_strategy import DensityNetCDFStrategy
    from el_paso.saving_strategies.monthly_strategy import MonthlyFileStrategy
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

PRBEMName: TypeAlias = InternalName
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
    "B_local",
    "R0",
    "InvMu",
    "InvK",
]

StandardName: TypeAlias = PRBEMName | GFZVarNames

MagneticFieldLiteral: TypeAlias = Literal["T89", "T01", "T01s", "TS04", "TS05", "T04s", "T96", "OP77Q", "OP77"]
MFSFormats: TypeAlias = Literal["nc", "cdf", "h5", "mat", ".nc", ".cdf", ".h5", ".mat"]
DataOrgFileFormat: TypeAlias = Literal[".mat", ".pickle"]

TimeInterval: TypeAlias = tuple[datetime, datetime]
SavedDataDict: TypeAlias = dict[InternalName | Literal["metadata"], Any]
MonthlyDataDict: TypeAlias = SavedDataDict

if TYPE_CHECKING:
    DataStandardInstance: TypeAlias = DataStandard[Any]
    DataStandardClass: TypeAlias = type[DataStandard[Any]]
    SavingStrategyInstance: TypeAlias = SavingStrategy
    SavingStrategyClass: TypeAlias = type[SavingStrategy]
    VariableDict: TypeAlias = dict[InternalName, Variable]
    VariableMapping: TypeAlias = Mapping[InternalName, Variable]

SaveFileWriter: TypeAlias = Callable[[Path, SavedDataDict], None]
SaveFileLoader: TypeAlias = Callable[[Path], SavedDataDict]
MonthlyFormatWriter: TypeAlias = SaveFileWriter
MonthlyFormatLoader: TypeAlias = SaveFileLoader
SingleFileFormatWriter: TypeAlias = Callable[[Path, dict[str, Any]], None]


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ConsistencyCheck": ("el_paso.data_standard", "ConsistencyCheck"),
    "DataOrgStandard": ("el_paso.data_standards", "DataOrgStandard"),
    "DataOrgStrategy": ("el_paso.saving_strategies.data_org_strategy", "DataOrgStrategy"),
    "DataStandard": ("el_paso.data_standard", "DataStandard"),
    "DensityNetCDFStrategy": ("el_paso.saving_strategies.density_netcdf_strategy", "DensityNetCDFStrategy"),
    "MonthlyFileStrategy": ("el_paso.saving_strategies.monthly_strategy", "MonthlyFileStrategy"),
    "OutputFile": ("el_paso.saving_strategy", "OutputFile"),
    "PRBEMStandard": ("el_paso.data_standards", "PRBEMStandard"),
    "SavingStrategy": ("el_paso.saving_strategy", "SavingStrategy"),
    "SingleFileStrategy": ("el_paso.saving_strategies.single_file_strategy", "SingleFileStrategy"),
    "Variable": ("el_paso.variable", "Variable"),
    "VariableInfo": ("el_paso.data_standard", "VariableInfo"),
    "VariableMetadata": ("el_paso.variable", "VariableMetadata"),
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
    return sorted([*globals(), *_LAZY_EXPORTS])


__all__ = [
    "ConsistencyCheck",
    "DataOrgFileFormat",
    "DataOrgStandard",
    "DataOrgStrategy",
    "DataStandard",
    "DataStandardClass",
    "DataStandardInstance",
    "DensityNetCDFStrategy",
    "GFZVarNames",
    "InternalName",
    "MFSFormats",
    "MagneticFieldLiteral",
    "MonthlyDataDict",
    "MonthlyFileStrategy",
    "MonthlyFormatLoader",
    "MonthlyFormatWriter",
    "OutputFile",
    "PRBEMName",
    "PRBEMStandard",
    "SaveFileLoader",
    "SaveFileWriter",
    "SavedDataDict",
    "SavingStrategy",
    "SavingStrategyClass",
    "SavingStrategyInstance",
    "SingleFileFormatWriter",
    "SingleFileStrategy",
    "StandardName",
    "TimeInterval",
    "Variable",
    "VariableDict",
    "VariableInfo",
    "VariableMapping",
    "VariableMetadata",
]
