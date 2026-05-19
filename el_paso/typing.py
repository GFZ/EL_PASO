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
    from el_paso.data_standard import ConsistencyCheck, DataStandard, VariableInfo
    from el_paso.data_standards import GFZStandard, PRBEMStandard
    from el_paso.saving_strategies.density_netcdf_strategy import DensityNetCDFStrategy
    from el_paso.saving_strategies.gfz_strategy import GFZStrategy
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
    "B_sat",
    "B_total",
    "R0",
    "InvMu",
    "InvK",
]

StandardName: TypeAlias = PRBEMName | GFZVarNames | Literal["metadata"]

MagneticFieldLiteral: TypeAlias = Literal["T89", "T01", "T01s", "TS04", "TS05", "T04s", "T96", "OP77Q", "OP77"]
MFSFormats: TypeAlias = Literal["nc", "cdf", "h5", "mat", ".nc", ".cdf", ".h5", ".mat"]
TimeInterval: TypeAlias = tuple[datetime, datetime]
SavedDataDict: TypeAlias = dict[InternalName | Literal["metadata"], Any]
FileLoader: TypeAlias = Callable[[Path], dict[StandardName, Any]]


class FileWriter(Protocol):  # noqa: D101
    def __call__(
        self,
        file_path: Path,
        data_dict: SavedDataDict,
        data_standard: DataStandard,
    ) -> None: ...


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ConsistencyCheck": ("el_paso.data_standard", "ConsistencyCheck"),
    "GFZStandard": ("el_paso.data_standards", "GFZStandard"),
    "GFZStrategy": ("el_paso.saving_strategies.gfz_strategy", "GFZStrategy"),
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
    "DataStandard",
    "DensityNetCDFStrategy",
    "FileLoader",
    "FileWriter",
    "GFZStandard",
    "GFZStrategy",
    "GFZVarNames",
    "InternalName",
    "MFSFormats",
    "MagneticFieldLiteral",
    "MonthlyFileStrategy",
    "OutputFile",
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
]
