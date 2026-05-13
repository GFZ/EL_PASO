# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402, I001

# useful custom IRBEM aliases
IRBEM_SYSAXIS_GDZ = 0
IRBEM_SYSAXIS_GEO = 1
IRBEM_SYSAXIS_GSM = 2
IRBEM_SYSAXIS_GSE = 3
IRBEM_SYSAXIS_SM = 4
IRBEM_SYSAXIS_GEI = 5
IRBEM_SYSAXIS_MAG = 6
IRBEM_SYSAXIS_SPH = 7  # (geo in spherical)

# package wide variables
_release_mode = False
_release_msg: str = ""

from el_paso.release_mode import activate_release_mode, get_release_msg, is_in_release_mode
from el_paso.variable import Variable
from el_paso import physics, processing, saving_strategies, units, data_standards
from el_paso.save import save
from el_paso.processing import TimeBinMethod
from el_paso.download import download
from el_paso.extract_variables_from_files import extract_variables_from_files, ExtractionInfo
from el_paso.load_indices_solar_wind_parameters import load_indices_solar_wind_parameters
from el_paso.data_standard import InternalName


__all__ = [
    # Public constants
    "IRBEM_SYSAXIS_GDZ",
    "IRBEM_SYSAXIS_GEI",
    "IRBEM_SYSAXIS_GEO",
    "IRBEM_SYSAXIS_GSE",
    "IRBEM_SYSAXIS_GSM",
    "IRBEM_SYSAXIS_MAG",
    "IRBEM_SYSAXIS_SM",
    "ExtractionInfo",
    "InternalName",
    "TimeBinMethod",
    "Variable",
    "activate_release_mode",
    "data_standards",
    "download",
    "extract_variables_from_files",
    "get_release_msg",
    "is_in_release_mode",
    "load_indices_solar_wind_parameters",
    "physics",
    "processing",
    "save",
    "saving_strategies",
    "units",
]

__version__ = "1.0.3rc0"
