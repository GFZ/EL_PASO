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
_release_mode: bool = False
_release_msg: str = ""
exit_after_download: bool = False
skip_download: bool = False

from el_paso.release_mode import activate_release_mode, get_release_msg, is_in_release_mode
from el_paso.variable import Variable
from el_paso.saving_strategy import SavingStrategy
from el_paso import physics, processing, saving_strategies, units, data_standards, utils, typing
from el_paso.save import save
from el_paso.processing import TimeBinMethod
from el_paso.dataset import DataSet
from el_paso.download import download
from el_paso.extract_variables_from_files import extract_variables_from_files, ExtractionInfo
from el_paso.load_indices_solar_wind_parameters import load_indices_solar_wind_parameters
from el_paso.logger import setup_logging
from el_paso import dataset
from el_paso import recipes

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
    "SavingStrategy",
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
    "recipes",
    "save",
    "saving_strategies",
    "setup_logging",
    "typing",
    "units",
    "utils",
]

from el_paso.cache import cleanup_stale_cache as _cleanup_stale_cache

_cleanup_stale_cache()

__version__ = "2.1.3rc0"
