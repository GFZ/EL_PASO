# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

import lazy_loader as lazy

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

_CONSTANTS = [
    "IRBEM_SYSAXIS_GDZ",
    "IRBEM_SYSAXIS_GEI",
    "IRBEM_SYSAXIS_GEO",
    "IRBEM_SYSAXIS_GSE",
    "IRBEM_SYSAXIS_GSM",
    "IRBEM_SYSAXIS_MAG",
    "IRBEM_SYSAXIS_SM",
    "IRBEM_SYSAXIS_SPH",
]

# Submodules and their attributes are resolved on first access (SPEC 1), so that
# `import el_paso` does not pay for astropy, scipy, swvo and friends up front.
# Set EAGER_IMPORT=1 to turn this off and surface broken imports immediately.
__getattr__, _lazy_dir, _lazy_all = lazy.attach(
    __name__,
    submodules=[
        "cli",
        "data_standards",
        "dataset",
        "physics",
        "processing",
        "recipes",
        "saving_strategies",
        "typing",
        "units",
        "utils",
    ],
    submod_attrs={
        "cli.recipe_cli": ["build_recipe_command", "run_recipe_cli"],
        "dataset": [
            "DataSet",
            "DatasetMetadata",
            "GFZDataSet",
            "GFZMetaData",
            "PRBEMDataSet",
            "PRBEMMetaData",
        ],
        "download": ["download"],
        "download_omm": ["download_omm"],
        "extract_variables_from_files": ["ExtractionInfo", "extract_variables_from_files"],
        "load_indices_solar_wind_parameters": ["load_indices_solar_wind_parameters"],
        "logger": ["setup_logging"],
        "processing": ["TimeBinMethod"],
        "release_mode": ["activate_release_mode", "get_release_msg", "is_in_release_mode"],
        "save": ["save"],
        "saving_strategy": ["SavingStrategy"],
        "variable": ["Variable"],
    },
)

__all__ = [*_CONSTANTS, *_lazy_all]  # noqa: PLE0604  (lazy_loader supplies the names)


def __dir__() -> list[str]:
    return sorted(__all__)


# Kept eager: this runs on import by design, and pulls in nothing heavy.
from el_paso.cache import cleanup_stale_cache as _cleanup_stale_cache  # noqa: E402

_cleanup_stale_cache()

__version__ = "2.1.3rc1"
