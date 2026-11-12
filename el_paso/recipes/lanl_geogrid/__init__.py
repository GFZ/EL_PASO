# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Literal

import lazy_loader as lazy

# Longitude slots (degrees east) that have LowRes SOPA_ESP files in the NOAA archive for 2008 to 2018.
LANLGeoGridSatellite = Literal[
    "000",
    "010",
    "020",
    "030",
    "040",
    "050",
    "060",
    "070",
    "080",
    "090",
    "100",
    "110",
    "120",
    "130",
    "140",
    "150",
    "190",
    "200",
    "210",
    "220",
    "230",
    "240",
    "250",
    "260",
    "270",
    "300",
    "310",
    "320",
]

__getattr__, _lazy_dir, _lazy_all = lazy.attach(
    __name__,
    submod_attrs={
        "process_lanl_geogrid_mpa_electrons": [
            "lanl_geogrid_mpa_electron_strategy",
            "process_lanl_geogrid_mpa_electrons",
        ],
        "process_lanl_geogrid_mpa_ions": [
            "lanl_geogrid_mpa_ion_strategy",
            "process_lanl_geogrid_mpa_ions",
        ],
        "process_lanl_geogrid_sopa_esp_electrons": [
            "lanl_geogrid_sopa_esp_electron_strategy",
            "process_lanl_geogrid_sopa_esp_electrons",
        ],
        "process_lanl_geogrid_sopa_protons": [
            "lanl_geogrid_sopa_proton_strategy",
            "process_lanl_geogrid_sopa_protons",
        ],
    },
)

__all__ = ["LANLGeoGridSatellite", *_lazy_all]  # noqa: PLE0604  (lazy_loader supplies the names)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from el_paso.recipes.lanl_geogrid.process_lanl_geogrid_mpa_electrons import (
        lanl_geogrid_mpa_electron_strategy,
        process_lanl_geogrid_mpa_electrons,
    )
    from el_paso.recipes.lanl_geogrid.process_lanl_geogrid_mpa_ions import (
        lanl_geogrid_mpa_ion_strategy,
        process_lanl_geogrid_mpa_ions,
    )
    from el_paso.recipes.lanl_geogrid.process_lanl_geogrid_sopa_esp_electrons import (
        lanl_geogrid_sopa_esp_electron_strategy,
        process_lanl_geogrid_sopa_esp_electrons,
    )
    from el_paso.recipes.lanl_geogrid.process_lanl_geogrid_sopa_protons import (
        lanl_geogrid_sopa_proton_strategy,
        process_lanl_geogrid_sopa_protons,
    )

    __all__ = [
        "LANLGeoGridSatellite",
        "lanl_geogrid_mpa_electron_strategy",
        "lanl_geogrid_mpa_ion_strategy",
        "lanl_geogrid_sopa_esp_electron_strategy",
        "lanl_geogrid_sopa_proton_strategy",
        "process_lanl_geogrid_mpa_electrons",
        "process_lanl_geogrid_mpa_ions",
        "process_lanl_geogrid_sopa_esp_electrons",
        "process_lanl_geogrid_sopa_protons",
    ]
