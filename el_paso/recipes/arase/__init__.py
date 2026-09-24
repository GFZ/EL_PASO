# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, _lazy_dir, _lazy_all = lazy.attach(
    __name__,
    submod_attrs={
        "get_arase_orbit_variables": ["get_arase_orbit_level_2_variables", "get_arase_orbit_level_3_variables"],
        "process_arase_mepe": [
            "arase_mepe_gfz_strategy",
            "arase_mepe_h5_strategy",
            "arase_mepe_netcdf_strategy",
            "process_arase_mepe",
        ],
        "process_arase_pwe_densities": ["arase_pwe_densities_strategy", "process_arase_pwe_density"],
        "process_arase_xep": ["arase_xep_gfz_strategy", "arase_xep_strategy", "process_arase_xep"],
        "process_arase_xep_realtime": [
            "arase_xep_realtime_gfz_strategy",
            "arase_xep_realtime_strategy",
            "process_arase_xep_real_time",
        ],
    },
)

__all__ = [*_lazy_all]  # noqa: PLE0604  (lazy_loader supplies the names)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from el_paso.recipes.arase.get_arase_orbit_variables import (
        get_arase_orbit_level_2_variables,
        get_arase_orbit_level_3_variables,
    )
    from el_paso.recipes.arase.process_arase_mepe import (
        arase_mepe_gfz_strategy,
        arase_mepe_h5_strategy,
        arase_mepe_netcdf_strategy,
        process_arase_mepe,
    )
    from el_paso.recipes.arase.process_arase_pwe_densities import (
        arase_pwe_densities_strategy,
        process_arase_pwe_density,
    )
    from el_paso.recipes.arase.process_arase_xep import arase_xep_gfz_strategy, arase_xep_strategy, process_arase_xep
    from el_paso.recipes.arase.process_arase_xep_realtime import (
        arase_xep_realtime_gfz_strategy,
        arase_xep_realtime_strategy,
        process_arase_xep_real_time,
    )

    __all__ = [
        "arase_mepe_gfz_strategy",
        "arase_mepe_h5_strategy",
        "arase_mepe_netcdf_strategy",
        "arase_pwe_densities_strategy",
        "arase_xep_gfz_strategy",
        "arase_xep_realtime_gfz_strategy",
        "arase_xep_realtime_strategy",
        "arase_xep_strategy",
        "get_arase_orbit_level_2_variables",
        "get_arase_orbit_level_3_variables",
        "process_arase_mepe",
        "process_arase_pwe_density",
        "process_arase_xep",
        "process_arase_xep_real_time",
    ]
