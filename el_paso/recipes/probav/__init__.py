# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, _lazy_dir, _lazy_all = lazy.attach(
    __name__,
    submod_attrs={
        "process_ept_electron_fluxes": [
            "probav_ept_electron_gfz_strategy",
            "probav_ept_electron_netcdf_strategy",
            "process_ept_electron_fluxes",
        ],
        "process_ept_proton_fluxes": [
            "probav_ept_proton_gfz_strategy",
            "probav_ept_proton_netcdf_strategy",
            "process_ept_proton_fluxes",
        ],
    },
)

__all__ = [*_lazy_all]  # noqa: PLE0604  (lazy_loader supplies the names)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from el_paso.recipes.probav.process_ept_electron_fluxes import (
        probav_ept_electron_gfz_strategy,
        probav_ept_electron_netcdf_strategy,
        process_ept_electron_fluxes,
    )
    from el_paso.recipes.probav.process_ept_proton_fluxes import (
        probav_ept_proton_gfz_strategy,
        probav_ept_proton_netcdf_strategy,
        process_ept_proton_fluxes,
    )

    __all__ = [
        "probav_ept_electron_gfz_strategy",
        "probav_ept_electron_netcdf_strategy",
        "probav_ept_proton_gfz_strategy",
        "probav_ept_proton_netcdf_strategy",
        "process_ept_electron_fluxes",
        "process_ept_proton_fluxes",
    ]
