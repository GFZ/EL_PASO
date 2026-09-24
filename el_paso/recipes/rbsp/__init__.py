# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Literal

import lazy_loader as lazy

RBSPSatellite = Literal["a", "b"]

__getattr__, _lazy_dir, _lazy_all = lazy.attach(
    __name__,
    submod_attrs={
        "process_rbsp_ect_combined": [
            "process_rbsp_ect_combined",
            "rbsp_ect_combined_gfz_strategy",
            "rbsp_ect_combined_netcdf_strategy",
        ],
        "process_rbsp_efw_emfisis_density_combined": ["process_rbsp_efw_emfisis_density_combined"],
        "process_rbsp_emfisis_waves": ["process_rbsp_emfisis_waves", "rbsp_emfisis_waves_strategy"],
        "process_rbsp_hope_electrons": [
            "process_rbsp_hope_electrons",
            "rbsp_hope_electron_gfz_strategy",
            "rbsp_hope_electron_netcdf_strategy",
        ],
        "process_rbsp_hope_protons": [
            "process_rbsp_hope_protons",
            "rbsp_hope_proton_gfz_strategy",
            "rbsp_hope_proton_netcdf_strategy",
        ],
        "process_rbsp_mageis_electrons": ["process_rbsp_mageis_electrons", "rbsp_mageis_electron_strategy"],
        "process_rbsp_mageis_protons": [
            "process_rbsp_mageis_protons",
            "rbsp_mageis_proton_gfz_strategy",
            "rbsp_mageis_proton_netcdf_strategy",
        ],
        "process_rbsp_rbspice_protons": [
            "process_rbsp_rbspice_protons",
            "rbsp_rbspice_proton_gfz_strategy",
            "rbsp_rbspice_proton_netcdf_strategy",
        ],
    },
)

__all__ = ["RBSPSatellite", *_lazy_all]  # noqa: PLE0604  (lazy_loader supplies the names)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from el_paso.recipes.rbsp.process_rbsp_ect_combined import (
        process_rbsp_ect_combined,
        rbsp_ect_combined_gfz_strategy,
        rbsp_ect_combined_netcdf_strategy,
    )
    from el_paso.recipes.rbsp.process_rbsp_efw_emfisis_density_combined import (
        process_rbsp_efw_emfisis_density_combined,
    )
    from el_paso.recipes.rbsp.process_rbsp_emfisis_waves import process_rbsp_emfisis_waves, rbsp_emfisis_waves_strategy
    from el_paso.recipes.rbsp.process_rbsp_hope_electrons import (
        process_rbsp_hope_electrons,
        rbsp_hope_electron_gfz_strategy,
        rbsp_hope_electron_netcdf_strategy,
    )
    from el_paso.recipes.rbsp.process_rbsp_hope_protons import (
        process_rbsp_hope_protons,
        rbsp_hope_proton_gfz_strategy,
        rbsp_hope_proton_netcdf_strategy,
    )
    from el_paso.recipes.rbsp.process_rbsp_mageis_electrons import (
        process_rbsp_mageis_electrons,
        rbsp_mageis_electron_strategy,
    )
    from el_paso.recipes.rbsp.process_rbsp_mageis_protons import (
        process_rbsp_mageis_protons,
        rbsp_mageis_proton_gfz_strategy,
        rbsp_mageis_proton_netcdf_strategy,
    )
    from el_paso.recipes.rbsp.process_rbsp_rbspice_protons import (
        process_rbsp_rbspice_protons,
        rbsp_rbspice_proton_gfz_strategy,
        rbsp_rbspice_proton_netcdf_strategy,
    )

    __all__ = [
        "RBSPSatellite",
        "process_rbsp_ect_combined",
        "process_rbsp_efw_emfisis_density_combined",
        "process_rbsp_emfisis_waves",
        "process_rbsp_hope_electrons",
        "process_rbsp_hope_protons",
        "process_rbsp_mageis_electrons",
        "process_rbsp_mageis_protons",
        "process_rbsp_rbspice_protons",
        "rbsp_ect_combined_gfz_strategy",
        "rbsp_ect_combined_netcdf_strategy",
        "rbsp_emfisis_waves_strategy",
        "rbsp_hope_electron_gfz_strategy",
        "rbsp_hope_electron_netcdf_strategy",
        "rbsp_hope_proton_gfz_strategy",
        "rbsp_hope_proton_netcdf_strategy",
        "rbsp_mageis_electron_strategy",
        "rbsp_mageis_proton_gfz_strategy",
        "rbsp_mageis_proton_netcdf_strategy",
        "rbsp_rbspice_proton_gfz_strategy",
        "rbsp_rbspice_proton_netcdf_strategy",
    ]
