# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Literal

import lazy_loader as lazy

ESANGRMSatellite = Literal["EDRS-C", "S6-MF", "S6-B", "MTG-S1", "MTG-I1"]

__getattr__, _lazy_dir, _lazy_all = lazy.attach(
    __name__,
    submod_attrs={
        "process_ngrm_satellite": ["esa_ngrm_strategy", "process_ngrm_electron_fluxes"],
    },
)

__all__ = ["ESANGRMSatellite", *_lazy_all]  # noqa: PLE0604  (lazy_loader supplies the names)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from el_paso.recipes.esa.process_ngrm_satellite import esa_ngrm_strategy, process_ngrm_electron_fluxes

    __all__ = ["ESANGRMSatellite", "esa_ngrm_strategy", "process_ngrm_electron_fluxes"]
