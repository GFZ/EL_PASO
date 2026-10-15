# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Literal

import lazy_loader as lazy

DMSPSatellite = Literal["f17"]

__getattr__, _lazy_dir, _lazy_all = lazy.attach(
    __name__,
    submod_attrs={
        "process_dmsp_ssj_electrons": ["dmsp_ssj_electron_strategy", "process_dmsp_ssj_electrons"],
    },
)

__all__ = ["DMSPSatellite", *_lazy_all]  # noqa: PLE0604  (lazy_loader supplies the names)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from el_paso.recipes.dmsp.process_dmsp_ssj_electrons import (
        dmsp_ssj_electron_strategy,
        process_dmsp_ssj_electrons,
    )

    __all__ = ["DMSPSatellite", "dmsp_ssj_electron_strategy", "process_dmsp_ssj_electrons"]
