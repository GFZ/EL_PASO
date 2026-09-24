# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Literal

import lazy_loader as lazy

POESSatellite = Literal[
    "metop1",
    "metop2",
    "metop3",
    "noaa05",
    "noaa06",
    "noaa07",
    "noaa08",
    "noaa10",
    "noaa12",
    "noaa14",
    "noaa15",
    "noaa16",
    "noaa17",
    "noaa18",
    "noaa19",
]

__getattr__, _lazy_dir, _lazy_all = lazy.attach(
    __name__,
    submod_attrs={
        "process_poes_meped": ["poes_meped_strategy", "process_poes_meped_electron"],
        "process_poes_ted": ["poes_ted_strategy", "process_poes_ted_electron"],
    },
)

__all__ = ["POESSatellite", *_lazy_all]  # noqa: PLE0604  (lazy_loader supplies the names)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from el_paso.recipes.poes.process_poes_meped import poes_meped_strategy, process_poes_meped_electron
    from el_paso.recipes.poes.process_poes_ted import poes_ted_strategy, process_poes_ted_electron

    __all__ = [
        "POESSatellite",
        "poes_meped_strategy",
        "poes_ted_strategy",
        "process_poes_meped_electron",
        "process_poes_ted_electron",
    ]
