# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Jaskirat Singh
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Literal

import lazy_loader as lazy

ELFINSatellite = Literal["ela", "elb"]

__getattr__, _lazy_dir, _lazy_all = lazy.attach(
    __name__,
    submod_attrs={
        "process_elfin_epd_electrons": [
            "elfin_epd_electron_strategy",
            "elfin_epd_electron_gfz_strategy",
            "elfin_epd_electron_netcdf_strategy",
            "process_elfin_epd_electrons",
        ],
    },
)

__all__ = ["ELFINSatellite", *_lazy_all]  # noqa: PLE0604  (lazy_loader supplies the names)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from el_paso.recipes.elfin.process_elfin_epd_electrons import (
        elfin_epd_electron_gfz_strategy,
        elfin_epd_electron_netcdf_strategy,
        elfin_epd_electron_strategy,
        process_elfin_epd_electrons,
    )

    __all__ = [
        "ELFINSatellite",
        "elfin_epd_electron_gfz_strategy",
        "elfin_epd_electron_netcdf_strategy",
        "elfin_epd_electron_strategy",
        "process_elfin_epd_electrons",
    ]
