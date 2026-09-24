# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Literal

import lazy_loader as lazy

GOESRSatellite = Literal["goes18", "goes19"]
GOESRealtimeSatellite = Literal["primary", "secondary"]

__getattr__, _lazy_dir, _lazy_all = lazy.attach(
    __name__,
    submod_attrs={
        "process_goes_r_mps_high": [
            "goes_r_mps_high_gfz_strategy",
            "goes_r_mps_high_netcdf_strategy",
            "process_goes_r_mps_high",
        ],
        "process_goes_realtime": [
            "goes_realtime_gfz_strategy",
            "goes_realtime_netcdf_strategy",
            "process_goes_real_time",
        ],
    },
)

__all__ = ["GOESRSatellite", "GOESRealtimeSatellite", *_lazy_all]  # noqa: PLE0604  (lazy_loader supplies the names)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from el_paso.recipes.goes.process_goes_r_mps_high import (
        goes_r_mps_high_gfz_strategy,
        goes_r_mps_high_netcdf_strategy,
        process_goes_r_mps_high,
    )
    from el_paso.recipes.goes.process_goes_realtime import (
        goes_realtime_gfz_strategy,
        goes_realtime_netcdf_strategy,
        process_goes_real_time,
    )

    __all__ = [
        "GOESRSatellite",
        "GOESRealtimeSatellite",
        "goes_r_mps_high_gfz_strategy",
        "goes_r_mps_high_netcdf_strategy",
        "goes_realtime_gfz_strategy",
        "goes_realtime_netcdf_strategy",
        "process_goes_r_mps_high",
        "process_goes_real_time",
    ]
