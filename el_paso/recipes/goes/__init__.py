# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

from typing import Literal

GOESRSatellite = Literal["goes18", "goes19"]
GOESRealtimeSatellite = Literal["primary", "secondary"]

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
