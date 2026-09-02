# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

from typing import Literal

poes_satellite_literal = Literal[
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

from el_paso.recipes.poes.process_poes_meped import process_poes_meped_electron
from el_paso.recipes.poes.process_poes_ted import process_poes_ted_electron

__all__ = [
    "poes_satellite_literal",
    "process_poes_meped_electron",
    "process_poes_ted_electron",
]
