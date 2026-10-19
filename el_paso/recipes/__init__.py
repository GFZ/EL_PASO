# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import lazy_loader as lazy

# Each satellite's recipes are resolved on first access (SPEC 1), so that e.g.
# `el_paso.recipes.rbsp` does not also pay for arase/esa/goes/gps/poes/probav/dmsp/themis.
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "arase",
        "dmsp",
        "esa",
        "goes",
        "gps",
        "model",
        "poes",
        "probav",
        "rbsp",
        "solar_wind_indices",
        "themis",
    ],
    submod_attrs={
        "model": ["process_lcds"],
        "solar_wind_indices": ["process_solar_wind_indices"],
    },
)

if TYPE_CHECKING:
    from el_paso.recipes import arase, dmsp, esa, goes, gps, model, poes, probav, rbsp, solar_wind_indices, themis
    from el_paso.recipes.model import process_lcds
    from el_paso.recipes.solar_wind_indices import process_solar_wind_indices

    __all__ = [
        "arase",
        "dmsp",
        "esa",
        "goes",
        "gps",
        "model",
        "poes",
        "probav",
        "process_lcds",
        "process_solar_wind_indices",
        "rbsp",
        "solar_wind_indices",
        "themis",
    ]
