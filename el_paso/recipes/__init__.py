# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import lazy_loader as lazy

# Each satellite's recipes are resolved on first access (SPEC 1), so that e.g.
# `el_paso.recipes.rbsp` does not also pay for arase/esa/goes/gps/poes/probav/dmsp.
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["arase", "dmsp", "esa", "goes", "gps", "poes", "probav", "rbsp"],
)

if TYPE_CHECKING:
    from el_paso.recipes import arase, dmsp, esa, goes, gps, poes, probav, rbsp

    __all__ = [
        "arase",
        "dmsp",
        "esa",
        "goes",
        "gps",
        "poes",
        "probav",
        "rbsp",
    ]
