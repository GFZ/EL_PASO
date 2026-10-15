# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""Recipes for products derived from a magnetic field model alone, without spacecraft data."""

from el_paso.recipes.model.process_lcds import lcds_strategy, process_lcds

__all__ = [
    "lcds_strategy",
    "process_lcds",
]
