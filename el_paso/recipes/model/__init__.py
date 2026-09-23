# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""Recipes for products derived from a magnetic field model alone, without spacecraft data."""

from el_paso.recipes.model.process_lcds import compute_lcds, lcds_strategy

__all__ = [
    "compute_lcds",
    "lcds_strategy",
]
