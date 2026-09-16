# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

from typing import Literal

ESANGRMSatellite = Literal["EDRS-C", "S6-MF", "S6-B", "MTG-S1", "MTG-I1"]

from el_paso.recipes.esa.process_ngrm_satellite import esa_ngrm_strategy, process_ngrm_electron_fluxes

__all__ = ["ESANGRMSatellite", "esa_ngrm_strategy", "process_ngrm_electron_fluxes"]
