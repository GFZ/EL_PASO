# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

from typing import Literal

DMSPSatellite = Literal["f17"]

from el_paso.recipes.dmsp.process_dmsp_ssj_electrons import dmsp_ssj_electron_strategy, process_dmsp_ssj_electrons

__all__ = ["DMSPSatellite", "dmsp_ssj_electron_strategy", "process_dmsp_ssj_electrons"]
