# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from el_paso.saving_strategies.monthly_rb_strategy import MonthlyRBStrategy

if TYPE_CHECKING:
    from el_paso.typing import InternalName


class MonthlyOmniFluxRBStrategy(MonthlyRBStrategy):
    """Save radiation-belt data, including the omnidirectional flux, into one monthly file per interval.

    This strategy extends `MonthlyRBStrategy` and additionally saves the omnidirectional
    flux (FEDO) together with its energy and local pitch-angle range.
    """

    def _get_output_file_entries(self) -> list[InternalName | tuple[InternalName, ...]]:
        """Return the monthly variable list plus FEDO and its energy and pitch-angle range."""
        return [*super()._get_output_file_entries(), "FEDO", "Energy_FEDO", "Alpha_range"]
