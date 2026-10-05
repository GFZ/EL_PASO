# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import el_paso as ep
import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.basic
def test_monthly_omni_flux_rb_strategy_saves_omnidirectional_flux(tmp_path: Path) -> None:
    args = (tmp_path, "Arase", "arase", "xep", "T89")
    standard = ep.data_standards.GFZStandard()
    strategy = ep.saving_strategies.MonthlyOmniFluxRBStrategy(*args, data_standard=standard)
    monthly_strategy = ep.saving_strategies.MonthlyRBStrategy(*args, data_standard=standard)

    names_to_save = strategy.output_files[0].names_to_save
    monthly_names = monthly_strategy.output_files[0].names_to_save

    assert names_to_save == [*monthly_names, "FEDO", "Energy_FEDO", "Alpha_range"]
