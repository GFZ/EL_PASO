# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from astropy import units as u

import el_paso as ep

rng = np.random.default_rng(1337)

@pytest.mark.basic
def test_save_raises_warning_when_var_is_empty(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:

    variables_to_save: dict[ep.typing.InternalName, Any] = {
        "FEDU": ep.Variable(original_unit=u.dimensionless_unscaled, data=rng.normal((20, 21))),
        "Alpha": ep.Variable(original_unit=u.dimensionless_unscaled, data=rng.normal((10, 11))),
        "B_Calc": ep.Variable(original_unit=u.dimensionless_unscaled, data=np.full((51,), np.nan)),
    }

    save_path = tmp_path / ("test.nc")
    strategy = ep.saving_strategies.SingleFileStrategy(file_path=save_path)

    with caplog.at_level(logging.WARNING, logger="ep"):
        ep.save(
            variables_to_save,
            strategy,
            start_time=datetime(2013, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2013, 1, 2, tzinfo=timezone.utc),
        )

    assert any(
        r.levelno == logging.WARNING and "Variable B_Calc only holds NaN values!" in r.getMessage()
        for r in caplog.records
    )
