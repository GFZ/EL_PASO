# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0


from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from astropy import units as u

import el_paso as ep

rng = np.random.default_rng(1337)


@pytest.mark.parametrize("file_format", [".mat", ".h5", ".nc", ".cdf"])
@pytest.mark.basic
def test_basic_single_file_strategy(tmp_path: Path, file_format: str) -> None:
    variables_to_save: dict[ep.typing.InternalName, Any] = {
        "FEDU": ep.Variable(original_unit=u.dimensionless_unscaled, data=rng.normal((20, 21))),
        "Alpha": ep.Variable(original_unit=u.dimensionless_unscaled, data=rng.normal((10, 11))),
        "B_Calc": ep.Variable(original_unit=u.dimensionless_unscaled, data=rng.normal((51,))),
    }

    save_path = tmp_path / ("test" + file_format)
    strategy = ep.saving_strategies.SingleFileStrategy(file_path=save_path)
    ep.save(
        variables_to_save,
        strategy,
        start_time=datetime(2013, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2013, 1, 2, tzinfo=timezone.utc),
    )

    assert save_path.exists()
