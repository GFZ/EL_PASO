# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from el_paso import TimeBinMethod


@pytest.mark.basic
def test_mean() -> None:
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = TimeBinMethod.Mean(data)
    assert result == np.mean(data)


@pytest.mark.basic
def test_mean_drop_percent() -> None:
    data = np.array([10, 6.3, 3, 5.3, 6, 9, 7, 8, 2, 1, 5], dtype=np.float64)
    result = TimeBinMethod.Mean(data, drop_percent=10)
    assert result == np.mean(np.sort(data)[1:-1])
