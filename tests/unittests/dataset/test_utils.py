# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

import numpy as np
import pytest

from el_paso.dataset import utils


@pytest.mark.basic
def test_join_var():
    a = np.array([1, 2])
    b = np.array([3, 4])
    result = utils.join_var(a, b)
    np.testing.assert_array_equal(result, [1, 2, 3, 4])


@pytest.mark.basic
def test_round_seconds():
    dt1 = datetime(2024, 1, 1, 12, 0, 0, 600_000, tzinfo=timezone.utc)
    dt2 = datetime(2024, 1, 1, 12, 0, 0, 300_000, tzinfo=timezone.utc)
    assert utils.round_seconds(dt1).second == 1
    assert utils.round_seconds(dt2).second == 0


@pytest.mark.basic
def test_python2matlab_and_matlab2python_roundtrip():
    dt1 = datetime(2024, 4, 16, 15, 30, 0, tzinfo=timezone.utc)
    matlab_time = utils.python2matlab(dt1)
    dt2: datetime = utils.matlab2python(matlab_time)  # ty:ignore[invalid-assignment]
    assert dt2.year == dt1.year
    assert dt2.month == dt1.month
    assert dt2.day == dt1.day


@pytest.mark.basic
def test_matlab2python_iterable():
    dt1 = datetime(2024, 4, 16, 15, 30, 0, tzinfo=timezone.utc)
    matlab_time = utils.python2matlab(dt1)
    dt2: list[datetime] = utils.matlab2python([matlab_time])  # ty:ignore[invalid-assignment]
    assert isinstance(dt2, list)
    assert dt2[0] == dt1


@pytest.mark.basic
def test_pol2cart_and_cart2pol():
    theta = np.array([0, np.pi / 2, np.pi])
    radius = np.array([1, 1, 1])
    x, y = utils.pol2cart(theta, radius)
    theta2, r2 = utils.cart2pol(x, y)
    np.testing.assert_allclose(theta % (2 * np.pi), theta2 % (2 * np.pi), atol=1e-5)
    np.testing.assert_allclose(radius, r2, atol=1e-5)
