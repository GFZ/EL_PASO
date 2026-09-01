# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import numpy as np
import pytest
from astropy import units as u

import el_paso as ep
from el_paso.data_standard import ConsistencyCheck, VariableInfo, _assert_sorted
from el_paso.typing import FixedDimensionName, InternalName

TIME_LEN = 100
ENERGY_LEN = 20
PITCH_ANGLE_LEN = 50


@pytest.mark.basic
def test_consistency_correct():
    consistency_check = ConsistencyCheck()

    consistency_check.check_size(TIME_LEN, "Time", "call1")
    consistency_check.check_size(TIME_LEN, "Time", "call2")

    consistency_check.check_size(ENERGY_LEN, "Energy", "call1")
    consistency_check.check_size(ENERGY_LEN, "Energy", "call2")

    consistency_check.check_size(PITCH_ANGLE_LEN, "Pitch Angle", "call1")
    consistency_check.check_size(PITCH_ANGLE_LEN, "Pitch Angle", "call2")


@pytest.mark.basic
def test_consistency_time_wrong():
    consistency_check = ConsistencyCheck()

    consistency_check.check((TIME_LEN,), ["Time"], "call1")

    with pytest.raises(ValueError, match=r"Length mismatch! Time length of variable call1: *"):
        consistency_check.check((TIME_LEN + 1,), ["Time"], "call2")


@pytest.mark.basic
def test_consistency_energy_wrong():
    consistency_check = ConsistencyCheck()

    consistency_check.check((TIME_LEN, ENERGY_LEN, PITCH_ANGLE_LEN), ["Time", "Energy", "Alpha"], "call1")
    consistency_check.check((TIME_LEN, ENERGY_LEN, PITCH_ANGLE_LEN), ["Time", "Energy", "Alpha"], "call2")

    with pytest.raises(ValueError, match=r"Length mismatch! Energy length of variable call1: *"):
        consistency_check.check((TIME_LEN, ENERGY_LEN+1, PITCH_ANGLE_LEN), ["Time", "Energy", "Alpha"], "call2")

@pytest.mark.basic
def test_numbers():
    consistency_check = ConsistencyCheck()

    consistency_check.check((TIME_LEN, 2, PITCH_ANGLE_LEN), ["Time", 2, "Alpha"], "call1")
    consistency_check.check((TIME_LEN, 2, PITCH_ANGLE_LEN), ["Time", 2, "Alpha"], "call2")

    with pytest.raises(ValueError, match="Length mismatch! Variable call2 should have length 2, but encountered 3!"):
        consistency_check.check((TIME_LEN, 3, PITCH_ANGLE_LEN), ["Time", 2, "Alpha"], "call2")


@pytest.mark.basic
def test_consistency_same_dim_different_size_raises():
    """Second call with the same dimension name but a different size must raise."""
    consistency_check = ConsistencyCheck()

    consistency_check.check_size(TIME_LEN, "Time", "var_a")

    with pytest.raises(ValueError):  # noqa: PT011
        consistency_check.check_size(TIME_LEN + 1, "Time", "var_b")


@pytest.mark.basic
def test_data_standard_repr_and_str():
    """DataStandard.__repr__ / __str__ should return a non-empty string without raising."""
    standard = ep.data_standards.GFZStandard()

    r = repr(standard)
    s = str(standard)

    assert isinstance(r, str)
    assert r
    assert isinstance(s, str)
    assert s
    assert "GFZStandard" in r


def _make_var_info(
    dependencies: list[InternalName | FixedDimensionName],
    sorted_along: tuple[InternalName, Literal["ascending", "descending"]] | None = None,
) -> VariableInfo[str]:
    return VariableInfo(
        standard_name="test",
        description="",
        unit=u.dimensionless_unscaled,
        dependencies=dependencies,
        sorted_along=sorted_along,
    )


@pytest.mark.basic
def test_assert_sorted_ascending_1d_passes() -> None:
    data = np.array([1.0, 2.0, 3.0, 4.0])
    info = _make_var_info(["Epoch"], sorted_along=("Epoch", "ascending"))
    _assert_sorted(data, info, "test_var")


@pytest.mark.basic
def test_assert_sorted_ascending_1d_raises() -> None:
    data = np.array([1.0, 3.0, 2.0, 4.0])
    info = _make_var_info(["Epoch"], sorted_along=("Epoch", "ascending"))
    with pytest.raises(ValueError, match="must be sorted ascending"):
        _assert_sorted(data, info, "test_var")


@pytest.mark.basic
def test_assert_sorted_descending_1d_passes() -> None:
    data = np.array([4.0, 3.0, 2.0, 1.0])
    info = _make_var_info(["Epoch"], sorted_along=("Epoch", "descending"))
    _assert_sorted(data, info, "test_var")


@pytest.mark.basic
def test_assert_sorted_descending_1d_raises() -> None:
    data = np.array([4.0, 2.0, 3.0, 1.0])
    info = _make_var_info(["Epoch"], sorted_along=("Epoch", "descending"))
    with pytest.raises(ValueError, match="must be sorted descending"):
        _assert_sorted(data, info, "test_var")


@pytest.mark.basic
def test_assert_sorted_ascending_2d_along_second_axis() -> None:
    data = np.array([[10.0, 20.0, 30.0], [5.0, 15.0, 25.0]])
    info = _make_var_info(["Epoch", "Alpha"], sorted_along=("Alpha", "ascending"))
    _assert_sorted(data, info, "test_var")


@pytest.mark.basic
def test_assert_sorted_ascending_2d_along_second_axis_raises() -> None:
    data = np.array([[10.0, 30.0, 20.0], [5.0, 15.0, 25.0]])
    info = _make_var_info(["Epoch", "Alpha"], sorted_along=("Alpha", "ascending"))
    with pytest.raises(ValueError, match=r"must be sorted ascending.*Alpha.*axis 1"):
        _assert_sorted(data, info, "test_var")


@pytest.mark.basic
def test_assert_sorted_along_first_axis_of_2d() -> None:
    data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    info = _make_var_info(["Epoch", "Alpha"], sorted_along=("Epoch", "ascending"))
    _assert_sorted(data, info, "test_var")


@pytest.mark.basic
def test_assert_sorted_along_first_axis_of_2d_raises() -> None:
    data = np.array([[2.0, 10.0], [1.0, 20.0], [3.0, 30.0]])
    info = _make_var_info(["Epoch", "Alpha"], sorted_along=("Epoch", "ascending"))
    with pytest.raises(ValueError, match=r"must be sorted ascending.*Epoch.*axis 0"):
        _assert_sorted(data, info, "test_var")


@pytest.mark.basic
def test_standardize_variable_checks_sorting() -> None:
    """standardize_variable must raise when sorted_along is violated."""
    standard = ep.data_standards.GFZStandard()

    standard.variable_infos["Epoch"] = standard.variable_infos["Epoch"]._replace(
        sorted_along=("Epoch", "ascending"),
    )

    unsorted_data = np.array([3.0, 1.0, 2.0])
    var = ep.Variable(original_unit=ep.units.posixtime, data=unsorted_data)

    with pytest.raises(ValueError, match="must be sorted ascending"):
        standard.standardize_variable("Epoch", var, reset_consistency_check=True)
