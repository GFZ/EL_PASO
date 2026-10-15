# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0


from typing import TYPE_CHECKING

import el_paso as ep
import numpy as np
import pytest
from astropy import units as u

if TYPE_CHECKING:
    from el_paso.processing.interpolate_in_time import InterpolationMethod


@pytest.mark.basic
def test_interpolate_max_gap_seconds():
    orig_times = np.asarray([0.0, 2.0, 10.0, 12.0])
    orig_data = np.asarray([0.0, 2.0, 10.0, 12.0])

    time_var = ep.Variable(data=orig_times, original_unit=ep.units.posixtime)
    var_test = ep.Variable(data=orig_data, original_unit=u.km)
    variables = {"var_test": var_test}

    target_times = np.asarray([1.0, 5.0, 11.0])
    target_time_var = ep.Variable(data=target_times, original_unit=ep.units.posixtime)

    interpolation_method_dict: dict[str, InterpolationMethod] = {"var_test": "linear"}

    ep.processing.interpolate_in_time(
        time_variable=time_var,
        variables=variables,
        interpolation_method_dict=interpolation_method_dict,
        target_time_variable=target_time_var,
        max_gap_seconds=4.0,
    )

    result_data = var_test.get_data()

    # 1.0 falls into a gap of 2.0 seconds (<= 4.0) -> Should interpolate perfectly
    assert np.isclose(result_data[0], 1.0)
    # 5.0 falls into a gap of 8.0 seconds (> 4.0)  -> Should mask out to NaN
    assert np.isnan(result_data[1])
    # 11.0 falls into a gap of 2.0 seconds (<= 4.0) -> Should interpolate perfectly
    assert np.isclose(result_data[2], 11.0)


@pytest.mark.basic
def test_interpolate_nan_gaps():
    orig_times = np.asarray([0.0, 1.0, 2.0, 3.0])
    orig_data = np.asarray([10.0, np.nan, 30.0, 40.0])

    time_var = ep.Variable(data=orig_times, original_unit=ep.units.posixtime)
    var_test = ep.Variable(data=orig_data, original_unit=u.km)
    variables = {"var_test": var_test}

    target_times = np.asarray([0.5, 1.5, 2.5])
    target_time_var = ep.Variable(data=target_times, original_unit=ep.units.posixtime)

    interpolation_method_dict: dict[str, InterpolationMethod] = {"var_test": "linear"}

    ep.processing.interpolate_in_time(
        time_variable=time_var,
        variables=variables,
        interpolation_method_dict=interpolation_method_dict,
        target_time_variable=target_time_var,
    )

    result_data = var_test.get_data()

    # 0.5 borders the NaN at 1.0 -> Should resolve to NaN
    assert np.isnan(result_data[0])
    # 1.5 borders the NaN at 1.0 -> Should resolve to NaN
    assert np.isnan(result_data[1])
    # 2.5 is bounded entirely by valid data points (2.0 and 3.0) -> Should interpolate to 35.0
    assert np.isclose(result_data[2], 35.0)


@pytest.mark.basic
def test_interpolate_exact_matches_are_protected():
    orig_times = np.asarray([0.0, 10.0, 20.0])
    orig_data = np.asarray([0.0, 100.0, np.nan])

    time_var = ep.Variable(data=orig_times, original_unit=ep.units.posixtime)
    var_test = ep.Variable(data=orig_data, original_unit=u.km)
    variables = {"var_test": var_test}

    target_times = np.asarray([10.0, 15.0, 20.0])
    target_time_var = ep.Variable(data=target_times, original_unit=ep.units.posixtime)

    interpolation_method_dict: dict[str, InterpolationMethod] = {"var_test": "linear"}

    ep.processing.interpolate_in_time(
        time_variable=time_var,
        variables=variables,
        interpolation_method_dict=interpolation_method_dict,
        target_time_variable=target_time_var,
        max_gap_seconds=5.0,
    )

    result_data = var_test.get_data()

    # 10.0 matches an original exact measurement. Despite max_gap_seconds=5.0, it should be protected.
    assert np.isclose(result_data[0], 100.0)
    # 15.0 is whithin the large gap -> should be NaN
    assert np.isnan(result_data[1])
    # 20.0 matches an exact NaN measurement -> Should remain NaN
    assert np.isnan(result_data[2])
