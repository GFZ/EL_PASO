# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

import el_paso as ep

mag_field_list = ["Dip", "OP77", "T89", "T01s", "TS04"]


@pytest.mark.parametrize("mag_field", mag_field_list)
@pytest.mark.basic
def test_magnetic_field(mag_field: Literal["T89", "OP77", "TS04", "T01s"], skip_if_unreachable: Callable[..., None]):
    if mag_field in ("T89", "T01s", "TS04"):
        skip_if_unreachable("https://omniweb.gsfc.nasa.gov", "https://spdf.gsfc.nasa.gov")

    true_data = {
        "Dip": (110.12, 110.12, 110.12),
        "OP77": (92.3, 97.27, 106.77),
        "T89": (82.31, 90.91, 96.11),
        "T01s": (40.19, 162.85, 329.75),
        "TS04": (26.91, 92.01, 156.14),
    }

    start_time = datetime(2024, 5, 10, 16, tzinfo=timezone.utc)
    end_time = datetime(2024, 5, 11, 0, tzinfo=timezone.utc)

    time_list: list[float] = []
    curr_time = start_time

    while curr_time <= end_time:
        time_list.append(curr_time.timestamp())
        curr_time += timedelta(minutes=30)

    time_var = ep.Variable(data=np.asarray(time_list), original_unit=ep.units.posixtime)

    xgeo_data = np.tile(np.array([0, 6.6, 0]), (len(time_var.get_data()), 1))
    xgeo_var = ep.Variable(data=xgeo_data, original_unit=ep.units.RE)

    variables_to_compute: ep.processing.VariableRequest = [
        ("B_Calc", mag_field),
    ]

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=time_var,
        xgeo_var=xgeo_var,
        variables_to_compute=variables_to_compute,
        irbem_options=ep.processing.magnetic_field_utils.IrbemOptions(),
        num_cores=12,
    )

    mag_field_data = magnetic_field_variables["B_Calc_" + mag_field].get_data("nT")
    min_value = np.round(mag_field_data.min(), 2)
    mean_value = np.round(mag_field_data.mean(), 2)
    max_value = np.round(mag_field_data.max(), 2)

    assert min_value == true_data[mag_field][0]
    assert mean_value == true_data[mag_field][1]
    assert max_value == true_data[mag_field][2]


@pytest.mark.basic
def test_mlt_mlt_eq_equal():

    start_time = datetime(2024, 5, 10, 16, tzinfo=timezone.utc)
    time_var = ep.Variable(data=np.asarray([start_time.timestamp()]), original_unit=ep.units.posixtime)

    xgeo_data = np.array([[0, 6.6, 0]])
    xgeo_var = ep.Variable(data=xgeo_data, original_unit=ep.units.RE)

    variables_to_compute: ep.processing.VariableRequest = [
        ("MLT", "OP77"),
        ("MLT_Eq", "OP77"),
    ]

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=time_var,
        xgeo_var=xgeo_var,
        variables_to_compute=variables_to_compute,
        irbem_options=ep.processing.magnetic_field_utils.IrbemOptions(),
        num_cores=12,
    )

    mlt = np.round(magnetic_field_variables["MLT_OP77"].get_data())
    mlt_eq = np.round(magnetic_field_variables["MLT_Eq_OP77"].get_data())
    assert mlt == mlt_eq
