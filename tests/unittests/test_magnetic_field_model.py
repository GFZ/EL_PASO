# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
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


def _leo_request(lat_lon_deg: list[tuple[float, float]]) -> tuple[ep.Variable, ep.Variable]:
    """Time and position variables for points at 500 km altitude, given as geocentric latitude and longitude."""
    radius = 1 + 500 / 6371.2
    lat, lon = np.radians(np.asarray(lat_lon_deg)).T
    x_geo = np.column_stack(
        [radius * np.cos(lat) * np.cos(lon), radius * np.cos(lat) * np.sin(lon), radius * np.sin(lat)]
    )

    epoch = datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp()
    time_var = ep.Variable(data=np.full(len(x_geo), epoch), original_unit=ep.units.posixtime)
    xgeo_var = ep.Variable(data=x_geo, original_unit=ep.units.RE)

    return time_var, xgeo_var


LOSS_CONE_REQUEST: ep.processing.VariableRequest = [
    ("Alpha_LC", "Dip"),
    ("Alpha_LC_Eq", "Dip"),
    ("Alpha_DLC", "Dip"),
    ("Alpha_DLC_Eq", "Dip"),
]


@pytest.mark.basic
def test_drift_loss_cone_equals_bounce_loss_cone_in_centred_dipole():
    # With no South Atlantic Anomaly, every longitude on a drift shell has the same foot point field, so nothing
    # mirrors lower anywhere else on the drift than it does locally: the two loss cones coincide.
    time_var, xgeo_var = _leo_request([(-40, -20), (-33, 10), (55, 100), (-55, 150)])
    irbem_options = ep.processing.magnetic_field_utils.IrbemOptions(
        drift_shell_resolution=1,
        internal_field_model=ep.processing.magnetic_field_utils.InternalFieldModel.CENTERED_DIPOLE,
    )

    out = ep.processing.compute_magnetic_field_variables(
        time_var, xgeo_var, LOSS_CONE_REQUEST, irbem_options, num_cores=4, cache_dir=None
    )

    # half the default bisection tolerance of 0.1 degrees, plus a little slack
    np.testing.assert_allclose(
        out["Alpha_DLC_Eq_Dip"].get_data("deg"), out["Alpha_LC_Eq_Dip"].get_data("deg"), rtol=0, atol=0.06
    )


@pytest.mark.basic
def test_drift_loss_cone_in_igrf():
    # Two points under the South Atlantic Anomaly and two well away from it.
    time_var, xgeo_var = _leo_request([(-40, -20), (-33, 10), (55, 100), (-55, 150)])
    irbem_options = ep.processing.magnetic_field_utils.IrbemOptions(drift_shell_resolution=1)

    out = ep.processing.compute_magnetic_field_variables(
        time_var, xgeo_var, LOSS_CONE_REQUEST, irbem_options, num_cores=4, cache_dir=None
    )

    dlc = out["Alpha_DLC_Dip"].get_data("deg")
    dlc_eq = out["Alpha_DLC_Eq_Dip"].get_data("deg")
    lc = out["Alpha_LC_Dip"].get_data("deg")
    lc_eq = out["Alpha_LC_Eq_Dip"].get_data("deg")

    assert np.isfinite(dlc).all()
    assert np.isfinite(dlc_eq).all()

    # The drift loss cone can never be narrower than the bounce loss cone, in either frame.
    assert (dlc >= lc).all()
    assert (dlc_eq >= lc_eq).all()
    assert (dlc <= 90).all()

    # Under the anomaly the local field line is already the weakest on the drift shell, so part of the locally
    # observable distribution is stably trapped. Away from it, all of it drifts into the anomaly and is lost.
    assert (dlc[:2] < 80).all()
    np.testing.assert_allclose(dlc[2:], 90)
    assert (dlc_eq[2:] - lc_eq[2:] > 0.5).all()
