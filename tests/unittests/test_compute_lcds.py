# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime, timezone
from typing import Literal

import h5py
import numpy as np
import pandas as pd
import pytest
from astropy import units as u

import el_paso as ep
from el_paso.processing import compute_LCDS
from el_paso.processing.magnetic_field_utils import IrbemOptions

# ---------------------------------------------------------------------------
# Reference LCDS L* from MATLAB (LCDS2 ver4), model T89, 2013-03-17, 2-hourly.
# Source: MP_nopos_LCDS2_20130301to20130331_{lstar,alpha}_T89_ver4.mat
# Rows = timestamps (LCDS_TIMES_UTC); columns = pitch angles (ALPHA_EQ_DEG).
# ---------------------------------------------------------------------------

ALPHA_EQ_DEG = (10.0, 50.0, 90.0)            # first / middle / last channel
ALPHA_EQ_RAD = (0.17453293, 0.87266463, 1.57079633)

LCDS_TIMES_UTC = (
    "2013-03-17T00:00:00",
    "2013-03-17T02:00:00",
    "2013-03-17T04:00:00",
    "2013-03-17T06:00:00",
    "2013-03-17T08:00:00",
    "2013-03-17T10:00:00",
    "2013-03-17T12:00:00",
    "2013-03-17T14:00:00",
    "2013-03-17T16:00:00",
    "2013-03-17T18:00:00",
    "2013-03-17T20:00:00",
    "2013-03-17T22:00:00",
)

LCDS_LSTAR_REF_T89 = np.array([
    [7.880390, 7.708010, 6.614410],
    [8.062970, 7.866900, 6.788370],
    [8.064740, 7.949160, 6.871560],
    [5.358870, 5.405360, 4.835760],
    [5.351260, 5.405980, 4.819060],
    [5.345900, 5.413110, 4.817160],
    [5.344380, 5.417560, 4.817890],
    [5.315130, 5.375610, 4.808890],
    [5.346030, 5.409220, 4.837040],
    [5.377300, 5.449220, 4.840690],
    [5.324520, 5.384800, 4.809580],
    [5.316690, 5.373820, 4.796410],
])

@pytest.mark.basic
def test_lcds_runs():

    start_time = datetime(2013, 3, 17, 0, tzinfo=timezone.utc)
    end_time = datetime(2013, 3, 17, 22, tzinfo=timezone.utc)

    datetimes = pd.date_range(start_time, end_time, freq="2h").to_pydatetime()  # ty:ignore[unresolved-attribute]
    posixtimes = [t.timestamp() for t in datetimes]

    pa_eq = [10.0, 50.0, 90.0]
    pa_eq = np.tile(pa_eq, (len(posixtimes), 1))

    time_var = ep.Variable(data=np.asarray(posixtimes), original_unit=ep.units.posixtime)
    pa_eq_var = ep.Variable(data=pa_eq, original_unit=u.deg)

    lcds = compute_LCDS(
        time_var,
        pa_eq_var,
        "T89",
        IrbemOptions(),
        num_cores=12,
    )

    lcds_data = lcds.get_data()

    matlab_solution = np.asarray(LCDS_LSTAR_REF_T89)

    assert lcds_data == pytest.approx(matlab_solution, abs=0.1)
