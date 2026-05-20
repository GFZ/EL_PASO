# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0


from datetime import datetime, timezone

import cdflib
import numpy as np
import pytest
from astropy import units as u

import el_paso as ep

# Define a common reference time point
REF_DATETIME = datetime(2023, 10, 27, 10, 0, 0, 99999, tzinfo=timezone.utc)

# Create Astropy quantities for the reference time in each unit
REF_POSIXTIME_QUANTITY = u.Quantity(REF_DATETIME.timestamp(), ep.units.posixtime)
REF_TT2000_QUANTITY = u.Quantity(
    cdflib.cdfepoch.timestamp_to_tt2000(REF_POSIXTIME_QUANTITY.value),
    ep.units.tt2000,
)
REF_CDF_EPOCH_QUANTITY = u.Quantity(
    cdflib.cdfepoch.timestamp_to_cdfepoch(REF_POSIXTIME_QUANTITY.value),
    ep.units.cdf_epoch,
)
REF_DATENUM_QUANTITY = u.Quantity(7.391864166666666e05, ep.units.datenum)  # calcualted using Matlab

# Parameterized test data for circular conversions
# Each tuple is: (start_quantity, target_unit_1, target_unit_2)
test_data = [
    (REF_TT2000_QUANTITY, ep.units.datenum, ep.units.posixtime),
    (REF_POSIXTIME_QUANTITY, ep.units.tt2000, ep.units.datenum),
    (REF_CDF_EPOCH_QUANTITY, ep.units.posixtime, ep.units.datenum),
    (REF_DATENUM_QUANTITY, ep.units.cdf_epoch, ep.units.tt2000),
]


@pytest.mark.basic
@pytest.mark.parametrize(
    ("start_q", "target_unit_1", "target_unit_2"),
    test_data,
)
def test_circular_conversion_astropy(start_q: u.Quantity, target_unit_1: u.UnitBase, target_unit_2: u.UnitBase) -> None:
    """Tests a circular conversion path using Astropy quantities.

    This test verifies that converting a quantity through a sequence of units
    and back to its original unit results in a value that is numerically
    equivalent to the starting value.
    """
    # Perform the conversion steps
    intermediate_q1 = start_q.to(target_unit_1)
    intermediate_q2 = intermediate_q1.to(target_unit_2)
    final_q = intermediate_q2.to(start_q.unit)

    # Assert that the final value is close to the starting value
    assert np.allclose(final_q.value, start_q.value)
