# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from astropy import units as u

import el_paso as ep
from el_paso import Variable


def test_merge_sorts_chronologically_and_converts_units():
    # self: 3 samples in keV, at t = 0, 10, 20
    self_var = Variable(data=np.array([1.0, 2.0, 3.0]), original_unit=u.keV)
    self_var.metadata.source_files = ["file_a.cdf"]

    self_time = Variable(data=np.array([0.0, 10.0, 20.0]), original_unit=ep.units.posixtime)

    # other: 2 samples in eV (convertible to keV), at t = 5, 15
    other_var = Variable(data=np.array([4000.0, 5000.0]), original_unit=u.eV)
    other_var.metadata.source_files = ["file_b.cdf"]

    other_time = Variable(data=np.array([5.0, 15.0]), original_unit=ep.units.posixtime)

    merged_time = self_var.merge(self_time, other_var, other_time)

    # merged timestamps are sorted chronologically
    np.testing.assert_array_equal(
        merged_time.get_data(ep.units.posixtime),
        np.array([0.0, 5.0, 10.0, 15.0, 20.0]),
    )

    # self's data was updated in place, converted to its own unit (keV) and interleaved by time
    np.testing.assert_allclose(self_var.get_data(u.keV), np.array([1.0, 4.0, 2.0, 5.0, 3.0]))

    # self's unit is unchanged (merge converts other_variable's data, not self's unit)
    assert self_var.metadata.unit == u.keV

    # source files from both variables are merged in order, without duplicates
    assert self_var.metadata.source_files == ["file_a.cdf", "file_b.cdf"]

    # merged time variable carries over description/standard_name/cadence from time_variable,
    # and merges source_files from both time variables (both empty here)
    assert merged_time.metadata.source_files == []


def test_merge_raises_on_trailing_dimension_mismatch():
    self_var = Variable(data=np.zeros((3, 2)), original_unit=u.keV)
    self_time = Variable(data=np.array([0.0, 10.0, 20.0]), original_unit=ep.units.posixtime)

    other_var = Variable(data=np.zeros((2, 3)), original_unit=u.keV)
    other_time = Variable(data=np.array([5.0, 15.0]), original_unit=ep.units.posixtime)

    with pytest.raises(ValueError, match="Can only merge variables"):
        self_var.merge(self_time, other_var, other_time)
