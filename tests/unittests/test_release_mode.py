# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import el_paso as ep
import pytest
from astropy import units as u


@pytest.mark.basic
def test_release_mode_basic():

    var_before = ep.Variable(original_unit=u.km)
    assert len(var_before.metadata.processing_notes) == 0

    ep.activate_release_mode("test_user", "test_email@test.test", ".", dirty_ok=True)

    var_after = ep.Variable(original_unit=u.km)

    assert "test_user" in var_after.metadata.processing_notes
    assert "test_email@test.test" in var_after.metadata.processing_notes
