# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import importlib
from datetime import datetime, timezone

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


@pytest.mark.basic
def test_release_mode_date_is_utc(monkeypatch: pytest.MonkeyPatch):
    release_mode_mod = importlib.import_module("el_paso.release_mode")

    class FakeDatetime(datetime):
        """UTC is already on the next day, while the naive 'local' time is still on the previous one."""

        @classmethod
        def now(cls, tz: timezone | None = None) -> "FakeDatetime":
            if tz is None:
                return cls(2020, 1, 1, 23)
            return cls(2020, 1, 2, 1, tzinfo=tz)

    monkeypatch.setattr(release_mode_mod, "datetime", FakeDatetime)

    ep.activate_release_mode("test_user", "test_email@test.test", ".", dirty_ok=True)

    assert "02-Jan-2020" in ep._release_msg
