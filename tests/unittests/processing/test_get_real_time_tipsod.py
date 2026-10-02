# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

import importlib
from datetime import datetime, timedelta, timezone

import el_paso as ep
import numpy as np
import pytest
from astropy import units as u


@pytest.mark.basic
def test_empty_time_bins_are_nan_filled(monkeypatch: pytest.MonkeyPatch) -> None:
    tipsod_mod = importlib.import_module("el_paso.processing.get_real_time_tipsod")

    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = np.array([(t0 + timedelta(minutes=i)).timestamp() for i in range(3)])

    # data only in the first and in the last bin, none in the second one
    sample_times = [t0 + timedelta(seconds=10), t0 + timedelta(minutes=2, seconds=10)]
    result = {
        "Data": [
            {
                "Time": sample_times,
                "Coordinates": [
                    {"X": 1.0, "Y": 2.0, "Z": 3.0},
                    {"X": 4.0, "Y": 5.0, "Z": 6.0},
                ],
            }
        ]
    }

    class FakeSscWs:
        def get_locations(self, *_args: object, **_kwargs: object) -> dict:
            return result

    monkeypatch.setattr(tipsod_mod, "SscWs", FakeSscWs)

    var = tipsod_mod.get_real_time_tipsod(timestamps, "FAKE-SAT")
    data = var.get_data(u.km)

    # one row per input timestamp, gap is NaN
    assert data.shape == (3, 3)
    np.testing.assert_allclose(data[0], [1.0, 2.0, 3.0])
    assert np.isnan(data[1]).all()
    np.testing.assert_allclose(data[2], [4.0, 5.0, 6.0])
    assert isinstance(var, ep.Variable)
