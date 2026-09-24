# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for DataSet methods using actual processed reference files.

These tests load pre-existing .nc files from tests/system/data/processed/ and
verify that get_by_internal_name, identify_orbits, and linearize_trajectories
work correctly on real data. No network access is needed.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import el_paso as ep
import numpy as np
import pytest
from el_paso.dataset import GFZDataSet
from el_paso.dataset.identify_orbits import Trajectory

_PROCESSED = Path(__file__).parent / "data" / "processed"
_START = datetime(2017, 9, 8, tzinfo=timezone.utc)
_END = _START + timedelta(days=0.4, seconds=-1)


@pytest.fixture(scope="module")
def rbsp_dataset() -> GFZDataSet:
    return GFZDataSet(
        start_time=_START,
        end_time=_END,
        saving_strategy=ep.saving_strategies.MonthlyRBStrategy(
            _PROCESSED,
            "RBSP",
            "rbspa",
            "ect_combined",
            "T89",
            data_standard=ep.data_standards.GFZStandard(),
            file_format="nc",
        ),
        verbose=False,
    )


@pytest.mark.basic
def test_get_by_internal_name_epoch_matches_time(rbsp_dataset: GFZDataSet) -> None:
    result = rbsp_dataset.get_by_internal_name("Epoch")
    np.testing.assert_array_equal(result, rbsp_dataset.time)


@pytest.mark.basic
def test_get_by_internal_name_r_eq_matches_r0(rbsp_dataset: GFZDataSet) -> None:
    result = rbsp_dataset.get_by_internal_name("R_Eq")
    np.testing.assert_array_equal(result, rbsp_dataset.R0)


@pytest.mark.basic
def test_get_by_internal_name_l_star_matches_lstar(rbsp_dataset: GFZDataSet) -> None:
    result = rbsp_dataset.get_by_internal_name("L_star")
    np.testing.assert_array_equal(result, rbsp_dataset.Lstar)


@pytest.mark.basic
def test_get_by_internal_name_fedu_matches_flux(rbsp_dataset: GFZDataSet) -> None:
    result = rbsp_dataset.get_by_internal_name("FEDU")
    np.testing.assert_array_equal(result, rbsp_dataset.Flux)


@pytest.mark.basic
def test_identify_orbits_returns_trajectories(rbsp_dataset: GFZDataSet) -> None:
    trajectories = rbsp_dataset.identify_orbits(orbit_type="R", apply_smoothing=False)

    assert isinstance(trajectories, list)
    assert len(trajectories) > 0
    assert all(isinstance(t, Trajectory) for t in trajectories)


@pytest.mark.basic
def test_identify_orbits_covers_full_time_range(rbsp_dataset: GFZDataSet) -> None:
    trajectories = rbsp_dataset.identify_orbits(orbit_type="R", apply_smoothing=False)

    assert trajectories[0].start == 0
    assert trajectories[-1].end == len(rbsp_dataset.R0) - 1


@pytest.mark.basic
def test_identify_orbits_lstar_mode_returns_trajectories(rbsp_dataset: GFZDataSet) -> None:
    trajectories = rbsp_dataset.identify_orbits(orbit_type="L*", apply_smoothing=False)

    assert isinstance(trajectories, list)
    assert len(trajectories) > 0


@pytest.mark.basic
def test_linearize_trajectories_returns_correct_shape(rbsp_dataset: GFZDataSet) -> None:
    trajectories = rbsp_dataset.identify_orbits(orbit_type="R", apply_smoothing=False)
    lin_x, bend_time = rbsp_dataset.linearize_trajectories(trajectories, orbit_type="R")

    assert len(lin_x) == len(rbsp_dataset.R0)
    assert len(bend_time) == len(rbsp_dataset.R0)


@pytest.mark.basic
def test_linearize_trajectories_x_axis_monotonic(rbsp_dataset: GFZDataSet) -> None:
    trajectories = rbsp_dataset.identify_orbits(orbit_type="R", apply_smoothing=False)
    lin_x, _ = rbsp_dataset.linearize_trajectories(trajectories, orbit_type="R")

    diffs = np.diff(lin_x)
    assert np.all(diffs >= 0), "Linearized x-axis is not monotonically non-decreasing"
