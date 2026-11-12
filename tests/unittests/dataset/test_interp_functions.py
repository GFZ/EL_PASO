# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from el_paso.dataset.interp_functions import (
    _interp_flux_parallel,
    _interp_omni_flux_parallel,
    _linear_interp,
    interp_flux,
)

# ── _linear_interp ────────────────────────────────────────────────────────────


@pytest.mark.minimal
@pytest.mark.basic
@pytest.mark.parametrize(
    ("flux_left", "flux_right", "target", "left", "right", "expected"),
    [
        (0.0, 10.0, 5.0, 0.0, 10.0, 5.0),  # midpoint → 0.5 weight
        (0.0, 10.0, 0.0, 0.0, 10.0, 0.0),  # at left boundary → flux_left
        (0.0, 10.0, 10.0, 0.0, 10.0, 10.0),  # at right boundary → flux_right
        (100.0, 200.0, 7.0, 0.0, 10.0, 170.0),  # 70 % weight on right
    ],
)
def test_linear_interp(
    flux_left: float,
    flux_right: float,
    target: float,
    left: float,
    right: float,
    expected: float,
) -> None:
    result = _linear_interp(flux_left, flux_right, target, left, right)
    assert result == pytest.approx(expected)


# ── _interp_flux_parallel ─────────────────────────────────────────────────────
#
# Test arrays  (nt=1, ne=4, na=3):
#   energy        : [[1.0, 2.0, 3.0, 4.0]]          shape (1, 4)
#   alpha_eq_model: [[10.0, 30.0, 60.0]]             shape (1, 3)
#   flux column 0 (alpha=10): [100, 200, 300, 400]
#   flux column 1 (alpha=30): [150, 250, 350, 450]
#   flux column 2 (alpha=60): [200, 300, 400, 500]
#
# For target (energy=2.0, alpha=20.0):
#   al_right=1 (alpha=30), al_left=0 (alpha=10)
#   flux_left  = interp(2.0 in [1,2,3,4], [100,200,300,400]) = 200
#   flux_right = interp(2.0 in [1,2,3,4], [150,250,350,450]) = 250
#   result     = linear(200, 250, 20, 10, 30) = 225


def _make_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (flux, energy, alpha_eq_model) for a (nt=1, ne=4, na=3) grid."""
    energy = np.array([[1.0, 2.0, 3.0, 4.0]])  # (1, 4)
    alpha = np.array([[10.0, 30.0, 60.0]])  # (1, 3)
    flux = np.array(
        [
            [
                [100.0, 150.0, 200.0],
                [200.0, 250.0, 300.0],
                [300.0, 350.0, 400.0],
                [400.0, 450.0, 500.0],
            ]
        ]
    )  # (1, 4, 3)
    return flux, energy, alpha


@pytest.mark.minimal
@pytest.mark.basic
def test_interp_flux_parallel_known_point() -> None:
    flux, energy, alpha = _make_arrays()
    result = _interp_flux_parallel(flux, energy, alpha, targets=[(2.0, 20.0)], it=0)
    assert len(result) == 1
    assert result[0] == pytest.approx(225.0)


@pytest.mark.minimal
@pytest.mark.basic
def test_interp_flux_parallel_on_grid_alpha() -> None:
    """Target exactly on a grid alpha value should return the flux at that alpha."""
    flux, energy, alpha = _make_arrays()
    # target alpha=30 is on-grid: al_right=2 (alpha=60), al_left=1 (alpha=30)
    # flux_left  = interp(2.0, [1,2,3,4], [150,250,350,450]) = 250
    # flux_right = interp(2.0, [1,2,3,4], [200,300,400,500]) = 300
    # result     = linear(250, 300, 30, 30, 60) = 250 + 0*(300-250) = 250
    result = _interp_flux_parallel(flux, energy, alpha, targets=[(2.0, 30.0)], it=0)
    assert result[0] == pytest.approx(250.0)


@pytest.mark.minimal
@pytest.mark.basic
def test_interp_flux_parallel_alpha_below_range_returns_nan() -> None:
    flux, energy, alpha = _make_arrays()
    result = _interp_flux_parallel(flux, energy, alpha, targets=[(2.0, 5.0)], it=0)
    assert np.isnan(result[0])


@pytest.mark.minimal
@pytest.mark.basic
def test_interp_flux_parallel_alpha_above_range_returns_nan() -> None:
    flux, energy, alpha = _make_arrays()
    result = _interp_flux_parallel(flux, energy, alpha, targets=[(2.0, 70.0)], it=0)
    assert np.isnan(result[0])


@pytest.mark.minimal
@pytest.mark.basic
def test_interp_flux_parallel_nan_flux_returns_nan() -> None:
    flux, energy, alpha = _make_arrays()
    flux_nan = flux.copy()
    flux_nan[0, :, 0] = np.nan  # invalidate the left-alpha column
    result = _interp_flux_parallel(flux_nan, energy, alpha, targets=[(2.0, 20.0)], it=0)
    assert np.isnan(result[0])


@pytest.mark.minimal
@pytest.mark.basic
def test_interp_flux_parallel_multiple_targets() -> None:
    flux, energy, alpha = _make_arrays()
    targets = [(2.0, 20.0), (2.0, 5.0)]  # first valid, second out-of-range
    result = _interp_flux_parallel(flux, energy, alpha, targets=targets, it=0)
    assert result[0] == pytest.approx(225.0)
    assert np.isnan(result[1])


# ── _interp_omni_flux_parallel ────────────────────────────────────────────────


@pytest.mark.minimal
@pytest.mark.basic
def test_interp_omni_flux_parallel_known_points() -> None:
    energy = np.array([[1.0, 2.0, 3.0, 4.0]])
    flux = np.array([[100.0, 200.0, 300.0, 400.0]])
    result = _interp_omni_flux_parallel(flux, energy, [2.5, 0.5, 5.0], it=0)
    assert result[0] == pytest.approx(250.0)
    assert np.isnan(result[1])  # below energy range
    assert np.isnan(result[2])  # above energy range


@pytest.mark.minimal
@pytest.mark.basic
def test_interp_omni_flux_parallel_skips_nan_and_needs_two_points() -> None:
    energy = np.array([[1.0, 2.0, 3.0, 4.0]])
    flux = np.array([[100.0, np.nan, 300.0, 400.0]])
    assert _interp_omni_flux_parallel(flux, energy, [2.0], it=0)[0] == pytest.approx(200.0)

    flux_sparse = np.array([[np.nan, np.nan, 300.0, np.nan]])
    assert np.isnan(_interp_omni_flux_parallel(flux_sparse, energy, [3.0], it=0)[0])


# ── interp_flux (flux_name dispatch) ──────────────────────────────────────────


class _FakeDataSet:
    """Minimal stand-in for DataSet exposing what interp_flux needs."""

    _verbose = False

    def __init__(self, variables: dict[str, np.ndarray]) -> None:
        self._variables = variables

    def get_by_internal_name(self, name: str) -> np.ndarray:
        return self._variables[name]


def _make_dataset(nt: int = 2) -> _FakeDataSet:
    flux3d, energy, alpha = _make_arrays()
    return _FakeDataSet(
        {
            "Epoch": np.arange(nt, dtype=float),
            "FEDU": np.repeat(flux3d, nt, axis=0),
            "Energy_FEDU": np.repeat(energy, nt, axis=0),
            "Alpha_Eq": np.repeat(alpha, nt, axis=0),
            "FEDO": np.repeat(np.array([[10.0, 20.0, 30.0, 40.0]]), nt, axis=0),
            "Energy_FEDO": np.array([1.0, 2.0, 3.0, 4.0]),  # time-independent grid
            "FPDU": np.repeat(np.array([[[1.0], [2.0], [3.0], [4.0]]]), nt, axis=0),  # single PA bin
            "Energy_FPDU": np.repeat(energy, nt, axis=0),
        }
    )


@pytest.mark.minimal
@pytest.mark.basic
def test_interp_flux_default_is_fedu() -> None:
    ds = _make_dataset()
    result = interp_flux(ds, [2.0], [20.0], target_type="TargetPairs", n_threads=1)  # ty:ignore[invalid-argument-type]
    assert result.shape == (2, 1)
    np.testing.assert_allclose(result, 225.0)


@pytest.mark.minimal
@pytest.mark.basic
def test_interp_flux_omnidirectional() -> None:
    ds = _make_dataset()
    result = interp_flux(ds, [1.5, 3.5], flux_name="FEDO", n_threads=1)  # ty:ignore[invalid-argument-type]
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, [[15.0, 35.0], [15.0, 35.0]])


@pytest.mark.minimal
@pytest.mark.basic
def test_interp_flux_single_pitch_angle_bin_is_spin_averaged() -> None:
    ds = _make_dataset()
    result = interp_flux(ds, 2.5, flux_name="FPDU", n_threads=1)  # ty:ignore[invalid-argument-type]
    assert result.shape == (2, 1)
    np.testing.assert_allclose(result, 2.5)


@pytest.mark.minimal
@pytest.mark.basic
def test_interp_flux_omnidirectional_rejects_target_al() -> None:
    ds = _make_dataset()
    with pytest.raises(ValueError, match="target_al"):
        interp_flux(ds, [2.0], [20.0], flux_name="FEDO", n_threads=1)  # ty:ignore[invalid-argument-type]


@pytest.mark.minimal
@pytest.mark.basic
def test_interp_flux_directional_requires_target_al() -> None:
    ds = _make_dataset()
    with pytest.raises(ValueError, match="required"):
        interp_flux(ds, [2.0], flux_name="FEDU", n_threads=1)  # ty:ignore[invalid-argument-type]
