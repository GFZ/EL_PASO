# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from el_paso.dataset.interp_functions import _interp_flux_parallel, _linear_interp

# ── _linear_interp ────────────────────────────────────────────────────────────


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


@pytest.mark.basic
def test_interp_flux_parallel_known_point() -> None:
    flux, energy, alpha = _make_arrays()
    result = _interp_flux_parallel(flux, energy, alpha, targets=[(2.0, 20.0)], it=0)
    assert len(result) == 1
    assert result[0] == pytest.approx(225.0)


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


@pytest.mark.basic
def test_interp_flux_parallel_alpha_below_range_returns_nan() -> None:
    flux, energy, alpha = _make_arrays()
    result = _interp_flux_parallel(flux, energy, alpha, targets=[(2.0, 5.0)], it=0)
    assert np.isnan(result[0])


@pytest.mark.basic
def test_interp_flux_parallel_alpha_above_range_returns_nan() -> None:
    flux, energy, alpha = _make_arrays()
    result = _interp_flux_parallel(flux, energy, alpha, targets=[(2.0, 70.0)], it=0)
    assert np.isnan(result[0])


@pytest.mark.basic
def test_interp_flux_parallel_nan_flux_returns_nan() -> None:
    flux, energy, alpha = _make_arrays()
    flux_nan = flux.copy()
    flux_nan[0, :, 0] = np.nan  # invalidate the left-alpha column
    result = _interp_flux_parallel(flux_nan, energy, alpha, targets=[(2.0, 20.0)], it=0)
    assert np.isnan(result[0])


@pytest.mark.basic
def test_interp_flux_parallel_multiple_targets() -> None:
    flux, energy, alpha = _make_arrays()
    targets = [(2.0, 20.0), (2.0, 5.0)]  # first valid, second out-of-range
    result = _interp_flux_parallel(flux, energy, alpha, targets=targets, it=0)
    assert result[0] == pytest.approx(225.0)
    assert np.isnan(result[1])
