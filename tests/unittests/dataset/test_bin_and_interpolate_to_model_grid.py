# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D107
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

import el_paso.dataset.bin_and_interpolate_to_model_grid as bai

# Empty-slice means and 0/0 divisions are part of the current implementation.
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")
T0 = datetime(2013, 1, 1, tzinfo=timezone.utc)

# =====================================================================================
# helpers
# =====================================================================================


def make_grids(
    R_1d: object,
    V_1d: object = (1.0,),
    K_1d: object = (1.0,),
    P_1d: object | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Build (grid_R, grid_V, grid_K, grid_P) with shape (P, R, V, K)."""
    R_1d = np.asarray(R_1d, dtype=float)
    V_1d = np.asarray(V_1d, dtype=float)
    K_1d = np.asarray(K_1d, dtype=float)
    n_P = 1 if P_1d is None else len(P_1d)
    shape = (n_P, len(R_1d), len(V_1d), len(K_1d))

    grid_R = np.broadcast_to(R_1d[None, :, None, None], shape).copy()
    grid_V = np.broadcast_to(V_1d[None, None, :, None], shape).copy()
    grid_K = np.broadcast_to(K_1d[None, None, None, :], shape).copy()
    grid_P = None if P_1d is None else np.broadcast_to(np.asarray(P_1d, dtype=float)[:, None, None, None], shape).copy()
    return grid_R, grid_V, grid_K, grid_P


def minutes(*offsets: float) -> list[datetime]:
    return [T0 + timedelta(minutes=m) for m in offsets]


class StubDataSet:
    """Minimal stand-in for `DataSet` exposing only what the function touches.

    Keeps the unit tests independent of file IO, saving strategies and data
    standards; the real `DataSet` is exercised in `TestRealDataSetIntegration`.
    """

    _INTERNAL = ("InvMu", "InvK", "R_Eq", "L_star")

    def __init__(self, **attrs: object) -> None:
        self.__dict__.update(attrs)

    def get_by_internal_name(self, name: str) -> np.ndarray:
        if name not in self._INTERNAL or name not in self.__dict__:
            msg = f"StubDataSet has no internal variable {name!r}"
            raise ValueError(msg)
        return self.__dict__[name]


def exponential_psd(n_time: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PSD = 10 ** (V + 2*K) sampled on V=[1, 2.2, 3], K=[1, 2].

    `log10(PSD)` is linear in V and in K, which is exactly the family the V-K
    interpolation reproduces, so expected values can be written down by hand.
    The V nodes are deliberately unevenly spaced, so a log-in-V scheme cannot
    pass these tests by accident.
    """
    V = np.broadcast_to(np.array([1.0, 2.2, 3.0])[None, :, None], (n_time, 3, 2)).copy()
    K = np.tile(np.array([1.0, 2.0])[None, :], (n_time, 1))
    psd = 10.0 ** (V + 2 * K[:, None, :])
    return psd, V, K


def interpolate_step(
    grid_K_1d: np.ndarray,
    grid_V: np.ndarray,
    K_data: np.ndarray,
    V_data: np.ndarray,
    psd: np.ndarray,
    it: int = 0,
    max_relative_distance: float = 0.25,
) -> np.ndarray:
    """Call the per-time-step worker with the tolerance in its positional slot."""
    return bai._parallel_func_VK(grid_K_1d, grid_V, K_data, V_data, psd, max_relative_distance, it)


GRID_V_1d = [1.5, 2.5]
GRID_K_1d = [1.5, 1.2]


def expected_psd(V_value: float, K_value: float) -> float:
    return 10.0 ** (V_value + 2 * K_value)


# =====================================================================================
# _linear_interp
# =====================================================================================


@pytest.mark.basic
class TestLinearInterp:
    def test_returns_endpoints(self) -> None:
        assert bai._linear_interp(2.0, 8.0, 0.0, 0.0, 1.0) == pytest.approx(2.0)
        assert bai._linear_interp(2.0, 8.0, 1.0, 0.0, 1.0) == pytest.approx(8.0)

    def test_interpolates_linearly(self) -> None:
        assert bai._linear_interp(2.0, 8.0, 0.25, 0.0, 1.0) == pytest.approx(3.5)

    def test_extrapolates_without_clamping(self) -> None:
        """No bounds check -- callers rely on index guards instead."""
        assert bai._linear_interp(2.0, 8.0, 2.0, 0.0, 1.0) == pytest.approx(14.0)

    def test_is_symmetric_in_endpoint_order(self) -> None:
        forward = bai._linear_interp(2.0, 8.0, 0.25, 0.0, 1.0)
        backward = bai._linear_interp(8.0, 2.0, 0.25, 1.0, 0.0)
        assert forward == pytest.approx(backward)


# =====================================================================================
# _get_time_bins / _get_time_indices
# =====================================================================================


@pytest.mark.basic
class TestTimeBinning:
    def test_bins_are_centred_on_timestamps(self) -> None:
        assert bai._get_time_bins([0.0, 10.0, 20.0]) == [-5.0, 5.0, 15.0, 25.0]

    def test_returns_one_more_edge_than_timestamps(self) -> None:
        stamps = [0.0, 3.0, 6.0, 9.0, 12.0]
        assert len(bai._get_time_bins(stamps)) == len(stamps) + 1

    def test_requires_at_least_two_timestamps(self) -> None:
        with pytest.raises(ValueError, match="At least two time steps"):
            bai._get_time_bins([0.0])

    @pytest.mark.parametrize(
        ("timestamps", "worst_index"),
        [
            ([0.0, 10.0, 100.0], 2),  # gap widens at the end
            ([0.0, 10.0, 20.0, 29.0], 3),  # gap narrows at the end
            ([0.0, 1.0, 11.0, 21.0], 2),  # first interval is the odd one out
        ],
    )
    def test_rejects_non_uniform_spacing(self, timestamps: list[float], worst_index: int) -> None:
        with pytest.raises(ValueError, match="uniformly spaced") as excinfo:
            bai._get_time_bins(timestamps)
        assert f"index {worst_index}" in str(excinfo.value)

    @pytest.mark.parametrize(
        ("timestamp", "expected"),
        [
            (-6.0, -1),  # before the first edge
            (-5.0, 0),  # exactly on the first edge -> included
            (0.0, 0),
            (4.9, 0),
            (5.0, 1),  # bins are left-closed, right-open
            (14.9, 1),
            (24.9, 2),
            (25.0, -1),  # exactly on the last edge -> excluded
            (99.0, -1),  # after the last edge
        ],
    )
    def test_index_assignment(self, timestamp: float, expected: int) -> None:
        bins = bai._get_time_bins([0.0, 10.0, 20.0])
        assert bai._get_time_indices([timestamp], bins)[0] == expected


# =====================================================================================
# _bin_in_time
# =====================================================================================


@pytest.mark.basic
class TestBinInTime:
    @staticmethod
    def _psd(values: object) -> np.ndarray:
        return np.asarray(values, dtype=float).reshape(-1, 1, 1, 1, 1)

    def test_output_shape_follows_sim_time(self) -> None:
        sim_time = minutes(0, 10, 20)
        out = bai._bin_in_time(minutes(0), sim_time, np.full((1, 2, 3, 4, 5), 1.0))
        assert out.shape == (3, 2, 3, 4, 5)

    def test_averages_geometrically_within_a_bin(self) -> None:
        """Averaging happens in log10 space -> geometric, not arithmetic, mean."""
        out = bai._bin_in_time(minutes(-4, 4), minutes(0, 10), self._psd([1.0, 100.0]))
        assert out[0, 0, 0, 0, 0] == pytest.approx(10.0)

    def test_empty_bin_is_nan(self) -> None:
        out = bai._bin_in_time(minutes(0), minutes(0, 10), self._psd([5.0]))
        assert np.isnan(out[1, 0, 0, 0, 0])

    def test_nan_samples_are_ignored_not_propagated(self) -> None:
        out = bai._bin_in_time(minutes(-4, 4), minutes(0, 10), self._psd([np.nan, 100.0]))
        assert out[0, 0, 0, 0, 0] == pytest.approx(100.0)

    def test_samples_outside_the_time_range_are_dropped(self) -> None:
        out = bai._bin_in_time(minutes(0, 60), minutes(0, 10), self._psd([5.0, 1e9]))
        assert out[0, 0, 0, 0, 0] == pytest.approx(5.0)
        assert np.isnan(out[1, 0, 0, 0, 0])

    def test_accepts_datetimes_wrapped_in_arrays(self) -> None:
        wrapped = np.empty(2, dtype=object)
        for i, stamp in enumerate(minutes(-4, 4)):
            wrapped[i] = np.asarray([stamp])
        out = bai._bin_in_time(wrapped, minutes(0, 10), self._psd([1.0, 100.0]))
        assert out[0, 0, 0, 0, 0] == pytest.approx(10.0)


# =====================================================================================
# _bin_in_space
# =====================================================================================


@pytest.mark.basic
class TestBinInSpace:
    R_1d = (3.0, 4.0, 5.0, 6.0)
    P_1d = (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)

    def _run(
        self,
        psd_values: object,
        R_data: object,
        P_data: object | None = None,
        *,
        with_P: bool = True,
    ) -> np.ndarray:
        psd = np.asarray(psd_values, dtype=float).reshape(-1, 1, 1)
        grid_R, _, _, grid_P = make_grids(self.R_1d, P_1d=self.P_1d if with_P else None)
        P_data = np.zeros(psd.shape[0]) if P_data is None else np.asarray(P_data, dtype=float)
        return bai._bin_in_space(psd, P_data, np.asarray(R_data, dtype=float), grid_R, grid_P)

    def test_output_shape_with_azimuthal_grid(self) -> None:
        out = self._run([2.0], [4.0])
        assert out.shape == (1, len(self.P_1d), len(self.R_1d), 1, 1)

    def test_output_shape_without_azimuthal_grid(self) -> None:
        out = self._run([2.0], [4.0], with_P=False)
        assert out.shape == (1, 1, len(self.R_1d), 1, 1)

    def test_time_axis_is_preserved(self) -> None:
        """Space binning scatters each sample into a cell; it does not average in time."""
        out = self._run([2.0, 4.0], [4.0, 4.0])
        assert out.shape[0] == 2
        assert out[0, 0, 1, 0, 0] == pytest.approx(2.0)
        assert out[1, 0, 1, 0, 0] == pytest.approx(4.0)

    def test_sample_lands_in_nearest_r_cell(self) -> None:
        out = self._run([7.0], [4.4])
        assert out[0, 0, 1, 0, 0] == pytest.approx(7.0)
        assert np.isnan(np.delete(out[0, 0, :, 0, 0], 1)).all()

    def test_azimuth_wraps_around_2pi(self) -> None:
        """P just below 2*pi is closest to the P=0 cell, not to the 3*pi/2 cell."""
        out = self._run([7.0], [4.0], P_data=[2 * np.pi - 0.1])
        assert out[0, 0, 1, 0, 0] == pytest.approx(7.0)

    @pytest.mark.parametrize(("R_value", "kept"), [(3.5, True), (3.49, False), (5.5, True), (5.51, False)])
    def test_half_a_cell_of_margin_is_required_at_the_grid_edges(self, R_value: float, *, kept: bool) -> None:
        out = self._run([7.0], [R_value])
        assert bool(np.any(np.isfinite(out))) is kept

    def test_all_nan_sample_is_skipped(self) -> None:
        out = self._run([np.nan], [4.0])
        assert np.isnan(out).all()

    def test_partially_nan_sample_keeps_finite_entries(self) -> None:
        psd = np.array([[[2.0, np.nan]]])
        grid_R, _, _, _ = make_grids(self.R_1d)
        out = bai._bin_in_space(psd, np.zeros(1), np.array([4.0]), grid_R, None)
        assert out[0, 0, 1, 0, 0] == pytest.approx(2.0)
        assert np.isnan(out[0, 0, 1, 0, 1])


# =====================================================================================
# _parallel_func_VK / _interpolate_in_V_K
# =====================================================================================


class TestInterpolateOrBinInVK:
    def test_reproduces_an_exponential_exactly(self) -> None:
        psd, V, K = exponential_psd(1)
        _, grid_V, grid_K, _ = make_grids([4.0], V_1d=GRID_V_1d, K_1d=GRID_K_1d)

        out = interpolate_step(grid_K[0, 0, 0, :], grid_V, K, V, psd)

        expected = np.array([[expected_psd(V_val, K_val) for K_val in GRID_K_1d] for V_val in GRID_V_1d])
        np.testing.assert_allclose(out, expected, rtol=1e-10)

    def test_midpoint_in_V_is_the_geometric_mean_of_its_neighbours(self) -> None:
        """The defining property: linear in log10(PSD), linear in V (not in log V)."""
        V = np.array([[[1.0], [2.2], [3.0]]])
        K = np.array([[1.0, 2.0]])
        V = np.concatenate([V, V], axis=2)
        psd = np.array([[[1.0e-3, 1.0e-3], [1.0e3, 1.0e3], [1.0e5, 1.0e5]]])
        _, grid_V, grid_K, _ = make_grids([4.0], V_1d=[(1.0 + 2.2) / 2], K_1d=[1.5])

        out = interpolate_step(grid_K[0, 0, 0, :], grid_V, K, V, psd)

        assert out[0, 0] == pytest.approx(np.sqrt(1.0e-3 * 1.0e3))

    def test_handles_descending_K(self) -> None:
        """K is stored descending for some missions; the result must not change."""
        psd_asc, V, K_asc = exponential_psd(1)
        _, grid_V, grid_K, _ = make_grids([4.0], V_1d=GRID_V_1d, K_1d=GRID_K_1d)
        ascending = interpolate_step(grid_K[0, 0, 0, :], grid_V, K_asc, V, psd_asc)

        psd_desc = psd_asc[:, :, ::-1]
        K_desc = K_asc[:, ::-1]
        V_desc = V[:, :, ::-1]
        descending = interpolate_step(grid_K[0, 0, 0, :], grid_V, K_desc, V_desc, psd_desc)

        np.testing.assert_allclose(descending, ascending, rtol=1e-10)

    def test_grid_point_outside_the_V_range_is_nan(self) -> None:
        psd, V, K = exponential_psd(1)
        _, grid_V, grid_K, _ = make_grids([4.0], V_1d=[0.5, 5.0], K_1d=[1.5])
        out = interpolate_step(grid_K[0, 0, 0, :], grid_V, K, V, psd)
        assert np.isnan(out).all()

    def test_grid_point_outside_the_K_range_is_nan(self) -> None:
        psd, V, K = exponential_psd(1)
        _, grid_V, grid_K, _ = make_grids([4.0], V_1d=[1.5], K_1d=[0.5, 5.0])
        out = interpolate_step(grid_K[0, 0, 0, :], grid_V, K, V, psd)
        assert np.isnan(out).all()

    def test_all_nan_K_row_yields_all_nan(self) -> None:
        psd, V, K = exponential_psd(1)
        K = np.full_like(K, np.nan)
        _, grid_V, grid_K, _ = make_grids([4.0], V_1d=GRID_V_1d, K_1d=GRID_K_1d)

        out = interpolate_step(grid_K[0, 0, 0, :], grid_V, K, V, psd)

        assert np.isnan(out).all()

    def test_nan_in_one_V_column_does_not_break_the_other_bracket(self) -> None:
        """Regression: the sort direction must ignore NaNs in *both* K columns.

        A NaN anywhere in the right-hand column used to make the column look
        descending, which sent the index search the wrong way and NaN-ed out every
        grid point at that K -- even when the interpolation bracket itself was finite.
        """
        psd, V, K = exponential_psd(1)
        V_with_nan = V.copy()
        V_with_nan[0, -1, 1] = np.nan  # outside the bracket used for V_val = 1.5
        _, grid_V, grid_K, _ = make_grids([4.0], V_1d=[1.5], K_1d=[1.5])

        out = interpolate_step(grid_K[0, 0, 0, :], grid_V, K, V_with_nan, psd)

        assert out[0, 0] == pytest.approx(expected_psd(1.5, 1.5))

    @staticmethod
    def _single_observation(
        V_values: object, K_values: object, psd_values: object
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        V_values = np.asarray(V_values, dtype=float)
        K_values = np.asarray(K_values, dtype=float)
        psd = np.asarray(psd_values, dtype=float).reshape(1, len(V_values), len(K_values))
        V = np.broadcast_to(V_values[None, :, None], psd.shape).copy()
        K = K_values[None, :].copy()
        return psd, V, K

    def test_single_K_observation_is_binned_to_the_nearest_grid_point(self) -> None:
        """One K value cannot be bracketed, so the observation is used as it is."""
        psd, V, K = self._single_observation([1.0, 3.0], [2.0], [[7.0], [700.0]])
        _, grid_V, grid_K, _ = make_grids([4.0], V_1d=[2.0], K_1d=[2.2])

        out = interpolate_step(grid_K[0, 0, 0, :], grid_V, K, V, psd)

        # binned in K, still interpolated in V: geometric mean of 7 and 700 at V=2
        assert out[0, 0] == pytest.approx(np.sqrt(7.0 * 700.0))

    def test_single_V_observation_is_binned_to_the_nearest_grid_point(self) -> None:
        psd, V, K = self._single_observation([2.0], [1.0, 3.0], [[7.0, 700.0]])
        _, grid_V, grid_K, _ = make_grids([4.0], V_1d=[2.2], K_1d=[2.0])

        out = interpolate_step(grid_K[0, 0, 0, :], grid_V, K, V, psd)

        assert out[0, 0] == pytest.approx(np.sqrt(7.0 * 700.0))

    def test_single_observation_in_both_dimensions_is_used_verbatim(self) -> None:
        psd, V, K = self._single_observation([2.0], [2.0], [[7.0]])
        _, grid_V, grid_K, _ = make_grids([4.0], V_1d=[2.2], K_1d=[2.2])

        out = interpolate_step(grid_K[0, 0, 0, :], grid_V, K, V, psd)

        assert out[0, 0] == pytest.approx(7.0)

    @pytest.mark.parametrize(
        ("grid_value", "is_used"),
        [(2.0, True), (2.5, True), (2.50001, False), (1.5, True), (1.49999, False), (4.0, False)],
    )
    def test_nearest_binning_respects_the_distance_tolerance(self, grid_value: float, *, is_used: bool) -> None:
        """The observation at V=K=2 may be moved by at most 25% of its own value."""
        psd, V, K = self._single_observation([2.0], [2.0], [[7.0]])
        _, grid_V, grid_K, _ = make_grids([4.0], V_1d=[grid_value], K_1d=[grid_value])

        out = interpolate_step(grid_K[0, 0, 0, :], grid_V, K, V, psd)

        assert np.isfinite(out[0, 0]) == is_used

    @pytest.mark.parametrize(("percent", "is_used"), [(5.0, False), (60.0, True)])
    def test_distance_tolerance_is_configurable(self, percent: float, *, is_used: bool) -> None:
        psd, V, K = self._single_observation([2.0], [2.0], [[7.0]])
        _, grid_V, grid_K, _ = make_grids([4.0], V_1d=[3.0], K_1d=[3.0])

        out = interpolate_step(grid_K[0, 0, 0, :], grid_V, K, V, psd, max_relative_distance=percent / 100)

        assert np.isfinite(out[0, 0]) == is_used

    def test_a_column_reduced_to_one_finite_value_is_binned(self) -> None:
        """The fallback keys off usable observations, not just axis length.

        A V axis of length three with two NaNs behaves like a single observation.
        """
        psd, V, K = self._single_observation([1.0, 2.0, 3.0], [2.0], [[np.nan], [7.0], [np.nan]])
        V[0, 0, 0] = np.nan
        V[0, 2, 0] = np.nan
        _, grid_V, grid_K, _ = make_grids([4.0], V_1d=[2.2], K_1d=[2.0])

        out = interpolate_step(grid_K[0, 0, 0, :], grid_V, K, V, psd)

        assert out[0, 0] == pytest.approx(7.0)

    def test_rejects_a_negative_distance_tolerance(self) -> None:
        psd, V, K = exponential_psd(2)
        _, grid_V, grid_K, _ = make_grids([4.0], V_1d=GRID_V_1d, K_1d=GRID_K_1d)

        with pytest.raises(ValueError, match="max_relative_distance_percent must not be negative"):
            bai._interpolate_or_bin_in_V_K(psd, V, K, grid_V, grid_K, n_processes=1, max_relative_distance_percent=-1.0)

    @pytest.mark.parametrize("n_processes", [0, -1])
    def test_rejects_a_non_positive_process_count(self, n_processes: int) -> None:
        psd, V, K = exponential_psd(2)
        _, grid_V, grid_K, _ = make_grids([4.0], V_1d=GRID_V_1d, K_1d=GRID_K_1d)

        with pytest.raises(ValueError, match="n_processes must be at least 1"):
            bai._interpolate_or_bin_in_V_K(psd, V, K, grid_V, grid_K, n_processes=n_processes)

    def test_process_count_does_not_change_the_result(self) -> None:
        psd, V, K = exponential_psd(6)
        psd = psd * np.arange(1, 7)[:, None, None]
        _, grid_V, grid_K, _ = make_grids([4.0], V_1d=GRID_V_1d, K_1d=GRID_K_1d)

        serial = bai._interpolate_or_bin_in_V_K(psd, V, K, grid_V, grid_K, n_processes=1)
        parallel = bai._interpolate_or_bin_in_V_K(psd, V, K, grid_V, grid_K, n_processes=3)

        np.testing.assert_array_equal(serial, parallel)

    def test_pool_result_matches_serial_evaluation(self) -> None:
        n_time = 4
        psd, V, K = exponential_psd(n_time)
        psd = psd * np.array([1.0, 2.0, 3.0, 4.0])[:, None, None]
        _, grid_V, grid_K, _ = make_grids([4.0], V_1d=GRID_V_1d, K_1d=GRID_K_1d)

        parallel = bai._interpolate_or_bin_in_V_K(psd, V, K, grid_V, grid_K)
        serial = np.asarray([interpolate_step(grid_K[0, 0, 0, :], grid_V, K, V, psd, it) for it in range(n_time)])

        assert parallel.shape == (n_time, grid_V.shape[2], grid_V.shape[3])
        np.testing.assert_allclose(parallel, serial, rtol=1e-12)


# =====================================================================================
# bin_and_interpolate_to_model_grid -- end to end
# =====================================================================================


@pytest.mark.basic
class TestBinAndInterpolateEndToEnd:
    def test_plasmasphere_path_bins_density_in_P_R_and_time(self) -> None:
        """1-D variable, singleton V/K grid, azimuthal grid present -> pure binning."""
        P_1d = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
        grid_R, grid_V, grid_K, grid_P = make_grids([3.0, 4.0, 5.0, 6.0], P_1d=P_1d)
        data_time = minutes(0, 5, 10, 15)
        sim_time = minutes(0, 10)

        dataset = StubDataSet(
            density=np.array([10.0, 1000.0, 20.0, 20.0]),
            P=np.array([0.1, 0.1, np.pi, np.pi]),
            R_Eq=np.array([4.0, 4.0, 5.0, 5.0]),
            datetime=data_time,
        )

        out = bai.bin_and_interpolate_to_model_grid(
            dataset, sim_time, grid_R, grid_V, grid_K, grid_P=grid_P, target_var_name="density"
        )

        assert out.shape == (2, 4, 4, 1, 1)
        assert out[0, 0, 1, 0, 0] == pytest.approx(10.0)  # sample 0 -> P=0,  R=4
        assert out[1, 0, 1, 0, 0] == pytest.approx(1000.0)  # sample 1 -> P=0,  R=4
        assert out[1, 2, 2, 0, 0] == pytest.approx(20.0)  # sample 2 -> P=pi, R=5
        # sample 3 sits on the last time-bin edge and is dropped
        assert np.count_nonzero(np.isfinite(out)) == 3

    def test_psd_path_interpolates_then_bins_by_lstar(self) -> None:
        """No azimuthal grid -> R comes from L_star[:, -1] and the P axis is a singleton."""
        n_time = 3
        psd, V, K = exponential_psd(n_time)
        grid_R, grid_V, grid_K, _ = make_grids([3.0, 4.0, 5.0, 6.0], V_1d=GRID_V_1d, K_1d=GRID_K_1d)
        dataset = StubDataSet(
            PSD=psd,
            InvV=V,
            InvK=K,
            P=np.zeros(n_time),
            L_star=np.column_stack([np.full(n_time, 9.9), np.array([4.0, 4.0, 5.0])]),
            datetime=minutes(0, 5, 10),
        )

        out = bai.bin_and_interpolate_to_model_grid(dataset, minutes(0, 10), grid_R[0], grid_V[0], grid_K[0])

        assert out.shape == (2, 1, 4, 2, 2)
        assert out[0, 0, 1, 0, 0] == pytest.approx(expected_psd(GRID_V_1d[0], GRID_K_1d[0]))
        assert out[1, 0, 2, 1, 1] == pytest.approx(expected_psd(GRID_V_1d[1], GRID_K_1d[1]))
        assert np.isnan(out[:, 0, 0, :, :]).all()
        assert np.isnan(out[:, 0, 3, :, :]).all()

    def test_three_dimensional_grids_are_promoted_to_four_dimensions(self) -> None:
        grid_R, grid_V, grid_K, _ = make_grids([3.0, 4.0, 5.0, 6.0])
        dataset = StubDataSet(
            density=np.array([10.0]),
            P=np.zeros(1),
            L_star=np.full((1, 2), 4.0),
            datetime=minutes(0),
        )
        kwargs = {"target_var_name": "density"}

        from_3d = bai.bin_and_interpolate_to_model_grid(
            dataset, minutes(0, 10), grid_R[0], grid_V[0], grid_K[0], **kwargs
        )
        from_4d = bai.bin_and_interpolate_to_model_grid(dataset, minutes(0, 10), grid_R, grid_V, grid_K, **kwargs)

        np.testing.assert_array_equal(np.nan_to_num(from_3d, nan=-1), np.nan_to_num(from_4d, nan=-1))

    def test_mu_or_V_switch_selects_V_coordinate(self) -> None:
        """`mu_or_V='Mu'` must read InvMu, `'V'` must read InvV."""
        n_time = 2
        psd, V, K = exponential_psd(n_time)
        grid_R, grid_V, grid_K, _ = make_grids([3.0, 4.0, 5.0, 6.0], V_1d=GRID_V_1d, K_1d=GRID_K_1d)
        dataset = StubDataSet(
            PSD=psd,
            InvV=V,
            InvMu=V * 1e6,  # far outside the model grid -> everything must be NaN
            InvK=K,
            P=np.zeros(n_time),
            L_star=np.full((n_time, 2), 4.0),
            datetime=minutes(0, 5),
        )

        with_V = bai.bin_and_interpolate_to_model_grid(dataset, minutes(0, 10), grid_R, grid_V, grid_K, mu_or_V="V")
        with_Mu = bai.bin_and_interpolate_to_model_grid(dataset, minutes(0, 10), grid_R, grid_V, grid_K, mu_or_V="Mu")

        assert np.any(np.isfinite(with_V))
        assert np.isnan(with_Mu).all()

    def test_singleton_V_grid_skips_interpolation(self) -> None:
        """grid_V of size 1 while grid_K is larger -> data passed through untouched."""
        n_time = 1
        psd, V, K = exponential_psd(n_time)
        grid_R, grid_V, grid_K, _ = make_grids([3.0, 4.0, 5.0, 6.0], V_1d=[1.5], K_1d=GRID_K_1d)
        dataset = StubDataSet(
            PSD=psd,
            InvV=V,
            InvK=K,
            P=np.zeros(n_time),
            L_star=np.full((n_time, 2), 4.0),
            datetime=minutes(0),
        )

        out = bai.bin_and_interpolate_to_model_grid(dataset, minutes(0, 10), grid_R, grid_V, grid_K)

        # V/K axes keep the *data* dimensions (3 x 2), not the model ones
        assert out.shape == (2, 1, 4, 3, 2)
        np.testing.assert_allclose(out[0, 0, 1, :, :], psd[0], rtol=1e-10)

    def test_options_are_forwarded_to_the_interpolation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_interpolate(  # noqa: ANN202
            psd_in,  # noqa: ANN001
            _V,  # noqa: ANN001
            _K,  # noqa: ANN001
            grid_V,  # noqa: ANN001
            grid_K,  # noqa: ANN001
            n_processes=None,  # noqa: ANN001
            max_relative_distance_percent=None,  # noqa: ANN001
        ):
            captured["n_processes"] = n_processes
            captured["max_relative_distance_percent"] = max_relative_distance_percent
            return np.full((psd_in.shape[0], grid_V.shape[2], grid_K.shape[3]), 1.0e5)

        monkeypatch.setattr(bai, "_interpolate_or_bin_in_V_K", fake_interpolate)

        n_time = 2
        psd, V, K = exponential_psd(n_time)
        grid_R, grid_V, grid_K, _ = make_grids([3.0, 4.0, 5.0, 6.0], V_1d=GRID_V_1d, K_1d=GRID_K_1d)
        dataset = StubDataSet(
            PSD=psd,
            InvV=V,
            InvK=K,
            P=np.zeros(n_time),
            L_star=np.full((n_time, 2), 4.0),
            datetime=minutes(0, 5),
        )

        bai.bin_and_interpolate_to_model_grid(
            dataset,
            minutes(0, 10),
            grid_R,
            grid_V,
            grid_K,
            n_processes=3,
            max_relative_distance_percent=10.0,
        )

        assert captured["n_processes"] == 3
        assert captured["max_relative_distance_percent"] == 10.0

    def test_irregular_sim_time_is_rejected(self) -> None:
        """The uniformity error must surface from the top-level call, not be swallowed."""
        grid_R, grid_V, grid_K, _ = make_grids([3.0, 4.0, 5.0, 6.0])
        dataset = StubDataSet(
            density=np.array([10.0, 20.0]),
            P=np.zeros(2),
            L_star=np.full((2, 2), 4.0),
            datetime=minutes(0, 5),
        )

        with pytest.raises(ValueError, match="uniformly spaced"):
            bai.bin_and_interpolate_to_model_grid(
                dataset,
                minutes(0, 10, 45),
                grid_R,
                grid_V,
                grid_K,
                target_var_name="density",
            )


# =====================================================================================
# sanity checks
# =====================================================================================


@pytest.mark.basic
class TestSanityChecks:
    @staticmethod
    def _dataset() -> StubDataSet:
        return StubDataSet(
            density=np.array([10.0, 20.0]),
            P=np.zeros(2),
            L_star=np.full((2, 2), 4.0),
            datetime=minutes(0, 5),
        )

    @staticmethod
    def _call(dataset: StubDataSet) -> np.ndarray:
        grid_R, grid_V, grid_K, _ = make_grids([3.0, 4.0, 5.0, 6.0])
        return bai.bin_and_interpolate_to_model_grid(
            dataset, minutes(0, 10), grid_R, grid_V, grid_K, target_var_name="density"
        )

    def test_space_binning_out_of_range_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(bai, "_bin_in_space", lambda *_, **__: np.full((2, 1, 4, 1, 1), 1e9))
        with pytest.raises(ValueError, match="inconsitency in space binning"):
            self._call(self._dataset())

    def test_time_binning_out_of_range_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(bai, "_bin_in_time", lambda *_, **__: np.full((2, 1, 4, 1, 1), 1e-9))
        with pytest.raises(ValueError, match="inconsitency in time binning"):
            self._call(self._dataset())

    def test_vk_interpolation_out_of_range_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        n_time = 1
        psd, V, K = exponential_psd(n_time)
        grid_R, grid_V, grid_K, _ = make_grids([3.0, 4.0, 5.0, 6.0], V_1d=GRID_V_1d, K_1d=GRID_K_1d)
        dataset = StubDataSet(
            PSD=psd,
            InvV=V,
            InvK=K,
            P=np.zeros(n_time),
            L_star=np.full((n_time, 2), 4.0),
            datetime=minutes(0),
        )
        monkeypatch.setattr(bai, "_interpolate_or_bin_in_V_K", lambda *_, **__: np.full((1, 2, 2), 1e9))

        with pytest.raises(ValueError, match="inconsitency in V-K interpolation"):
            bai.bin_and_interpolate_to_model_grid(dataset, minutes(0, 10), grid_R, grid_V, grid_K)

    def test_valid_pipeline_does_not_raise(self) -> None:
        assert np.any(np.isfinite(self._call(self._dataset())))


@pytest.mark.basic
def test_full_pipeline_against_hard_coded_values() -> None:
    """One pass through all three stages, with every expected number written out.

    The input is exponential in V and K (`log10(PSD) = V + 2*K + offset`), so every
    stage stays exact in log space: the V-K interpolation lands on `V + 2*K`, and
    the geometric mean over a time bin averages the offsets. Expected values are
    therefore given as exponents.

    Exercised in one go: interpolation onto the model V-K grid, the azimuth wrap at
    2*pi, the half-cell margin at the R edges, and geometric averaging in time.
    """
    n_time = 4
    _, V, K = exponential_psd(n_time)
    offsets = np.array([0.0, 1.0, 3.0, 0.0])
    psd = 10.0 ** (V + 2 * K[:, None, :] + offsets[:, None, None])

    grid_R, grid_V, grid_K, grid_P = make_grids(
        [3.0, 4.0, 5.0, 6.0],
        V_1d=GRID_V_1d,
        K_1d=GRID_K_1d,
        P_1d=[0.0, np.pi / 2, np.pi, 3 * np.pi / 2],
    )

    dataset = StubDataSet(
        PSD=psd,
        InvV=V,
        InvK=K,
        # sample 1 sits just below 2*pi and must wrap into the P=0 cell;
        # sample 3 is inside the R range but within half a cell of its inner edge,
        # so it must be dropped
        P=np.array([0.1, 2 * np.pi - 0.1, 0.05, 0.0]),
        R_Eq=np.array([4.0, 4.0, 4.2, 3.2]),
        datetime=minutes(0, 5, 10, 15),
    )

    out = bai.bin_and_interpolate_to_model_grid(
        dataset, minutes(0, 10, 20), grid_R, grid_V, grid_K, grid_P=grid_P, n_processes=1
    )

    assert out.shape == (3, 4, 4, 2, 2)

    # V + 2*K on the model grid, for V in (1.5, 2.5) and K in (1.5, 1.2)
    interpolated_exponents = np.array([[4.5, 3.9], [5.5, 4.9]])

    # time bin 0 holds sample 0 alone (offset 0)
    np.testing.assert_allclose(np.log10(out[0, 0, 1, :, :]), interpolated_exponents, rtol=1e-12)
    # time bin 1 holds samples 1 and 2 in the same cell -> offsets 1 and 3 average to 2
    np.testing.assert_allclose(np.log10(out[1, 0, 1, :, :]), interpolated_exponents + 2.0, rtol=1e-12)
    # time bin 2 would hold sample 3, which was dropped in space
    assert np.isnan(out[2]).all()

    # nothing landed anywhere else: two cells, four V-K points each
    assert np.count_nonzero(np.isfinite(out)) == 8
