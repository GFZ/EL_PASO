# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from icecream import ic
from matplotlib import pyplot as plt
from tqdm import tqdm

if TYPE_CHECKING:
    from datetime import datetime

    from numpy.typing import NDArray

    from el_paso.dataset import DataSet


def bin_and_interpolate_to_model_grid(
    self: DataSet,
    sim_time: list[datetime],
    grid_R: NDArray[np.float64],
    grid_mu_V: NDArray[np.float64],
    grid_K: NDArray[np.float64],
    grid_P: NDArray[np.float64] | None = None,
    debug_plot_settings: DebugPlotSettings | None = None,
    target_var_name: Literal["PSD", "density"] = "PSD",
    mu_or_V: Literal["Mu", "V"] = "V",
    n_processes: int | None = None,
    max_relative_distance_percent: float = 25.0,
) -> NDArray[np.float64]:
    """Bin and interpolate a dataset variable onto a model's grid and time axis.

    The variable named by `target_var_name` (e.g. ``PSD``) is processed in three
    steps: (1) it is interpolated onto the model's V-K (or Mu-K) grid, (2) it is
    binned in space onto the model's R (or L*) and, if `grid_P` is given, P/MLT
    grid, and (3) it is binned in time onto `sim_time`. After each step, a sanity
    check ensures the resulting values stay within the range of the original data.

    Args:
        self (RBMDataSet): The RBMDataSet instance this method operates on.
        sim_time (list[datetime]): Target simulation time steps to bin the data onto.
        grid_R (NDArray[np.float64]): Radial distance (R or L*) grid of the model.
        grid_mu_V (NDArray[np.float64]): Mu or V grid of the model, matching `mu_or_V`.
        grid_K (NDArray[np.float64]): Second adiabatic invariant (K) grid of the model.
        grid_P (NDArray[np.float64] | None, optional): Azimuthal (P/MLT) grid of the
            model. If `None`, the data is binned by R (or L*) only, as for a
            plasmasphere model. Defaults to `None`.
        debug_plot_settings (DebugPlotSettings | None, optional): If provided, debug
            figures are generated for each simulation time step. Defaults to `None`.
        target_var_name (Literal["PSD", "density"], optional): Name of the dataset
            attribute to bin and interpolate. Defaults to `"PSD"`.
        mu_or_V (Literal["Mu", "V"], optional): Whether to use `InvMu` or `InvV` as the
            velocity-space coordinate when interpolating onto `grid_mu_V`. Defaults to `"V"`.
        n_processes (int | None, optional): Number of worker processes used for the
            V-K step. `1` runs in the calling process, which keeps tracebacks intact
            and suits debugging or short time series. Defaults to `None`, meaning the
            number of CPUs available to this process.
        max_relative_distance_percent (float, optional): How far a single observation
            may be moved when it is binned onto the nearest V or K grid point, as a
            percentage of the observed value itself. Applies only where a dimension
            holds one usable observation and interpolation is impossible.
            Defaults to `25.0`.

    Returns:
        NDArray[np.float64]: The values of `target_var_name`, binned and interpolated
        onto `sim_time` and the provided model grids.

    Raises:
        ValueError: If the V-K interpolation, the spatial binning, or the temporal
            binning produces values outside the range of the original data,
            indicating an inconsistency in the binning/interpolation.
    """
    # make sure everything is 4D

    if grid_R.ndim == 3:
        grid_R = grid_R[np.newaxis, ...]
    if grid_mu_V.ndim == 3:
        grid_mu_V = grid_mu_V[np.newaxis, ...]
    if grid_K.ndim == 3:
        grid_K = grid_K[np.newaxis, ...]

    target_var_init = getattr(self, target_var_name)

    # 1. interpolate to V-K

    if grid_R.shape[2] > 1 and grid_R.shape[3] > 1:
        if target_var_init.ndim == 1:
            target_var_init = target_var_init[:, np.newaxis, np.newaxis]

        mu_or_V_arr = self.get_by_internal_name("InvMu") if mu_or_V == "Mu" else self.InvV
        if grid_mu_V.shape[2] > 1:
            psd_interp = _interpolate_or_bin_in_V_K(
                target_var_init,
                mu_or_V_arr,
                self.get_by_internal_name("InvK"),
                grid_mu_V,
                grid_K,
                n_processes,
                max_relative_distance_percent,
            )
        else:
            psd_interp = target_var_init

        # sanity check
        if np.min(target_var_init) > np.min(psd_interp) or np.max(target_var_init) < np.max(psd_interp):
            msg = "Found inconsitency in V-K interpolation. Aborting..."
            raise (ValueError(msg))
    else:
        if target_var_init.ndim == 1:  # plasmasphere
            target_var_init = target_var_init[:, np.newaxis, np.newaxis]

        psd_interp = target_var_init

    # 2. Bin in space

    R_or_Lstar_arr = (
        self.get_by_internal_name("R_Eq") if grid_P is not None else self.get_by_internal_name("L_star")[:, -1]
    )

    psd_binned_in_space = _bin_in_space(psd_interp, self.P, R_or_Lstar_arr, grid_R, grid_P)
    # sanity check
    if np.min(target_var_init) > np.min(psd_binned_in_space) or np.max(target_var_init) < np.max(psd_binned_in_space):
        msg = "Found inconsitency in space binning. Aborting..."
        raise (ValueError(msg))

    # 3. Bin in time
    psd_binned_in_time = _bin_in_time(self.datetime, sim_time, psd_binned_in_space)  # ty:ignore[invalid-argument-type]
    # sanity check
    if np.min(target_var_init) > np.min(psd_binned_in_time) or np.max(target_var_init) < np.max(psd_binned_in_time):
        msg = "Found inconsitency in time binning. Aborting..."
        raise (ValueError(msg))

    if debug_plot_settings:
        if debug_plot_settings.target_K is not None:
            plot_debug_figures(
                self,
                psd_binned_in_time,
                sim_time,  # ty:ignore[invalid-argument-type]
                grid_P,
                grid_R,
                grid_mu_V,
                grid_K,
                mu_or_V,
                debug_plot_settings,
            )
        else:
            plot_debug_figures_plasmasphere(
                self,
                psd_binned_in_time,
                sim_time,  # ty:ignore[invalid-argument-type]
                grid_P,
                grid_R,
                debug_plot_settings,
            )

    return psd_binned_in_time


def _get_time_bins(timestamps: list[float]) -> list[float]:

    if len(timestamps) < 2:
        msg = f"At least two time steps are required to determine the bin width, got {len(timestamps)}."
        raise ValueError(msg)

    diffs = np.diff(np.asarray(timestamps, dtype=float))
    dt = diffs[0]

    if not np.allclose(diffs, dt, rtol=1e-6, atol=0.0):
        worst = int(np.argmax(np.abs(diffs - dt)))
        msg = (
            "Time steps must be uniformly spaced to be binned, but the spacing changes from "
            f"{dt} to {diffs[worst]} at index {worst + 1}."
        )
        raise ValueError(msg)

    bins = [timestamps[0] - dt / 2]
    for i in range(len(timestamps)):
        bins.append(bins[i] + dt)

    return bins


def _get_time_indices(data_timestamps: list[float], time_bins: list[float]) -> NDArray[np.float32]:
    time_indices = np.digitize(data_timestamps, time_bins)
    time_indices = time_indices - 1
    time_indices = np.where(time_indices == len(time_bins) - 1, -1, time_indices)

    return time_indices


def _bin_in_time(
    data_time: NDArray[np.object_],
    sim_time: NDArray[np.object_],
    data_psd: NDArray[np.float64],
) -> NDArray[np.float64]:
    psd_binned = np.full(
        (
            len(sim_time),
            data_psd.shape[1],
            data_psd.shape[2],
            data_psd.shape[3],
            data_psd.shape[4],
        ),
        np.nan,
    )

    if isinstance(data_time[0], np.ndarray):
        data_time = np.asarray([t[0] for t in data_time])  # ty:ignore[invalid-assignment]

    sim_timestamps = [t.timestamp() for t in sim_time]
    data_timestamps = [t.timestamp() for t in data_time]
    time_indices = _get_time_indices(data_timestamps, _get_time_bins(sim_timestamps))

    for i, _ in tqdm(enumerate(sim_time)):
        psd_binned[i, ...] = np.power(10, np.nanmean(np.log10(data_psd[time_indices == i, ...]), axis=0))

    return psd_binned


def _nearest_azimuth_index(P_value: float, grid_P_1d: NDArray[np.float64]) -> int:
    """Index of the grid azimuth closest to `P_value`, measured across the 2*pi wrap."""
    difference = np.abs(P_value - grid_P_1d)
    return int(np.argmin(np.minimum(difference, 2 * np.pi - difference)))


def _bin_in_space(
    psd_in: NDArray[np.float64],
    P_data: NDArray[np.float64],
    R_data: NDArray[np.float64],
    grid_R: NDArray[np.float64],
    grid_P: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Scatter each sample onto the (P, R) cell of the model grid it falls into.

    The sample axis is carried through untouched, so every output slice receives at
    most one sample and no averaging happens here -- samples sharing a cell are
    averaged later, per simulation time step, by `_bin_in_time`.

    Samples closer than half a cell to either end of the R grid are dropped, so a
    sample is never assigned to a cell that would extend beyond the model domain.

    Args:
        psd_in: Values to bin, shaped (sample, V, K).
        P_data: Azimuth of each sample, in radians. Ignored if `grid_P` is `None`.
        R_data: Radial distance (R or L*) of each sample.
        grid_R: Radial grid of the model, shaped (P, R, V, K).
        grid_P: Azimuthal grid of the model, shaped like `grid_R`. If `None`, the
            output keeps a singleton azimuth axis and `P_data` is not used.

    Returns:
        The values of `psd_in`, placed on the model grid and shaped
        (sample, P, R, V, K). Cells without a sample are `NaN`.
    """
    grid_R_1d = grid_R[0, :, 0, 0]
    grid_P_1d = None if grid_P is None else grid_P[:, 0, 0, 0]
    n_P = 1 if grid_P is None else grid_P.shape[0]

    psd_binned = np.full(
        (psd_in.shape[0], n_P, grid_R.shape[1], psd_in.shape[1], psd_in.shape[2]),
        np.nan,
    )

    dR = grid_R_1d[1] - grid_R_1d[0]

    for it in range(psd_in.shape[0]):
        if R_data[it] - dR / 2 < grid_R_1d[0] or R_data[it] + dR / 2 > grid_R_1d[-1]:
            continue

        r_idx = int(np.argmin(np.abs(R_data[it] - grid_R_1d)))
        p_idx = 0 if grid_P_1d is None else _nearest_azimuth_index(P_data[it], grid_P_1d)

        psd_binned[it, p_idx, r_idx, :, :] = psd_in[it, :, :]

    return psd_binned


def _interpolate_or_bin_in_V_K(
    psd_in: NDArray[np.float64],
    V_data: NDArray[np.float64],
    K_data: NDArray[np.float64],
    grid_V: NDArray[np.float64],
    grid_K: NDArray[np.float64],
    n_processes: int | None = None,
    max_relative_distance_percent: float = 25.0,
) -> NDArray[np.float64]:
    """Move `psd_in` onto the model's V-K grid, one time step per task.

    Where a dimension holds two or more usable observations the values are
    interpolated; where it holds only one they are binned onto the nearest grid
    point, since a single observation cannot be bracketed. Data sets with a single
    energy channel or a single pitch angle therefore still contribute, instead of
    dropping out entirely.

    Args:
        psd_in: Values to place on the grid, shaped (time, V, K).
        V_data: V (or Mu) of each data point, shaped like `psd_in`.
        K_data: K of each data point, shaped (time, K).
        grid_V: V grid of the model, shaped (P, R, V, K).
        grid_K: K grid of the model, shaped like `grid_V`.
        n_processes: Number of worker processes. `1` runs in the calling process,
            which keeps tracebacks intact and is the better choice for debugging or
            for short time series. Defaults to the number of CPUs available to this
            process.
        max_relative_distance_percent: How far a single observation may be moved
            when it is binned, as a percentage of the observed V or K itself.
            Observations further from every grid point than this are discarded.
            Has no effect where interpolation is possible. Defaults to `25.0`.

    Returns:
        The values of `psd_in` on the model's V-K grid, shaped (time, V, K).

    Raises:
        ValueError: If `n_processes` is smaller than 1, or if
            `max_relative_distance_percent` is negative.
    """
    if n_processes is None:
        n_processes = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    if n_processes < 1:
        msg = f"n_processes must be at least 1, got {n_processes}."
        raise ValueError(msg)
    if max_relative_distance_percent < 0:
        msg = f"max_relative_distance_percent must not be negative, got {max_relative_distance_percent}."
        raise ValueError(msg)

    grid_K_1d = grid_K[0, 0, 0, :]
    func = partial(
        _parallel_func_VK,
        grid_K_1d,
        grid_V,
        K_data,
        V_data,
        psd_in,
        max_relative_distance_percent / 100,
    )
    time_steps = range(psd_in.shape[0])

    if n_processes == 1:
        return np.asarray([func(it) for it in tqdm(time_steps)])

    with Pool(n_processes) as p:
        return np.asarray(list(tqdm(p.imap(func, time_steps), total=len(time_steps))))


def _sort_direction(values: NDArray[np.float64]) -> int:
    """+1 if the finite entries ascend, -1 otherwise.

    `searchsorted` needs an ascending array, so descending axes (K is stored
    descending for some missions) are searched after multiplying by -1.
    """
    finite = np.isfinite(values)
    return 1 if np.all(np.diff(values[finite]) >= 0) else -1


def _bracket_indices(
    values: NDArray[np.float64],
    target: float,
    max_relative_distance: float,
) -> tuple[int, int] | None:
    """Indices of the two values surrounding `target`, or twice the same index.

    A repeated index means the axis holds a single usable observation, which is
    binned onto the nearest grid point rather than interpolated. `None` means the
    target cannot be represented: it falls outside the data, or the single
    observation sits further away than `max_relative_distance` relative to itself.
    """
    finite = np.isfinite(values)
    n_finite = int(np.count_nonzero(finite))

    if n_finite == 0:
        return None

    if n_finite == 1:
        idx = int(np.argmax(finite))
        if np.abs(target - values[idx]) > max_relative_distance * np.abs(values[idx]):
            return None
        return idx, idx

    direction = _sort_direction(values)
    idx_left = int(np.searchsorted(direction * values, direction * target, side="right")) - 1
    idx_right = idx_left + 1

    if idx_left == -1 or idx_right >= len(values):
        return None

    return idx_left, idx_right


def _linear_interp(
    PSD_left: float,
    PSD_right: float,
    target_value: float,
    left_value: float,
    right_value: float,
) -> float:
    """Interpolate between two values, or return the single one they collapse to.

    A degenerate bracket (`left_value == right_value`) means the axis held one
    usable observation, which is binned onto the grid point instead. The weight
    would be 0/0, so it is never computed.
    """
    if left_value == right_value:
        return PSD_left

    a = (target_value - left_value) / (right_value - left_value)
    return PSD_left + a * (PSD_right - PSD_left)


def _parallel_func_VK(
    grid_K_1d: NDArray[np.float64],
    grid_V: NDArray[np.float64],
    K_data: NDArray[np.float64],
    V_data: NDArray[np.float64],
    psd_in: NDArray[np.float64],
    max_relative_distance: float,
    it: int,
) -> NDArray[np.float64]:
    """Place a single time step on the model's V-K grid.

    Interpolation is linear in `log10(PSD)` and linear in V and K, so
    ``PSD = C * 10**(a*V + b*K)`` is reproduced exactly. Grid points that can be
    neither interpolated nor binned are left as `NaN`.
    """
    psd_interp = np.full((grid_V.shape[2], grid_V.shape[3]), np.nan)

    K_data_it = K_data[it, :]
    if np.all(np.isnan(K_data_it)) or np.all(np.isnan(psd_in[it, :, :])):
        return psd_interp

    for iK, K_val in enumerate(grid_K_1d):
        K_bracket = _bracket_indices(K_data_it, K_val, max_relative_distance)
        if K_bracket is None:
            continue

        K_idx_left, K_idx_right = K_bracket
        V_left = V_data[it, :, K_idx_left]
        V_right = V_data[it, :, K_idx_right]

        for iV, V_val in enumerate(grid_V[0, 0, :, iK]):
            V_bracket_left = _bracket_indices(V_left, V_val, max_relative_distance)
            V_bracket_right = _bracket_indices(V_right, V_val, max_relative_distance)

            if V_bracket_left is None or V_bracket_right is None:
                continue

            V_idx_left_left, V_idx_left_right = V_bracket_left
            V_idx_right_left, V_idx_right_right = V_bracket_right

            log_psd_left = _linear_interp(
                np.log10(psd_in[it, V_idx_left_left, K_idx_left]),
                np.log10(psd_in[it, V_idx_left_right, K_idx_left]),
                V_val,
                V_left[V_idx_left_left],
                V_left[V_idx_left_right],
            )

            log_psd_right = _linear_interp(
                np.log10(psd_in[it, V_idx_right_left, K_idx_right]),
                np.log10(psd_in[it, V_idx_right_right, K_idx_right]),
                V_val,
                V_right[V_idx_right_left],
                V_right[V_idx_right_right],
            )

            psd_interp[iV, iK] = np.power(
                10,
                _linear_interp(
                    log_psd_left,
                    log_psd_right,
                    K_val,
                    K_data_it[K_idx_left],
                    K_data_it[K_idx_right],
                ),
            )

    return psd_interp


@dataclass
class DebugPlotSettings:
    """Settings controlling debug plot generation in `bin_and_interpolate_to_model_grid`.

    Attributes:
        folder_path (Path): Directory in which the generated debug figures are saved.
        satellite_name (str): Satellite name, used in the output figure file names.
        target_V (float | None): Target V value to highlight in the V-K debug plot.
            If `None`, the plasmasphere debug plots are generated instead.
        target_K (float | None): Target K value to highlight in the V-K debug plot.
            If `None`, the plasmasphere debug plots are generated instead.
    """

    folder_path: Path
    satellite_name: str
    target_V: float | None = None  # noqa: N815
    target_K: float | None = None  # noqa: N815


def plot_debug_figures_plasmasphere(  # noqa: D103
    data_set: DataSet,
    psd_binned: NDArray[np.float64],
    sim_time: NDArray[np.object_],
    grid_P: NDArray[np.float64] | None,
    grid_R: NDArray[np.float64],
    debug_plot_settings: DebugPlotSettings,
) -> None:
    dt = sim_time[1] - sim_time[0]

    fig = plt.figure(figsize=(19.20, 8))
    plt.rcParams["axes.axisbelow"] = False

    R_or_Lstar_arr = data_set.R0

    for it, sim_time_curr in enumerate(tqdm(sim_time)):
        sat_time_idx = np.argwhere(np.abs(np.asarray(data_set.datetime) - sim_time_curr) <= dt / 2)

        ax0 = fig.add_subplot(121, projection="polar")
        ax1 = fig.add_subplot(122)

        # plot satellite trajectory on PxR grid
        # [x_sat, y_sat] = pol2cart(self.P, self.R)  # noqa: ERA001
        ax0.scatter(
            data_set.P[sat_time_idx],
            R_or_Lstar_arr[sat_time_idx],
            c=np.log10(data_set.density[sat_time_idx]),
            marker="D",
            vmin=0,
            vmax=4,
            cmap="jet",
        )
        ax0.set_ylim(1, 6.6)
        ax0.set_title("Orbit")
        ax0.set_rlim([0, 6.6])  # ty:ignore[unresolved-attribute]
        ax0.set_theta_offset(np.pi)  # ty:ignore[unresolved-attribute]

        grid_X = grid_R[:, :, 0, 0] * np.cos(grid_P[:, :, 0, 0])  # ty:ignore[not-subscriptable]

        grid_Y = grid_R[:, :, 0, 0] * np.sin(grid_P[:, :, 0, 0])  # ty:ignore[not-subscriptable]

        pc = ax1.pcolormesh(
            grid_X,
            grid_Y,
            np.squeeze(np.log10(psd_binned[it, :, :, :, :])),
            vmin=0,
            vmax=4,
            cmap="jet",
            edgecolors="k",
            linewidth=0.1,
        )
        ax1.set_title("Assimilation input")
        ax1.set_xlim(np.max(grid_R), -np.max(grid_R))
        ax1.set_ylim(np.max(grid_R), -np.max(grid_R))
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")

        fig.colorbar(pc, ax=ax1)

        fig.savefig(Path(debug_plot_settings.folder_path) / f"{debug_plot_settings.satellite_name}_{sim_time_curr}.png")

        fig.clf()

        if np.any(data_set.P[sat_time_idx] < 0.1):
            ic(psd_binned[it, 0, :, :, :])


def plot_debug_figures(  # noqa: D103
    data_set: DataSet,
    psd_binned: NDArray[np.float64],
    sim_time: NDArray[np.object_],
    grid_P: NDArray[np.float64] | None,
    grid_R: NDArray[np.float64],
    grid_V: NDArray[np.float64],
    grid_K: NDArray[np.float64],
    mu_or_V: Literal["Mu", "V"],
    debug_plot_settings: DebugPlotSettings,
) -> None:
    dt = sim_time[1] - sim_time[0]

    fig = plt.figure(figsize=(19.20, 5))
    plt.rcParams["axes.axisbelow"] = False

    data_set_V_or_Mu = data_set.InvMu if mu_or_V == "Mu" else data_set.InvV

    R_or_Lstar_arr = data_set.R0 if grid_P is not None else data_set.Lstar[:, -1]

    for it, sim_time_curr in enumerate(tqdm(sim_time)):
        sat_time_idx = np.argwhere(np.abs(np.asarray(data_set.datetime) - sim_time_curr) <= dt / 2)

        K_idx = np.argmin(
            np.abs(grid_K[0, 0, 0, :] - debug_plot_settings.target_K)  # ty:ignore[unsupported-operator]
        )
        V_idx = np.argmin(
            np.abs(grid_V[0, 0, :, K_idx] - debug_plot_settings.target_V)  # ty:ignore[unsupported-operator]
        )

        V_lim_min = np.log10(0.9 * np.min([np.nanmin(data_set_V_or_Mu), np.min(grid_V)]))
        V_lim_max = np.log10(1.1 * np.max([np.nanmax(data_set_V_or_Mu), np.max(grid_V)]))

        K_lim_min = np.log10(0.9 * np.min([np.nanmin(data_set.InvK), np.min(grid_K)]))
        K_lim_max = np.log10(1.1 * np.max([np.nanmax(data_set.InvK), np.max(grid_K)]))

        ax0 = fig.add_subplot(131, projection="polar")
        ax1 = fig.add_subplot(132)
        ax2 = fig.add_subplot(133)

        # plot satellite trajectory on PxR grid
        # [x_sat, y_sat] = pol2cart(self.P, self.R)  # noqa: ERA001

        ax0.scatter(
            data_set.P[sat_time_idx],
            R_or_Lstar_arr[sat_time_idx],
            c="k",
            marker="D",
        )
        ax0.set_ylim(1, 6.6)
        ax0.set_title("Orbit")
        ax0.set_theta_offset(np.pi)  # ty:ignore[unresolved-attribute]

        ax1.vlines(
            [np.log10(np.min(grid_V)), np.log10(np.max(grid_V))],
            np.log10(np.min(grid_K)),
            np.log10(np.max(grid_K)),
        )
        ax1.hlines(
            [np.log10(np.min(grid_K)), np.log10(np.max(grid_K))],
            np.log10(np.min(grid_V)),
            np.log10(np.max(grid_V)),
        )
        ax1.scatter(
            np.log10(grid_V[0, 0, :, :]),
            np.log10(grid_K[0, 0, :, :]),
            c="b",
            s=10,
        )

        for iV in range(data_set_V_or_Mu.shape[1]):  # ty:ignore[index-out-of-bounds]
            sc = ax1.scatter(
                np.log10(data_set_V_or_Mu[sat_time_idx, iV, :]),
                np.log10(data_set.InvK[sat_time_idx, :]),
                c=np.log10(data_set.PSD[sat_time_idx, iV, :]),
                marker="D",
                vmin=-1,
                vmax=3,
                cmap="jet",
            )

        # sc = ax1.scatter(np.log10(data_set_V_or_Mu[sat_time_idx,0,:]), np.log10(data_set.InvK[sat_time_idx,:]),
        #                     c=np.log10(data_set.PSD[sat_time_idx,0,:]), marker="D", vmin=-1, vmax=3, cmap="jet")
        # sc = ax1.scatter(np.log10(data_set_V_or_Mu[sat_time_idx,-1,:]), np.log10(data_set.InvK[sat_time_idx,:]),
        #                     c=np.log10(data_set.PSD[sat_time_idx,-1,:]), marker="D", vmin=-1, vmax=3, cmap="jet")

        ax1.scatter(
            np.log10(grid_V[0, 0, V_idx, K_idx]),
            np.log10(grid_K[0, 0, V_idx, K_idx]),
            c="r",
            s=15,
            marker="x",
        )
        ax1.set_title("V-K of satellite and simulation grid")
        ax1.set_xlim(V_lim_min, V_lim_max)
        ax1.set_ylim(K_lim_min, K_lim_max)
        ax1.set_xlabel("log10 V")
        ax1.set_ylabel("log10 K")

        fig.colorbar(sc, ax=ax1)

        if grid_P is not None:
            grid_X = grid_R[:, :, 0, 0] * np.cos(grid_P[:, :, 0, 0])
            grid_Y = grid_R[:, :, 0, 0] * np.sin(grid_P[:, :, 0, 0])

            pc = ax2.pcolormesh(
                grid_X,
                grid_Y,
                np.any(np.isfinite(psd_binned[it, :, :, :, :]), axis=(2, 3)),
                vmin=-1,
                vmax=5,
                cmap="jet",
                edgecolors="k",
                linewidth=0.1,
            )
            ax2.set_title("Assimilation input")
            ax2.set_xlim(np.max(grid_R), -np.max(grid_R))
            ax2.set_ylim(np.max(grid_R), -np.max(grid_R))
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")

            fig.colorbar(pc, ax=ax2)
        else:
            grid_X, grid_Y = np.meshgrid(sim_time, grid_R[0, :, 0, 0])
            pc = ax2.pcolormesh(
                grid_X,
                grid_Y,
                np.log10(psd_binned[:, 0, :, V_idx, K_idx]).T,
                vmin=-1,
                vmax=5,
                cmap="jet",
                edgecolors=None,
                linewidth=0.1,
                shading="nearest",
            )
            ax2.set_title("Assimilation input")
            ax2.set_ylim(0, np.max(grid_R))
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Lstar")

            fig.colorbar(pc, ax=ax2)

        fig.savefig(Path(debug_plot_settings.folder_path) / f"{debug_plot_settings.satellite_name}_{sim_time_curr}.png")

        fig.clf()
