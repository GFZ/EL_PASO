# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from collections.abc import Iterable
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import numpy as np
from richpool import p_map

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from el_paso.dataset import DataSet
    from el_paso.typing import InternalName


class TargetType(Enum):
    """Specifies how two target coordinate vectors should be combined for interpolation.

    Attributes:
        TargetPairs: Interpolate at each pair formed by zipping the two target
            vectors together; the two vectors must have the same length.
        TargetMeshGrid: Interpolate at every combination of values from the two
            target vectors (the Cartesian product), producing a 2D grid of results.
    """

    TargetPairs = 0
    TargetMeshGrid = 1


TARGETS: TypeAlias = list[tuple[float | int, float | int]]


def _linear_interp(
    flux_left: float,
    flux_right: float,
    target_value: float,
    left_value: float,
    right_value: float,
) -> float:
    a = (target_value - left_value) / (right_value - left_value)
    return flux_left + a * (flux_right - flux_left)


def _interp_flux_parallel(
    flux: NDArray[np.float64],
    energy: NDArray[np.float64],
    alpha_eq_model: NDArray[np.float64],
    targets: list[tuple[float, float]],
    it: int,
) -> list[float]:
    result: list[float] = []

    for _, (target_en_single, target_al_single) in enumerate(targets):
        # find left and right alpha indices
        # first find the two al levels, where en points must exist

        al_right_idx = np.searchsorted(alpha_eq_model[it, :], target_al_single, side="right")
        al_left_idx = al_right_idx - 1

        if al_right_idx == 0 or al_right_idx >= len(alpha_eq_model[it, :]):
            result.append(np.nan)
            continue

        finite_idx = np.argwhere(np.isfinite(energy[it, :]) & np.isfinite(flux[it, :, al_left_idx]))
        if finite_idx.size == 0:
            result.append(np.nan)
            continue

        energy_interp = np.squeeze(energy[it, finite_idx])
        flux_interp = np.squeeze(flux[it, finite_idx, al_left_idx])
        assert np.all(np.diff(energy_interp) > 0)

        flux_left = float(np.interp(target_en_single, energy_interp, flux_interp, left=np.nan, right=np.nan))

        finite_idx = np.argwhere(np.isfinite(energy[it, :]) & np.isfinite(flux[it, :, al_right_idx]))
        if finite_idx.size == 0:
            result.append(np.nan)
            continue

        energy_interp = np.squeeze(energy[it, finite_idx])
        flux_interp = np.squeeze(flux[it, finite_idx, al_right_idx])
        assert np.all(np.diff(energy_interp) > 0)

        flux_right = float(np.interp(target_en_single, energy_interp, flux_interp, left=np.nan, right=np.nan))

        result.append(
            _linear_interp(
                flux_left,
                flux_right,
                target_al_single,
                alpha_eq_model[it, al_left_idx],
                alpha_eq_model[it, al_right_idx],
            )
        )

    return result


def interp_flux(
    self: DataSet,
    target_en: float | list[float] | NDArray[np.float64],
    target_al: float | list[float],
    target_type: TargetType | Literal["TargetPairs", "TargetMesh"],
    n_threads: int = 10,
    flux_internal_name: InternalName = "FEDU",
    energy_internal_name: InternalName = "Energy_FEDU",
) -> NDArray[np.float64]:
    """Interpolate flux to requested (energy, pitch angle) targets for every time.

    For each time step, the flux variable named by `flux_internal_name` is interpolated
    in energy (using the grid named by `energy_internal_name`) at the two pitch angles
    (from `self.alpha_eq_model`) bracketing each target pitch angle, and the two
    resulting flux values are then linearly interpolated in pitch angle to the target.

    Args:
        self (DataSet): The DataSet instance this method operates on.
        target_en (float | list[float] | NDArray[np.float64]): Target energy value(s)
            to interpolate to.
        target_al (float | list[float]): Target equatorial pitch angle value(s) to
            interpolate to.
        target_type (TargetType | Literal["TargetPairs", "TargetMesh"]): How to combine
            `target_en` and `target_al`. `TargetPairs` interpolates each
            `(energy, pitch angle)` pair (the two vectors must have the same length),
            producing a result of shape `(time, N)`. `TargetMeshGrid` interpolates
            every combination of the two vectors, producing a result of shape
            `(time, n_en, n_al)`.
        n_threads (int, optional): Number of worker processes used to parallelize the
            interpolation over time. Defaults to `10`.
        flux_internal_name (InternalName, optional): Internal name of the flux variable
            to interpolate (e.g. `"FEDU"` for electrons, `"FPDU"` for protons).
            Defaults to `"FEDU"`.
        energy_internal_name (InternalName, optional): Internal name of the matching
            energy grid (e.g. `"Energy_FEDU"`, `"Energy_FPDU"`). Must correspond to
            the species of `flux_internal_name`. Defaults to `"Energy_FEDU"`.

    Returns:
        NDArray[np.float64]: The interpolated flux values, with shape `(time, N)` for
        `TargetType.TargetPairs` or `(time, n_en, n_al)` for `TargetType.TargetMeshGrid`.

    Raises:
        AssertionError: If `target_type` is `TargetType.TargetPairs` and `target_en`
            and `target_al` do not have the same length.
    """
    if not isinstance(target_en, Iterable):
        target_en = [target_en]
    if not isinstance(target_al, Iterable):
        target_al = [target_al]

    if isinstance(target_type, str):
        target_type = TargetType[target_type]

    epoch = self.get_by_internal_name("Epoch")
    flux = self.get_by_internal_name(flux_internal_name)
    energy = self.get_by_internal_name(energy_internal_name)
    alpha_eq = self.get_by_internal_name("Alpha_Eq")

    if target_type == TargetType.TargetPairs:
        assert len(target_en) == len(  # ty:ignore[invalid-argument-type]
            target_al  # ty:ignore[invalid-argument-type]
        ), "For TargetType.Pairs, the target vectors must have the same size!"

        result_arr = np.empty((len(epoch), len(target_en)))  # ty:ignore[invalid-argument-type]
        targets = cast("TARGETS", list(zip(target_en, target_al, strict=False)))
    else:
        result_arr = np.empty((len(epoch), len(target_en), len(target_al)))  # ty:ignore[invalid-argument-type]
        targets = cast("TARGETS", list(itertools.product(target_en, target_al)))

    func = partial(
        _interp_flux_parallel,
        flux,
        energy,
        alpha_eq,
        targets,
    )

    chunksize = max(1, len(epoch) // n_threads // 4)  # same as multiprocessing.Pool's default
    parallel_results = p_map(
        func,
        range(len(epoch)),
        num_cpus=n_threads,
        chunksize=chunksize,
        desc="Interpolating flux",
        disable=not self._verbose,
    )

    for i in range(result_arr.shape[0]):
        if target_type == TargetType.TargetPairs:
            for t, _ in enumerate(targets):
                result_arr[i, t] = parallel_results[i][t]
        else:
            for ie, ia in itertools.product(
                range(len(target_en)),  # ty:ignore[invalid-argument-type]
                range(len(target_al)),  # ty:ignore[invalid-argument-type]
            ):
                result_arr[i, ie, ia] = parallel_results[i][ie * len(target_al) + ia]  # ty:ignore[invalid-argument-type]

    return result_arr


def _interp_psd_parallel(
    psd: NDArray[np.float64],
    invmu: NDArray[np.float64],
    invk: NDArray[np.float64],
    targets: list[tuple[float, float]],
    it: int,
) -> list[float]:
    """Interpolate PSD at time index `it` to (mu_target, K_target) pairs in `targets`.

    Shapes per time slice:
      psd[it]   -> (nE, nA)
      invmu[it] -> (nE, nA)
      invk[it]  -> (nA,)
    """
    out: list[float] = []

    # ---- 0) Extract this time slice
    psd_i = psd[it, :, :]  # (nE, nA)
    mu_i = invmu[it, :, :]  # (nE, nA)
    K_row = invk[it, :]  # (nA,)

    # ---- 1) Drop NaN K bins and the corresponding columns in PSD/mu
    finite_k = np.isfinite(K_row)
    if not np.any(finite_k):
        # No valid K at this time -> all NaN
        return [np.nan] * len(targets)

    K_use = K_row[finite_k]  # (nA_valid,)
    psd_use = psd_i[:, finite_k]  # (nE, nA_valid)
    mu_use = mu_i[:, finite_k]  # (nE, nA_valid)

    # If after masking we have fewer than 2 K points, we cannot bracket
    if K_use.size < 2:
        return [np.nan] * len(targets)

    # ---- 2) Ensure K ascending for searchsorted; if descending, flip columns
    if K_use[1] < K_use[0]:
        K_use = K_use[::-1]
        psd_use = psd_use[:, ::-1]
        mu_use = mu_use[:, ::-1]

    # ---- 3) For each (mu*, K*) target: 1D along mu, then linear across K
    for _, (mu_t, K_t) in enumerate(targets):
        # 3a) Bracket in K
        k_right = np.searchsorted(K_use, K_t, side="right")
        k_left = k_right - 1
        if k_right == 0 or k_right >= K_use.size:
            out.append(np.nan)
            continue

        # 3b) Interp along mu at LEFT K
        mu_L = mu_use[:, k_left]
        psd_L = psd_use[:, k_left]
        okL = np.isfinite(mu_L) & np.isfinite(psd_L)
        if not np.any(okL):
            out.append(np.nan)
            continue

        xL = np.asarray(mu_L[okL], dtype=float)
        yL = np.asarray(psd_L[okL], dtype=float)
        if xL.size < 2:
            out.append(np.nan)
            continue
        if not np.all(np.diff(xL) > 0):
            order = np.argsort(xL)
            xL, yL = xL[order], yL[order]
            xL, idx = np.unique(xL, return_index=True)
            yL = yL[idx]
            if xL.size < 2:
                out.append(np.nan)
                continue

        psd_left = float(np.interp(mu_t, xL, yL, left=np.nan, right=np.nan))

        # 3c) Interp along mu at RIGHT K
        mu_R = mu_use[:, k_right]
        psd_R = psd_use[:, k_right]
        okR = np.isfinite(mu_R) & np.isfinite(psd_R)
        if not np.any(okR):
            out.append(np.nan)
            continue

        xR = np.asarray(mu_R[okR], dtype=float)
        yR = np.asarray(psd_R[okR], dtype=float)
        if xR.size < 2:
            out.append(np.nan)
            continue
        if not np.all(np.diff(xR) > 0):
            order = np.argsort(xR)
            xR, yR = xR[order], yR[order]
            xR, idx = np.unique(xR, return_index=True)
            yR = yR[idx]
            if xR.size < 2:
                out.append(np.nan)
                continue

        psd_right = float(np.interp(mu_t, xR, yR, left=np.nan, right=np.nan))

        if not np.isfinite(psd_left) or not np.isfinite(psd_right):
            out.append(np.nan)
            continue

        # 3d) Linear across K to K_t
        val = _linear_interp(psd_left, psd_right, K_t, K_use[k_left], K_use[k_right])
        out.append(val)

    return out


def interp_psd(
    self: DataSet,
    target_mu: float | list[float] | NDArray[np.float64],
    target_K: float | list[float] | NDArray[np.float64],
    target_type: TargetType | Literal["TargetPairs", "TargetMesh"],
    n_threads: int = 10,
) -> NDArray[np.float64]:
    """Interpolate phase space density (PSD) to requested (mu, K) targets for every time.

    For each time step, `self.PSD` is interpolated in `self.InvMu` at the two K values
    (from `self.InvK`) bracketing each target K, and the two resulting PSD values are
    then linearly interpolated in K to the target.

    Args:
        self (DataSet): The DataSet instance this method operates on.
        target_mu (float | list[float] | NDArray[np.float64]): Target first adiabatic
            invariant (mu) value(s) to interpolate to.
        target_K (float | list[float] | NDArray[np.float64]): Target second adiabatic
            invariant (K) value(s) to interpolate to.
        target_type (TargetType | Literal["TargetPairs", "TargetMesh"]): How to combine
            `target_mu` and `target_K`. `TargetPairs` interpolates each `(mu, K)` pair
            (the two vectors must have the same length), producing a result of shape
            `(time, N)`. `TargetMeshGrid` interpolates every combination of the two
            vectors, producing a result of shape `(time, n_mu, n_K)`.
        n_threads (int, optional): Number of worker processes used to parallelize the
            interpolation over time. Defaults to `10`.

    Returns:
        NDArray[np.float64]: The interpolated PSD values, with shape `(time, N)` for
        `TargetType.TargetPairs` or `(time, n_mu, n_K)` for `TargetType.TargetMeshGrid`.

    Raises:
        AssertionError: If `target_type` is `TargetType.TargetPairs` and `target_mu`
            and `target_K` do not have the same length.
    """
    if not isinstance(target_mu, Iterable):
        target_mu = [target_mu]
    if not isinstance(target_K, Iterable):
        target_K = [target_K]

    if isinstance(target_type, str):
        target_type = TargetType[target_type]

    epoch = self.get_by_internal_name("Epoch")
    psd = self.get_by_internal_name("PSD")
    inv_mu = self.get_by_internal_name("InvMu")
    inv_k = self.get_by_internal_name("InvK")

    if target_type == TargetType.TargetPairs:
        assert len(target_mu) == len(target_K), "For TargetType.Pairs, mu and K vectors must have the same size!"  # ty:ignore[invalid-argument-type]
        result_arr = np.empty((len(epoch), len(target_mu)))  # ty:ignore[invalid-argument-type]
        targets = cast("TARGETS", list(zip(target_mu, target_K, strict=False)))
    else:
        result_arr = np.empty((len(epoch), len(target_mu), len(target_K)))  # ty:ignore[invalid-argument-type]
        targets = cast("TARGETS", list(itertools.product(target_mu, target_K)))

    # parallel over time (same pattern as interp_flux)
    func = partial(_interp_psd_parallel, psd, inv_mu, inv_k, targets)

    chunksize = max(1, len(epoch) // n_threads // 4)  # same as multiprocessing.Pool's default
    parallel_results = p_map(
        func,
        range(len(epoch)),
        num_cpus=n_threads,
        chunksize=chunksize,
        desc="Interpolating PSD",
        disable=not self._verbose,
    )

    # pack results back like interp_flux
    if target_type == TargetType.TargetPairs:
        for i in range(result_arr.shape[0]):
            for t, _ in enumerate(targets):
                result_arr[i, t] = parallel_results[i][t]
    else:
        n_mu, n_K = len(target_mu), len(target_K)  # ty:ignore[invalid-argument-type]
        for i in range(result_arr.shape[0]):
            for im, iK in itertools.product(range(n_mu), range(n_K)):
                result_arr[i, im, iK] = parallel_results[i][im * n_K + iK]

    return result_arr
