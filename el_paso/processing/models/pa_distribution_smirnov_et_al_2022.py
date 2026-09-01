# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Artem Smirnov, Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from datetime import datetime, timezone
from functools import cache, lru_cache
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
from numpy.typing import NDArray

import el_paso as ep

_DATA_URL = "https://datapub.gfz-potsdam.de/download/10.5880.GFZ.2.7.2022.001daedaf/2022-001_Smirnov-et-al_Data.zip"
_DATA_DIR = ep.utils.get_el_paso_model_data_path()
_COEFS_DIR = _DATA_DIR / "2022-001_Smirnov-et-al_Data" / "2022-001_Smirnov-et-al_PAD_model"

_ENERGY_SNAP_REL_TOLERANCE = 0.5
_FITTED_ENERGIES = np.asarray([37, 56, 78, 106, 142, 179, 220, 246, 342, 453, 588, 735, 871, 1080, 1650])
_FITTED_L_RANGE = (3, 6)
_PDYN_HIGH_THRESHOLD = 5.5

# Column indices in the Pijk_*.dat files: A_n -> (c0, c1, c2)
_A_COEF_COLUMNS = ((3, 4, 5), (6, 7, 8), (9, 10, 11))


def get_pa_distribution_smirnov_et_al_2022(
    time_var: ep.Variable,
    pa: NDArray[np.float64],
    L: NDArray[np.float64],
    MLT: NDArray[np.float64],
    energies: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Construct equatorial electron pitch angle distributions from the Smirnov et al. 2022 model.

    Evaluates the empirical polynomial model of Smirnov et al. (2022), which
    parameterizes normalized pitch angle distributions as a sum of odd sine
    harmonics, ``A1*sin(a) + A3*sin(3a) + A5*sin(5a)``. The coefficients depend
    on L-shell, magnetic local time, and solar wind dynamic pressure.

    Solar wind dynamic pressure is loaded automatically for the time range
    spanned by ``time_var``. The model coefficient files are downloaded on first
    use and cached under ``~/.elpaso``.

    The model is only fitted for L between 3 and 6 and for the 15 MagEIS energy
    channels listed in ``_FITTED_ENERGIES``. Inputs outside that domain are not
    extrapolated: L-shells are clamped to the fitted range, and each requested
    energy is snapped to the nearest fitted channel (with a warning if the
    relative difference exceeds 50%). Timesteps with Pdyn above 5.5 nPa use the
    saturated coefficient set, in which the coefficients no longer depend on Pdyn.

    Args:
        time_var: Time stamps as an ep.Variable, convertible to POSIX time.
            Defines the time grid (length ``n_times``) for all other inputs.
        pa: Pitch angles in radians, with shape ``(n_times, n_pitch_angles)``.
        L: L-shell values with shape ``(n_times,)``. Values outside the fitted
            range of 3 to 6 are clamped to that range.
        MLT: Magnetic local time values in hours, with shape ``(n_times,)``.
        energies: Energy channels in keV, with shape ``(n_energy_channels,)``.
            Each value is snapped to the nearest fitted MagEIS channel; values
            need not match a fitted channel exactly.

    Raises:
        ValueError: If the input arrays do not have consistent shapes.

    Returns:
        Normalized pitch angle distributions as a 3D array with shape
        ``(n_times, n_energy_channels, n_pitch_angles)``. Values are the
        dimensionless PAD shape, not absolute flux.

    References:
        Smirnov et al. (2022), Space Weather, https://doi.org/10.1029/2022SW003053
        Model coefficients: https://doi.org/10.5880/GFZ.2.7.2022.001 (CC BY 4.0)
    """
    times = time_var.get_data(ep.units.posixtime).astype(np.float64)

    _check_shapes(pa, times, L, MLT, energies)

    start_time = datetime.fromtimestamp(times[0], tz=timezone.utc)
    end_time = datetime.fromtimestamp(times[-1], tz=timezone.utc)

    L = np.clip(L, a_min=_FITTED_L_RANGE[0], a_max=_FITTED_L_RANGE[1])

    pa_distributions = np.empty((len(times), len(energies), pa.shape[1]))

    sw_vars = ep.load_indices_solar_wind_parameters(start_time, end_time, ["Pdyn"], time_var)
    p_dyn = sw_vars["Pdyn"].get_data().astype(np.float64)

    for ie, energy in enumerate(energies):
        a1, a3, a5 = _get_coefs(energy, L, MLT, p_dyn)
        pa_distributions[:, ie, :] = _apply_coefs(pa, a1, a3, a5)

    return pa_distributions


def _apply_coefs(
    pa: NDArray[np.float64], A1: NDArray[np.float64], A3: NDArray[np.float64], A5: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate the pitch angle distribution, normalized per steradian.

    The shape is normalized such that its integral over solid angle is unity,
    so that multiplying an omnidirectional flux by the result directly yields
    the corresponding directional (unidirectional) flux:

        j(alpha) = j_omni * ghat(alpha)

    Because the sine harmonics are orthogonal on [0, pi], only the first
    harmonic contributes to the solid angle integral, and the normalization
    factor reduces to ``pi**2 * A1``.

    Args:
        pa: Pitch angles in radians, with shape ``(n_times, n_pitch_angles)``.
        A1: First harmonic coefficients with shape ``(n_times,)``.
        A3: Third harmonic coefficients with shape ``(n_times,)``.
        A5: Fifth harmonic coefficients with shape ``(n_times,)``.

    Returns:
        The normalized distribution in units of 1/sr, with shape
        ``(n_times, n_pitch_angles)``.
    """
    if np.any(A1 <= 0):
        msg = f"Non-positive A1 encountered at {np.sum(A1 <= 0)} timestep(s); normalization is undefined there."
        raise ValueError(msg)

    denom = np.pi**2 * A1[:, np.newaxis]
    return (
        A1[:, np.newaxis] * np.sin(pa) + A3[:, np.newaxis] * np.sin(3 * pa) + A5[:, np.newaxis] * np.sin(5 * pa)
    ) / denom


@cache
def _load_coefs_arr(filename: str) -> NDArray[np.float64]:
    return np.loadtxt(_ensure_coefficient_files() / filename)


def _design_matrix(
    coefs_arr: NDArray[np.float64], L: NDArray[np.float64], MLT: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate cos(MLT)^i * sin(MLT)^j * L^k for every polynomial term.

    Returns shape (..., n_terms), broadcasting over the shape of L and MLT.
    """
    angle = MLT / 24 * 2 * np.pi
    cos_mlt = np.cos(angle)[..., np.newaxis]
    sin_mlt = np.sin(angle)[..., np.newaxis]
    l_shell = np.asarray(L, dtype=np.float64)[..., np.newaxis]

    return (cos_mlt ** coefs_arr[:, 0]) * (sin_mlt ** coefs_arr[:, 1]) * (l_shell ** coefs_arr[:, 2])


def _coefs_from_file(
    filename: str, L: NDArray[np.float64], MLT: NDArray[np.float64], Pdyn: NDArray[np.float64]
) -> tuple[NDArray[np.float64], ...]:
    coefs_arr = _load_coefs_arr(filename)
    design = _design_matrix(coefs_arr, L, MLT)

    return tuple(
        design @ coefs_arr[:, c0] * Pdyn**2 + design @ coefs_arr[:, c1] * Pdyn + design @ coefs_arr[:, c2]
        for c0, c1, c2 in _A_COEF_COLUMNS
    )


@lru_cache(maxsize=1)
def _ensure_coefficient_files() -> Path:
    """Return the directory containing the Pijk_*.dat files, downloading if needed.

    Data: Smirnov et al. (2022), GFZ Data Services, CC BY 4.0.
    https://doi.org/10.5880/GFZ.2.7.2022.001
    """
    if any(_COEFS_DIR.glob("Pijk_*.dat")):
        return _COEFS_DIR

    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    request = Request(_DATA_URL, headers={"User-Agent": "el_paso"})
    with urlopen(request, timeout=120) as response:  # noqa: S310
        payload = response.read()

    with ZipFile(BytesIO(payload)) as zf:
        zf.extractall(_DATA_DIR)

    if not any(_COEFS_DIR.glob("Pijk_*.dat")):
        msg = f"No Pijk_*.dat files found in {_COEFS_DIR} after extracting {_DATA_URL}."
        raise RuntimeError(msg)

    return _COEFS_DIR


def _get_coefs(
    energy: float, L: NDArray[np.float64], MLT: NDArray[np.float64], Pdyn: NDArray[np.float64]
) -> tuple[NDArray[np.float64], ...]:
    """Calculate the A1, A3 and A5 coefficients for given L, MLT and Pdyn.

    Vectorized: L, MLT and Pdyn may be scalars or broadcastable arrays. Timesteps
    with Pdyn above 5.5 nPa use the saturated (_HIGH) coefficient set, in which
    the coefficients no longer depend on Pdyn.

    Args:
        energy: Energy in keV. Snapped to the nearest fitted MagEIS channel.
        L: L-shell value, between 3 and 6.
        MLT: Magnetic local time value in hours.
        Pdyn: Solar wind dynamic pressure in nPa.

    Warns:
        UserWarning: If ``energy`` differs from the nearest fitted channel by
            more than 50%.

    Returns:
        The A1, A3 and A5 coefficients, with the broadcast shape of the inputs.
    """
    L, MLT, Pdyn = np.broadcast_arrays(
        np.asarray(L, dtype=np.float64),
        np.asarray(MLT, dtype=np.float64),
        np.asarray(Pdyn, dtype=np.float64),
    )

    closest_idx = np.argmin(np.abs(energy - _FITTED_ENERGIES))
    closest_energy = _FITTED_ENERGIES[closest_idx]

    if abs(energy - closest_energy) > _ENERGY_SNAP_REL_TOLERANCE * closest_energy:
        msg = (
            f"Requested energy {energy} keV is not close to any fitted MagEIS channel; "
            f"using the nearest fitted channel at {closest_energy} keV "
            f"({abs(energy - closest_energy) / closest_energy:.0%} relative difference). "
            f"Fitted channels are: {_FITTED_ENERGIES.tolist()} keV."
        )
        warnings.warn(msg, stacklevel=2)

    coefs_low = _coefs_from_file(f"Pijk_{int(closest_energy)}_keV.dat", L, MLT, Pdyn)

    is_high = Pdyn > _PDYN_HIGH_THRESHOLD
    if not np.any(is_high):
        return coefs_low

    # In the saturated regime the coefficients are Pdyn-independent, which is
    # equivalent to evaluating the polynomial at Pdyn = 0.
    coefs_high = _coefs_from_file(f"Pijk_{int(closest_energy)}_keV_HIGH.dat", L, MLT, np.asarray(0.0))

    return tuple(np.where(is_high, high, low) for low, high in zip(coefs_low, coefs_high, strict=True))


def _check_shapes(
    pa_local: NDArray[np.float64],
    times: NDArray[np.float64],
    L: NDArray[np.float64],
    MLT: NDArray[np.float64],
    energies: NDArray[np.float64],
) -> None:

    n_times = len(times)

    if pa_local.ndim != 2:
        msg = f"Expected 'pa_local' to be 2D (n_times, n_pitch_angles). Got shape: {pa_local.shape}"
        raise ValueError(msg)

    if pa_local.shape[0] != n_times:
        msg = (
            f"Shape mismatch between 'pa_local' and 'times'. "
            f"Expected pa_local.shape[0] == {n_times} (len(times)). Got: {pa_local.shape[0]}"
        )
        raise ValueError(msg)

    if L.ndim != 1:
        msg = f"Expected 'L' to be 1D (n_times,). Got shape: {L.shape}"
        raise ValueError(msg)

    if L.shape[0] != n_times:
        msg = (
            f"Shape mismatch between 'L' and 'times'. Expected L.shape[0] == {n_times} (len(times)). Got: {L.shape[0]}"
        )
        raise ValueError(msg)

    if MLT.ndim != 1:
        msg = f"Expected 'MLT' to be 1D (n_times,). Got shape: {MLT.shape}"
        raise ValueError(msg)

    if MLT.shape[0] != n_times:
        msg = (
            f"Shape mismatch between 'MLT' and 'times'. "
            f"Expected MLT.shape[0] == {n_times} (len(times)). Got: {MLT.shape[0]}"
        )
        raise ValueError(msg)

    if energies.ndim != 1:
        msg = f"Expected 'energies' to be 1D (n_energy_channels). Got shape: {energies.shape}"
        raise ValueError(msg)
