# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

import numpy as np
from astropy import units as u
from scipy.integrate import trapezoid

import el_paso as ep
from el_paso.processing.models import get_pa_distribution_smirnov_et_al_2022

_OMNI_FLUX_UNIT = (u.cm**2 * u.s * u.keV) ** (-1)
_DIRECTIONAL_FLUX_UNIT = (u.cm**2 * u.s * u.sr * u.keV) ** (-1)


def construct_pitch_angle_distribution(
    omni_flux_var: ep.Variable,
    pa_local_var: ep.Variable,
    pa_eq_var: ep.Variable,
    flux_type: Literal["omni", "spin_average"],
    method: Literal["sin", "Smirnov2022"] = "sin",
    time_var: ep.Variable | None = None,
    L_var: ep.Variable | None = None,
    MLT_var: ep.Variable | None = None,
    energy_var: ep.Variable | None = None,
) -> ep.Variable:
    r"""Construct a pitch angle distribution from omni-directional or spin-averaged flux.

    The chosen ``method`` provides only the *shape* $f(\alpha)$ of the pitch angle
    distribution (PAD) as a function of the local pitch angle $\alpha$. This shape
    is then normalized according to ``flux_type``, so that multiplying the input
    flux by the normalized shape yields a directional flux consistent with the
    input.

    Methods (shape only):

    - ``"sin"``: sine-shaped PAD in the equatorial pitch angle $\alpha_{eq}$,
      mapped to the local pitch angle via the adiabatic invariant
      $\sin \alpha_{eq} = \sin \alpha \, \sin \alpha_{eq,max}$, which gives
      $f(\alpha) \propto \sin \alpha$. Requires no additional inputs.
    - ``"Smirnov2022"``: empirical PAD model of Smirnov et al. (2022), depending
      on L-shell, magnetic local time, solar wind dynamic pressure and energy.
      Requires ``time_var``, ``L_var``, ``MLT_var`` and ``energy_var``.

    Normalization (``flux_type``):

    - ``"omni"``: the input is a true omni-directional flux $J_{omni}$, i.e. the
      directional flux integrated over solid angle (units of
      $\mathrm{cm}^{-2} \mathrm{s}^{-1} \mathrm{keV}^{-1}$, no steradian). The
      shape is normalized such that its integral over solid angle is unity:
      $$2\pi \int_0^{\pi} f(\alpha) \sin \alpha \, d\alpha = 1$$
    - ``"spin_average"``: the input is a spin-averaged directional flux,
      already per steradian but averaged over the local pitch
      angles sampled during a spin, *not* integrated over solid angle. The shape
      is normalized such that its mean over the local pitch angle is unity:
      $$\frac{1}{\alpha_{max} - \alpha_{min}} \int_{\alpha_{min}}^{\alpha_{max}} f(\alpha) \, d\alpha = 1$$

    Args:
        omni_flux_var: Flux as an `el_paso` variable (interpreted according to
            ``flux_type``), with dimensions (time, energy_channels).
        pa_local_var: Local pitch angles as an `el_paso` variable, with
            dimensions (time, pitch_angles).
        pa_eq_var: Equatorial pitch angles as an `el_paso` variable, with
            dimensions (time, pitch_angles).
        method: The method used to construct the PAD shape. Defaults to ``"sin"``.
        flux_type: How the input flux is interpreted for normalization.
        time_var: Time stamps as an `el_paso` variable. Required for
            ``"Smirnov2022"``, ignored otherwise.
        L_var: L-shell values as an `el_paso` variable, with dimensions (time,).
            Required for ``"Smirnov2022"``, ignored otherwise.
        MLT_var: Magnetic local time in hours as an `el_paso` variable, with
            dimensions (time,). Required for ``"Smirnov2022"``, ignored otherwise.
        energy_var: Energy channels as an `el_paso` variable, with dimensions
            (energy_channels,). Required for ``"Smirnov2022"``, ignored otherwise.

    Raises:
        ValueError: If an unsupported method or flux type is provided, or if a
            required variable is missing for the selected method.

    Returns:
        A new `el_paso` variable containing the directional flux as a 3D array
        with dimensions (time, energy_channels, pitch_angles), in units of
        $\mathrm{cm}^{-2} \mathrm{s}^{-1} \mathrm{sr}^{-1} \mathrm{keV}^{-1}$.

    References:
        Smirnov et al. (2022), Space Weather, https://doi.org/10.1029/2022SW003053
    """
    omni_flux = omni_flux_var.get_data(_OMNI_FLUX_UNIT)
    pa_local = pa_local_var.get_data(u.rad).astype(np.float64)
    pa_eq = pa_eq_var.get_data(u.rad).astype(np.float64)

    omni_flux = np.atleast_3d(omni_flux)

    match method:
        case "sin":
            pa_eq_max = np.max(pa_eq, axis=1)
            pad_shape = np.sin(pa_local) * np.sin(pa_eq_max[:, np.newaxis])

        case "Smirnov2022":
            _require_vars(
                method,
                time_var=time_var,
                L_var=L_var,
                MLT_var=MLT_var,
                energy_var=energy_var,
            )

            assert time_var is not None
            assert L_var is not None
            assert MLT_var is not None
            assert energy_var is not None

            L = L_var.get_data().astype(np.float64)
            MLT = MLT_var.get_data(u.hour).astype(np.float64)
            energies = energy_var.get_data(u.keV).astype(np.float64)

            pad_shape = get_pa_distribution_smirnov_et_al_2022(
                time_var=time_var,
                pa=pa_eq,
                L=L,
                MLT=MLT,
                energies=energies[0, :],
            )

        case _:
            msg = f"Encountered invalid method to construct pitch angle distribution: {method}!"
            raise ValueError(msg)

    # Ensure shape is (time, energy_channels_or_1, pitch_angles) for broadcasting.
    if pad_shape.ndim == 2:
        pad_shape = pad_shape[:, np.newaxis, :]

    norm = _compute_normalization(pad_shape, pa_local, flux_type)
    directional_flux = omni_flux * pad_shape / norm

    result_var = ep.Variable(data=directional_flux, original_unit=_DIRECTIONAL_FLUX_UNIT)

    if method == "sin":
        shape_note = (
            "sin(alpha_local) * sin(alpha_eq_max), mapped from local to equatorial "
            "pitch angle via the adiabatic invariant"
        )
    else:
        shape_note = (
            "the empirical Smirnov et al. (2022) model, using L-shell, MLT, solar "
            "wind dynamic pressure and energy as inputs"
        )

    norm_note = "solid-angle integral" if flux_type == "omni" else "mean over local pitch angle"

    result_var.metadata.add_processing_note(
        f"Constructed directional flux from {flux_type!r} flux using the {method!r} pitch angle "
        f"distribution shape, computed as {shape_note}. Normalized so that the {norm_note} of the "
        "PAD shape is unity."
    )

    return result_var


def _compute_normalization(
    pad_shape: np.ndarray,
    pa_local: np.ndarray,
    flux_type: Literal["omni", "spin_average"],
) -> np.ndarray:
    r"""Compute the normalization factor for a PAD shape.

    For ``flux_type="omni"``, the returned factor is the integral of the shape
    $f(\alpha)$ over solid angle:
    $$N = 2\pi \int_0^{\pi} f(\alpha) \sin \alpha \, d\alpha$$
    For ``flux_type="spin_average"``, it is instead the mean of the shape over
    the sampled pitch angle range:
    $$N = \frac{1}{\alpha_{max} - \alpha_{min}} \int_{\alpha_{min}}^{\alpha_{max}} f(\alpha) \, d\alpha$$

    The pitch angle axis is sorted before integrating, since pitch angle grids
    are not guaranteed to be monotonic. If the grid only covers
    $0 \leq \alpha \leq \pi/2$, mirror symmetry about $\alpha = \pi/2$ is assumed
    and the solid angle integral is doubled accordingly.

    Args:
        pad_shape: Unnormalized PAD shape, with dimensions
            (time, energy_channels_or_1, pitch_angles).
        pa_local: Local pitch angles in radians, with dimensions
            (time, pitch_angles).
        flux_type: Either ``"omni"`` or ``"spin_average"``.

    Raises:
        ValueError: If an unsupported flux type is provided.

    Returns:
        Normalization factors broadcastable against ``pad_shape``, with
        dimensions (time, energy_channels_or_1, 1).
    """
    sort_idx = np.argsort(pa_local, axis=-1)
    pa_sorted = np.take_along_axis(pa_local, sort_idx, axis=-1)
    shape_sorted = np.take_along_axis(pad_shape, sort_idx[:, np.newaxis, :], axis=-1)

    pa_broadcast = pa_sorted[:, np.newaxis, :]

    match flux_type:
        case "omni":
            # Half-range grids (0-90 deg) are completed by mirror symmetry.
            half_range = np.max(pa_sorted, axis=-1) <= (np.pi / 2 + 1e-3)
            symmetry_factor = np.where(half_range, 2.0, 1.0)[:, np.newaxis, np.newaxis]

            integral = trapezoid(shape_sorted * np.sin(pa_broadcast), x=pa_broadcast, axis=-1)
            norm = 2 * np.pi * integral[..., np.newaxis] * symmetry_factor

        case "spin_average":
            # Mean over the sampled local pitch angle range.
            span = (pa_sorted[:, -1] - pa_sorted[:, 0])[:, np.newaxis, np.newaxis]
            integral = trapezoid(shape_sorted, x=pa_broadcast, axis=-1)
            norm = integral[..., np.newaxis] / span

        case _:
            msg = f"Encountered invalid flux_type: {flux_type}!"
            raise ValueError(msg)

    return norm


def _require_vars(method: str, **variables: ep.Variable | None) -> None:
    """Raise if any variable required by the selected method is missing.

    Args:
        method: Name of the selected method, used in the error message.
        **variables: Mapping of argument name to variable, which must not be None.

    Raises:
        ValueError: If one or more of the given variables is None.
    """
    missing = sorted(name for name, var in variables.items() if var is None)

    if missing:
        msg = f"Method {method!r} requires the following arguments, but they were not provided: {', '.join(missing)}."
        raise ValueError(msg)
