# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u  # type: ignore[reportMissingTypeStubs]
from matplotlib import pyplot as plt

from el_paso.utils import timed_function

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from el_paso import Variable

logger = logging.getLogger(__name__)


def _fold_pitch_angles_and_flux(
    pitch_angles: NDArray[np.float64], flux: NDArray[np.float64], *, produce_statistic_plot: bool
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # Normalize pitch angles to the range [0, 180]
    pitch_angles_abs = np.abs(pitch_angles)

    # Fold around 90 degrees
    folded_pitch_angles = np.minimum(pitch_angles_abs, 180 - pitch_angles_abs)
    folded_pitch_angles = np.round(folded_pitch_angles, 1).astype(np.float64)

    # Create an array to hold the folded flux values
    n_time, n_energy, _ = flux.shape
    unique_angles = np.unique(folded_pitch_angles[~np.isnan(folded_pitch_angles)])

    folded_flux = np.full((n_time, n_energy, len(unique_angles)), np.nan)

    if produce_statistic_plot:
        fig, axes = plt.subplots(len(unique_angles), n_energy, figsize=(5 * n_energy, 5 * len(unique_angles)))  # type: ignore[reportUnknownMemberType]

    for i, angle in enumerate(unique_angles):
        mask = folded_pitch_angles != angle

        mask = np.repeat(mask[:, np.newaxis, :], n_energy, axis=1)

        masked_flux = np.ma.masked_array(flux, mask=mask)  # type: ignore[reportUnknownMemberType]

        if produce_statistic_plot:
            diff = np.log10(masked_flux) - np.log10(np.flip(masked_flux, axis=2))  # type: ignore[reportUnknownArgumentType]
            for ie in range(n_energy):
                diff_energy = diff[:, ie, :].flatten()
                mask_diff = diff_energy > 0

                axes[i, ie].hist(diff_energy[mask_diff].compressed(), bins=10)  # type: ignore[reportUnknownArgumentType]
                axes[i, ie].set_title(f"Energy = {ie}, alpha = {angle}")  # type: ignore[reportUnknownArgumentType]

        folded_flux[:, :, i] = np.nanmean(masked_flux, axis=2)  # type: ignore[reportUnknownArgumentType]

    # add time dimension
    unique_angles = np.tile(unique_angles.reshape(1, -1), (n_time, 1))

    if produce_statistic_plot:
        fig.savefig("folded_pitch_angle_statistics.png")  # type: ignore[reportUnknownArgumentType]

    return folded_flux, unique_angles


@timed_function()
def fold_pitch_angles_and_flux(
    flux_var: Variable, pa_local_var: Variable, *, produce_statistic_plot: bool = False
) -> None:
    """Folds pitch angles and corresponding flux values around 90 degrees.

    This function modifies the input `flux_var` and `pa_local_var` in place
    by folding the pitch angle data and averaging the corresponding flux values
    around 90 degrees. It assumes that pitch angles are symmetric around 90 degrees.

    Args:
        flux_var (Variable): A Variable object containing the flux data.
            This object will be modified in place with the folded flux data.
            Expected to have a shape compatible with (time, energy_bins, pitch_angle_bins).
        pa_local_var (Variable): A Variable object containing the local pitch angle data.
            This object will be modified in place with the unique folded pitch angles in degrees.
            Expected to have a shape compatible with (time, pitch_angle_bins).
        produce_statistic_plot (bool, optional): If True, a statistical plot
            related to the folding process will be produced. Defaults to False.

    Raises:
        ValueError:
            - If local pitch angles are found to change over time (they are assumed constant).
            - If the time dimensions of `flux_var` and `pa_local_var` do not match.
            - If the pitch angle dimension of `flux_var` does not match the
              second dimension of `pa_local_var`.
    """
    logger.info("Folding pitch angles and flux ...")

    flux = flux_var.get_data().astype(np.float64)
    pa_local = pa_local_var.get_data(u.deg).astype(np.float64)

    if np.all(np.repeat(pa_local[0, :][np.newaxis, :], pa_local.shape[0], axis=0) != pa_local):
        msg = "We assume that local pitch angles do not change in time!"
        raise ValueError(msg)

    if len(flux) != len(pa_local):
        msg = f"Size of time dimensions of flux ({len(flux)}) and local pitch angles ({len(pa_local)}) do not match!"
        raise ValueError(msg)

    if flux.shape[2] != pa_local.shape[1]:
        msg = f"Dimension mismatch: flux: {flux.shape}, pitch angle: {pa_local.shape}"
        raise ValueError(msg)

    folded_flux, unique_angles = _fold_pitch_angles_and_flux(
        pa_local, flux, produce_statistic_plot=produce_statistic_plot
    )

    flux_var.set_data(folded_flux, "same")
    flux_var.metadata.add_processing_note("Folded around 90 degrees local pitch angle")

    pa_local_var.set_data(unique_angles, u.deg)
    pa_local_var.metadata.add_processing_note("Folded around 90 degrees local pitch angle")
