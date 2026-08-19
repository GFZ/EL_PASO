# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
import pandas as pd
from scipy.interpolate import make_splrep
from scipy.signal import find_peaks

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from el_paso.dataset import DataSet


class Trajectory(NamedTuple):
    """A single orbit segment, identified between two consecutive radial extrema.

    Attributes:
        start (int): Index into the time/distance arrays where the trajectory starts.
        end (int): Index into the time/distance arrays where the trajectory ends.
        direction (Literal["inbound", "outbound"]): Whether the radial distance is
            decreasing ("inbound") or increasing ("outbound") over the trajectory.
    """

    start: int
    end: int
    direction: Literal["inbound", "outbound"]


def _identify_orbits(
    time: list, distance: NDArray[np.floating], minimal_distance: int, *, apply_smoothing: bool
) -> list[Trajectory]:
    distance_filled = pd.Series(distance).interpolate(method="linear", limit_direction="both").to_numpy()

    if apply_smoothing:
        timestamps = [t.timestamp() for t in time]
        distance_filled = make_splrep(timestamps, distance_filled, s=0)(timestamps)
        distance_filled = typing.cast("NDArray[np.floating]", distance_filled)

    peaks, _ = find_peaks(distance_filled, distance=minimal_distance)
    troughs, _ = find_peaks(-distance_filled, distance=minimal_distance)
    extrema = np.sort(np.concatenate((peaks, troughs)))
    extrema = typing.cast("NDArray[np.int32]", extrema)

    diffs = np.diff(distance_filled)
    in_out_bound_label = "inbound" if np.median(diffs[0 : extrema[0]]) < 0 else "outbound"
    orbits: list[Trajectory] = [Trajectory(0, int(extrema[0]), in_out_bound_label)]

    for i in range(1, len(extrema)):
        in_out_bound_label = "inbound" if np.median(diffs[extrema[i - 1] : extrema[i]]) < 0 else "outbound"

        orbits.append(Trajectory(extrema[i - 1] + 1, extrema[i], in_out_bound_label))

    in_out_bound_label = "inbound" if np.median(diffs[extrema[-1] :]) < 0 else "outbound"
    orbits.append(Trajectory(extrema[-1] + 1, len(distance) - 1, in_out_bound_label))

    return orbits


def identify_orbits(
    self: DataSet,
    orbit_type: Literal["R", "L*"] = "R",
    minimal_distance: int = 60,
    *,
    apply_smoothing: bool = True,
) -> list[Trajectory]:
    """Split the dataset's time series into individual orbit segments.

    The radial distance (``R0`` or the last column of ``Lstar``, depending on
    `orbit_type`) is interpolated to fill any gaps and, optionally, smoothed with a
    spline. Local minima and maxima (perigee/apogee or their L* equivalents) are then
    used as the boundaries between consecutive inbound/outbound trajectories.

    Args:
        self (DataSet): The DataSet instance this method operates on.
        orbit_type (Literal["R", "L*"], optional): Which radial distance to use to
            identify orbits: ``"R"`` uses `self.R0`, ``"L*"`` uses the last column of
            `self.Lstar`. Defaults to `"R"`.
        minimal_distance (int, optional): Minimal number of samples between two
            consecutive extrema, passed to `scipy.signal.find_peaks` as `distance`.
            Defaults to `60`.
        apply_smoothing (bool, optional): If `True`, smooth the radial distance with a
            cubic spline before locating extrema. Defaults to `True`.

    Returns:
        list[Trajectory]: The list of inbound/outbound trajectories that together
        cover the full time series.
    """
    dist = self.get_by_internal_name("R_Eq") if orbit_type == "R" else self.get_by_internal_name("L_star")[:, -1]

    return _identify_orbits(self.datetime, dist, minimal_distance, apply_smoothing=apply_smoothing)
