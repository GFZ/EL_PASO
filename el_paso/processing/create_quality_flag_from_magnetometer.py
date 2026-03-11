# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Alwin Roy
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from astropy import units as u

import el_paso as ep


def magnetometer_quality_flags(variables: dict, threshold: int = 500) -> np.ndarray:
    """Compute a quality mask for magnetometer data using Bt.

    The mask flags samples that are considered valid based on two criteria:
    1. Magnetic field magnitude (Bt) must be positive.
    2. Consecutive differences |ΔBt| must be below the spike threshold.

    Args:
        variables (dict): Dictionary containing variable objects. Must include
            the key "Bt". Each variable must implement ``get_data()``.
        threshold(int, optional): Spike detection threshold in nT.
            Defaults to 500.

    Returns:
        np.ndarray: Boolean array where ``True`` indicates valid data and
        ``False`` indicates flagged samples.
    """
    bt_var = variables["Bt"]
    bt = bt_var.get_data(u.nT)

    mask_positive = bt > 0

    delta_b = np.diff(bt, prepend=bt[0])
    mask_spike = np.abs(delta_b) < threshold

    mask = mask_positive & mask_spike

    mask = ep.Variable(
        data=mask_positive & mask_spike,
        original_unit=u.dimensionless_unscaled,
    )

    return mask
