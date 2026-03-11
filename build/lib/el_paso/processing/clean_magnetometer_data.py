# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Alwin Roy
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from astropy import units as u


def clean_magnetometer_variables(variables: dict) -> dict:
    """Clean magnetometer data using Bt only.

    Steps:
    1. Keep only samples where Bt > 0.
    2. Remove spikes where |ΔBt| >= 500 nT between consecutive samples.
    3. Apply the resulting mask consistently to all input variables.

    Args:
        variables (dict): Dictionary of variable objects with 'Bt' key included.
                          Each variable must implement `.get_data()` -> np.ndarray
                          and `.set_data(filtered_data, mode)`.

    Returns:
        dict: Updated dictionary with filtered data arrays applied to all variables.

    Raises:
        ValueError: If there is a mismatch in the length of the data arrays.
    """
    # Extract the magnetic field magnitude
    bt_var = variables["Bt"]
    bt = bt_var.get_data(u.nT)  # Ensure data is in nT

    # Step 1: Filter for Bt > 0
    mask_positive = bt > 0

    # Step 2: Compute deltaB (prepend 0 to match original length)
    delta_b = np.diff(bt, prepend=bt[0])

    # Step 3: Filter out spikes (absolute change less than 500 nT)
    mask_spike = np.abs(delta_b) < 500 * u.nT  # Ensure spike threshold is in nT

    # Combined mask
    mask = mask_positive & mask_spike

    # Step 4: Apply consistent mask to all variables
    for var in variables.values():
        data = var.get_data(u.nT)  # Ensure all data is in nT
        if data.shape[0] != mask.shape[0]:
            message = f"Data length mismatch for variable. Expected {mask.shape[0]}, got {data.shape[0]}"
            raise ValueError(message)
        var.set_data(data[mask], unit="same")

    variables.add_processing_note("Data cleaned using clean_agnetometer_data")

    return variables
