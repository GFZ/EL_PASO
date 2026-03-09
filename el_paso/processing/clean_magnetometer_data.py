import numpy as np


def clean_magnetometer_variables(variables):
    """
    Clean magnetometer data using Bt only.

    Steps:
    1. Keep only samples where Bt > 0.
    2. Remove spikes where |ΔBt| >= 500 nT between consecutive samples.
    3. Apply the resulting mask consistently to all input variables.

    Parameters
    ----------
    variables : dict
        Dictionary of variable objects with 'Bt' key included.
        Each variable must implement .get_data() -> np.ndarray and
        .set_data(filtered_data, mode).

    Returns
    -------
    dict
        Updated dictionary with filtered data arrays applied to all variables.
    """

    # Extract the magnetic field magnitude
    Bt_var = variables["Bt"]
    Bt = np.asarray(Bt_var.get_data())

    # Step 1: Filter for Bt > 0
    mask_positive = Bt > 0

    # Step 2: Compute deltaB (prepend 0 to match original length)
    deltaB = np.diff(Bt, prepend=Bt[0])

    # Step 3: Filter out spikes (absolute change less than 500 nT)
    mask_spike = np.abs(deltaB) < 500

    # Combined mask
    mask = mask_positive & mask_spike

    # Step 4: Apply consistent mask to all variables
    for var in variables.values():
        data = np.asarray(var.get_data())
        if data.shape[0] != mask.shape[0]:
            raise ValueError(
                f"Data length mismatch for variable. Expected {mask.shape[0]}, got {data.shape[0]}"
            )
        var.set_data(data[mask], unit="same")

    return variables
