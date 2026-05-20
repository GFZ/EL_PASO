# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import el_paso as ep


def test_calculate_w_parameters() -> None:
    start_time = datetime(2015, 3, 17, 0, 0, tzinfo=timezone.utc)
    end_time = start_time + timedelta(days=1)

    w_vars = ep.load_indices_solar_wind_parameters(start_time, end_time, ["W_params"], w_parameter_method="Calculation")

    true_max_values = [6.9, 7.34, 5.6, 15.75, 8.7, 42.95]

    w_params = w_vars["W_params"][0].get_data()
    assert w_params.shape[1] == 6

    for i in range(6):
        assert np.max(w_params[:, i]) == pytest.approx(true_max_values[i], abs=0.05)


@pytest.mark.visual
def test_w_parameters_comparison() -> None:
    start_time = datetime(2015, 3, 17, 0, 0, tzinfo=timezone.utc)
    end_time = start_time + timedelta(days=1)

    w_vars_calc = ep.load_indices_solar_wind_parameters(
        start_time, end_time, ["W_params"], w_parameter_method="Calculation"
    )

    w_params_calc = w_vars_calc["W_params"][0].get_data().astype(np.float64)
    time_var = w_vars_calc["W_params"][1]

    w_vars_website = ep.load_indices_solar_wind_parameters(
        start_time, end_time, ["W_params"], target_time_variable=time_var, w_parameter_method="TsyWebsite"
    )

    w_params_website = w_vars_website["W_params"].get_data().astype(np.float64)

    assert w_params_calc.shape == w_params_website.shape

    _, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    for i in range(6):
        axes[i].plot(w_params_calc[:, i], label=f"W{i} calc", linestyle="--", marker="o")
        axes[i].plot(w_params_website[:, i], label=f"W{i} website", linestyle="-", marker="x")
        axes[i].set_title(f"Comparison: W{i}")
        axes[i].set_ylabel("Value")
        axes[i].legend()
        axes[i].grid()

    axes[-1].set_xlabel("Index")
    plt.tight_layout()
    plt.savefig(f"{Path(__file__).parent / 'test_w_parameters_comparison.png'}")
