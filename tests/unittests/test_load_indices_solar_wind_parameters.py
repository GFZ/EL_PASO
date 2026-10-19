# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import importlib
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import el_paso as ep
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.mark.minimal
@pytest.mark.basic
def test_calculate_w_parameters() -> None:
    start_time = datetime(2015, 3, 17, 0, 0, tzinfo=timezone.utc)
    end_time = start_time + timedelta(days=1)

    w_vars = ep.load_indices_solar_wind_parameters(start_time, end_time, ["W_params"], w_parameter_method="Calculation")

    true_max_values = [6.94, 7.36, 5.64, 15.9, 8.71, 42.95]

    w_params = w_vars["W_params"][0].get_data()
    assert w_params.shape[1] == 6

    for i in range(6):
        assert np.max(w_params[:, i]) == pytest.approx(true_max_values[i], abs=0.05)


@pytest.mark.minimal
@pytest.mark.basic
def test_fill_sw_controls_edge_gap_filling_for_high_res_omni_params(monkeypatch: pytest.MonkeyPatch) -> None:
    load_sw_module = importlib.import_module("el_paso.load_indices_solar_wind_parameters")

    idx = pd.date_range("2020-01-01", periods=5, freq="h", tz="UTC")
    raw_df = pd.DataFrame({"pdyn": [np.nan, 1.0, np.nan, 3.0, np.nan]}, index=idx)

    monkeypatch.setattr(load_sw_module, "_cache_omni_high_res", lambda *args, **kwargs: raw_df.copy())

    start_time = datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)
    end_time = datetime(2020, 1, 1, 5, 0, tzinfo=timezone.utc)

    filled = ep.load_indices_solar_wind_parameters(start_time, end_time, ["Pdyn"], fill_sw=True)
    data_filled = filled["Pdyn"][0].get_data()
    assert not np.any(np.isnan(data_filled))
    assert data_filled[0] == pytest.approx(1.0)  # back-filled from index 1
    assert data_filled[2] == pytest.approx(2.0)  # linearly interpolated regardless of fill_sw
    assert data_filled[-1] == pytest.approx(3.0)  # held forward by interpolate() itself

    unfilled = ep.load_indices_solar_wind_parameters(start_time, end_time, ["Pdyn"], fill_sw=False)
    data_unfilled = unfilled["Pdyn"][0].get_data()
    assert np.isnan(data_unfilled[0])  # the leading gap is left unfilled
    assert data_unfilled[2] == pytest.approx(2.0)  # linearly interpolated regardless of fill_sw
    assert data_unfilled[-1] == pytest.approx(3.0)  # held forward by interpolate() itself


@pytest.mark.visual
def test_w_parameters_comparison(skip_if_unreachable: Callable[..., None]) -> None:
    skip_if_unreachable("https://geo.phys.spbu.ru")

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
