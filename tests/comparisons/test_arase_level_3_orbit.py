# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import functools
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from matplotlib import pyplot as plt

import el_paso as ep
from el_paso.dataset import DataSet
from el_paso.recipes.arase import (
    get_arase_orbit_level_3_variables,
    process_arase_mepe,
)


@functools.cache
def arase_orbit_el_paso(
    start_time: datetime, end_time: datetime, mag_field: Literal["T89", "TS04", "OP77Q"]
) -> DataSet:
    Path("tests/comparisons/raw_data").mkdir(exist_ok=True)
    Path("tests/comparisons/processed_data").mkdir(exist_ok=True)

    process_arase_mepe(
        start_time,
        end_time,
        mag_field,
        num_cores=48,
        use_level_3_orbit_data=False,
        cadence=timedelta(minutes=1),
        raw_data_path=Path("tests/comparisons/raw_data"),
        processed_data_path=Path("tests/comparisons/processed_data"),
    )
    data_standard_instance = ep.data_standards.GFZStandard()

    saving_strategy = ep.saving_strategies.MonthlyRBStrategy(
        Path("tests/comparisons/processed_data/"),
        "Arase",
        "arase",
        "mepe",
        mag_field=mag_field,
        file_format="nc",
        data_standard=data_standard_instance,
    )

    return DataSet(
        saving_strategy,
        start_time,
        end_time,
        verbose=True,
    )


mag_field_list = ["TS04", "T89", "OP77Q"]


@pytest.mark.parametrize("mag_field", mag_field_list)
@pytest.mark.visual
def test_arase_level_3_orbit_comparison(mag_field: Literal["TS04", "T89", "OP77Q"]) -> None:
    start_time = datetime(2017, 9, 8, tzinfo=timezone.utc)
    end_time = start_time + timedelta(days=0, hours=23, minutes=59, seconds=59)

    Path("tests/comparisons/raw_data").mkdir(exist_ok=True)

    orbit_vars = get_arase_orbit_level_3_variables(
        start_time, end_time, mag_field, raw_data_path="tests/comparisons/raw_data"
    )

    orbit_vars["R0"].apply_thresholds_on_data(0, 50)

    arase_data_el_paso = arase_orbit_el_paso(start_time, end_time, mag_field)

    keys_to_compare = ["R0", "MLT"]
    orbit_time = [datetime.fromtimestamp(t, tz=timezone.utc) for t in orbit_vars["Epoch"].get_data(ep.units.posixtime)]

    _, axs = plt.subplots(len(keys_to_compare), 1, figsize=(10, 6), sharex=True)

    for i, key in enumerate(keys_to_compare):
        match key:
            case "R0":
                el_paso_data = arase_data_el_paso.R0
                axs[i].set_ylabel("R0 (RE)")
            case "MLT":
                el_paso_data = arase_data_el_paso.MLT
                axs[i].set_ylabel("MLT (h)")

        orbit_data = orbit_vars[key].get_data().astype(np.float64)

        axs[i].plot(orbit_time, orbit_data, "r", label="From Arase Team")
        axs[i].plot(arase_data_el_paso.datetime, el_paso_data, "b:", label="From EL-PASO")

    plt.legend()
    plt.savefig(f"{Path(__file__).parent / f'arase_level_3_orbit_test_{mag_field}.png'}")
