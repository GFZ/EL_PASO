# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import el_paso as ep
import numpy as np
import pytest
from astropy import units as u
from el_paso.recipes.solarwind.process_solar_wind_indices import (
    process_solar_wind_indices,
    solar_wind_indices_strategy,
)
from el_paso.saving_strategies.solar_wind_indices_strategy import ALL_SW_INDICES


@pytest.mark.basic
def test_strategy_saves_all_sw_indices_regardless_of_mag_field(tmp_path: Path) -> None:
    strategy = solar_wind_indices_strategy(tmp_path)

    assert len(strategy.output_files) == 1
    assert set(strategy.output_files[0].names_to_save) == {"Epoch", *ALL_SW_INDICES}


@pytest.mark.basic
def test_recipe_writes_a_monthly_solar_wind_indices_file(tmp_path: Path) -> None:
    start_time = datetime(2013, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2013, 1, 2, tzinfo=timezone.utc)
    num_samples = 24
    kp_var = ep.Variable(original_unit=u.dimensionless_unscaled, data=np.full(num_samples, 3.0))

    with patch(
        "el_paso.load_indices_solar_wind_parameters",
        return_value={"Kp": kp_var},
    ) as mock_load:
        process_solar_wind_indices(start_time, end_time, processed_data_path=tmp_path, bin_cadence=timedelta(hours=1))

    # the time grid handed to the index loader covers [start_time, end_time) at the requested cadence
    call_args = mock_load.call_args.args
    assert call_args[0] == start_time
    assert call_args[1] == end_time
    assert set(call_args[2]) == set(ALL_SW_INDICES)
    time_var = call_args[3]
    assert time_var.get_data(ep.units.posixtime).shape == (num_samples,)

    file_path = tmp_path / "SolarWindIndices_20130101to20130131.nc"
    assert file_path.exists()
