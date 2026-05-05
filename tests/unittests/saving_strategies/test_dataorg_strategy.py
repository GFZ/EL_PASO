# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0


import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from astropy import units as u  # type: ignore[reportMissingTypeStubs]
from scipy.io import savemat  # type: ignore[reportMissingTypeStubs]

import el_paso as ep

rng = np.random.default_rng(1337)


@pytest.mark.parametrize("file_format", [".mat", ".pickle"])
@pytest.mark.basic
def test_basic_dataorg_strategy(tmp_path: Path, file_format: Literal[".mat", ".pickle"]) -> None:
    start_time = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time = [start_time]
    for _ in range(100):
        time.append(time[-1] + timedelta(hours=1))
    end_time = time[-1]

    time = [t.timestamp() for t in time]
    time_var = ep.Variable(original_unit=ep.units.posixtime, data=np.asarray(time))

    variables_to_save = {
        "time": time_var,
        "Flux": ep.Variable(
            original_unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),  # type: ignore[reportUnknownArgumentType]
            data=rng.normal(size=(len(time), 11, 5)),
        ),
        "Lstar": ep.Variable(original_unit=u.dimensionless_unscaled, data=rng.normal(size=(len(time), 5))),
    }

    strategy = ep.saving_strategies.DataOrgStrategy(
        base_data_path=tmp_path,
        mission="mission",
        satellite="satellite",
        instrument="instrument",
        kext="T89",
        file_format=file_format,
    )

    ep.save(variables_to_save, strategy, start_time=start_time, end_time=end_time, time_var=time_var)

    save_path = (
        tmp_path
        / "MISSION"
        / "satellite"
        / "Processed_Mat_Files"
        / ("satellite_instrument_20150101to20150131_flux_ver4" + file_format)
    )
    assert save_path.exists()


@pytest.mark.basic
@pytest.mark.parametrize("file_format", [".mat", ".pickle"])
def test_dataorg_append_data_merges_existing_file(tmp_path: Path, file_format: Literal[".mat", ".pickle"]) -> None:
    existing_data = {
        "time": np.array([[1.0], [2.0], [4.0]]),
        "Flux": np.array([[10.0], [12.0], [40.0]]),
        "metadata": {"time": {"unit": "s"}, "Flux": {"unit": "1"}},
    }
    new_data = {
        "time": np.array([[2.0], [3.0]]),
        "Flux": np.array([[20.0], [30.0]]),
        "metadata": {"time": {"unit": "s"}, "Flux": {"unit": "1"}},
    }

    file_path = tmp_path / f"flux{file_format}"
    if file_format == ".mat":
        savemat(file_path, existing_data)
    else:
        with file_path.open("wb") as file:
            pickle.dump(existing_data, file)

    strategy = ep.saving_strategies.DataOrgStrategy(
        base_data_path=tmp_path,
        mission="mission",
        satellite="satellite",
        instrument="instrument",
        kext="T89",
    )

    if file_format == ".pickle":
        with pytest.warns(FutureWarning, match=r"Appending to '\.pickle' files is deprecated"):
            merged_data = strategy.append_data(file_path, new_data)
    else:
        merged_data = strategy.append_data(file_path, new_data)

    np.testing.assert_array_equal(merged_data["time"], np.array([[1.0], [2.0], [3.0], [4.0]]))
    np.testing.assert_array_equal(merged_data["Flux"], np.array([[10.0], [20.0], [30.0], [40.0]]))
