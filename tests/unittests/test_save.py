# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import el_paso as ep
import netCDF4 as nC
import numpy as np
import pytest
from astropy import units as u

rng = np.random.default_rng(1337)


@pytest.mark.basic
def test_save_raises_warning_when_var_is_empty(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:

    variables_to_save: dict[ep.typing.InternalName, Any] = {
        "FEDU": ep.Variable(original_unit=u.dimensionless_unscaled, data=rng.normal((20, 21))),
        "Alpha": ep.Variable(original_unit=u.dimensionless_unscaled, data=rng.normal((10, 11))),
        "B_Calc": ep.Variable(original_unit=u.dimensionless_unscaled, data=np.full((51,), np.nan)),
    }

    save_path = tmp_path / ("test.nc")
    strategy = ep.saving_strategies.SingleFileStrategy(file_path=save_path)

    with caplog.at_level(logging.WARNING, logger="ep"):
        ep.save(
            variables_to_save,
            strategy,
            start_time=datetime(2013, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2013, 1, 2, tzinfo=timezone.utc),
        )

    assert any(
        r.levelno == logging.WARNING and "Variable B_Calc only holds NaN values!" in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.basic
def test_time_independent_variable_is_written_to_every_daily_file(tmp_path: Path) -> None:
    """A variable that is not time dependent belongs in full in each interval's file.

    `DailyWaveStrategy` splits the range into days and `save` truncates each time dependent
    variable to the day being written. The frequency axis carries no time dimension, so it
    must survive truncation intact and be repeated in every file, rather than being cut down
    or dropped.
    """
    n_freq = 4
    samples_per_day = 6

    start_time = datetime(2024, 5, 10, tzinfo=timezone.utc)
    end_time = datetime(2024, 5, 11, 23, 59, tzinfo=timezone.utc)

    # Six samples on each of the two days.
    day_one = np.array([start_time.timestamp() + 3600 * i for i in range(samples_per_day)])
    day_two = day_one + 24 * 3600
    times = np.concatenate([day_one, day_two])

    frequencies = np.logspace(1, 3, n_freq)
    psd = np.arange(times.size * n_freq, dtype=np.float64).reshape(times.size, n_freq)

    variables_to_save: dict[ep.typing.InternalName, ep.Variable] = {
        "Epoch": ep.Variable(ep.units.posixtime, data=times),
        "Wave_frequency": ep.Variable(u.Hz, data=frequencies),
        "Magnetic_Power_Spectral_Density": ep.Variable((u.nT) ** 2 / u.Hz, data=psd),
    }

    saving_strategy = ep.saving_strategies.DailyWaveStrategy(
        tmp_path, "THEMIS", "tha", "FFT", ep.data_standards.GFZStandard()
    )

    ep.save(
        variables_to_save,
        saving_strategy,
        start_time,
        end_time,
        time_var=variables_to_save["Epoch"],
    )

    out_dir = tmp_path / "THEMIS" / "tha"
    expected_files = [out_dir / "tha_fft_20240510.nc", out_dir / "tha_fft_20240511.nc"]

    for file_path in expected_files:
        assert file_path.exists(), f"{file_path.name} was not written"

        with nC.Dataset(file_path) as dataset:
            # The frequency axis is repeated in full, not split across the two files.
            np.testing.assert_allclose(dataset.variables["freq"][:], frequencies)
            assert dataset.dimensions["Wave_frequency"].size == n_freq

            # The time dependent variables carry only this day's samples.
            assert dataset.dimensions["Epoch"].size == samples_per_day
            assert dataset.variables["BB"].shape == (samples_per_day, n_freq)

    # Each file holds its own half of the power spectral density.
    with nC.Dataset(expected_files[0]) as first, nC.Dataset(expected_files[1]) as second:
        np.testing.assert_allclose(first.variables["BB"][:], psd[:samples_per_day])
        np.testing.assert_allclose(second.variables["BB"][:], psd[samples_per_day:])


@pytest.mark.basic
def test_interval_without_records_is_skipped(tmp_path: Path) -> None:
    """An interval containing no samples is skipped, leaving no file behind.

    A variable that is not time dependent survives truncation, so the data dictionary stays
    non-empty even when the interval holds no records. `save` therefore checks the time
    variable itself before writing; otherwise the writer would create the file and only then
    fail on the missing time axis, leaving an empty file on disk.
    """
    n_freq = 4
    n_time = 6

    # Two days requested, every sample inside the first.
    start_time = datetime(2024, 5, 10, tzinfo=timezone.utc)
    end_time = datetime(2024, 5, 11, 23, 59, tzinfo=timezone.utc)
    times = np.array([start_time.timestamp() + 3600 * i for i in range(n_time)])

    variables_to_save: dict[ep.typing.InternalName, ep.Variable] = {
        "Epoch": ep.Variable(ep.units.posixtime, data=times),
        "Wave_frequency": ep.Variable(u.Hz, data=np.logspace(1, 3, n_freq)),
        "Magnetic_Power_Spectral_Density": ep.Variable((u.nT) ** 2 / u.Hz, data=np.full((n_time, n_freq), 1e-5)),
    }

    saving_strategy = ep.saving_strategies.DailyWaveStrategy(
        tmp_path, "THEMIS", "tha", "FFT", ep.data_standards.GFZStandard()
    )

    ep.save(
        variables_to_save,
        saving_strategy,
        start_time,
        end_time,
        time_var=variables_to_save["Epoch"],
    )

    out_dir = tmp_path / "THEMIS" / "tha"
    assert (out_dir / "tha_fft_20240510.nc").exists()
    assert not (out_dir / "tha_fft_20240511.nc").exists()
