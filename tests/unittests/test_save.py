# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import el_paso as ep
import netCDF4 as nC
import numpy as np
import pytest
from astropy import units as u
from el_paso.processing.magnetic_field_utils.construct_maginput import MagInputResult
from el_paso.processing.magnetic_field_utils.mag_field_enum import MagneticField
from el_paso.saving_strategy import OutputFile

rng = np.random.default_rng(1337)


class _SWStubStrategy(ep.SavingStrategy):
    """Minimal concrete strategy that just records the variables_dict it was asked to save."""

    def __init__(self) -> None:
        self.data_standard = ep.data_standards.GFZStandard()
        self.output_files = [OutputFile("probe", [])]
        self.mag_field = "T89"
        self.received_variables_dict: ep.typing.VariablesDict | None = None

    def get_time_intervals_to_save(self, start_time: datetime, end_time: datetime) -> list:
        return [(start_time, end_time)]

    def get_file_path(self, interval_start: datetime, interval_end: datetime, output_file: OutputFile) -> Path:  # noqa: ARG002
        return Path("/tmp/unused.nc")  # noqa: S108

    def get_file_path_stem(self) -> Path:
        return Path("/tmp/unused")  # noqa: S108

    def get_file_name_stem(self) -> str:
        return "unused"

    def get_target_variables(
        self,
        output_file: OutputFile,
        variables_dict: ep.typing.VariablesDict,
        time_var: ep.Variable | None,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> None:
        del output_file, time_var, start_time, end_time
        self.received_variables_dict = variables_dict


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


@pytest.mark.basic
def test_save_sw_merges_indices_via_construct_maginput() -> None:
    """save_sw=True must fetch indices through construct_maginput, not a fresh loader call.

    construct_maginput is cached, so reusing it (rather than calling
    load_indices_solar_wind_parameters directly) lets a save_sw=True call reuse whatever a prior
    compute_magnetic_field_variables call in the same process already loaded.
    """
    base_ts = datetime(2013, 1, 1, 12, tzinfo=timezone.utc).timestamp()
    time_var = ep.Variable(original_unit=ep.units.posixtime, data=base_ts + np.array([1.0, 2.0, 3.0]))
    kp_var = ep.Variable(original_unit=u.dimensionless_unscaled, data=np.array([1.0, 2.0, 3.0]))
    fake_result = MagInputResult(maginput={}, indices_solar_wind={"Kp": kp_var})

    strategy = _SWStubStrategy()

    with (
        patch("el_paso.save.construct_maginput", return_value=fake_result) as mock_construct,
        patch("el_paso.save.get_saveable_sw_indices", return_value=["Kp"]) as mock_get_saveable,
    ):
        ep.save(
            {},
            strategy,
            start_time=datetime(2013, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2013, 1, 2, tzinfo=timezone.utc),
            time_var=time_var,
            save_sw=True,
            ignore_validation=True,
        )

    mock_construct.assert_called_once_with(time_var, MagneticField.T89)
    mock_get_saveable.assert_called_once_with(MagneticField.T89, strategy.data_standard)
    assert strategy.received_variables_dict == {"Kp": kp_var}


@pytest.mark.basic
def test_save_sw_false_by_default_does_not_call_construct_maginput() -> None:
    base_ts = datetime(2013, 1, 1, 12, tzinfo=timezone.utc).timestamp()
    time_var = ep.Variable(original_unit=ep.units.posixtime, data=base_ts + np.array([1.0, 2.0, 3.0]))
    strategy = _SWStubStrategy()

    with patch("el_paso.save.construct_maginput") as mock_construct:
        ep.save(
            {},
            strategy,
            start_time=datetime(2013, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2013, 1, 2, tzinfo=timezone.utc),
            time_var=time_var,
            ignore_validation=True,
        )

    mock_construct.assert_not_called()
    assert strategy.received_variables_dict == {}


@pytest.mark.basic
def test_save_sw_without_time_var_raises() -> None:
    strategy = _SWStubStrategy()

    with pytest.raises(ValueError, match="save_sw=True requires time_var"):
        ep.save(
            {},
            strategy,
            start_time=datetime(2013, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2013, 1, 2, tzinfo=timezone.utc),
            save_sw=True,
            ignore_validation=True,
        )


def _sw_append_variables(start_time: datetime, hours: int) -> dict[ep.typing.InternalName, ep.Variable]:
    base_ts = start_time.timestamp()
    times = base_ts + np.arange(hours) * 3600.0
    return {
        "Epoch": ep.Variable(original_unit=ep.units.posixtime, data=times),
        "MLT": ep.Variable(original_unit=u.hour, data=np.full(hours, 12.0)),
        "R_Eq": ep.Variable(original_unit=ep.units.RE, data=np.full(hours, 6.0)),
    }


@pytest.mark.basic
def test_save_sw_appends_solar_wind_indices_file(tmp_path: Path) -> None:
    """save_sw=True plus append=True must merge into the solar_wind_indices file by timestamp.

    The second call's 12-hour-overlapping Kp value must replace the first call's value for the
    shared timestamps, while the non-overlapping timestamps from both calls survive untouched.
    """
    strategy = ep.saving_strategies.GFZStrategy(
        base_data_path=tmp_path,
        mission="GOES",
        satellite="primary",
        instrument="MAGED",
        mag_field="T89",
        data_standard=ep.data_standards.GFZStandard(),
    )

    start_1 = datetime(2013, 1, 1, tzinfo=timezone.utc)
    end_1 = datetime(2013, 1, 2, tzinfo=timezone.utc)
    variables_1 = _sw_append_variables(start_1, hours=24)
    kp_1 = ep.Variable(original_unit=u.dimensionless_unscaled, data=np.full(24, 2.0))

    with patch(
        "el_paso.save.construct_maginput",
        return_value=MagInputResult(maginput={}, indices_solar_wind={"Kp": kp_1}),
    ):
        ep.save(
            variables_1, strategy, start_time=start_1, end_time=end_1, time_var=variables_1["Epoch"], save_sw=True
        )

    start_2 = datetime(2013, 1, 1, 12, tzinfo=timezone.utc)
    end_2 = datetime(2013, 1, 2, 12, tzinfo=timezone.utc)
    variables_2 = _sw_append_variables(start_2, hours=24)
    kp_2 = ep.Variable(original_unit=u.dimensionless_unscaled, data=np.full(24, 8.0))

    with patch(
        "el_paso.save.construct_maginput",
        return_value=MagInputResult(maginput={}, indices_solar_wind={"Kp": kp_2}),
    ):
        ep.save(
            variables_2,
            strategy,
            start_time=start_2,
            end_time=end_2,
            time_var=variables_2["Epoch"],
            save_sw=True,
            append=True,
        )

    sw_output_file = next(f for f in strategy.output_files if f.name == "solar_wind_indices")
    file_path = strategy.get_file_path(start_1, end_1, sw_output_file)
    assert file_path.exists()

    loaded_data = ep.utils.load_mat_data(file_path)
    kp_data = np.asarray(loaded_data[strategy.data_standard.get_standard_name("Kp")]).flatten()

    # 24 + 24 hourly samples with a 12-hour overlap merge down to 36 unique timestamps, not 48.
    assert kp_data.shape[0] == 36
    assert np.sum(kp_data == 2.0) == 12
    assert np.sum(kp_data == 8.0) == 24
