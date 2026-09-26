# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import el_paso as ep
import pytest
from el_paso.dataset import DataSet
from el_paso.recipes.poes import process_poes_meped_electron, process_poes_ted_electron


@pytest.mark.basic
def test_poes_ted_electron(
    tmpdir: Path,
    *,
    renew_solution: bool,
) -> None:

    start_time = datetime(2013, 9, 8, tzinfo=timezone.utc)
    end_time = start_time + timedelta(hours=4)

    processed_data_path = tmpdir

    process_poes_ted_electron(
        start_time=start_time,
        end_time=end_time,
        satellite="metop1",
        raw_data_path=Path(__file__).parent / "data" / "raw",
        processed_data_path=processed_data_path,
        num_cores=32,
        calculate_Lm_Lstar=True,
    )

    out_path = processed_data_path / "POES" / "metop1" / f"metop1_ted_{start_time:%Y%m%d}_T89.nc"
    assert out_path.exists()

    poes_proc = DataSet(
        start_time=start_time,
        end_time=end_time,
        saving_strategy=ep.saving_strategies.DailyLEORBStrategy(
            tmpdir,
            "POES",
            "metop1",
            "ted",
            "T89",
            data_standard=ep.data_standards.GFZStandard(),
            file_format="nc",
        ),
    )

    if renew_solution:
        # Copy every output file this strategy produces (e.g. "full" and "solar_wind_indices"),
        # not just the primary one, so a strategy that later gains more output groups stays covered.
        # get_file_path() needs the strategy's own rounded interval, not the raw start_time/end_time,
        # to compute the correct file name.
        dest_dir = Path(__file__).parent / "data" / "processed" / "POES" / "metop1"
        interval_start, interval_end = poes_proc.saving_strategy.get_time_intervals_to_save(start_time, end_time)[0]
        for output_file in poes_proc.saving_strategy.output_files:
            src_path = poes_proc.saving_strategy.get_file_path(interval_start, interval_end, output_file)
            if src_path.exists():
                shutil.copy(src_path, dest_dir)

    poes_true = DataSet(
        start_time=start_time,
        end_time=end_time,
        saving_strategy=ep.saving_strategies.DailyLEORBStrategy(
            Path(__file__).parent / "data" / "processed",
            "POES",
            "metop1",
            "ted",
            "T89",
            data_standard=ep.data_standards.GFZStandard(),
            file_format="nc",
        ),
    )

    poes_proc.assert_equal(poes_true)


@pytest.mark.basic
def test_poes_meped_electron(
    tmpdir: Path,
    *,
    renew_solution: bool,
) -> None:

    start_time = datetime(2013, 9, 8, tzinfo=timezone.utc)
    end_time = start_time + timedelta(hours=4)

    processed_data_path = tmpdir

    process_poes_meped_electron(
        start_time=start_time,
        end_time=end_time,
        satellite="noaa18",
        raw_data_path=Path(__file__).parent / "data" / "raw",
        processed_data_path=processed_data_path,
        num_cores=32,
    )

    out_path = processed_data_path / "POES" / "noaa18" / f"noaa18_meped_{start_time:%Y%m%d}_T89.nc"
    assert out_path.exists()

    poes_proc = DataSet(
        start_time=start_time,
        end_time=end_time,
        saving_strategy=ep.saving_strategies.DailyLEORBStrategy(
            tmpdir,
            "POES",
            "noaa18",
            "meped",
            "T89",
            data_standard=ep.data_standards.GFZStandard(),
            file_format="nc",
        ),
    )

    if renew_solution:
        # Copy every output file this strategy produces (e.g. "full" and "solar_wind_indices"),
        # not just the primary one, so a strategy that later gains more output groups stays covered.
        # get_file_path() needs the strategy's own rounded interval, not the raw start_time/end_time,
        # to compute the correct file name.
        dest_dir = Path(__file__).parent / "data" / "processed" / "POES" / "noaa18"
        interval_start, interval_end = poes_proc.saving_strategy.get_time_intervals_to_save(start_time, end_time)[0]
        for output_file in poes_proc.saving_strategy.output_files:
            src_path = poes_proc.saving_strategy.get_file_path(interval_start, interval_end, output_file)
            if src_path.exists():
                shutil.copy(src_path, dest_dir)

    poes_true = DataSet(
        start_time=start_time,
        end_time=end_time,
        saving_strategy=ep.saving_strategies.DailyLEORBStrategy(
            Path(__file__).parent / "data" / "processed",
            "POES",
            "noaa18",
            "meped",
            "T89",
            data_standard=ep.data_standards.GFZStandard(),
            file_format="nc",
        ),
    )

    poes_proc.assert_equal(poes_true)
