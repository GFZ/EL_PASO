# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from bisect import bisect_right
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from el_paso.extract_variables_from_files import _construct_file_list


@pytest.mark.basic
def test_construct_file_list_with_weekly_callable_cadence(tmp_path: Path) -> None:
    """A weekly callable file_cadence should walk forward via curr_time + 7 days."""
    start_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2023, 1, 22, tzinfo=timezone.utc)

    expected_dates = [
        datetime(2023, 1, 1, tzinfo=timezone.utc),
        datetime(2023, 1, 8, tzinfo=timezone.utc),
        datetime(2023, 1, 15, tzinfo=timezone.utc),
        datetime(2023, 1, 22, tzinfo=timezone.utc),
    ]
    for date in expected_dates:
        (tmp_path / f"ns41_{date.strftime('%Y%m%d')}_v1.10.ascii").touch()

    def weekly(time: datetime) -> datetime:
        return time + timedelta(days=7)

    file_path_template = tmp_path / r"ns41_YYYYMMDD_v1\.10\.ascii"

    file_paths, time_intervals = _construct_file_list(start_time, end_time, weekly, file_path_template)

    assert [path.name for path in file_paths] == [
        f"ns41_{date.strftime('%Y%m%d')}_v1.10.ascii" for date in expected_dates
    ]

    assert len(time_intervals) == len(expected_dates)
    for idx, date in enumerate(expected_dates):
        interval_start, interval_end = time_intervals[idx]
        assert interval_start == date
        if idx < len(expected_dates) - 1:
            assert interval_end == expected_dates[idx + 1] - timedelta(seconds=1)

    # The last chunk's next_time (2023-01-29) is beyond end_time, so it must clamp to end_time.
    last_start, last_end = time_intervals[-1]
    assert last_start == end_time
    assert last_end == end_time - timedelta(seconds=1)


@pytest.mark.basic
def test_construct_file_list_with_two_digit_year_placeholder(tmp_path: Path) -> None:
    """The YY (2-digit year) placeholder should work end-to-end for the real ns41 naming convention."""
    start_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2023, 1, 15, tzinfo=timezone.utc)

    expected_dates = [
        datetime(2023, 1, 1, tzinfo=timezone.utc),
        datetime(2023, 1, 8, tzinfo=timezone.utc),
        datetime(2023, 1, 15, tzinfo=timezone.utc),
    ]
    for date in expected_dates:
        (tmp_path / f"ns41_{date.strftime('%y%m%d')}_v1.10.ascii").touch()

    def weekly(time: datetime) -> datetime:
        return time + timedelta(days=7)

    file_path_template = tmp_path / r"ns41_YYMMDD_v1\.10\.ascii"

    file_paths, time_intervals = _construct_file_list(start_time, end_time, weekly, file_path_template)

    assert [path.name for path in file_paths] == [
        f"ns41_{date.strftime('%y%m%d')}_v1.10.ascii" for date in expected_dates
    ]
    assert len(time_intervals) == len(expected_dates)


@pytest.mark.basic
def test_construct_file_list_with_irregular_gapped_cadence(tmp_path: Path) -> None:
    """A lookup-table-driven callable should skip over a gap and produce one spanning interval."""
    available_dates = [
        datetime(2023, 1, 1, tzinfo=timezone.utc),
        datetime(2023, 1, 8, tzinfo=timezone.utc),
        datetime(2023, 1, 15, tzinfo=timezone.utc),
        # 2023-01-22 deliberately skipped to simulate an instrument outage / gap.
        datetime(2023, 1, 29, tzinfo=timezone.utc),
        datetime(2023, 2, 5, tzinfo=timezone.utc),
    ]
    for date in available_dates:
        (tmp_path / f"ns41_{date.strftime('%Y%m%d')}_v1.10.ascii").touch()

    def irregular_cadence(curr_time: datetime) -> datetime:
        idx = bisect_right(available_dates, curr_time)
        if idx < len(available_dates):
            return available_dates[idx]
        return curr_time + timedelta(days=365)

    start_time = available_dates[0]
    end_time = available_dates[-1]

    file_path_template = tmp_path / r"ns41_YYYYMMDD_v1\.10\.ascii"

    file_paths, time_intervals = _construct_file_list(start_time, end_time, irregular_cadence, file_path_template)

    assert [path.name for path in file_paths] == [
        f"ns41_{date.strftime('%Y%m%d')}_v1.10.ascii" for date in available_dates
    ]
    assert len(time_intervals) == len(available_dates)

    # The interval starting 2023-01-15 must span across the missing 2023-01-22 date, all the
    # way to the next available file on 2023-01-29.
    gap_start, gap_end = time_intervals[2]
    assert gap_start == datetime(2023, 1, 15, tzinfo=timezone.utc)
    assert gap_end == datetime(2023, 1, 29, tzinfo=timezone.utc) - timedelta(seconds=1)


@pytest.mark.basic
def test_construct_file_list_with_callable_cadence_skips_missing_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A date with no file on disk should be skipped with a warning, not crash the loop."""
    start_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2023, 1, 15, tzinfo=timezone.utc)

    # Deliberately do not create a file for 2023-01-08.
    present_dates = [
        datetime(2023, 1, 1, tzinfo=timezone.utc),
        datetime(2023, 1, 15, tzinfo=timezone.utc),
    ]
    for date in present_dates:
        (tmp_path / f"ns41_{date.strftime('%Y%m%d')}_v1.10.ascii").touch()

    def weekly(time: datetime) -> datetime:
        return time + timedelta(days=7)

    file_path_template = tmp_path / r"ns41_YYYYMMDD_v1\.10\.ascii"

    with caplog.at_level("WARNING"):
        file_paths, time_intervals = _construct_file_list(start_time, end_time, weekly, file_path_template)

    assert [path.name for path in file_paths] == [
        f"ns41_{date.strftime('%Y%m%d')}_v1.10.ascii" for date in present_dates
    ]
    assert len(time_intervals) == len(present_dates)
    assert "No file found for 2023-01-08" in caplog.text
