# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import functools
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

import el_paso as ep
from el_paso.utils import (
    enforce_utc_timezone,
    extract_version,
    fill_str_template_with_time,
    get_file_by_version,
    make_dict_hashable,
    timed_function,
)

REF_DT = datetime(2023, 11, 5, tzinfo=timezone.utc)


# ── fill_str_template_with_time ───────────────────────────────────────────────


@pytest.mark.basic
@pytest.mark.parametrize(
    ("template", "expected"),
    [
        ("file_yyyymmdd.nc", "file_20231105.nc"),
        ("file_YYYYMMDD.nc", "file_20231105.nc"),
        ("YYYY/MM/DD/data.csv", "2023/11/05/data.csv"),
        ("no_placeholder.nc", "no_placeholder.nc"),
    ],
)
def test_fill_str_template_with_time_placeholders(template: str, expected: str) -> None:
    assert fill_str_template_with_time(template, REF_DT) == expected


@pytest.mark.basic
def test_fill_str_template_multiple_occurrences() -> None:
    result = fill_str_template_with_time("YYYY_YYYY.nc", REF_DT)
    assert result == "2023_2023.nc"


@pytest.mark.basic
def test_fill_str_template_mixed_placeholders() -> None:
    result = fill_str_template_with_time("path/yyyymmdd/YYYY-MM-DD.nc", REF_DT)
    assert result == "path/20231105/2023-11-05.nc"


# ── extract_version ───────────────────────────────────────────────────────────


@pytest.mark.basic
def test_extract_version_standard() -> None:
    base, ver = extract_version("file_v1.2.3.nc")
    assert base == "file"
    assert str(ver) == "1.2.3"


@pytest.mark.basic
def test_extract_version_underscore_separators() -> None:
    _, ver = extract_version("file_v1_2_3.nc")
    assert str(ver) == "1.2.3"


@pytest.mark.basic
def test_extract_version_dash_separators() -> None:
    _, ver = extract_version("file_v1-2-3.nc")
    assert str(ver) == "1.2.3"


@pytest.mark.basic
def test_extract_version_no_version() -> None:
    base, ver = extract_version("file.nc")
    assert base == "file.nc"
    assert str(ver) == "0"


@pytest.mark.basic
def test_extract_version_path_input() -> None:
    base, ver = extract_version(Path("/some/dir/data_v2.0.0.nc"))
    assert "data" in base
    assert str(ver) == "2.0.0"


@pytest.mark.basic
def test_extract_version_no_extension_no_version() -> None:
    base, ver = extract_version("noext")
    assert base == "noext"
    assert str(ver) == "0"


# ── get_file_by_version ───────────────────────────────────────────────────────


@pytest.mark.basic
def test_get_file_by_version_latest_returns_highest() -> None:
    files = ["data_v1.0.0.nc", "data_v2.0.0.nc", "data_v1.5.0.nc"]
    assert get_file_by_version(files, "latest") == "data_v2.0.0.nc"


@pytest.mark.basic
def test_get_file_by_version_specific_match() -> None:
    files = ["data_v1.0.0.nc", "data_v2.0.0.nc"]
    assert get_file_by_version(files, "v1.0.0") == "data_v1.0.0.nc"


@pytest.mark.basic
def test_get_file_by_version_empty_iterable_returns_none() -> None:
    assert get_file_by_version([], "latest") is None


@pytest.mark.basic
def test_get_file_by_version_path_objects() -> None:
    files = [Path("data_v1.0.0.nc"), Path("data_v3.1.0.nc")]
    result = get_file_by_version(files, "latest")
    assert result == Path("data_v3.1.0.nc")


# ── enforce_utc_timezone ──────────────────────────────────────────────────────


@pytest.mark.basic
def test_enforce_utc_naive_gets_utc() -> None:
    dt = datetime(2023, 11, 5)  # noqa: DTZ001
    result = enforce_utc_timezone(dt)
    assert result.tzinfo == timezone.utc


@pytest.mark.basic
def test_enforce_utc_aware_unchanged() -> None:
    dt = datetime(2023, 11, 5, tzinfo=timezone.utc)
    result = enforce_utc_timezone(dt)
    assert result is dt


# ── timed_function ────────────────────────────────────────────────────────────


@pytest.mark.basic
def test_timed_function_returns_correct_result() -> None:
    @timed_function("test_add")
    def add(a: int, b: int) -> int:
        return a + b

    assert add(3, 4) == 7


@pytest.mark.basic
def test_timed_function_without_name() -> None:
    @timed_function()
    def multiply(a: int, b: int) -> int:
        return a * b

    assert multiply(6, 7) == 42


# ── make_dict_hashable ────────────────────────────────────────────────────────


@pytest.mark.basic
def test_make_dict_hashable_can_be_hashed() -> None:
    assert hash(make_dict_hashable({"a": 1, "b": 2})) == hash(make_dict_hashable({"b": 2, "a": 1}))


@pytest.mark.basic
def test_make_dict_hashable_with_variables_works_as_cache_key() -> None:
    # This is how compute_magnetic_field_variables hands indices_solar_wind to the functools.cache
    # on construct_maginput.
    calls: list[int] = []

    @functools.cache
    def cached_len(indices: dict[str, ep.Variable]) -> int:
        calls.append(1)
        return len(indices)

    kp = ep.Variable(data=np.array([1.0, 2.0]), original_unit=u.dimensionless_unscaled)
    indices = make_dict_hashable({"Kp": kp})

    assert cached_len(indices) == 1
    assert cached_len(indices) == 1
    assert len(calls) == 1
