# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import os
import time
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
from astropy import units as u

import el_paso as ep
from el_paso.cache import _CACHE_SUBDIR, cleanup_stale_cache, clear_cache, get_cache_dir
from el_paso.processing.magnetic_field_utils import IrbemOptions

if TYPE_CHECKING:
    from pathlib import Path


def _make_time_var() -> ep.Variable:
    return ep.Variable(original_unit=ep.units.posixtime, data=np.array([1.0, 2.0, 3.0]))


def _make_xgeo_var() -> ep.Variable:
    return ep.Variable(original_unit=ep.units.RE, data=np.array([[1.0, 0.0, 0.0]] * 3))


def _fake_result() -> dict[str, ep.Variable]:
    return {
        "B_Eq_T89": ep.Variable(original_unit=u.nT, data=np.array([100.0, 200.0, 300.0])),
        "R_Eq_T89": ep.Variable(original_unit=ep.units.RE, data=np.array([6.0, 6.1, 6.2])),
        "MLT_Eq_T89": ep.Variable(original_unit=u.hour, data=np.array([12.0, 12.1, 12.2])),
        "xGEO_Eq_T89": ep.Variable(original_unit=ep.units.RE, data=np.array([[1.0, 0.0, 0.0]] * 3)),
    }


@pytest.mark.basic
def test_cache_hit_skips_recomputation(tmp_path: Path) -> None:
    """Second call with identical inputs must return cached result without recomputing."""
    compute_mod = importlib.import_module("el_paso.processing.compute_magnetic_field_variables")

    call_count = 0
    expected = _fake_result()

    def mock_compute_core(*args, **kwargs) -> dict[str, ep.Variable]:  # noqa: ANN002, ANN003, ARG001
        nonlocal call_count
        call_count += 1
        return expected

    with patch.object(compute_mod, "_compute_core", side_effect=mock_compute_core):
        result1 = compute_mod.compute_magnetic_field_variables(
            time_var=_make_time_var(),
            xgeo_var=_make_xgeo_var(),
            variables_to_compute=[("B_Eq", "T89")],
            irbem_options=IrbemOptions(),
            num_cores=1,
            cache_dir=tmp_path / "cache",
        )

        result2 = compute_mod.compute_magnetic_field_variables(
            time_var=_make_time_var(),
            xgeo_var=_make_xgeo_var(),
            variables_to_compute=[("B_Eq", "T89")],
            irbem_options=IrbemOptions(),
            num_cores=1,
            cache_dir=tmp_path / "cache",
        )

    assert call_count == 1
    np.testing.assert_array_equal(
        result1["B_Eq_T89"].get_data(), result2["B_Eq_T89"].get_data()
    )


@pytest.mark.basic
def test_cache_miss_on_different_input(tmp_path: Path) -> None:
    """Changing input data must trigger a fresh computation."""
    compute_mod = importlib.import_module("el_paso.processing.compute_magnetic_field_variables")

    call_count = 0

    def mock_compute_core(*args, **kwargs) -> dict[str, ep.Variable]:  # noqa: ANN002, ANN003, ARG001
        nonlocal call_count
        call_count += 1
        return _fake_result()

    xgeo1 = _make_xgeo_var()
    xgeo2 = ep.Variable(original_unit=ep.units.RE, data=np.array([[2.0, 0.0, 0.0]] * 3))

    with patch.object(compute_mod, "_compute_core", side_effect=mock_compute_core):
        compute_mod.compute_magnetic_field_variables(
            time_var=_make_time_var(),
            xgeo_var=xgeo1,
            variables_to_compute=[("B_Eq", "T89")],
            irbem_options=IrbemOptions(),
            num_cores=1,
            cache_dir=tmp_path / "cache",
        )

        compute_mod.compute_magnetic_field_variables(
            time_var=_make_time_var(),
            xgeo_var=xgeo2,
            variables_to_compute=[("B_Eq", "T89")],
            irbem_options=IrbemOptions(),
            num_cores=1,
            cache_dir=tmp_path / "cache",
        )

    assert call_count == 2


@pytest.mark.basic
def test_overwrite_cache_forces_recomputation(tmp_path: Path) -> None:
    """overwrite_cache=True must recompute even when a cached result exists."""
    compute_mod = importlib.import_module("el_paso.processing.compute_magnetic_field_variables")

    call_count = 0

    def mock_compute_core(*args, **kwargs) -> dict[str, ep.Variable]:  # noqa: ANN002, ANN003, ARG001
        nonlocal call_count
        call_count += 1
        return _fake_result()

    with patch.object(compute_mod, "_compute_core", side_effect=mock_compute_core):
        compute_mod.compute_magnetic_field_variables(
            time_var=_make_time_var(),
            xgeo_var=_make_xgeo_var(),
            variables_to_compute=[("B_Eq", "T89")],
            irbem_options=IrbemOptions(),
            num_cores=1,
            cache_dir=tmp_path / "cache",
        )

        compute_mod.compute_magnetic_field_variables(
            time_var=_make_time_var(),
            xgeo_var=_make_xgeo_var(),
            variables_to_compute=[("B_Eq", "T89")],
            irbem_options=IrbemOptions(),
            num_cores=1,
            cache_dir=tmp_path / "cache",
            overwrite_cache=True,
        )

    assert call_count == 2


@pytest.mark.basic
def test_no_caching_when_cache_dir_is_none() -> None:
    """cache_dir=None must skip caching entirely."""
    compute_mod = importlib.import_module("el_paso.processing.compute_magnetic_field_variables")

    call_count = 0

    def mock_compute_core(*args, **kwargs) -> dict[str, ep.Variable]:  # noqa: ANN002, ANN003, ARG001
        nonlocal call_count
        call_count += 1
        return _fake_result()

    with patch.object(compute_mod, "_compute_core", side_effect=mock_compute_core):
        compute_mod.compute_magnetic_field_variables(
            time_var=_make_time_var(),
            xgeo_var=_make_xgeo_var(),
            variables_to_compute=[("B_Eq", "T89")],
            irbem_options=IrbemOptions(),
            num_cores=1,
            cache_dir=None,
        )

        compute_mod.compute_magnetic_field_variables(
            time_var=_make_time_var(),
            xgeo_var=_make_xgeo_var(),
            variables_to_compute=[("B_Eq", "T89")],
            irbem_options=IrbemOptions(),
            num_cores=1,
            cache_dir=None,
        )

    assert call_count == 2


@pytest.mark.basic
def test_clear_cache_removes_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """clear_cache() must remove the joblib_cache subdirectory."""
    monkeypatch.setenv("HOME", str(tmp_path))
    cache_dir = tmp_path / ".elpaso" / _CACHE_SUBDIR
    cache_dir.mkdir(parents=True)
    (cache_dir / "some_entry").mkdir()

    clear_cache()

    assert not cache_dir.exists()


@pytest.mark.basic
def test_cleanup_stale_cache_removes_old_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Entries older than max_age_days must be removed; recent ones kept."""
    monkeypatch.setenv("HOME", str(tmp_path))
    cache_dir = tmp_path / ".elpaso" / _CACHE_SUBDIR
    cache_dir.mkdir(parents=True)

    old_entry = cache_dir / "old_entry"
    old_entry.mkdir()
    old_mtime = time.time() - 10 * 86400
    os.utime(old_entry, (old_mtime, old_mtime))

    recent_entry = cache_dir / "recent_entry"
    recent_entry.mkdir()

    cleanup_stale_cache(max_age_days=7)

    assert not old_entry.exists()
    assert recent_entry.exists()


@pytest.mark.basic
def test_get_cache_dir_creates_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """get_cache_dir() must create the directory if it doesn't exist."""
    monkeypatch.setenv("HOME", str(tmp_path))

    result = get_cache_dir()

    assert result == tmp_path / ".elpaso" / _CACHE_SUBDIR
    assert result.is_dir()
