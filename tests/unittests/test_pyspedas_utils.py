# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pyspedas-to-EL-PASO conversion helpers in `el_paso.pyspedas_utils`.

These build synthetic tplot variables with `pyspedas.store_data` rather than downloading
anything, so they cover the conversion logic without touching the network.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import el_paso as ep
import numpy as np
import pyspedas
import pytest
from astropy import units as u
from el_paso.pyspedas_utils import (
    build_trange,
    load_with_local_fallback,
    set_pyspedas_data_dir,
    tplot_to_bin_variable,
    tplot_to_time_variable,
    tplot_to_variable,
    unpack_tplot,
)
from pyspedas.projects.themis import config as themis_config

if TYPE_CHECKING:
    from pathlib import Path

_N_TIME = 5
_N_FREQ = 3
_FIRST_TIME = datetime(2024, 5, 10, 20, 0, tzinfo=timezone.utc)


@pytest.fixture
def spectrogram_tplot_name() -> str:
    """Store a time/frequency spectrogram as a tplot variable and return its name."""
    name = "el_paso_test_spectrogram"
    times = np.array([(_FIRST_TIME.timestamp() + 60 * i) for i in range(_N_TIME)])
    values = np.arange(_N_TIME * _N_FREQ, dtype=np.float64).reshape(_N_TIME, _N_FREQ)
    bins = np.array([10.0, 100.0, 1000.0])

    pyspedas.store_data(name, data={"x": times, "y": values, "v": bins})
    return name


@pytest.mark.minimal
@pytest.mark.basic
def test_build_trange_formats_both_bounds() -> None:
    start_time = datetime(2024, 5, 10, 20, 0, 0, tzinfo=timezone.utc)
    end_time = datetime(2024, 5, 11, 3, 30, 15, tzinfo=timezone.utc)

    assert build_trange(start_time, end_time) == ["2024-05-10 20:00:00", "2024-05-11 03:30:15"]


@pytest.mark.minimal
@pytest.mark.basic
def test_unpack_tplot_returns_times_values_and_bins(spectrogram_tplot_name: str) -> None:
    times, values, bins = unpack_tplot(spectrogram_tplot_name)

    assert times.shape == (_N_TIME,)
    assert values.shape == (_N_TIME, _N_FREQ)
    assert bins is not None
    np.testing.assert_allclose(bins, [10.0, 100.0, 1000.0])


@pytest.mark.minimal
@pytest.mark.basic
def test_tplot_to_variable_preserves_values_and_unit(spectrogram_tplot_name: str) -> None:
    psd_unit = (u.nT) ** 2 / u.Hz

    variable = tplot_to_variable(spectrogram_tplot_name, psd_unit)

    assert variable.metadata.unit == psd_unit
    np.testing.assert_allclose(
        variable.get_data(psd_unit), np.arange(_N_TIME * _N_FREQ, dtype=np.float64).reshape(_N_TIME, _N_FREQ)
    )


@pytest.mark.minimal
@pytest.mark.basic
def test_tplot_to_time_variable_is_posixtime(spectrogram_tplot_name: str) -> None:
    """Tplot stores POSIX seconds, so the time axis must survive the round trip untouched."""
    time_variable = tplot_to_time_variable(spectrogram_tplot_name)

    assert time_variable.metadata.unit == ep.units.posixtime

    first = time_variable.get_data(ep.units.posixtime)[0]
    assert datetime.fromtimestamp(first, tz=timezone.utc) == _FIRST_TIME


@pytest.mark.minimal
@pytest.mark.basic
def test_tplot_to_bin_variable_returns_the_frequency_axis(spectrogram_tplot_name: str) -> None:
    bin_variable = tplot_to_bin_variable(spectrogram_tplot_name, u.Hz)

    assert bin_variable.metadata.unit == u.Hz
    np.testing.assert_allclose(bin_variable.get_data(u.Hz), [10.0, 100.0, 1000.0])


@pytest.mark.minimal
@pytest.mark.basic
def test_tplot_to_bin_variable_collapses_a_per_record_axis() -> None:
    """Some products repeat the bin centres per record; the grid is fixed, so one row suffices."""
    name = "el_paso_test_repeated_bins"
    pyspedas.store_data(
        name,
        data={
            "x": np.arange(_N_TIME, dtype=np.float64),
            "y": np.zeros((_N_TIME, _N_FREQ)),
            "v": np.tile(np.array([1.0, 2.0, 3.0]), (_N_TIME, 1)),
        },
    )

    bin_variable = tplot_to_bin_variable(name, u.Hz)

    assert bin_variable.get_data().ndim == 1
    np.testing.assert_allclose(bin_variable.get_data(u.Hz), [1.0, 2.0, 3.0])


@pytest.mark.minimal
@pytest.mark.basic
def test_tplot_to_bin_variable_rejects_a_variable_without_bins() -> None:
    name = "el_paso_test_no_bins"
    pyspedas.store_data(name, data={"x": np.arange(_N_TIME, dtype=np.float64), "y": np.zeros(_N_TIME)})

    with pytest.raises(ValueError, match="no bin"):
        tplot_to_bin_variable(name, u.Hz)


@pytest.mark.minimal
@pytest.mark.basic
def test_missing_tplot_variable_raises() -> None:
    """A failed or empty download surfaces as a missing tplot variable, not as silent garbage."""
    with pytest.raises(ValueError, match="not found"):
        tplot_to_variable("el_paso_test_variable_that_does_not_exist", u.Hz)


@pytest.mark.minimal
@pytest.mark.basic
def test_set_pyspedas_data_dir_redirects_downloads(tmp_path: Path) -> None:
    """The loaders read CONFIG on each call, which is what makes `raw_data_path` work."""
    original = themis_config.CONFIG["local_data_dir"]
    try:
        set_pyspedas_data_dir("themis", tmp_path)

        assert themis_config.CONFIG["local_data_dir"] == str(tmp_path)
    finally:
        themis_config.CONFIG["local_data_dir"] = original


@pytest.mark.minimal
@pytest.mark.basic
def test_set_pyspedas_data_dir_works_for_other_projects(tmp_path: Path) -> None:
    """The helper is not THEMIS-specific; every pyspedas project shares the CONFIG pattern."""
    from pyspedas.projects.rbsp import config as rbsp_config  # noqa: PLC0415

    original = rbsp_config.CONFIG["local_data_dir"]
    try:
        set_pyspedas_data_dir("rbsp", tmp_path)

        assert rbsp_config.CONFIG["local_data_dir"] == str(tmp_path)
    finally:
        rbsp_config.CONFIG["local_data_dir"] = original


@pytest.mark.minimal
@pytest.mark.basic
def test_set_pyspedas_data_dir_rejects_an_unknown_project(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no configurable data directory"):
        set_pyspedas_data_dir("not_a_real_mission", tmp_path)


class _FakeLoader:
    """Records its calls like a pyspedas load routine; optionally fails the download-only step."""

    def __init__(self, *, fail_download: bool = False, loaded: list[str] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.fail_download = fail_download
        self.loaded = ["tha_fgs_gse"] if loaded is None else loaded

    def __call__(self, **kwargs: object) -> list[str]:
        self.calls.append(kwargs)
        if kwargs.get("downloadonly") and self.fail_download:
            msg = "remote index timed out"
            raise ConnectionError(msg)
        return [] if kwargs.get("downloadonly") else self.loaded


@pytest.mark.minimal
@pytest.mark.basic
def test_load_with_local_fallback_updates_then_loads_local_files() -> None:
    loader = _FakeLoader()

    loaded = load_with_local_fallback(loader, trange=["2024-05-10", "2024-05-11"], probe="a")

    assert loaded == ["tha_fgs_gse"]
    assert loader.calls == [
        {"trange": ["2024-05-10", "2024-05-11"], "probe": "a", "downloadonly": True},
        {"trange": ["2024-05-10", "2024-05-11"], "probe": "a", "no_update": True},
    ]


@pytest.mark.minimal
@pytest.mark.basic
def test_load_with_local_fallback_loads_local_files_if_update_fails(caplog: pytest.LogCaptureFixture) -> None:
    loader = _FakeLoader(fail_download=True)

    loaded = load_with_local_fallback(loader, probe="a")

    assert loaded == ["tha_fgs_gse"]
    assert loader.calls[-1] == {"probe": "a", "no_update": True}
    assert "using previously downloaded files only" in caplog.text


@pytest.mark.minimal
@pytest.mark.basic
def test_load_with_local_fallback_warns_if_nothing_is_loaded(caplog: pytest.LogCaptureFixture) -> None:
    loaded = load_with_local_fallback(_FakeLoader(loaded=[]), probe="a")

    assert loaded == []
    assert "loaded no data from the local files" in caplog.text
