# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

import ast
import importlib
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import el_paso as ep
import pytest
import requests


@pytest.mark.usefixtures("enable_downloads")
@pytest.mark.network
def test_esa_api(tmp_path: Path):

    client_id = os.environ.get("ESA_CLIENT_ID")
    client_secret = os.environ.get("ESA_CLIENT_SECRET")

    if not client_id or not client_secret:
        pytest.skip("ESA_CLIENT_ID/ESA_CLIENT_SECRET not set; skipping live ESA API test.")

    start_time = datetime(2025, 3, 17, tzinfo=timezone.utc)
    end_time = datetime(2025, 3, 17, 1, tzinfo=timezone.utc)

    url = "https://swe.ssa.esa.int/hapi/data?id=spase://SSA/NumericalData/D3S/d3s_edrsc_ngrm_spid204030252_science_ep_l1_gc_v3"
    file_name_stem = "EDRS-C_ngrm_YYYYMMDD_L1d.csv"

    ep.download(
        start_time,
        end_time,
        save_path=tmp_path,
        file_cadence="daily",
        download_url=url,
        file_name_stem="",
        rename_file_name_stem=file_name_stem,
        method="esa_swe",
        authentication_info=(client_id, client_secret),
        skip_existing=False,
    )

    assert len(list(tmp_path.glob("*"))) == 1


@pytest.mark.usefixtures("enable_downloads")
@pytest.mark.network
def test_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):

    download_mod = importlib.import_module("el_paso.download")

    def _get_page_content_longer_timeout(
        url: str, authentication_info: tuple[str, str]
    ) -> download_mod.requests.Response | None:
        response_of_content = download_mod.requests.get(
            url, stream=True, timeout=30, auth=download_mod.HTTPDigestAuth(*authentication_info)
        )

        if response_of_content.status_code == download_mod.ERROR_NOT_FOUND:
            return None

        response_of_content.raise_for_status()

        return response_of_content

    monkeypatch.setattr(download_mod, "_get_page_content", _get_page_content_longer_timeout)

    start_time = datetime(2013, 3, 17, tzinfo=timezone.utc)
    end_time = datetime(2013, 3, 17, 1, tzinfo=timezone.utc)

    url = "https://spdf.gsfc.nasa.gov/pub/data/rbsp/rbspa/l3/ect/hope/pitchangle/rel04/YYYY/"
    file_name_stem = "rbspa_rel04_ect-hope-pa-l3_YYYYMMDD_.{6}.cdf"

    ep.download(
        start_time,
        end_time,
        save_path=tmp_path,
        download_url=url,
        file_name_stem=file_name_stem,
        file_cadence="daily",
        method="request",
        skip_existing=True,
    )

    files = list(tmp_path.glob("*"))
    assert len(files) == 1

    data_path = tmp_path / "2013" / "03"

    assert not data_path.exists()

    ep.download(
        start_time,
        end_time,
        save_path=tmp_path,
        download_url=url,
        file_name_stem=file_name_stem,
        file_cadence="daily",
        method="request",
        skip_existing=True,
        sort_raw_files_by_time=True,
    )

    assert data_path.exists()
    assert len(list(data_path.glob("*"))) == 1


@pytest.mark.usefixtures("enable_downloads")
@pytest.mark.standard
def test_get_next_time_with_callable_cadence() -> None:
    """A callable file_cadence should be invoked with curr_time and its return value used directly."""
    download_mod = importlib.import_module("el_paso.download")

    curr_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def weekly(time: datetime) -> datetime:
        return time + timedelta(days=7)

    next_time = download_mod._get_next_time(curr_time, weekly)

    assert next_time == datetime(2024, 1, 8, tzinfo=timezone.utc)


@pytest.mark.usefixtures("enable_downloads")
@pytest.mark.standard
def test_download_with_callable_cadence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A callable file_cadence should drive ep.download the same way a built-in literal does."""
    directory_listing = MagicMock()
    directory_listing.text = "\n".join(
        f"ns41_{(datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=7 * i)).strftime('%Y%m%d')}_v1.10.ascii"
        for i in range(3)
    )

    download_mod = importlib.import_module("el_paso.download")

    download_mod._get_page_content.cache_clear()
    monkeypatch.setattr(download_mod, "_get_page_content", lambda _url, _auth: directory_listing)

    downloaded_urls: list[str] = []

    def mock_get(url: str, **_kwargs: object) -> MagicMock:
        downloaded_urls.append(url)
        resp = MagicMock()
        resp.status_code = 200
        resp.iter_content.return_value = [b"content"]
        return resp

    monkeypatch.setattr(download_mod.requests, "get", mock_get)

    def weekly(time: datetime) -> datetime:
        return time + timedelta(days=7)

    ep.download(
        start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 22, tzinfo=timezone.utc),
        save_path=tmp_path,
        file_cadence=weekly,
        download_url="https://fake.server/data/",
        file_name_stem=r"ns41_YYYYMMDD_v1\.10\.ascii",
        method="request",
        skip_existing=False,
    )

    assert len(downloaded_urls) == 3


@pytest.mark.usefixtures("enable_downloads")
@pytest.mark.standard
def test_download_single_file_cadence_runs_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """file_cadence="single_file" must create exactly one task instead of looping forever."""
    download_mod = importlib.import_module("el_paso.download")

    steps: list[tuple[datetime, datetime]] = []

    def mock_step(curr_time: datetime, next_time: datetime, **_kwargs: object) -> None:
        steps.append((curr_time, next_time))

    monkeypatch.setattr(download_mod, "_download_single_step", mock_step)

    start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2024, 1, 2, tzinfo=timezone.utc)

    ep.download(
        start_time=start_time,
        end_time=end_time,
        save_path=tmp_path,
        file_cadence="single_file",
        download_url="https://fake.server/data/",
        file_name_stem="file.txt",
    )

    assert steps == [(start_time, end_time)]


@pytest.mark.usefixtures("enable_downloads")
@pytest.mark.standard
@pytest.mark.parametrize(
    ("file_cadence", "start_time", "end_time", "expected_starts"),
    [
        (
            "daily",
            datetime(2013, 2, 28, 23, 50, tzinfo=timezone.utc),
            datetime(2013, 3, 3, 22, 10, tzinfo=timezone.utc),
            [
                datetime(2013, 2, 28, tzinfo=timezone.utc),
                *(datetime(2013, 3, d, tzinfo=timezone.utc) for d in (1, 2, 3)),
            ],
        ),
        (
            "monthly",
            datetime(2024, 1, 15, 12, tzinfo=timezone.utc),
            datetime(2024, 3, 1, 6, tzinfo=timezone.utc),
            [datetime(2024, m, 1, tzinfo=timezone.utc) for m in (1, 2, 3)],
        ),
    ],
)
def test_download_includes_end_file_when_start_is_late_in_its_day(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    file_cadence: str,
    start_time: datetime,
    end_time: datetime,
    expected_starts: list[datetime],
) -> None:
    """A start time later in its day than the end time must not drop the file of the end day or month.

    Stepping a full day from 2013-02-28 23:50 would land on 2013-03-03 23:50, past the end time, and skip
    the 03-03 file; the same holds for months stepped from a time of day later than the end's.
    """
    download_mod = importlib.import_module("el_paso.download")

    steps: list[datetime] = []
    monkeypatch.setattr(download_mod, "_download_single_step", lambda curr_time, **_kwargs: steps.append(curr_time))

    ep.download(
        start_time=start_time,
        end_time=end_time,
        save_path=tmp_path,
        file_cadence=file_cadence,  # ty:ignore[invalid-argument-type]
        download_url="https://fake.server/data/",
        file_name_stem="file_YYYYMMDD.cdf",
    )

    assert sorted(steps) == expected_starts


@pytest.mark.usefixtures("enable_downloads")
@pytest.mark.network
def test_ftp(tmp_path: Path):

    start_time = datetime(2024, 1, 3, tzinfo=timezone.utc)
    end_time = datetime(2024, 1, 4, tzinfo=timezone.utc)

    url = "ftp://ftp.gfz.de/pub/home/obs/Kp_ap_Ap_SN_F107/"
    file_name_stem = "Kp_ap_Ap_SN_F107_YYYY.txt"

    ep.download(
        start_time,
        end_time,
        save_path=tmp_path,
        download_url=url,
        file_name_stem=file_name_stem,
        file_cadence="daily",
        method="ftp",
        skip_existing=True,
    )

    files = list(tmp_path.glob("*"))
    assert len(files) == 1
    assert files[0].name == "Kp_ap_Ap_SN_F107_2024.txt"


@pytest.mark.usefixtures("enable_downloads")
@pytest.mark.standard
def test_exit_after_download(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch):
    download_module = importlib.import_module("el_paso.download")

    def run_download() -> None:
        ep.download(
            datetime(2000, 1, 1, tzinfo=timezone.utc),
            datetime(1999, 1, 1, tzinfo=timezone.utc),
            save_path="",
            download_url="",
            file_name_stem="",
            file_cadence="daily",
            method="request",
            skip_existing=True,
            sort_raw_files_by_time=True,
        )

    # nothing requested: neither download nor the exit check exits
    monkeypatch.setattr(ep, "exit_after_download", False)
    monkeypatch.delenv("EL_PASO_EXIT_AFTER_DOWNLOAD", raising=False)
    run_download()
    download_module.exit_if_download_only()

    # ep.download itself never exits, so recipes can download several products
    monkeypatch.setattr(ep, "exit_after_download", True)
    run_download()
    run_download()

    # the exit happens when the recipe starts processing data
    with pytest.raises(SystemExit) as sample_exception:
        download_module.exit_if_download_only()

    assert sample_exception.value.code == 0
    assert "Exiting after ep.download is completed!" in caplog.text

    monkeypatch.setattr(ep, "exit_after_download", False)
    monkeypatch.setenv("EL_PASO_EXIT_AFTER_DOWNLOAD", "True")
    run_download()

    with pytest.raises(SystemExit) as sample_exception:
        download_module.exit_if_download_only()

    assert sample_exception.value.code == 0

    # skipping the download takes precedence over exiting
    monkeypatch.setattr(ep, "skip_download", True)
    download_module.exit_if_download_only()


@pytest.mark.minimal
@pytest.mark.standard
def test_skip_download_via_ep_flag(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):

    monkeypatch.setattr(ep, "skip_download", True)

    was_called = False

    def spy_function(time: datetime) -> datetime:
        nonlocal was_called
        was_called = True
        return time

    monkeypatch.setattr(importlib.import_module("el_paso.download"), "enforce_utc_timezone", spy_function)

    ep.download(
        datetime(2000, 1, 1, tzinfo=timezone.utc),
        datetime(1999, 1, 1, tzinfo=timezone.utc),
        save_path="",
        download_url="",
        file_name_stem="",
        file_cadence="daily",
        method="request",
        skip_existing=True,
        sort_raw_files_by_time=True,
    )

    assert not was_called
    assert "Skipping ep.download" in caplog.text


@pytest.mark.minimal
@pytest.mark.standard
def test_skip_download_via_env_var(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):

    monkeypatch.setenv("EL_PASO_SKIP_DOWNLOAD", "True")

    was_called = False

    def spy_function(time: datetime) -> datetime:
        nonlocal was_called
        was_called = True
        return time

    monkeypatch.setattr(importlib.import_module("el_paso.download"), "enforce_utc_timezone", spy_function)

    ep.download(
        datetime(2000, 1, 1, tzinfo=timezone.utc),
        datetime(1999, 1, 1, tzinfo=timezone.utc),
        save_path="",
        download_url="",
        file_name_stem="",
        file_cadence="daily",
        method="request",
        skip_existing=True,
        sort_raw_files_by_time=True,
    )

    assert not was_called
    assert "Skipping ep.download" in caplog.text


@pytest.mark.minimal
@pytest.mark.standard
@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_falsy_env_values_neither_skip_nor_exit(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, value: str
) -> None:
    monkeypatch.setattr(ep, "skip_download", False)
    monkeypatch.setattr(ep, "exit_after_download", False)
    monkeypatch.setenv("EL_PASO_SKIP_DOWNLOAD", value)
    monkeypatch.setenv("EL_PASO_EXIT_AFTER_DOWNLOAD", value)

    was_called = False

    def spy_function(time: datetime) -> datetime:
        nonlocal was_called
        was_called = True
        return time

    monkeypatch.setattr(importlib.import_module("el_paso.download"), "enforce_utc_timezone", spy_function)

    ep.download(
        datetime(2000, 1, 1, tzinfo=timezone.utc),
        datetime(1999, 1, 1, tzinfo=timezone.utc),
        save_path="",
        download_url="",
        file_name_stem="",
        file_cadence="daily",
        method="request",
        skip_existing=True,
        sort_raw_files_by_time=True,
    )

    assert was_called
    assert "Skipping ep.download" not in caplog.text
    assert "Exiting after ep.download" not in caplog.text


@pytest.mark.usefixtures("enable_downloads")
@pytest.mark.standard
def test_skip_existing_false_does_not_overwrite_files_outside_time_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When skip_existing=False, only files within [start_time, end_time) should be re-downloaded."""
    for day in range(1, 6):
        (tmp_path / f"data_2013010{day}_v01.cdf").write_bytes(b"original")

    directory_listing = MagicMock()
    directory_listing.text = "\n".join(f"data_2013010{d}_v01.cdf" for d in range(1, 6))

    download_mod = importlib.import_module("el_paso.download")

    download_mod._get_page_content.cache_clear()
    monkeypatch.setattr(download_mod, "_get_page_content", lambda _url, _auth: directory_listing)

    downloaded_urls: list[str] = []

    def mock_get(url: str, **_kwargs: object) -> MagicMock:
        downloaded_urls.append(url)
        resp = MagicMock()
        resp.status_code = 200
        resp.iter_content.return_value = [b"new_content"]
        return resp

    monkeypatch.setattr(download_mod.requests, "get", mock_get)

    ep.download(
        start_time=datetime(2013, 1, 2, tzinfo=timezone.utc),
        end_time=datetime(2013, 1, 3, 23, 59, tzinfo=timezone.utc),
        save_path=tmp_path,
        file_cadence="daily",
        download_url="https://fake.server/data/",
        file_name_stem=r"data_YYYYMMDD_v\d+\.cdf",
        method="request",
        skip_existing=False,
    )

    # Files outside the time range must not be touched
    assert (tmp_path / "data_20130101_v01.cdf").read_bytes() == b"original"
    assert (tmp_path / "data_20130104_v01.cdf").read_bytes() == b"original"
    assert (tmp_path / "data_20130105_v01.cdf").read_bytes() == b"original"

    # Files inside the time range should have been overwritten
    assert (tmp_path / "data_20130102_v01.cdf").read_bytes() == b"new_content"
    assert (tmp_path / "data_20130103_v01.cdf").read_bytes() == b"new_content"

    # Only two download requests should have been made
    assert len(downloaded_urls) == 2
    assert any("data_20130102_v01.cdf" in url for url in downloaded_urls)
    assert any("data_20130103_v01.cdf" in url for url in downloaded_urls)


def _calls_function(tree: ast.AST, dotted_name: str) -> bool:
    return any(isinstance(node, ast.Call) and ast.unparse(node.func) == dotted_name for node in ast.walk(tree))


@pytest.mark.minimal
@pytest.mark.standard
def test_recipes_with_downloads_call_exit_if_download_only() -> None:
    recipes_dir = Path(ep.__file__).parent / "recipes"

    # Private helper modules (e.g. `rbsp/_emfisis_magnetometer.py`) download data for several recipes.
    # They must not exit themselves, since recipes call them before their other downloads, so their
    # download functions are checked where they are called instead.
    helper_modules = [path for path in recipes_dir.rglob("_*.py") if path.name != "__init__.py"]
    helper_download_functions = {
        node.name
        for path in helper_modules
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and _calls_function(node, "ep.download")
    }

    modules_with_download = []
    modules_without_exit = []

    for module_path in sorted(recipes_dir.rglob("*.py")):
        if module_path in helper_modules:
            continue

        tree = ast.parse(module_path.read_text())

        if _calls_function(tree, "ep.download") or any(
            _calls_function(tree, name) for name in helper_download_functions
        ):
            modules_with_download.append(module_path.name)

            if not (
                _calls_function(tree, "ep.exit_if_download_only")
                or _calls_function(tree, "get_arase_orbit_level_2_variables")
                or _calls_function(tree, "get_arase_orbit_level_3_variables")
            ):
                modules_without_exit.append(module_path.name)

    assert modules_with_download, "No recipes calling ep.download found, the test is not checking anything."
    assert not modules_without_exit, (
        "Recipes that call ep.download must call ep.exit_if_download_only() after all of their downloads "
        f"(needed for --exit-after-download): {modules_without_exit}"
    )


def _setup_fake_server(download_mod, monkeypatch: pytest.MonkeyPatch, get) -> list[float]:
    """Install a fake directory listing, the given requests.get replacement, and a recording time.sleep."""
    listing = MagicMock()
    listing.text = "data_20240101_v01.cdf"
    download_mod._get_page_content.cache_clear()
    monkeypatch.setattr(download_mod, "_get_page_content", lambda _url, _auth: listing)
    monkeypatch.setattr(download_mod.requests, "get", get)

    sleeps: list[float] = []
    monkeypatch.setattr(download_mod, "time", SimpleNamespace(sleep=sleeps.append))
    return sleeps


def _ok_response(content: bytes = b"content") -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.iter_content.return_value = [content]
    return resp


def _http_error_response(status: int, headers: dict[str, str] | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {}
    resp.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp)
    return resp


def _run_download(tmp_path: Path, **kwargs: object) -> None:
    ep.download(
        start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        save_path=tmp_path,
        file_cadence="daily",
        download_url="https://fake.server/data/",
        file_name_stem=r"data_YYYYMMDD_v\d+\.cdf",
        method="request",
        skip_existing=False,
        **kwargs,  # ty:ignore[invalid-argument-type]
    )


@pytest.mark.usefixtures("enable_downloads")
@pytest.mark.standard
def test_request_retries_once_after_connection_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    download_mod = importlib.import_module("el_paso.download")
    calls: list[str] = []

    def get(url: str, **_kwargs: object) -> MagicMock:
        calls.append(url)
        if len(calls) == 1:
            raise requests.exceptions.ConnectionError("boom")
        return _ok_response()

    sleeps = _setup_fake_server(download_mod, monkeypatch, get)
    _run_download(tmp_path)

    assert len(calls) == 2
    assert sleeps == [1.0]
    assert (tmp_path / "data_20240101_v01.cdf").read_bytes() == b"content"
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.usefixtures("enable_downloads")
@pytest.mark.standard
@pytest.mark.parametrize("status", [403, 404])
def test_request_does_not_retry_permanent_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    download_mod = importlib.import_module("el_paso.download")
    calls: list[str] = []

    def get(url: str, **_kwargs: object) -> MagicMock:
        calls.append(url)
        return _http_error_response(status)

    sleeps = _setup_fake_server(download_mod, monkeypatch, get)
    _run_download(tmp_path)

    assert len(calls) == 1
    assert sleeps == []
    assert not list(tmp_path.glob("*.cdf"))


@pytest.mark.usefixtures("enable_downloads")
@pytest.mark.standard
def test_request_honors_retry_after(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    download_mod = importlib.import_module("el_paso.download")
    calls: list[str] = []

    def get(url: str, **_kwargs: object) -> MagicMock:
        calls.append(url)
        if len(calls) == 1:
            return _http_error_response(429, {"Retry-After": "7"})
        return _ok_response()

    sleeps = _setup_fake_server(download_mod, monkeypatch, get)
    _run_download(tmp_path)

    assert sleeps == [7.0]
    assert (tmp_path / "data_20240101_v01.cdf").exists()


@pytest.mark.usefixtures("enable_downloads")
@pytest.mark.standard
def test_persistent_failure_is_retried_in_second_pass_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    download_mod = importlib.import_module("el_paso.download")
    calls: list[str] = []

    def get(url: str, **_kwargs: object) -> MagicMock:
        calls.append(url)
        raise requests.exceptions.ConnectionError("down")

    sleeps = _setup_fake_server(download_mod, monkeypatch, get)

    with caplog.at_level(logging.WARNING):
        _run_download(tmp_path)

    # 2 attempts in the first pass (initial + 1 retry) and 1 in the second pass
    assert len(calls) == 3
    assert len(sleeps) == 2
    assert "1 of 1 files could not be downloaded" in caplog.text
    assert not list(tmp_path.glob("*"))


@pytest.mark.usefixtures("enable_downloads")
@pytest.mark.standard
def test_failed_file_succeeds_in_second_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    download_mod = importlib.import_module("el_paso.download")
    calls: list[str] = []

    def get(url: str, **_kwargs: object) -> MagicMock:
        calls.append(url)
        if len(calls) <= 2:
            raise requests.exceptions.Timeout("slow")
        return _ok_response()

    _setup_fake_server(download_mod, monkeypatch, get)
    _run_download(tmp_path)

    assert (tmp_path / "data_20240101_v01.cdf").exists()


@pytest.mark.usefixtures("enable_downloads")
@pytest.mark.standard
def test_interrupted_stream_leaves_no_partial_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    download_mod = importlib.import_module("el_paso.download")

    def get(_url: str, **_kwargs: object) -> MagicMock:
        resp = MagicMock()
        resp.status_code = 200

        def chunks(**_kw: object):
            yield b"partial"
            raise requests.exceptions.ChunkedEncodingError("cut")

        resp.iter_content.side_effect = chunks
        return resp

    _setup_fake_server(download_mod, monkeypatch, get)
    _run_download(tmp_path, retries=0)

    assert not list(tmp_path.glob("*"))


@pytest.mark.usefixtures("enable_downloads")
@pytest.mark.standard
def test_ftp_connect_failure_is_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    download_mod = importlib.import_module("el_paso.download")
    attempts: list[int] = []

    class FakeFTP:
        def retrbinary(self, _cmd: str, callback) -> None:
            callback(b"ftp-content")

        def quit(self) -> None:
            pass

    def connect(_host: str, _auth: tuple[str, str], **_kwargs: object) -> FakeFTP:
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("refused")
        return FakeFTP()

    monkeypatch.setattr(download_mod, "_ftp_connect", connect)
    sleeps: list[float] = []
    monkeypatch.setattr(download_mod, "time", SimpleNamespace(sleep=sleeps.append))

    ep.download(
        start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        save_path=tmp_path,
        file_cadence="single_file",
        download_url="ftp://fake.server/dir",
        file_name_stem="file.txt",
        method="ftp",
    )

    assert len(attempts) == 2
    assert sleeps == [1.0]
    assert (tmp_path / "file.txt").read_bytes() == b"ftp-content"


@pytest.mark.usefixtures("enable_downloads")
@pytest.mark.standard
def test_unsupported_method_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported download method"):
        ep.download(
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            save_path=tmp_path,
            file_cadence="daily",
            download_url="https://fake.server/",
            file_name_stem="x",
            method="wget",  # ty:ignore[invalid-argument-type]
        )


@pytest.mark.usefixtures("enable_downloads")
@pytest.mark.standard
def test_esa_swe_failure_is_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    download_mod = importlib.import_module("el_paso.download")
    calls: list[dict] = []

    monkeypatch.setattr(download_mod, "_get_esa_swe_access_token", lambda *_args: "token")

    def get(url: str, **kwargs: object) -> MagicMock:
        calls.append({"url": url, **kwargs})
        if len(calls) == 1:
            return _http_error_response(503)
        resp = _ok_response(b"esa-content")
        resp.text = ""
        return resp

    monkeypatch.setattr(download_mod.requests, "get", get)
    sleeps: list[float] = []
    monkeypatch.setattr(download_mod, "time", SimpleNamespace(sleep=sleeps.append))

    ep.download(
        start_time=datetime(2025, 3, 17, tzinfo=timezone.utc),
        end_time=datetime(2025, 3, 17, 1, tzinfo=timezone.utc),
        save_path=tmp_path,
        file_cadence="single_file",
        download_url="https://swe.ssa.esa.int/hapi/data?id=fake",
        file_name_stem="",
        rename_file_name_stem="esa_YYYYMMDD.csv",
        method="esa_swe",
        authentication_info=("id", "secret"),
        skip_existing=False,
    )

    assert len(calls) == 2
    assert calls[0]["headers"] == {"Authorization": "Bearer token"}
    assert sleeps == [1.0]
    assert (tmp_path / "esa_20250317.csv").read_bytes() == b"esa-content"
    assert not list(tmp_path.glob("*.part"))
