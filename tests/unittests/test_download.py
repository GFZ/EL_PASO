# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from astropy import units as u

import el_paso as ep


def test_esa_api(tmp_path: Path, skip_if_unreachable: Callable[..., None]):

    skip_if_unreachable("https://swe.ssa.esa.int", "https://sso.s2p.esa.int")

    client_id = os.environ.get("ESA_CLIENT_ID")
    client_secret = os.environ.get("ESA_CLIENT_SECRET")

    if client_id is None:
        msg = "Client ID not found!"
        raise ValueError(msg)

    if client_secret is None:
        msg = "Client secret not found!."
        raise ValueError(msg)

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


@pytest.mark.basic
def test_request(tmp_path: Path, skip_if_unreachable: Callable[..., None], monkeypatch: pytest.MonkeyPatch):

    skip_if_unreachable("https://spdf.gsfc.nasa.gov")

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


@pytest.mark.basic
def test_ftp(tmp_path: Path, skip_if_unreachable: Callable[..., None]):
    skip_if_unreachable("ftp://gssc.esa.int")

    username = os.environ.get("GSSC_USER")
    password = os.environ.get("GSSC_PASSWORD")

    if username is None:
        msg = "GSSC username not found!"
        raise ValueError(msg)

    if password is None:
        msg = "GSSC password not found!"
        raise ValueError(msg)

    start_time = datetime(2024, 1, 3, tzinfo=timezone.utc)
    end_time = datetime(2024, 1, 4, tzinfo=timezone.utc)

    url = "ftp://gssc.esa.int/emu/galileo_gssc_emu_gsat0215_sd_l1/YYYY"
    file_name_stem = "galileo_gssc_emu_gsat0215_sd_l1_YYYYMMDD_V*.cdf.gz"

    ep.download(
        start_time,
        end_time,
        save_path=tmp_path,
        download_url=url,
        file_name_stem=file_name_stem,
        file_cadence="daily",
        method="ftp",
        authentication_info=(username, password),
        skip_existing=True,
    )

    files = list(tmp_path.glob("*"))
    assert len(files) == 1
    assert files[0].name == "galileo_gssc_emu_gsat0215_sd_l1_20240103_V01.cdf"

    data_path = tmp_path / "2024" / "01"

    assert not data_path.exists()

    ep.download(
        start_time,
        end_time,
        save_path=tmp_path,
        download_url=url,
        file_name_stem=file_name_stem,
        file_cadence="daily",
        method="ftp",
        authentication_info=(username, password),
        skip_existing=True,
        sort_raw_files_by_time=True,
    )

    assert data_path.exists()
    assert len(list(data_path.glob("*"))) == 1


def test_exit_after_download(caplog: pytest.LogCaptureFixture):

    # test if the programs exits; it should not
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

    ep.exit_after_download = True

    with pytest.raises(SystemExit) as sample_exception:
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

    assert sample_exception.value.code == 1
    assert "Exiting after ep.download is completed!" in caplog.text

    ep.exit_after_download = False
    os.environ["EL_PASO_EXIT_AFTER_DOWNLOAD"] = "True"

    with pytest.raises(SystemExit) as sample_exception:
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

    assert sample_exception.value.code == 1
    assert "Exiting after ep.download is completed!" in caplog.text


def test_skip_download_via_ep_flag(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):

    ep.skip_download = True

    was_called = False

    def spy_function() -> None:
        nonlocal was_called
        was_called = True

    monkeypatch.setattr(ep.utils, "enforce_utc_timezone", spy_function)

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


def test_skip_download_via_env_var(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):

    os.environ["EL_PASO_SKIP_DOWNLOAD"] = "True"

    was_called = False

    def spy_function() -> None:
        nonlocal was_called
        was_called = True

    monkeypatch.setattr(ep.utils, "enforce_utc_timezone", spy_function)

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


@pytest.mark.basic
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
        end_time=datetime(2013, 1, 4, tzinfo=timezone.utc),
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
