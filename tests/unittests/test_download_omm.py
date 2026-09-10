# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

import csv
import importlib
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import el_paso as ep

_ISS_OMM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ndm xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<omm id="CCSDS_OMM_VERS" version="3.0">
<header>
<COMMENT>GENERATED VIA SPACE-TRACK.ORG API</COMMENT>
<CREATION_DATE>2024-01-01T01:16:16</CREATION_DATE>
<ORIGINATOR>18 SPCS</ORIGINATOR>
</header>
<body><segment>
<metadata>
<OBJECT_NAME>ISS (ZARYA)</OBJECT_NAME>
<OBJECT_ID>1998-067A</OBJECT_ID>
<CENTER_NAME>EARTH</CENTER_NAME>
<REF_FRAME>TEME</REF_FRAME>
<TIME_SYSTEM>UTC</TIME_SYSTEM>
<MEAN_ELEMENT_THEORY>SGP4</MEAN_ELEMENT_THEORY>
</metadata>
<data>
<meanElements>
<EPOCH>2024-01-01T00:18:14.850432</EPOCH>
<MEAN_MOTION>15.49961425</MEAN_MOTION>
<ECCENTRICITY>0.00033470</ECCENTRICITY>
<INCLINATION>51.6422</INCLINATION>
<RA_OF_ASC_NODE>68.6294</RA_OF_ASC_NODE>
<ARG_OF_PERICENTER>343.4617</ARG_OF_PERICENTER>
<MEAN_ANOMALY>78.0593</MEAN_ANOMALY>
</meanElements>
<tleParameters>
<EPHEMERIS_TYPE>0</EPHEMERIS_TYPE>
<CLASSIFICATION_TYPE>U</CLASSIFICATION_TYPE>
<NORAD_CAT_ID>25544</NORAD_CAT_ID>
<ELEMENT_SET_NO>999</ELEMENT_SET_NO>
<REV_AT_EPOCH>43247</REV_AT_EPOCH>
<BSTAR>0.00029758000000</BSTAR>
<MEAN_MOTION_DOT>0.00016541</MEAN_MOTION_DOT>
<MEAN_MOTION_DDOT>0.0000000000000</MEAN_MOTION_DDOT>
</tleParameters>
<userDefinedParameters>
<USER_DEFINED parameter="SEMIMAJOR_AXIS">6794.976</USER_DEFINED>
<USER_DEFINED parameter="OBJECT_TYPE">PAYLOAD</USER_DEFINED>
<USER_DEFINED parameter="DECAY_DATE"></USER_DEFINED>
</userDefinedParameters>
</data>
</segment></body>
</omm>
<omm id="CCSDS_OMM_VERS" version="3.0">
<header>
<COMMENT>GENERATED VIA SPACE-TRACK.ORG API</COMMENT>
<CREATION_DATE>2024-01-01T06:46:17</CREATION_DATE>
<ORIGINATOR>18 SPCS</ORIGINATOR>
</header>
<body><segment>
<metadata>
<OBJECT_NAME>ISS (ZARYA)</OBJECT_NAME>
<OBJECT_ID>1998-067A</OBJECT_ID>
<CENTER_NAME>EARTH</CENTER_NAME>
<REF_FRAME>TEME</REF_FRAME>
<TIME_SYSTEM>UTC</TIME_SYSTEM>
<MEAN_ELEMENT_THEORY>SGP4</MEAN_ELEMENT_THEORY>
</metadata>
<data>
<meanElements>
<EPOCH>2024-01-01T04:40:56.161344</EPOCH>
<MEAN_MOTION>15.49966131</MEAN_MOTION>
<ECCENTRICITY>0.00034720</ECCENTRICITY>
<INCLINATION>51.6400</INCLINATION>
<RA_OF_ASC_NODE>67.7274</RA_OF_ASC_NODE>
<ARG_OF_PERICENTER>339.1814</ARG_OF_PERICENTER>
<MEAN_ANOMALY>20.9033</MEAN_ANOMALY>
</meanElements>
<tleParameters>
<EPHEMERIS_TYPE>0</EPHEMERIS_TYPE>
<CLASSIFICATION_TYPE>U</CLASSIFICATION_TYPE>
<NORAD_CAT_ID>25544</NORAD_CAT_ID>
<ELEMENT_SET_NO>999</ELEMENT_SET_NO>
<REV_AT_EPOCH>43250</REV_AT_EPOCH>
<BSTAR>0.00029045000000</BSTAR>
<MEAN_MOTION_DOT>0.00014230</MEAN_MOTION_DOT>
<MEAN_MOTION_DDOT>0.0000000000000</MEAN_MOTION_DDOT>
</tleParameters>
</data>
</segment></body>
</omm>
</ndm>
"""

_EMPTY_OMM_XML = '<?xml version="1.0" encoding="UTF-8"?>\n<ndm></ndm>\n'


def _omm_segment(*, object_name: str, norad_cat_id: str, epoch: str) -> str:
    """Build one minimal <omm> element, for constructing multi-satellite fixtures."""
    return f"""<omm id="CCSDS_OMM_VERS" version="3.0">
<header><CREATION_DATE>{epoch}</CREATION_DATE><ORIGINATOR>18 SPCS</ORIGINATOR></header>
<body><segment>
<metadata><OBJECT_NAME>{object_name}</OBJECT_NAME><OBJECT_ID>2000-001A</OBJECT_ID></metadata>
<data>
<meanElements><EPOCH>{epoch}</EPOCH><MEAN_MOTION>1.0</MEAN_MOTION><ECCENTRICITY>0.0</ECCENTRICITY>
<INCLINATION>0.0</INCLINATION><RA_OF_ASC_NODE>0.0</RA_OF_ASC_NODE><ARG_OF_PERICENTER>0.0</ARG_OF_PERICENTER>
<MEAN_ANOMALY>0.0</MEAN_ANOMALY></meanElements>
<tleParameters><EPHEMERIS_TYPE>0</EPHEMERIS_TYPE><CLASSIFICATION_TYPE>U</CLASSIFICATION_TYPE>
<NORAD_CAT_ID>{norad_cat_id}</NORAD_CAT_ID><ELEMENT_SET_NO>999</ELEMENT_SET_NO><REV_AT_EPOCH>1</REV_AT_EPOCH>
<BSTAR>0.0</BSTAR><MEAN_MOTION_DOT>0.0</MEAN_MOTION_DOT><MEAN_MOTION_DDOT>0.0</MEAN_MOTION_DDOT></tleParameters>
</data>
</segment></body>
</omm>"""


def _multi_satellite_xml(*segments: str) -> str:
    return '<?xml version="1.0" encoding="UTF-8"?>\n<ndm>\n' + "\n".join(segments) + "\n</ndm>\n"


def test_parse_omm_xml_extracts_all_fields() -> None:
    download_omm_mod = importlib.import_module("el_paso.download_omm")
    records = download_omm_mod._parse_omm_xml(_ISS_OMM_XML)

    assert len(records) == 2

    first = records[0]
    assert first["OBJECT_NAME"] == "ISS (ZARYA)"
    assert first["EPOCH"] == "2024-01-01T00:18:14.850432"
    assert first["MEAN_MOTION"] == "15.49961425"
    assert first["NORAD_CAT_ID"] == "25544"
    assert first["BSTAR"] == "0.00029758000000"
    # userDefinedParameters extension fields are flattened in by their 'parameter' attribute
    assert first["SEMIMAJOR_AXIS"] == "6794.976"
    assert first["OBJECT_TYPE"] == "PAYLOAD"
    assert first["DECAY_DATE"] == ""

    second = records[1]
    assert second["EPOCH"] == "2024-01-01T04:40:56.161344"
    assert second["REV_AT_EPOCH"] == "43250"
    # second record has no userDefinedParameters block
    assert "SEMIMAJOR_AXIS" not in second


def test_parse_omm_xml_empty_response() -> None:
    download_omm_mod = importlib.import_module("el_paso.download_omm")
    assert download_omm_mod._parse_omm_xml(_EMPTY_OMM_XML) == []


@pytest.mark.parametrize(
    ("object_name", "expected"),
    [
        ("ISS (ZARYA)", "ISS"),
        ("GALILEO 24 (GSAT0207)", "GALILEO_24"),
        ("NOAA 15", "NOAA_15"),
        ("STARLINK-1234", "STARLINK-1234"),
    ],
)
def test_sanitize_object_name(object_name: str, expected: str) -> None:
    download_omm_mod = importlib.import_module("el_paso.download_omm")
    assert download_omm_mod._sanitize_object_name(object_name) == expected


def test_parse_norad_ids() -> None:
    download_omm_mod = importlib.import_module("el_paso.download_omm")

    assert download_omm_mod._parse_norad_ids("25544") == [25544]
    assert download_omm_mod._parse_norad_ids("41859,25544") == [25544, 41859]
    assert download_omm_mod._parse_norad_ids("25544, 25544") == [25544]

    with pytest.raises(ValueError, match="must contain at least one"):
        download_omm_mod._parse_norad_ids("")


def test_build_gp_history_url_joins_multiple_ids() -> None:
    download_omm_mod = importlib.import_module("el_paso.download_omm")
    url = download_omm_mod._build_gp_history_url([25544, 41859], "format/xml")
    assert "NORAD_CAT_ID/25544,41859/" in url


def test_write_omm_csv_matches_canonical_field_set_and_order(tmp_path: Path) -> None:
    download_omm_mod = importlib.import_module("el_paso.download_omm")

    records = download_omm_mod._parse_omm_xml(_ISS_OMM_XML)
    file_path = tmp_path / "sample.csv"
    download_omm_mod._write_omm_csv(records, file_path)

    lines = file_path.read_text().splitlines()
    assert lines[0] == (
        "OBJECT_NAME,OBJECT_ID,EPOCH,MEAN_MOTION,ECCENTRICITY,INCLINATION,RA_OF_ASC_NODE,"
        "ARG_OF_PERICENTER,MEAN_ANOMALY,EPHEMERIS_TYPE,CLASSIFICATION_TYPE,NORAD_CAT_ID,"
        "ELEMENT_SET_NO,REV_AT_EPOCH,BSTAR,MEAN_MOTION_DOT,MEAN_MOTION_DDOT"
    )

    with file_path.open(newline="") as f:
        reread_records = list(csv.DictReader(f))

    assert len(reread_records) == 2
    assert reread_records[0]["OBJECT_NAME"] == "ISS (ZARYA)"
    assert reread_records[0]["EPOCH"] == "2024-01-01T00:18:14.850432"
    assert reread_records[0]["NORAD_CAT_ID"] == "25544"
    # userDefinedParameters extras (e.g. SEMIMAJOR_AXIS) are not part of the canonical field set
    assert "SEMIMAJOR_AXIS" not in reread_records[0]


def test_download_omm_missing_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SPACETRACK_USER", raising=False)
    monkeypatch.delenv("SPACETRACK_PASS", raising=False)

    with pytest.raises(ValueError, match="Space-Track username not found"):
        ep.download_omm(
            norad_ids="25544",
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
            save_path=tmp_path,
        )


def test_download_omm_end_time_before_start_time(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SPACETRACK_USER", "fake_user")
    monkeypatch.setenv("SPACETRACK_PASS", "fake_pass")

    with pytest.raises(ValueError, match="'end_time' must be after 'start_time'"):
        ep.download_omm(
            norad_ids="25544",
            start_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            save_path=tmp_path,
        )


def _patch_login_returning(monkeypatch: pytest.MonkeyPatch, xml_text: str) -> list[str]:
    """Patch `_login_spacetrack` to return a fake session serving `xml_text`; return requested URLs."""
    download_omm_mod = importlib.import_module("el_paso.download_omm")
    requested_urls: list[str] = []

    def fake_login(_username: str, _password: str) -> MagicMock:
        session = MagicMock()

        def fake_get(url: str, **_kwargs: object) -> MagicMock:
            requested_urls.append(url)
            response = MagicMock()
            response.status_code = 200
            response.text = xml_text
            response.raise_for_status = lambda: None
            return response

        session.get.side_effect = fake_get
        return session

    monkeypatch.setattr(download_omm_mod, "_login_spacetrack", fake_login)
    return requested_urls


@pytest.mark.basic
def test_download_omm_range_mode_writes_parsed_files_and_discards_xml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SPACETRACK_USER", "fake_user")
    monkeypatch.setenv("SPACETRACK_PASS", "fake_pass")

    requested_urls = _patch_login_returning(monkeypatch, _ISS_OMM_XML)

    ep.download_omm(
        norad_ids="25544",
        start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
        save_path=tmp_path,
        skip_existing=False,
    )

    # One combined request for the whole requested range.
    assert len(requested_urls) == 1
    assert "NORAD_CAT_ID/25544/" in requested_urls[0]
    assert "format/xml" in requested_urls[0]

    # Written under a per-satellite directory named from the sanitized OBJECT_NAME.
    omm_files = list(tmp_path.glob("**/*.csv"))
    assert len(omm_files) == 1
    assert omm_files[0].name == "omm_20240101.csv"
    assert omm_files[0].parent.name == "ISS"

    assert list(tmp_path.glob("**/*.xml")) == []

    content = omm_files[0].read_text()
    assert content.splitlines()[0].startswith("OBJECT_NAME,OBJECT_ID,EPOCH,")
    assert "ISS (ZARYA)" in content
    assert "<omm" not in content


@pytest.mark.basic
def test_download_omm_range_mode_multi_satellite_writes_separate_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SPACETRACK_USER", "fake_user")
    monkeypatch.setenv("SPACETRACK_PASS", "fake_pass")

    xml_text = _multi_satellite_xml(
        _omm_segment(object_name="ISS (ZARYA)", norad_cat_id="25544", epoch="2024-01-01T01:00:00"),
        _omm_segment(object_name="GALILEO 24 (GSAT0207)", norad_cat_id="41859", epoch="2024-01-01T02:00:00"),
    )
    requested_urls = _patch_login_returning(monkeypatch, xml_text)

    ep.download_omm(
        norad_ids="25544,41859",
        start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
        save_path=tmp_path,
        skip_existing=False,
    )

    # One combined request naming both satellites, not one query per satellite.
    assert len(requested_urls) == 1
    assert "NORAD_CAT_ID/25544,41859/" in requested_urls[0]

    written = {p.parent.name: p.name for p in tmp_path.glob("**/*.csv")}
    assert written == {"ISS": "omm_20240101.csv", "GALILEO_24": "omm_20240101.csv"}


@pytest.mark.basic
def test_download_omm_skip_existing_does_not_overwrite_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """skip_existing gates the file *write*; every requested satellite is still queried together."""
    monkeypatch.setenv("SPACETRACK_USER", "fake_user")
    monkeypatch.setenv("SPACETRACK_PASS", "fake_pass")

    existing_file = tmp_path / "ISS" / "omm_20240101.csv"
    existing_file.parent.mkdir(parents=True)
    existing_file.write_text("OBJECT_NAME\nplaceholder\n")

    requested_urls = _patch_login_returning(monkeypatch, _ISS_OMM_XML)

    ep.download_omm(
        norad_ids="25544",
        start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
        save_path=tmp_path,
        skip_existing=True,
    )

    assert len(requested_urls) == 1  # the query still happens
    assert existing_file.read_text() == "OBJECT_NAME\nplaceholder\n"  # but the file is untouched


@pytest.mark.basic
def test_download_omm_instant_mode_multi_satellite(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SPACETRACK_USER", "fake_user")
    monkeypatch.setenv("SPACETRACK_PASS", "fake_pass")

    xml_text = _multi_satellite_xml(
        _omm_segment(object_name="ISS (ZARYA)", norad_cat_id="25544", epoch="2024-01-02T01:00:00"),
        _omm_segment(object_name="GALILEO 24 (GSAT0207)", norad_cat_id="41859", epoch="2024-01-01T02:00:00"),
    )
    requested_urls = _patch_login_returning(monkeypatch, xml_text)

    ep.download_omm(
        norad_ids="25544,41859",
        start_time=datetime(2024, 1, 2, 12, tzinfo=timezone.utc),
        save_path=tmp_path,
        skip_existing=False,
    )

    assert len(requested_urls) == 1
    assert "NORAD_CAT_ID/25544,41859/" in requested_urls[0]
    assert "--" in requested_urls[0]  # a bounded lookback window, not an unbounded "<"

    written = {p.parent.name: p.name for p in tmp_path.glob("**/*.csv")}
    assert written == {"ISS": "omm_20240102.csv", "GALILEO_24": "omm_20240102.csv"}


@pytest.mark.basic
def test_download_omm_live(tmp_path: Path, skip_if_unreachable: Callable[..., None]) -> None:
    skip_if_unreachable("https://www.space-track.org")

    username = os.environ.get("SPACETRACK_USER")
    password = os.environ.get("SPACETRACK_PASS")

    if username is None or password is None:
        pytest.skip("SPACETRACK_USER/SPACETRACK_PASS not set; skipping live Space-Track test.")

    ep.download_omm(
        norad_ids="25544",
        start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
        save_path=tmp_path,
        skip_existing=False,
    )

    omm_files = list(tmp_path.glob("**/*.csv"))
    assert len(omm_files) == 1
    assert omm_files[0].parent.name == "ISS"

    with omm_files[0].open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) >= 1
    assert rows[0]["NORAD_CAT_ID"] == "25544"
    assert list(tmp_path.glob("**/*.xml")) == []
