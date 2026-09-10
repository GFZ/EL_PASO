# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import requests

from el_paso.download import _get_next_time
from el_paso.utils import enforce_utc_timezone, fill_str_template_with_time, timed_function

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

_SPACETRACK_BASE_URL = "https://www.space-track.org"
_SPACETRACK_LOGIN_URL = f"{_SPACETRACK_BASE_URL}/ajaxauth/login"
_SPACETRACK_QUERY_URL = f"{_SPACETRACK_BASE_URL}/basicspacedata/query"
_SPACETRACK_TIMEOUT_SECONDS = 60
_SPACETRACK_RATE_LIMIT_STATUS = 429
_SPACETRACK_DEFAULT_RETRY_AFTER_SECONDS = 60.0
_SPACETRACK_MAX_RETRIES = 3

_QUERY_DATETIME_FORMAT = "%Y-%m-%d"

# How far back to look for the nearest preceding elset in "individual instant" mode. Satellites
# are typically re-tracked well within this window; one with no elset at all in it is likely
# decayed, lost, or otherwise no longer actively tracked.
_INSTANT_LOOKBACK_DAYS = 30

_OMM_CSV_FIELDS = (
    "OBJECT_NAME",
    "OBJECT_ID",
    "EPOCH",
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
    "EPHEMERIS_TYPE",
    "CLASSIFICATION_TYPE",
    "NORAD_CAT_ID",
    "ELEMENT_SET_NO",
    "REV_AT_EPOCH",
    "BSTAR",
    "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT",
)

_BRACKETED_SUFFIX_RE = re.compile(r"\s*\([^)]*\)")


def _sanitize_object_name(object_name: str) -> str:
    """Turn an OMM OBJECT_NAME into a filesystem-safe per-satellite directory name.

    Strips any parenthesized suffix (e.g. "ISS (ZARYA)" becomes "ISS") and replaces remaining
    spaces with underscores (e.g. "GALILEO 24" becomes "GALILEO_24").
    """
    return _BRACKETED_SUFFIX_RE.sub("", object_name).strip().replace(" ", "_")


def _spacetrack_credentials(username: str | None, password: str | None) -> tuple[str, str]:
    if username is None:
        username = os.environ.get("SPACETRACK_USER")
    if password is None:
        password = os.environ.get("SPACETRACK_PASS")

    if username is None:
        msg = "Space-Track username not found! Either load it from environment variables or pass it as an argument."
        raise ValueError(msg)

    if password is None:
        msg = "Space-Track password not found! Either load it from environment variables or pass it as an argument."
        raise ValueError(msg)

    return username, password


def _parse_norad_ids(norad_ids: str) -> list[int]:
    parsed = sorted({int(norad_id.strip()) for norad_id in norad_ids.split(",") if norad_id.strip()})

    if not parsed:
        msg = "'norad_ids' must contain at least one NORAD catalog ID."
        raise ValueError(msg)

    return parsed


@cache
def _login_spacetrack(username: str, password: str) -> requests.Session:
    """Authenticate with Space-Track and return a session carrying the login cookie.

    Cached per (username, password) so a single process reuses one session (and thus one login
    request) across multiple `download_omm` calls.
    """
    session = requests.Session()
    response = session.post(
        _SPACETRACK_LOGIN_URL,
        data={"identity": username, "password": password},
        timeout=_SPACETRACK_TIMEOUT_SECONDS,
    )

    if response.status_code != requests.codes.ok:
        msg = f"Space-Track login failed: {response.text}"
        raise ValueError(msg)

    return session


def _get_with_retry(session: requests.Session, url: str) -> requests.Response:
    """GET `url`, backing off on Space-Track's 429 (rate limit: 30/min, 300/hr per its docs)."""
    response = session.get(url, timeout=_SPACETRACK_TIMEOUT_SECONDS)

    for _ in range(_SPACETRACK_MAX_RETRIES):
        if response.status_code != _SPACETRACK_RATE_LIMIT_STATUS:
            return response

        wait_seconds = float(response.headers.get("Retry-After", _SPACETRACK_DEFAULT_RETRY_AFTER_SECONDS))
        logger.warning(f"Space-Track rate limit hit, waiting {wait_seconds:.0f}s before retrying: {url}")
        time.sleep(wait_seconds)
        response = session.get(url, timeout=_SPACETRACK_TIMEOUT_SECONDS)

    return response


def _build_gp_history_url(norad_ids: Sequence[int], query_suffix: str) -> str:
    """Build one combined query for every requested NORAD ID.

    Space-Track's own docs ask that queries be built this way (a comma-delimited list of
    NORAD_CAT_IDs in one request) rather than issuing one query per satellite.
    """
    ids_str = ",".join(str(norad_id) for norad_id in norad_ids)
    return f"{_SPACETRACK_QUERY_URL}/class/gp_history/NORAD_CAT_ID/{ids_str}/{query_suffix}"


def _parse_omm_xml(xml_content: str) -> list[dict[str, str]]:
    """Parse a Space-Track Orbit Mean-Elements Message (OMM) XML response.

    Args:
        xml_content (str): The raw XML text of a Space-Track `gp`/`gp_history` query response.

    Returns:
        list[dict[str, str]]: One dict per element set, in document order. Includes every field
        Space-Track's XML carries (metadata, mean elements, TLE parameters, and any
        userDefinedParameters extras); `download_omm` itself only persists the canonical
        OMM/TLE subset (see `_OMM_CSV_FIELDS`).
    """
    root = ET.fromstring(xml_content)  # noqa: S314

    records: list[dict[str, str]] = []

    for omm_elem in root.findall("omm"):
        for segment in omm_elem.findall("./body/segment"):
            record: dict[str, str] = {}

            metadata = segment.find("metadata")
            if metadata is not None:
                record.update({child.tag: child.text or "" for child in metadata})

            data = segment.find("data")
            if data is not None:
                mean_elements = data.find("meanElements")
                if mean_elements is not None:
                    record.update({child.tag: child.text or "" for child in mean_elements})

                tle_parameters = data.find("tleParameters")
                if tle_parameters is not None:
                    record.update({child.tag: child.text or "" for child in tle_parameters})

                user_defined_parameters = data.find("userDefinedParameters")
                if user_defined_parameters is not None:
                    for user_defined in user_defined_parameters.findall("USER_DEFINED"):
                        parameter_name = user_defined.get("parameter")
                        if parameter_name:
                            record[parameter_name] = user_defined.text or ""

            if record:
                records.append(record)

    return records


def _parse_epoch(epoch: str) -> datetime:
    return enforce_utc_timezone(datetime.fromisoformat(epoch))


def _write_omm_csv(records: list[dict[str, str]], file_path: Path) -> None:
    """Write parsed OMM records to disk as CSV, restricted to the canonical OMM/TLE field set."""
    with file_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_OMM_CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in _OMM_CSV_FIELDS})


def _target_path(
    save_path: Path, object_dir: str, file_name_stem: str, time: datetime, *, sort_raw_files_by_time: bool
) -> Path:
    """Resolve one satellite/chunk's output path."""
    base = save_path / object_dir

    if sort_raw_files_by_time:
        base = base / fill_str_template_with_time("YYYY/MM/", time)

    return base / fill_str_template_with_time(file_name_stem, time)


def _download_omm_instant(
    norad_ids: list[int],
    target_time: datetime,
    save_path: Path,
    file_name_stem: str,
    username: str,
    password: str,
    *,
    sort_raw_files_by_time: bool,
    skip_existing: bool,
) -> None:
    session = _login_spacetrack(username, password)

    # Bound the query to a lookback window else ISS can exhaust the whole limit
    # before a less frequently updated one's rows are ever reached.
    lower_bound = target_time - timedelta(days=_INSTANT_LOOKBACK_DAYS)
    epoch_predicate = f"{lower_bound.strftime(_QUERY_DATETIME_FORMAT)}--{target_time.strftime(_QUERY_DATETIME_FORMAT)}"
    url = _build_gp_history_url(norad_ids, f"EPOCH/{epoch_predicate}/orderby/NORAD_CAT_ID,EPOCH desc/format/xml")

    response = _get_with_retry(session, url)
    response.raise_for_status()
    records = _parse_omm_xml(response.text)

    # Grouped by NORAD_CAT_ID ascending, EPOCH descending, so the first record seen for each ID
    # is its most recent elset before `target_time`.
    found_norad_ids: set[str] = set()
    for record in records:
        norad_id = record.get("NORAD_CAT_ID")
        if not norad_id or norad_id in found_norad_ids:
            continue
        found_norad_ids.add(norad_id)

        object_dir = _sanitize_object_name(record.get("OBJECT_NAME", norad_id))
        target_path = _target_path(
            save_path, object_dir, file_name_stem, target_time, sort_raw_files_by_time=sort_raw_files_by_time
        )

        if skip_existing and target_path.exists():
            logger.info(f"File already exists, skipping write: {target_path}")
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        _write_omm_csv([record], target_path)
        logger.info(f"Downloaded successfully: {target_path}")

    for norad_id in norad_ids:
        if str(norad_id) not in found_norad_ids:
            logger.warning(f"No OMM elset found at or before {target_time.isoformat()} for NORAD ID {norad_id}.")


@timed_function("download_omm")
def download_omm(
    norad_ids: str,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    save_path: str | Path = "./OMM",
    file_name_stem: str = "omm_YYYYMMDD.csv",
    username: str | None = None,
    password: str | None = None,
    *,
    sort_raw_files_by_time: bool = False,
    skip_existing: bool = True,
) -> None:
    """Download Orbit Mean-Elements Message (OMM) data from Space-Track for one or more NORAD IDs.

    Every requested NORAD ID is queried together in a single combined request per call, as
    Space-Track's own documentation asks (querying satellites individually is discouraged and
    counts harder against its 30/min, 300/hr rate limit). Each satellite's parsed element set(s)
    are written under their own subdirectory, named from the OMM's OBJECT_NAME field with any
    parenthesized suffix removed and spaces replaced by underscores (e.g. "ISS (ZARYA)" becomes
    the directory "ISS"), so `save_path` ends up holding one subdirectory per satellite.

    Args:
        norad_ids (str): Comma-separated NORAD catalog ID(s) to download OMM data for, e.g.
            "25544" or "25544,41859".
        start_time (datetime | None, optional): Start of the time range to download. If `end_time` is None, this is
            instead treated as the single instant to fetch the nearest preceding element set for. If not given,
            today's date at 00:00:00 UTC is used.
        end_time (datetime | None, optional): End of the time range to download. If None,
            `start_time` is treated as an individual instant: the most recent element set with an
            EPOCH strictly before `start_time` is downloaded. Defaults to None.
        save_path (str | Path, optional): Base directory the per-satellite subdirectories and
            parsed CSV files are written under. Defaults to "./OMM".
        file_name_stem (str, optional): Time-templated file name (see `fill_str_template_with_time`)
            for each chunk's output file, joined onto `save_path`/<satellite>. Defaults to
            "omm_YYYYMMDD.csv".
        username (str | None, optional): Space-Track username. If None, read from the
            `SPACETRACK_USER` environment variable. Defaults to None.
        password (str | None, optional): Space-Track password. If None, read from the
            `SPACETRACK_PASS` environment variable. Defaults to None.
        sort_raw_files_by_time (bool, optional): If True, creates subdirectories for each year and
            month under each satellite's directory (e.g. 'YYYY/MM/'). If not given, files are
            written directly under it.
        skip_existing (bool, optional): If True, skip writing (but not querying, since every
            requested satellite is fetched together in one request either way) a satellite/chunk's
            output file if it already exists. Defaults to True.

    Raises:
        ValueError: If `norad_ids` is empty, if `username`/`password` is not provided and not
            available via the `SPACETRACK_USER`/`SPACETRACK_PASS` environment variables, if
            Space-Track login fails, or if `end_time` is not None and not after `start_time`.
    """
    username, password = _spacetrack_credentials(username, password)
    norad_id_list = _parse_norad_ids(norad_ids)

    if not start_time:
        start_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    start_time = enforce_utc_timezone(start_time)
    save_path = Path(save_path)

    if end_time is None:
        _download_omm_instant(
            norad_id_list,
            start_time,
            save_path,
            file_name_stem,
            username,
            password,
            sort_raw_files_by_time=sort_raw_files_by_time,
            skip_existing=skip_existing,
        )
        return

    end_time = enforce_utc_timezone(end_time)

    if end_time <= start_time:
        msg = "'end_time' must be after 'start_time' (omit 'end_time' to query a single instant)."
        raise ValueError(msg)

    chunks: list[tuple[datetime, datetime]] = []
    curr_time = start_time
    while curr_time < end_time:
        next_time = _get_next_time(curr_time, "daily")
        next_time = end_time if next_time is None else min(next_time, end_time)
        chunks.append((curr_time, next_time))
        curr_time = next_time

    session = _login_spacetrack(username, password)

    start_str = start_time.strftime(_QUERY_DATETIME_FORMAT)
    end_str = end_time.strftime(_QUERY_DATETIME_FORMAT)
    query_suffix = f"EPOCH/{start_str}--{end_str}/orderby/NORAD_CAT_ID,EPOCH asc/format/xml"
    url = _build_gp_history_url(norad_id_list, query_suffix)

    response = _get_with_retry(session, url)
    response.raise_for_status()
    records = _parse_omm_xml(response.text)

    records_by_norad_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        norad_id = record.get("NORAD_CAT_ID")
        if norad_id:
            records_by_norad_id[norad_id].append(record)

    for norad_id in norad_id_list:
        satellite_records = records_by_norad_id.get(str(norad_id), [])

        if not satellite_records:
            logger.warning(
                f"No OMM records found between {start_time.isoformat()} and {end_time.isoformat()} "
                f"for NORAD ID {norad_id}."
            )
            continue

        object_dir = _sanitize_object_name(satellite_records[0].get("OBJECT_NAME", str(norad_id)))

        for chunk_start, chunk_end in chunks:
            chunk_records = [
                record
                for record in satellite_records
                if "EPOCH" in record and chunk_start <= _parse_epoch(record["EPOCH"]) < chunk_end
            ]

            if not chunk_records:
                continue

            target_path = _target_path(
                save_path, object_dir, file_name_stem, chunk_start, sort_raw_files_by_time=sort_raw_files_by_time
            )

            if skip_existing and target_path.exists():
                logger.info(f"File already exists, skipping write: {target_path}")
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            _write_omm_csv(chunk_records, target_path)
            logger.info(f"Downloaded successfully: {target_path}")
