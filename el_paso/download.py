# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ftplib
import gzip
import io
import json
import logging
import os
import re
import shutil
import sys
import typing
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import cache, partial
from pathlib import Path
from typing import Literal

import requests
from requests.auth import HTTPDigestAuth
from richpool import JoblibPool

import el_paso as ep
from el_paso.utils import enforce_utc_timezone, fill_str_template_with_time, get_file_by_version, timed_function

if typing.TYPE_CHECKING:
    from el_paso.typing import FileCadence

ERROR_NOT_FOUND = 404
logger = logging.getLogger(__name__)


def _download_single_step(
    curr_time: datetime,
    next_time: datetime,
    save_path: Path,
    method: str,
    download_url: str,
    file_name_stem: str,
    download_arguments_prefixes: str,
    download_arguments_suffixes: str,
    authentication_info: tuple[str, str],
    rename_file_name_stem: str | None,
    *,
    skip_existing: bool,
    sort_raw_files_by_time: bool,
) -> None:
    """Worker function to handle a single time chunk download."""
    match method:
        case "request":
            _requests_download(
                curr_time,
                save_path,
                download_url,
                file_name_stem,
                authentication_info,
                rename_file_name_stem,
                skip_existing=skip_existing,
                sort_raw_files_by_time=sort_raw_files_by_time,
            )
        case "wget":
            _wget_download(curr_time, save_path, download_url, download_arguments_prefixes, download_arguments_suffixes)
        case "ftp":
            _ftp_download(
                curr_time,
                save_path,
                download_url,
                file_name_stem,
                authentication_info,
                rename_file_name_stem,
                skip_existing=skip_existing,
                sort_raw_files_by_time=sort_raw_files_by_time,
            )
        case "esa_swe":
            _esa_swe_download(
                authentication_info,
                download_url,
                start_time=curr_time,
                end_time=next_time,
                save_path=save_path,
                rename_file_name_stem=rename_file_name_stem,
                skip_existing=skip_existing,
            )


def _download_single_step_safe(task: tuple[datetime, datetime], **kwargs: object) -> None:
    """Wraps `_download_single_step` so one failed task logs and doesn't abort the rest of the batch."""
    curr_time, next_time = task
    try:
        _download_single_step(curr_time=curr_time, next_time=next_time, **kwargs)  # ty:ignore[invalid-argument-type]
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Download for date {curr_time} generated an exception: {exc}")


@timed_function()
def download(
    start_time: datetime,
    end_time: datetime,
    save_path: str | Path,
    file_cadence: FileCadence,
    download_url: str,
    file_name_stem: str,
    download_arguments_prefixes: str = "",
    download_arguments_suffixes: str = "",
    method: Literal["request", "wget", "ftp", "esa_swe"] = "request",
    authentication_info: tuple[str, str] = ("", ""),
    rename_file_name_stem: str | None = None,
    *,
    sort_raw_files_by_time: bool = False,
    skip_existing: bool = True,
    max_threads: int = 4,
) -> None:
    """Download satellite data files within a specified time range and cadence.

    Examples can be found in the 'examples' and 'tutorials' folder.

    Args:
        start_time (datetime): The start of the time range for downloading files. Must be timezone-aware (UTC).
        end_time (datetime): The end of the time range for downloading files. Must be timezone-aware (UTC).
        save_path (str | Path): Directory path where downloaded files will be saved.
        file_cadence (FileCadence): Frequency of file downloads.
            - "daily": Download files for each day in the range.
            - "monthly": Download files for each month in the range.
            - "single_file": Download a single file.
            - A callable taking the current time and returning the next file boundary
              time, for cadences that don't fit a fixed interval (e.g. irregular
              weekly files).
        download_url (str): Base URL for downloading files.
        file_name_stem (str): Stem for the file name to be downloaded.
        download_arguments_prefixes (str, optional): Additional arguments to prefix to the download command
                                                     (used with wget). Defaults to "".
        download_arguments_suffixes (str, optional): Additional arguments to suffix to the download command
                                                     (used with wget). Defaults to "".
        method (Literal["request", "wget", "ftp", "esa_swe"], optional): Download method to use. "request" uses the
                                                       Python requests library, "wget" uses the system wget command,
                                                       "ftp" uses Python's ftplib against an FTP server (the host is
                                                       parsed out of `download_url`, and `file_name_stem` is the
                                                       exact remote file name to download), and
                                                       "esa_swe" uses the ESA Space Weather Service API (requires
                                                       `authentication_info` and `rename_file_name_stem`).
                                                       Defaults to "request".
        authentication_info (tuple[str, str], optional): Tuple of (username, password) for authentication.
                                                           For "ftp", an empty username falls back to anonymous
                                                           login. Defaults to ("", "").
        rename_file_name_stem (str | None, optional): If provided, rename the downloaded file to this stem.
                                                      Defaults to None.
        skip_existing (bool, optional): If True, skip downloading files that already exist. Defaults to True.
        sort_raw_files_by_time (bool, optional):
                If True, creates subdirectories for each year and month (e.g., 'YYYY/MM/').
                This helps organize a large number of downloaded files. Defaults to False.
        max_threads (int, optional): Maximum number of threads used for downloading. Defaults to 4.

    Raises:
        NotImplementedError: If "monthly" cadence or an unsupported cadence is specified.
    """
    if ep.skip_download or os.getenv("EL_PASO_SKIP_DOWNLOAD"):
        logger.info("Skipping ep.download!")
        return

    start_time = enforce_utc_timezone(start_time)
    end_time = enforce_utc_timezone(end_time)

    save_path = Path(save_path)

    curr_time = start_time
    tasks = []

    while curr_time < end_time:
        next_time = _get_next_time(curr_time, file_cadence)
        next_time = end_time if next_time is None else min(next_time, end_time)

        tasks.append((curr_time, next_time))
        curr_time = next_time

    if len(tasks) > 1:
        logger.info(f"Starting parallel download with {max_threads} threads for {len(tasks)} files...")

    with JoblibPool(processes=max_threads, backend="threading") as pool:
        pool.map(
            partial(
                _download_single_step_safe,
                save_path=save_path,
                method=method,
                download_url=download_url,
                file_name_stem=file_name_stem,
                download_arguments_prefixes=download_arguments_prefixes,
                download_arguments_suffixes=download_arguments_suffixes,
                authentication_info=authentication_info,
                rename_file_name_stem=rename_file_name_stem,
                skip_existing=skip_existing,
                sort_raw_files_by_time=sort_raw_files_by_time,
            ),
            tasks,
            desc="Downloading",
        )

    if ep.exit_after_download or os.getenv("EL_PASO_EXIT_AFTER_DOWNLOAD"):
        logger.info("Exiting after ep.download is completed!")
        sys.exit(0)


def _get_next_time(curr_time: datetime, file_cadence: FileCadence) -> datetime | None:
    if callable(file_cadence):
        return file_cadence(curr_time)

    match file_cadence:
        case "daily":
            curr_time += timedelta(days=1)
            return curr_time

        case "monthly":
            msg = "Monthly file cadence has not been implemented yet!"
            raise NotImplementedError(msg)

        case "single_file":
            return None

        case _:
            msg = "File cadence must be 'single_file', 'daily', 'monthly', or a callable"
            raise NotImplementedError(msg)


def _requests_download(
    current_time: datetime,
    save_path: Path,
    download_url: str,
    file_name_stem: str,
    authentication_info: tuple[str, str],
    rename_file_name_stem: str | None,
    *,
    skip_existing: bool,
    sort_raw_files_by_time: bool,
) -> None:
    """Download a file using the requests library."""
    if sort_raw_files_by_time:
        parent_folders = fill_str_template_with_time("YYYY/MM/", current_time)
        save_path = save_path / parent_folders

    save_path = Path(fill_str_template_with_time(str(save_path), current_time))
    save_path.mkdir(exist_ok=True, parents=True)

    url = fill_str_template_with_time(download_url, current_time)
    file_name_stem = fill_str_template_with_time(file_name_stem, current_time)

    try:
        response_of_content = _get_page_content(url, authentication_info)

        if response_of_content is None:
            return

        found_files = typing.cast(
            "list[str]", re.findall(rf"{file_name_stem}", response_of_content.text, flags=re.IGNORECASE)
        )
        latest_file_name = get_file_by_version(found_files, version="latest")

        if latest_file_name is None:
            msg = f"No file found matching the pattern {file_name_stem} in the response from {url}"
            logger.warning(msg)
            return

        if rename_file_name_stem is None:
            save_file_name = latest_file_name
        else:
            save_file_name = fill_str_template_with_time(rename_file_name_stem, current_time)

        if skip_existing and (save_path / save_file_name).exists():
            logger.info(f"File already exists, skipping download: {save_path / save_file_name}")
            return

        response = requests.get(
            f"{url}/{latest_file_name}",
            stream=True,
            timeout=30,
            auth=HTTPDigestAuth(*authentication_info),
        )

        if response.status_code == ERROR_NOT_FOUND:
            msg = f"File not found on server: {url}"
            logger.warning(msg)
            return

        response.raise_for_status()

        with (save_path / save_file_name).open("wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        logger.info(f"Downloaded successfully: {save_path / save_file_name}")

    except requests.exceptions.RequestException as e:
        logger.info(f"Error downloading file from {url}: {e}")


@cache
def _get_page_content(url: str, authentication_info: tuple[str, str]) -> requests.Response | None:
    response_of_content = requests.get(url, stream=True, timeout=10, auth=HTTPDigestAuth(*authentication_info))

    if response_of_content.status_code == ERROR_NOT_FOUND:
        msg = f"File not found on server: {url}"
        logger.warning(msg)
        return None

    response_of_content.raise_for_status()

    return response_of_content


def _wget_download(
    current_time: datetime,
    save_path: Path,
    download_url: str,
    download_arguments_prefixes: str,
    download_arguments_suffixes: str,
) -> None:
    Path(fill_str_template_with_time(str(save_path), current_time)).parents[0].mkdir(exist_ok=True, parents=True)

    # Replace "yyyymmdd" or "YYYYMMDD" in url, prefix, and suffix with the parsed string
    url = fill_str_template_with_time(download_url, current_time)
    prefix = fill_str_template_with_time(download_arguments_prefixes, current_time)
    suffix = fill_str_template_with_time(download_arguments_suffixes, current_time)
    download_command = f"wget {prefix} {url} {suffix}"
    logger.info(download_command)

    # Execute the download command
    try:
        os.system(download_command)  # noqa: S605  # ty:ignore[deprecated]
    except Exception as e:  # noqa: BLE001
        logger.info(f"Error downloading file using command {download_command}: {e}")


def _ftp_connect(
    host: str,
    authentication_info: tuple[str, str],
    *,
    port: int = 21,
    timeout: float = 30.0,
) -> ftplib.FTP:
    if not host:
        msg = "FTP download URL is missing a host"
        raise ValueError(msg)

    user, password = authentication_info
    ftp = ftplib.FTP(timeout=timeout)  # noqa: S321
    ftp.connect(host, port=port)
    if user:
        ftp.login(user, password)
    else:
        ftp.login()
    ftp.set_pasv(True)
    return ftp


def _ftp_download(
    current_time: datetime,
    save_path: Path,
    download_url: str,
    file_name_stem: str,
    authentication_info: tuple[str, str],
    rename_file_name_stem: str | None,
    *,
    skip_existing: bool,
    sort_raw_files_by_time: bool,
) -> None:
    """Download a file from an FTP server using ftplib.

    `download_url` must be of the form "ftp://host/remote/dir" (may contain time placeholders).
    `file_name_stem` is the exact remote file name (may contain time placeholders).
    """
    if sort_raw_files_by_time:
        parent_folders = fill_str_template_with_time("YYYY/MM/", current_time)
        save_path = save_path / parent_folders

    save_path = Path(fill_str_template_with_time(str(save_path), current_time))
    save_path.mkdir(exist_ok=True, parents=True)

    url = fill_str_template_with_time(download_url, current_time)
    remote_file_name = fill_str_template_with_time(file_name_stem, current_time)

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "ftp":
        msg = f"FTP download URL must start with 'ftp://', got: {url}"
        raise ValueError(msg)

    remote_dir = parsed.path or "/"

    try:
        ftp = _ftp_connect(parsed.hostname or "", authentication_info)
    except OSError as e:
        logger.warning(f"Error connecting to FTP server for {url}: {e}")
        return

    try:
        is_gzipped = remote_file_name.endswith(".gz")

        if rename_file_name_stem is None:
            save_file_name = remote_file_name[: -len(".gz")] if is_gzipped else remote_file_name
        else:
            save_file_name = fill_str_template_with_time(rename_file_name_stem, current_time)

        if skip_existing and (save_path / save_file_name).exists():
            logger.info(f"File already exists, skipping download: {save_path / save_file_name}")
            return

        remote_file_path = f"{remote_dir.rstrip('/')}/{remote_file_name}"

        if is_gzipped:
            buffer = io.BytesIO()
            ftp.retrbinary(f"RETR {remote_file_path}", buffer.write)
            buffer.seek(0)
            with gzip.GzipFile(fileobj=buffer) as decompressed_stream, (save_path / save_file_name).open("wb") as file:
                shutil.copyfileobj(decompressed_stream, file)
        else:
            with (save_path / save_file_name).open("wb") as file:
                ftp.retrbinary(f"RETR {remote_file_path}", file.write)

        logger.info(f"Downloaded successfully: {save_path / save_file_name}")

    except ftplib.all_errors as e:
        logger.warning(f"Error downloading file from {url}: {e}")
    finally:
        try:
            ftp.quit()
        except ftplib.all_errors:
            ftp.close()


def _esa_swe_download(
    authentication_info: tuple[str, str],
    download_url: str,
    start_time: datetime,
    end_time: datetime,
    save_path: Path,
    rename_file_name_stem: str | None,
    *,
    skip_existing: bool,
) -> None:
    if rename_file_name_stem is None:
        msg = "'rename_file_name_stem' is required for method 'esa_swe'!"
        raise ValueError(msg)

    if authentication_info[0] == "" or authentication_info[1] == "":
        msg = "'authentication_info' must be provided for method 'esa_swe'!"
        raise ValueError(msg)

    if "https://swe.ssa.esa.int/hapi/" in download_url:
        start_time_str = start_time.isoformat(timespec="seconds").split("+")[0] + "Z"
        end_time_str = end_time.isoformat(timespec="seconds").split("+")[0] + "Z"

        download_url = f"{download_url}&start={start_time_str}&stop={end_time_str}&format=csv"
        token_scope = "swe_hapiserver"  # noqa: S105

    elif "https://sso-csr-ucl-ac-be.content.swe.s2p.esa.int/" in download_url:
        download_url = fill_str_template_with_time(download_url, start_time)
        token_scope = "swe_contentproxy"  # noqa: S105

    else:
        msg = f"Encountered unknown url type for ESA download: {download_url}"
        raise ValueError(msg)

    save_path = Path(fill_str_template_with_time(str(save_path), start_time))
    save_path.mkdir(exist_ok=True, parents=True)

    save_file_name = fill_str_template_with_time(rename_file_name_stem, start_time)

    if skip_existing and (save_path / save_file_name).exists():
        logger.info(f"File already exists, skipping download: {save_path / save_file_name}")
        return

    try:
        access_token = _get_esa_swe_access_token(authentication_info[0], authentication_info[1], token_scope)
        data_response = requests.get(
            download_url,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30,
            stream=True,
        )
        if token_scope == "swe_hapiserver" and '"code":1201' in data_response.text:  # noqa: S105
            logger.error(f"HAPI server error 1201: {json.loads(data_response.text)['status']['message']}")
            return
        data_response.raw.decode_content = False

        if data_response.status_code == ERROR_NOT_FOUND:
            msg = f"File not found on server: {download_url}"
            logger.warning(msg)
            return

        data_response.raise_for_status()

        if download_url.endswith(".gz"):
            with (
                gzip.GzipFile(fileobj=data_response.raw) as decompressed_stream,
                (save_path / save_file_name).open("wb") as file,
            ):
                shutil.copyfileobj(decompressed_stream, file)
        else:
            with (save_path / save_file_name).open("wb") as file:
                for chunk in data_response.iter_content(chunk_size=8192):
                    file.write(chunk)

        logger.info(f"Downloaded successfully: {save_path / save_file_name}")

    except requests.exceptions.RequestException:
        logger.exception(f"Error downloading file from {download_url}")


@cache
def _get_esa_swe_access_token(
    client_id: str, client_secret: str, token_scope: Literal["swe_hapiserver", "swe_contentproxy"]
) -> str:
    response = requests.post(
        "https://sso.s2p.esa.int/realms/swe/protocol/openid-connect/token",
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials",
            "scope": token_scope,
        },
        timeout=30,
    )

    token_data = response.json()
    access_token = token_data["access_token"]

    return access_token
