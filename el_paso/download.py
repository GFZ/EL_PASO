# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import re
import typing
from datetime import datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Literal

import requests
from requests.auth import HTTPDigestAuth

from el_paso.utils import enforce_utc_timezone, fill_str_template_with_time, get_file_by_version, timed_function

ERROR_NOT_FOUND = 404
logger = logging.getLogger(__name__)


@timed_function()
def download(
    start_time: datetime,
    end_time: datetime,
    save_path: str | Path,
    file_cadence: Literal["daily", "monthly", "single_file"],
    download_url: str,
    file_name_stem: str,
    download_arguments_prefixes: str = "",
    download_arguments_suffixes: str = "",
    method: Literal["request", "wget"] = "request",
    authentification_info: tuple[str, str] = ("", ""),
    rename_file_name_stem: str | None = None,
    *,
    sort_raw_files_by_time: bool = False,
    skip_existing: bool = True,
) -> None:
    """Download satellite data files within a specified time range and cadence.

    Examples can be found in the 'examples' and 'tutorials' folder.

    Args:
        start_time (datetime): The start of the time range for downloading files. Must be timezone-aware (UTC).
        end_time (datetime): The end of the time range for downloading files. Must be timezone-aware (UTC).
        save_path (str | Path): Directory path where downloaded files will be saved.
        file_cadence (Literal["daily", "monthly", "single_file"]): Frequency of file downloads.
            - "daily": Download files for each day in the range.
            - "monthly": Download files for each month in the range.
            - "single_file": Download a single file.
        download_url (str): Base URL for downloading files.
        file_name_stem (str): Stem for the file name to be downloaded.
        download_arguments_prefixes (str, optional): Additional arguments to prefix to the download command
                                                     (used with wget). Defaults to "".
        download_arguments_suffixes (str, optional): Additional arguments to suffix to the download command
                                                     (used with wget). Defaults to "".
        method (Literal["request", "wget"], optional): Download method to use. Either "request" (Python requests) or
                                                       "wget" (system wget). Defaults to "request".
        authentification_info (tuple[str, str], optional): Tuple of (username, password) for authentication.
                                                           Defaults to ("", "").
        rename_file_name_stem (str | None, optional): If provided, rename the downloaded file to this stem.
                                                      Defaults to None.
        skip_existing (bool, optional): If True, skip downloading files that already exist. Defaults to True.
        sort_raw_files_by_time (bool, optional):
                If True, creates subdirectories for each year and month (e.g., 'YYYY/MM/').
                This helps organize a large number of downloaded files. Defaults to False.

    Raises:
        NotImplementedError: If "monthly" cadence or an unsupported cadence is specified.

    Returns:
        None

    """
    start_time = enforce_utc_timezone(start_time)
    end_time = enforce_utc_timezone(end_time)

    save_path = Path(save_path)

    curr_time = start_time

    while True:
        match method:
            case "request":
                _requests_download(
                    curr_time,
                    save_path,
                    download_url,
                    file_name_stem,
                    authentification_info,
                    rename_file_name_stem,
                    skip_existing=skip_existing,
                    sort_raw_files_by_time=sort_raw_files_by_time,
                )
            case "wget":
                _wget_download(
                    curr_time, save_path, download_url, download_arguments_prefixes, download_arguments_suffixes
                )

        match file_cadence:
            case "daily":
                curr_time += timedelta(days=1)
                if curr_time > end_time:
                    break

            case "monthly":
                msg = "Monthly file cadence has not been implemented yet!"
                raise NotImplementedError(msg)

            case "single_file":
                break

            case _:
                msg = "File cadence must be 'single_file', 'daily', or 'monthly'"
                raise NotImplementedError(msg)


def _requests_download(
    current_time: datetime,
    save_path: Path,
    download_url: str,
    file_name_stem: str,
    authentification_info: tuple[str, str],
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
        response_of_content = _get_page_content(url, authentification_info)

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
            timeout=10,
            auth=HTTPDigestAuth(*authentification_info),
        )  # type: ignore[reportUnknownMemberType]

        if response.status_code == ERROR_NOT_FOUND:
            msg = f"File not found on server: {url}"
            logger.warning(msg)
            return

        response.raise_for_status()

        with (save_path / save_file_name).open("wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        logger.info(f"Downloaded successfully: {save_path / latest_file_name}")

    except requests.exceptions.RequestException as e:
        logger.info(f"Error downloading file from {url}: {e}")


@cache
def _get_page_content(url: str, authentification_info: tuple[str, str]) -> requests.Response | None:
    response_of_content = requests.get(url, stream=True, timeout=10, auth=HTTPDigestAuth(*authentification_info))  # type: ignore[reportUnknownMemberType]

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
        os.system(download_command)  # noqa: S605
    except Exception as e:  # noqa: BLE001
        logger.info(f"Error downloading file using command {download_command}: {e}")
