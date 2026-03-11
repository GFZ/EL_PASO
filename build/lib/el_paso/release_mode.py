# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
import logging
from datetime import datetime, timezone
from pathlib import Path

import git

import el_paso as ep

logger = logging.getLogger("__name__")


def activate_release_mode(
    user_name: str, email_address: str, el_paso_repository_path: str | Path, *, dirty_ok: bool = False
) -> None:
    """Activates the package's release mode and validates the repository state.

    This function enables a special mode for data processing that records key
    information about the execution environment. It checks if the package is
    installed and validates that the Git repository is in a clean state (no
    uncommitted changes) unless `dirty_ok` is set to `True`. Upon successful
    activation, it stores metadata about the user, version, and commit hash.
    This information is appended to the metadata of any processed variables.

    Args:
        user_name (str): The name of the user activating release mode.
        email_address (str): The email address of the user.
        el_paso_repository_path (str | Path): The path to the EL-PASO Git repository.
        dirty_ok (bool, optional): If `True`, allows activation even if the
            repository has uncommitted changes. Defaults to `False`.

    Raises:
        importlib.metadata.PackageNotFoundError: If the 'el_paso' package is not
            found in the Python environment, indicating it's not properly installed.
        ValueError: If the Git repository is not clean and `dirty_ok` is `False`.
    """
    try:
        el_paso_version = importlib.metadata.version("el_paso")
    except importlib.metadata.PackageNotFoundError:
        logger.exception("EL-PASO has to be installed when release mode is used!")
        raise

    el_paso_repo = git.Repo(el_paso_repository_path)
    commit_hash = el_paso_repo.head.commit.hexsha

    if el_paso_repo.is_dirty(index=True, working_tree=True, untracked_files=True):
        if dirty_ok:
            logger.warning("Dirty repository used for processing data in release mode!", stacklevel=2)
        else:
            msg = "Your EL-PASO repository contains changes! Please push your changes to process data in release mode!"
            raise ValueError(msg)

    date_now = datetime.now(timezone.utc).now()
    date_now_str = date_now.strftime("%d-%b-%Y")

    ep._release_msg = (  # noqa: SLF001 # type: ignore[Private]
        f"This variable was processed using the EL-PASO release mode on {date_now_str}.\n"
        f"User name: {user_name}, email address: {email_address}.\n"
        f"EL-PASO version: {el_paso_version}, git commit: {commit_hash}."
    )

    ep._release_mode = True  # noqa: SLF001 # type: ignore[Private]


def is_in_release_mode() -> bool:
    """Checks if the package's release mode is currently active.

    Returns:
        bool: `True` if release mode is active, `False` otherwise.
    """
    return ep._release_mode  # noqa: SLF001 # type: ignore[Private]


def get_release_msg() -> str:
    """Retrieves the message associated with the package's release mode.

    This message contains metadata about the execution environment, including
    user information, version, and commit hash, if release mode is active.

    Returns:
        str: The release mode message if active, otherwise an empty string.
    """
    return ep._release_msg  # noqa: SLF001 # type: ignore[Private]
