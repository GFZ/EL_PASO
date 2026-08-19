# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_SUBDIR = "joblib_cache"
_MAX_AGE_DAYS = 7

_exit_with_exception = False
_original_excepthook = sys.excepthook


def _excepthook_tracker(exc_type, exc_value, exc_tb) -> None:  # noqa: ANN001
    global _exit_with_exception
    _exit_with_exception = True
    _original_excepthook(exc_type, exc_value, exc_tb)


sys.excepthook = _excepthook_tracker


def get_cache_dir() -> Path:
    """Return the default joblib cache directory at ``$HOME/.elpaso/joblib_cache``.

    The directory is created if it does not exist.

    Returns:
        Path: The cache directory path.

    Raises:
        OSError: If the ``HOME`` environment variable is not set.
    """
    home_path = os.getenv("HOME")
    if home_path is None:
        msg = "HOME environment variable is not set!"
        raise OSError(msg)

    cache_dir = Path(home_path) / ".elpaso" / _CACHE_SUBDIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def cleanup_stale_cache(max_age_days: int = _MAX_AGE_DAYS) -> None:
    """Delete cache entries older than *max_age_days* from the default cache directory.

    This is a no-op if the cache directory does not exist.

    Args:
        max_age_days: Maximum age in days before an entry is removed.
    """
    home_path = os.getenv("HOME")
    if home_path is None:
        return

    cache_dir = Path(home_path) / ".elpaso" / _CACHE_SUBDIR
    if not cache_dir.exists():
        return

    cutoff = time.time() - max_age_days * 86400

    for entry in cache_dir.iterdir():
        if entry.is_dir() and entry.stat().st_mtime < cutoff:
            shutil.rmtree(entry)
            logger.debug("Removed stale cache entry: %s", entry)


def clear_cache() -> None:
    """Remove the entire joblib cache directory."""
    home_path = os.getenv("HOME")
    if home_path is None:
        return

    cache_dir = Path(home_path) / ".elpaso" / _CACHE_SUBDIR
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        logger.debug("Cleared cache directory: %s", cache_dir)


def clear_cache_on_success() -> None:
    """Remove the cache directory only if the interpreter is exiting without an unhandled exception.

    Intended as an ``atexit`` handler so the cache survives crashes but is
    cleaned up after a successful run.
    """
    if _exit_with_exception:
        logger.debug("Keeping cache — interpreter is exiting due to an exception.")
        return
    clear_cache()
