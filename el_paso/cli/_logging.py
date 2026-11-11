# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""Logging and reporting for the recipe commands: log files, levels and settings tables."""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from rich.console import Console

# Bound as a module (not by attribute) because `el_paso/__init__.py` imports the CLI, see recipe_cli.
import el_paso as ep
from el_paso.cli._parsing import _func_name

if TYPE_CHECKING:
    from pathlib import Path

    from rich.table import Table

    from el_paso.typing import Recipe

logger = logging.getLogger("el_paso.cli")

_LOG_FILE_FORMAT = "[%(levelname)-8s] %(asctime)s - %(name)s:%(lineno)d - %(message)s"
_LOG_FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"

_log_file_handler: logging.Handler | None = None


def _mission_name(func: Recipe) -> str:
    """Derive a recipe's mission name from its module path (`el_paso.recipes.<mission>.*`)."""
    parts = func.__module__.split(".")
    if parts[:2] == ["el_paso", "recipes"] and len(parts) > 2:
        return parts[2]

    return "misc"


def _resolve_log_file(logs_path: Path, func: Recipe, satellite: str | None) -> Path:
    """Resolve `--logs` into the concrete log file a recipe call should write to.

    A `logs_path` with a suffix (e.g. ``log.log``) is used as-is, as an explicit log file.
    Otherwise it is treated as a base directory, under which a dated, mission/satellite-scoped
    log file is created: ``<logs_path>/<YYYY>/<MM>/<DD>/<mission>/<satellite>/<recipe>.log``
    (the satellite segment is omitted for recipes that do not take one).
    """
    if logs_path.suffix:
        log_file = logs_path
    else:
        today = datetime.now(timezone.utc)
        log_dir = logs_path / f"{today:%Y}" / f"{today:%m}" / f"{today:%d}" / _mission_name(func)
        if satellite is not None:
            log_dir /= str(satellite)
        log_file = log_dir / f"{_func_name(func)}_{today:%H00}.log"

    log_file.parent.mkdir(parents=True, exist_ok=True)
    return log_file


def _log_level(*, verbose: int, quiet: bool) -> int:
    """Map ``-v``/``--quiet`` onto a logging level."""
    if quiet:
        return logging.WARNING
    if verbose >= 1:
        return logging.DEBUG
    return logging.INFO


def _configure_logging(
    *,
    verbose: int,
    quiet: bool,
    logs_path: Path,
    func: Recipe,
    satellite: str | None,
    configure_console: bool,
) -> None:
    """Set up logging for one recipe call."""
    level = _log_level(verbose=verbose, quiet=quiet)

    if configure_console:
        ep.setup_logging(level)
    else:
        logging.getLogger().setLevel(level)

    log_file = _resolve_log_file(logs_path, func, satellite)

    global _log_file_handler
    root_logger = logging.getLogger()
    if _log_file_handler is not None:
        root_logger.removeHandler(_log_file_handler)
        _log_file_handler.close()

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(logging.Formatter(_LOG_FILE_FORMAT, datefmt=_LOG_FILE_DATEFMT))
    root_logger.addHandler(file_handler)
    _log_file_handler = file_handler


def _log_table(table: Table) -> None:
    """Log a rich table line by line, so that it reaches the console and the log file alike."""
    buffer = io.StringIO()
    Console(file=buffer).print(table)
    for line in buffer.getvalue().splitlines():
        logger.info(line)
