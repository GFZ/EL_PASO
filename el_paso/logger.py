# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0


import logging
import sys
from pathlib import Path
from typing import ClassVar, Optional

# Get the package logger
logger = logging.getLogger("el_paso")
logger.addHandler(logging.NullHandler())


class _ColorFormatter(logging.Formatter):
    COLORS: ClassVar = {
        logging.DEBUG: "\033[36m",  # cyan
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }
    RESET = "\033[0m"

    def format(self, record) -> str:  # noqa: ANN001
        msg = super().format(record)
        color = self.COLORS.get(record.levelno, "")
        return f"{color}{msg}{self.RESET}"


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None, file_mode: str = "w") -> None:
    """Setup logging for the el_paso package and root logger.

    Args:
        level (str, optional): Logging level, by default is INFO
        log_file (Path, optional): Path to log file. If None, only console logging is enabled. I
                f provided, logs will be written to both console and file., by default None
        file_mode (str, optional): Mode for writing to the log file, by default is "w".
                Use "a" to append to an existing log file instead of overwriting it.
    """
    try:
        level = getattr(logging, level.upper())
    except AttributeError:
        msg = f"Invalid logging level: {level}. Use one of DEBUG, INFO, WARNING, ERROR, CRITICAL."
        raise ValueError(msg)  # noqa: B904

    # Configure root logger so all loggers inherit the formatting
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    log_format = "[%(levelname)-8s] %(asctime)s - %(name)s:%(lineno)d - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    if any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root_logger.handlers
    ):
        pass
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(_ColorFormatter(log_format, datefmt=datefmt))
        root_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=datefmt))
        root_logger.addHandler(file_handler)
