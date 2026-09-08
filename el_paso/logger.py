# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0


import logging

from swvo.logger import setup_logging

# Get the package logger
logger = logging.getLogger(__package__)
logger.addHandler(logging.NullHandler())

__all__ = ["logger", "setup_logging"]
