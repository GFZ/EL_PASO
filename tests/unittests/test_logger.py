# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

import el_paso as ep
import pytest

logger = logging.getLogger(__name__)


@pytest.mark.basic
def test_setup_logging_write_to_file(tmp_path: Path):

    log_file = tmp_path / "test.log"

    ep.setup_logging(log_file=log_file)

    logger.info("This is a test!")

    with log_file.open("r") as f:
        log_content = f.read()

    assert "tests.unittests.test_logger:" in log_content
    assert "This is a test!" in log_content


@pytest.mark.basic
def test_setup_logging_append_to_file(tmp_path: Path):

    log_file = tmp_path / "test_append.log"
    with log_file.open("w") as f:
        f.write("HEADER\n")

    ep.setup_logging(log_file=log_file, file_mode="a")

    logger.info("This is a test!")

    with log_file.open("r") as f:
        line1 = f.readline()
        assert line1 == "HEADER\n"

        line2 = f.readline()
        assert "tests.unittests.test_logger:" in line2
        assert "This is a test!" in line2
