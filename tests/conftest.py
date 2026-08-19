# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import socket
import typing
from collections.abc import Callable
from urllib.parse import urlparse

import netCDF4
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--renew_solution", action="store", default="false")


@pytest.fixture
def renew_solution(request: pytest.FixtureRequest) -> bool:
    def str2bool(v: str) -> bool:
        return v.lower() in ("yes", "true", "t", "1")

    option = request.config.getoption("--renew_solution")
    return str2bool(typing.cast("str", option))


_DEFAULT_PORTS_BY_SCHEME = {"https": 443, "http": 80, "ftp": 21}


def _is_reachable(url: str, timeout: float = 5.0) -> bool:
    parsed = urlparse(url)
    port = parsed.port or _DEFAULT_PORTS_BY_SCHEME.get(parsed.scheme, 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture
def skip_if_unreachable() -> Callable[..., None]:
    def _skip_if_unreachable(*urls: str, timeout: float = 5.0) -> None:
        for url in urls:
            if not _is_reachable(url, timeout=timeout):
                pytest.skip(f"External resource '{url}' is not reachable; skipping test that requires network access.")

    return _skip_if_unreachable
