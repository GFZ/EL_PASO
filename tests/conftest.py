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
    parser.addoption(
        "--indices_sw_param_data_path",
        action="store",
        default=None,
        help=(
            "Override the directory used for solar wind index/parameter data during "
            "this test run (sets EL_PASO_INDICES_SW_PARAM_DATA_PATH). Defaults to a "
            "per-session temporary directory if not given."
        ),
    )


@pytest.fixture
def renew_solution(request: pytest.FixtureRequest) -> bool:
    def str2bool(v: str) -> bool:
        return v.lower() in ("yes", "true", "t", "1")

    option = request.config.getoption("--renew_solution")
    return str2bool(typing.cast("str", option))


@pytest.fixture(autouse=True)
def indices_sw_param_data_path(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Point EL_PASO_INDICES_SW_PARAM_DATA_PATH at a CLI-provided or temporary directory.

    If `--indices_sw_param_data_path` is passed on the command line, that path is used for
    the whole test session (useful for reusing an already-populated cache across
    runs). Otherwise, a fresh session-scoped temporary directory is used so tests
    never touch a user's real EL_PASO_INDICES_SW_PARAM_DATA_PATH.
    """
    cli_path = request.config.getoption("--indices_sw_param_data_path")
    path = cli_path or tmp_path_factory.mktemp("sw_param_data")
    monkeypatch.setenv("EL_PASO_INDICES_SW_PARAM_DATA_PATH", str(path))



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
