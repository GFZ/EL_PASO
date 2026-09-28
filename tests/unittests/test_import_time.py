# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests guarding that ``el_paso`` and its CLI stay cheap to import."""

import subprocess
import sys

import pytest

_HEAVY_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "astropy",
    "xarray",
    "matplotlib",
    "cdflib",
    "sunpy",
    "pyspedas",
    "netCDF4",
    "skyfield",
    "sscws",
    "swvo",
    "joblib",
    "requests",
)

_IMPORT_TIME_BUDGET_SECONDS = 1.0
_CLI_TIME_BUDGET_SECONDS = 1.5

_CHEAP_CLI_ARGS = (
    pytest.param(["--help"], id="root_help"),
    pytest.param(["list"], id="list"),
    pytest.param(["poes", "--help"], id="mission_group_help"),
)


def _run_in_subprocess(code: str) -> str:
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _run_cli(args: list[str]) -> tuple[float, set[str]]:
    """Run the ``el-paso`` CLI with ``args`` in a subprocess."""
    code = (
        "import contextlib, io, sys, time\n"
        f"sys.argv = ['el-paso'] + {args!r}\n"
        "buf = io.StringIO()\n"
        "t0 = time.perf_counter()\n"
        "with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):\n"
        "    import el_paso.cli.app as cli_app\n"
        "    try:\n"
        "        cli_app.main()\n"
        "    except SystemExit:\n"
        "        pass\n"
        "t1 = time.perf_counter()\n"
        "print(t1 - t0)\n"
        "print(','.join(sorted(sys.modules)))\n"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    elapsed_line, modules_line = result.stdout.strip().splitlines()
    return float(elapsed_line), set(modules_line.split(","))


@pytest.mark.basic
def test_import_does_not_load_heavy_packages() -> None:
    """`import el_paso` must not pull in any heavy third-party dependency."""
    code = "import sys\nimport el_paso\nprint(','.join(sorted(sys.modules)))\n"
    loaded_modules = set(_run_in_subprocess(code).split(","))

    loaded_heavy = sorted(loaded_modules & set(_HEAVY_PACKAGES))
    assert not loaded_heavy, (
        f"`import el_paso` eagerly loaded heavy package(s) {loaded_heavy}. "
        "Move the offending import inside the function that needs it (or under "
        "`if TYPE_CHECKING:`) instead of importing it at module level."
    )


@pytest.mark.basic
def test_import_time_stays_within_budget() -> None:
    """`import el_paso` must complete within a small, fixed time budget."""
    _run_in_subprocess("import el_paso")

    code = "import time\nt0 = time.perf_counter()\nimport el_paso\nt1 = time.perf_counter()\nprint(t1 - t0)\n"
    elapsed_seconds = float(_run_in_subprocess(code))

    assert elapsed_seconds < _IMPORT_TIME_BUDGET_SECONDS, (
        f"`import el_paso` took {elapsed_seconds:.3f}s, exceeding the "
        f"{_IMPORT_TIME_BUDGET_SECONDS}s budget. This usually means a heavy "
        "dependency got imported eagerly; check with "
        "`python -X importtime -c 'import el_paso'`."
    )


@pytest.mark.basic
@pytest.mark.parametrize("args", _CHEAP_CLI_ARGS)
def test_cli_help_does_not_load_heavy_packages(args: list[str]) -> None:
    """Browsing the CLI (root/mission help, ``list``) must not import any recipe."""
    _elapsed, loaded_modules = _run_cli(args)

    loaded_heavy = sorted(loaded_modules & set(_HEAVY_PACKAGES))
    assert not loaded_heavy, (
        f"`el-paso {' '.join(args)}` eagerly loaded heavy package(s) {loaded_heavy}. "
        "This usually means a group's `get_command` built a real recipe command "
        "(importing its module) merely to render `--help` or `list`; see "
        "`_LazyHelpGroup` in el_paso/cli/app.py."
    )


@pytest.mark.basic
@pytest.mark.parametrize("args", _CHEAP_CLI_ARGS)
def test_cli_help_stays_within_time_budget(args: list[str]) -> None:
    """Browsing the CLI must complete within a small, fixed time budget."""
    _run_cli(args)

    elapsed_seconds, _loaded_modules = _run_cli(args)

    assert elapsed_seconds < _CLI_TIME_BUDGET_SECONDS, (
        f"`el-paso {' '.join(args)}` took {elapsed_seconds:.3f}s, exceeding the "
        f"{_CLI_TIME_BUDGET_SECONDS}s budget. This usually means a heavy "
        "dependency got imported eagerly while building the CLI."
    )
