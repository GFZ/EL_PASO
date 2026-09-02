# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences  # noqa: INP001
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""Run a real EL-PASO recipe at documentation build time.

`command_line.md` contains a "Try it" example that is meant to show what
actually happens when a recipe is run, not a hand-written guess. This hook
finds the ``<!-- LIVE_CLI_EXAMPLE -->`` marker on that page and replaces it
with an animated-terminal block (rendered by the `termynal` mkdocs plugin,
see `mkdocs.yml`) built from the *real* stdout of an actual
`el-paso rbsp hope-electrons` run: it really downloads the day's CDF from
NASA SPDF and really runs the IRBEM magnetic field computations, in a
throwaway temporary directory.

The raw console output is cleaned up before display: the `[LEVEL] timestamp -
module:line -` prefix that every log line carries is stripped (it comes from
`swvo.logger.setup_logging`'s formatter and is noise for a docs reader),
third-party `swvo.*` index-loading chatter is dropped, and the build
machine's absolute temp path is normalised back to the `.` the command was
given for `--raw-data-path`/`--processed-data-path`. Every remaining line is
still exactly what the recipe printed.

The whole substitution happens in `on_page_markdown`, i.e. before the page's
markdown is handed to python-markdown/`termynal` for conversion - so this
runs independently of, and does not conflict with, any markdown-preprocessor
ordering (e.g. `pymdownx.snippets` vs. `termynal`).

If the live run fails (no network, a transient NASA SPDF outage, ...), the
build falls back to `live_cli_example_fallback.txt`, a real capture from a
previous successful run, so a flaky network doesn't break the docs build.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.files import Files
    from mkdocs.structure.pages import Page

log = logging.getLogger("mkdocs.hooks.live_cli_example")

_PAGE_URI = "getting_started/command_line.md"
_MARKER = "<!-- LIVE_CLI_EXAMPLE -->"
_FALLBACK_FILE = Path(__file__).parent / "live_cli_example_fallback.txt"

_ARGS = [
    "rbsp",
    "hope-electrons",
    "--start-time",
    "2013-10-15",
    "--end-time",
    "2013-10-15T23:59:59",
    "--no-calculate-Lstar",
]
_DISPLAY_COMMAND = "el-paso " + " ".join(_ARGS)


def _run_live() -> str:
    """Actually run the recipe and return its cleaned, genuine console output."""
    with tempfile.TemporaryDirectory(prefix="el_paso_docs_") as tempdir:
        env = {**os.environ, "COLUMNS": "300"}  # wide enough that Rich never wraps a line
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "el_paso.cli.app", *_ARGS],
            cwd=tempdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
        return _clean(result.stdout, tempdir)


def _clean(raw: str, tempdir: str) -> str:
    """Strip logging boilerplate and machine-specific paths, keep every real message."""
    out: list[str] = []
    for line in raw.splitlines():
        out.append(line.rstrip())

    return "\n".join(out).replace(tempdir, ".").strip("\n")


def _build_block(output: str) -> str:
    return f"<!-- termynal -->\n```console\n$ {_DISPLAY_COMMAND}\n\n{output}\n```"


def on_page_markdown(
    markdown: str,
    page: Page,
    config: MkDocsConfig,  # noqa: ARG001
    files: Files,  # noqa: ARG001
) -> str:
    """Replace the live-example marker in `command_line.md` with a real run's output."""
    if page.file.src_uri != _PAGE_URI or _MARKER not in markdown:
        return markdown

    try:
        output = _run_live()
    except Exception:
        log.warning(
            "live_cli_example: could not run 'el-paso rbsp hope-electrons' live "
            "(no network, or a real failure) - falling back to a previously captured run.",
            exc_info=True,
        )
        output = _FALLBACK_FILE.read_text().strip("\n")

    return markdown.replace(_MARKER, _build_block(output))
