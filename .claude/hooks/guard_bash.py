#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""PreToolUse hook for Bash/Monitor: block full-suite pytest runs and any bumpver command.

Exit code 2 blocks the tool call and shows stderr to Claude. A command that cannot be
tokenized (e.g. unbalanced quotes) is allowed through; bash would reject it anyway.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path

SEPARATOR_CHARS = frozenset(";&|\n()")
SHELLS = {"bash", "sh", "zsh", "dash"}
SHELL_KEYWORDS = {"{", "}", "!", "if", "then", "elif", "else", "while", "until", "do"}
HEREDOC = re.compile(r"(?<!<)<<(-?)[ \t]*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")
BROAD_TARGETS = {".", "tests", "tests/unittests", "tests/system", "tests/comparisons"}
PYTEST_NO_RUN_FLAGS = {"-h", "--help", "-V", "--version", "--co", "--collect-only", "--markers", "--fixtures"}
PYTEST_OPTS_WITH_VALUE = {
    "-m",
    "-k",
    "-p",
    "-c",
    "-o",
    "-W",
    "-n",
    "--basetemp",
    "--rootdir",
    "--confcutdir",
    "--ignore",
    "--deselect",
    "--maxfail",
    "--durations",
    "--tb",
    "--log-cli-level",
    "--log-level",
    "--dist",
    "--cov",
    "--cov-report",
    "--renew_solution",
    "--indices_sw_param_data_path",
}
UV_RUN_OPTS_WITH_VALUE = {
    "--with",
    "--python",
    "-p",
    "--extra",
    "--group",
    "--package",
    "--project",
    "--directory",
    "--env-file",
    "--index",
}


def _split_heredocs(command: str) -> tuple[str, list[tuple[str, str]]]:
    """Remove heredoc bodies, returning the command plus (introducing line, body) pairs.

    A heredoc body is data for the program it is fed to, not shell commands.
    """
    lines = command.split("\n")
    kept: list[str] = []
    heredocs: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        kept.append(line)
        i += 1
        for match in HEREDOC.finditer(line):
            strip_tabs, delimiter = match.group(1) == "-", match.group(3)
            body: list[str] = []
            while i < len(lines):
                candidate = lines[i].lstrip("\t") if strip_tabs else lines[i]
                i += 1
                if candidate == delimiter:
                    break
                body.append(lines[i - 1])
            heredocs.append((line, "\n".join(body)))
    return "\n".join(kept), heredocs


def _segments(command: str) -> list[list[str]]:
    """Split a command into simple commands at `;`, `&&`, `|`, newlines, and parentheses.

    Newlines separate commands only outside quotes, so multi-line quoted strings stay intact.
    """
    lexer = shlex.shlex(command, posix=True, punctuation_chars="();<>|&\n")
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    segments: list[list[str]] = []
    current: list[str] = []
    for token in lexer:
        if token and set(token) <= SEPARATOR_CHARS:
            segments.append(current)
            current = []
        else:
            current.append(token)
    segments.append(current)
    return [s for s in segments if s]


def _program(segment: list[str]) -> str:
    tokens = _unwrap(segment)
    return Path(tokens[0]).name if tokens else ""


def _is_assignment(token: str) -> bool:
    name, sep, _ = token.partition("=")
    return bool(sep) and name.isidentifier()


def _skip_flags(tokens: list[str], opts_with_value: set[str] = frozenset()) -> list[str]:
    i = 0
    while i < len(tokens) and tokens[i].startswith("-"):
        i += 2 if tokens[i] in opts_with_value else 1
    return tokens[i:]


def _unwrap(tokens: list[str]) -> list[str]:
    """Strip env assignments and launcher prefixes down to the program actually run."""
    while tokens:
        while tokens and _is_assignment(tokens[0]):
            tokens = tokens[1:]
        if not tokens:
            break
        prog = Path(tokens[0]).name
        if prog in SHELL_KEYWORDS:
            tokens = tokens[1:]
        elif prog in {"env", "time", "nice", "nohup", "command", "exec"}:
            tokens = _skip_flags(tokens[1:])
        elif prog == "timeout":
            tokens = _skip_flags(tokens[1:])[1:]
        elif prog == "uv" and len(tokens) > 1 and tokens[1] == "run":
            tokens = _skip_flags(tokens[2:], UV_RUN_OPTS_WITH_VALUE)
        elif prog == "uvx":
            tokens = _skip_flags(tokens[1:], UV_RUN_OPTS_WITH_VALUE)
        elif prog == "pipx" and len(tokens) > 1 and tokens[1] == "run":
            tokens = _skip_flags(tokens[2:])
        elif prog.startswith("python"):
            i = 1
            while i < len(tokens) and tokens[i].startswith("-") and tokens[i] != "-m":
                i += 1
            if i + 1 < len(tokens) and tokens[i] == "-m":
                tokens = tokens[i + 1 :]
            else:
                break
        else:
            break
    return tokens


def _nested_script(prog: str, args: list[str]) -> str | None:
    """Return the inner command of `bash -c "..."`-style or `eval ...` invocations."""
    if prog == "eval":
        return " ".join(args)
    if prog in {"bash", "sh", "zsh", "dash"}:
        for i, arg in enumerate(args):
            if not arg.startswith("-"):
                return None
            if "c" in arg[1:] and not arg.startswith("--") and i + 1 < len(args):
                return args[i + 1]
    return None


def _pytest_targets(args: list[str]) -> list[str]:
    targets = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
        elif arg.startswith("-"):
            skip_next = arg in PYTEST_OPTS_WITH_VALUE
        else:
            targets.append(arg)
    return targets


def _normalize(path: str) -> str:
    path = path.split("::", 1)[0]
    while path.startswith("./"):
        path = path[2:]
    return path.rstrip("/") or "."


def violation(command: str) -> str | None:
    """Return the reason to block `command`, or None if it may run."""
    command, heredocs = _split_heredocs(command)
    for line, body in heredocs:
        try:
            feeds_shell = any(_program(segment) in SHELLS for segment in _segments(line))
        except ValueError:
            continue
        if feeds_shell and (reason := violation(body)):
            return reason
    for segment in _segments(command):
        tokens = _unwrap(segment)
        if not tokens:
            continue
        prog = Path(tokens[0]).name
        nested = _nested_script(prog, tokens[1:])
        if nested is not None and (reason := violation(nested)):
            return reason
        if prog == "bumpver":
            return (
                "Blocked: bumpver cuts, tags, and pushes a release. In EL_PASO only the user "
                "runs releases. Hand this step back to the user."
            )
        if prog in {"pytest", "py.test"}:
            if any(arg in PYTEST_NO_RUN_FLAGS for arg in tokens[1:]):
                continue
            targets = [_normalize(t) for t in _pytest_targets(tokens[1:])]
            if not targets or any(t in BROAD_TARGETS or "**" in t for t in targets):
                return (
                    "Blocked: this runs a whole test suite (all unit, system, or comparison tests). "
                    "Run only the tests covering the "
                    'files you changed, e.g. `pytest tests/unittests/test_<module>.py -m "not visual"`. '
                    "Find them with `grep -rl <module_name> tests/unittests` "
                    "(see the build-test-verify skill)."
                )
    return None


def main() -> int:
    """Read the hook payload from stdin and exit 2 to block a violating command."""
    try:
        payload = json.load(sys.stdin)
        command = payload.get("tool_input", {}).get("command", "")
        reason = violation(command)
    except (ValueError, AttributeError):
        return 0
    if reason:
        sys.stderr.write(reason + "\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
