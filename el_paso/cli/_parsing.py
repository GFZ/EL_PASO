# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""Parsing and formatting helpers shared by the command line builders.

Cadences, dates and docstrings are turned into command line values here, and values are rendered
back for display. Everything is re-exported from :mod:`el_paso.cli.recipe_cli`.
"""

from __future__ import annotations

import inspect
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dateutil.parser

if TYPE_CHECKING:
    from el_paso.typing import Recipe

_SECTION_RE = re.compile(r"^\s*(Args|Arguments|Returns|Raises|Yields|Note|Notes|Example|Examples|Attributes):\s*$")
_ARG_RE = re.compile(r"^(?P<name>\*{0,2}\w+)\s*(?:\((?P<type>[^)]*)\))?\s*:\s*(?P<help>.*)$")
_DEFAULTS_RE = re.compile(r"\s*Defaults? to\s+[^.]*\.?\s*$")

_CADENCE_UNITS = {
    "": 1.0,
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "m": 60.0,
    "min": 60.0,
    "mins": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hrs": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "d": 86400.0,
    "day": 86400.0,
    "days": 86400.0,
}
_CADENCE_RE = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[a-zA-Z]*)\s*$")


def parse_cadence(value: str | timedelta) -> timedelta:
    """Parse a cadence such as ``"5min"``, ``"10s"``, ``"1h"`` or ``"30"`` into a timedelta.

    A bare number is interpreted as seconds.

    Args:
        value (str | timedelta): The cadence to parse. A `timedelta` is returned unchanged,
            which lets Typer re-use the parser on its own default values.

    Returns:
        timedelta: The parsed cadence.

    Raises:
        ValueError: If the string is not a number followed by a known unit.
    """
    if isinstance(value, timedelta):
        return value

    match = _CADENCE_RE.match(value)
    if match is None or match.group("unit").lower() not in _CADENCE_UNITS:
        units = ", ".join(sorted(u for u in _CADENCE_UNITS if u))
        msg = f"Cannot parse cadence {value!r}. Expected a number with an optional unit ({units}), e.g. '5min'."
        raise ValueError(msg)

    return timedelta(seconds=float(match.group("value")) * _CADENCE_UNITS[match.group("unit").lower()])


def format_cadence(value: timedelta) -> str:
    """Render a timedelta the way :func:`parse_cadence` accepts it, for help output."""
    total_seconds = value.total_seconds()
    for unit, seconds_per_unit in (("d", 86400.0), ("h", 3600.0), ("min", 60.0)):
        if total_seconds >= seconds_per_unit and total_seconds % seconds_per_unit == 0:
            return f"{int(total_seconds // seconds_per_unit)}{unit}"
    return f"{total_seconds:g}s"


def parse_datetime(value: str | datetime) -> datetime:
    """Parse a date/time string into a UTC-aware datetime.

    Accepts anything :mod:`dateutil` understands, so both ``2013-03-16`` and
    ``2013-03-16T23:59:59`` work. Naive results are assumed to be UTC.

    Args:
        value (str | datetime): The value to parse. A `datetime` is returned with UTC
            enforced, which lets Typer re-use the parser on its own default values.

    Returns:
        datetime: The parsed, timezone-aware datetime.
    """
    from el_paso.utils import enforce_utc_timezone  # noqa: PLC0415

    if isinstance(value, datetime):
        return enforce_utc_timezone(value)

    return enforce_utc_timezone(dateutil.parser.parse(value))


def parse_docstring(docstring: str | None) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into its summary and its per-argument help texts.

    Args:
        docstring (str | None): The raw ``__doc__`` of a recipe function.

    Returns:
        tuple[str, dict[str, str]]: The summary line (everything up to the first blank
        line) and a mapping from argument name to its description with whitespace
        collapsed.
    """
    if not docstring:
        return "", {}

    lines = inspect.cleandoc(docstring).splitlines()

    summary_lines: list[str] = []
    for line in lines:
        if not line.strip():
            break
        summary_lines.append(line.strip())
    summary = " ".join(summary_lines)

    arg_help: dict[str, str] = {}
    in_args = False
    current: str | None = None

    for line in lines:
        section = _SECTION_RE.match(line)
        if section:
            in_args = section.group(1) in {"Args", "Arguments"}
            current = None
            continue

        if not in_args:
            continue

        stripped = line.strip()
        if not stripped:
            continue

        match = _ARG_RE.match(stripped)
        if match and not stripped.startswith("-"):
            current = match.group("name").lstrip("*")
            arg_help[current] = match.group("help").strip()
        elif current is not None:
            # a continuation line of the previous argument
            arg_help[current] = f"{arg_help[current]} {stripped}".strip()

    # Typer renders the actual default itself, so drop the docstrings' trailing
    # "Defaults to ..." sentence rather than showing it twice.
    return summary, {name: _DEFAULTS_RE.sub("", " ".join(text.split())).strip() for name, text in arg_help.items()}


def _func_name(func: Recipe) -> str:
    """Return a readable name for a recipe function."""
    return getattr(func, "__name__", repr(func))


def _expand_loops(kwargs: dict[str, Any], loop_names: list[str]) -> list[dict[str, Any]]:
    """Expand looped parameters into one keyword set per combination of values."""
    call_kwargs_list = [dict(kwargs)]

    for name in loop_names:
        values = call_kwargs_list[0].get(name)
        if not isinstance(values, list):
            continue

        call_kwargs_list = [{**call_kwargs, name: value} for call_kwargs in call_kwargs_list for value in values]

    return call_kwargs_list


_SECRET_PARAMETER_HINTS = ("password", "secret", "token", "credential")


def _format_value(name: str, value: Any) -> str:  # noqa: ANN401
    """Render one argument for display, hiding anything that looks like a credential."""
    if value is not None and any(hint in name.lower() for hint in _SECRET_PARAMETER_HINTS):
        return "<hidden>"

    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, timedelta):
        return format_cadence(value)
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return "-"

    return str(value)
