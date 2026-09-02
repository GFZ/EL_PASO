# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""Build a unified Typer command line interface for any EL-PASO recipe.

Every recipe under :mod:`el_paso.recipes` is a plain Python function whose
signature already declares everything the command line needs: the time range,
the satellite, the magnetic field model, the binning cadence, the data paths and
so on. This module turns such a function into a Typer command automatically, so
that a recipe never has to hand-write argument parsing and can never drift out
of sync with its own command line.

The recipe functions themselves stay untouched. Typer does not understand the
annotations the recipes use (``str | Path`` unions, ``timedelta``, free-form
``datetime`` strings), so :func:`build_recipe_command` generates a *wrapper*
whose ``__signature__`` and ``__annotations__`` are Typer-compatible and which
converts the parsed values back before calling the recipe.

Typical use at the bottom of a recipe module::

    CLI_DEFAULTS = {
        "start_time": datetime(2013, 3, 16, tzinfo=timezone.utc),
        "end_time": datetime(2013, 3, 16, 23, 59, 59, tzinfo=timezone.utc),
    }

    if __name__ == "__main__":
        ep.run_recipe_cli(process_poes_meped_electron, defaults=CLI_DEFAULTS)
"""

from __future__ import annotations

import enum
import inspect
import logging
import re
import types
import typing
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, get_args, get_origin

import dateutil.parser
import typer
from rich.console import Console
from rich.table import Table

# `el_paso` is imported for its module-level flags and helpers, which are only read at
# call time. This module is imported from `el_paso/__init__.py`, so binding the module
# object here (rather than any of its attributes) keeps that import cycle safe.
import el_paso as ep
from el_paso.utils import enforce_utc_timezone

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

LOOP_PARAMETER_NAMES = frozenset({"satellite"})
"""Parameters that accept several values and make the recipe run once per value.

Every recipe names its spacecraft parameter ``satellite``. Pass ``loop_over`` to
:func:`build_recipe_command` to loop over a differently named parameter.
"""

_UNIVERSAL_PARAMETER_NAMES = ("verbose", "quiet", "dry_run", "skip_download", "exit_after_download", "version")
"""Options every recipe command carries; they are consumed by the command, not the recipe."""

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


def _func_name(func: Callable[..., Any]) -> str:
    """Return a readable name for a recipe function."""
    return getattr(func, "__name__", repr(func))


def _literal_choices(hint: Any) -> tuple[str, ...] | None:  # noqa: ANN401
    """Return the string members of a Literal annotation, or None if it is not one."""
    if get_origin(hint) is not Literal:
        return None

    members = get_args(hint)
    if not members or not all(isinstance(member, str) for member in members):
        return None

    return typing.cast("tuple[str, ...]", members)


def _make_choice_enum(name: str, members: Sequence[str]) -> type[enum.Enum]:
    """Build a str-based Enum so Typer renders the members as command line choices.

    Typer's own ``Literal`` support varies between versions, whereas a ``str`` Enum
    renders as choices everywhere. The value is converted back to a plain string
    before the recipe is called, so recipes keep their ``Literal`` annotations.
    """
    return enum.Enum(f"{name}_choices", {member: member for member in members}, type=str)


def _is_path_hint(hint: Any) -> bool:  # noqa: ANN401
    """Return True for ``Path``, ``str | Path`` and their optional variants."""
    if hint is Path:
        return True

    if get_origin(hint) in (types.UnionType, typing.Union):
        members = {member for member in get_args(hint) if member is not type(None)}
        return Path in members and members <= {Path, str}

    return False


def _strip_optional(hint: Any) -> tuple[Any, bool]:  # noqa: ANN401
    """Split ``X | None`` into ``(X, True)``; return ``(hint, False)`` otherwise."""
    if get_origin(hint) in (types.UnionType, typing.Union):
        members = [member for member in get_args(hint) if member is not type(None)]
        if len(members) == 1 and len(get_args(hint)) == 2:
            return members[0], True

    return hint, False


class _ParameterSpec(typing.NamedTuple):
    """How one recipe parameter is exposed on the command line."""

    name: str
    annotation: Any
    default: Any
    choice_enum: type[enum.Enum] | None
    is_loop: bool


def _option_names(name: str, *, is_flag: bool) -> list[str]:
    """Return the option spelling for a parameter: kebab-case, as is conventional."""
    kebab = name.replace("_", "-")

    if is_flag:
        return [f"--{kebab}/--no-{kebab}"]

    return [f"--{kebab}"]


def _build_parameter_spec(  # noqa: PLR0911
    name: str,
    hint: Any,  # noqa: ANN401
    default: Any,  # noqa: ANN401
    help_text: str,
    *,
    is_loop: bool,
) -> _ParameterSpec | None:
    """Map one recipe parameter onto a Typer option, or return None if it cannot be mapped."""
    inner_hint, _ = _strip_optional(hint)

    if inner_hint is datetime:
        option = typer.Option(
            *_option_names(name, is_flag=False),
            parser=parse_datetime,
            metavar="TIME",
            help=help_text,
            show_default=default.isoformat() if isinstance(default, datetime) else True,
        )
        return _ParameterSpec(name, Annotated[datetime, option], default, None, is_loop=False)

    if inner_hint is timedelta:
        option = typer.Option(
            *_option_names(name, is_flag=False),
            parser=parse_cadence,
            metavar="CADENCE",
            help=help_text,
            show_default=format_cadence(default) if isinstance(default, timedelta) else True,
        )
        return _ParameterSpec(name, Annotated[timedelta, option], default, None, is_loop=False)

    choices = _literal_choices(inner_hint)
    if choices is not None:
        choice_enum = _make_choice_enum(name, choices)
        option = typer.Option(*_option_names(name, is_flag=False), help=help_text)
        # The enum is created at runtime, so the annotation has to be assembled dynamically.
        annotation = Annotated[list[choice_enum], option] if is_loop else Annotated[choice_enum, option]  # ty:ignore[invalid-type-form]
        return _ParameterSpec(name, annotation, default, choice_enum, is_loop=is_loop)

    if _is_path_hint(inner_hint):
        option = typer.Option(*_option_names(name, is_flag=False), help=help_text)
        return _ParameterSpec(name, Annotated[Path, option], default, None, is_loop=False)

    if inner_hint is bool:
        option = typer.Option(*_option_names(name, is_flag=True), help=help_text)
        return _ParameterSpec(name, Annotated[bool, option], default, None, is_loop=False)

    if inner_hint in (int, float, str):
        option = typer.Option(*_option_names(name, is_flag=False), help=help_text)
        if is_loop:
            annotation = Annotated[list[inner_hint], option]
        elif default is None:
            annotation = Annotated[inner_hint | None, option]
        else:
            annotation = Annotated[inner_hint, option]
        return _ParameterSpec(name, annotation, default, None, is_loop=is_loop)

    return None


def build_recipe_command(
    func: Callable[..., None],
    *,
    defaults: dict[str, Any] | None = None,
    loop_over: Sequence[str] | None = None,
) -> Callable[..., None]:
    """Build a Typer-compatible command callable for a recipe function.

    The returned callable carries a synthesised ``__signature__`` and matching
    ``__annotations__`` that Typer can introspect. Calling it parses nothing: it
    receives already-converted values from Typer, restores the types the recipe
    expects, and invokes `func` — once per value of each looped parameter.

    Parameters whose annotation cannot be represented on a command line (such as a
    prebuilt `SavingStrategy`) are omitted from the command and keep their default.

    Args:
        func (Callable[..., None]): The recipe function to expose.
        defaults (dict[str, Any] | None): Overrides for the defaults taken from the
            signature. Use it for values that have no sensible library default, most
            notably the processing time range.
        loop_over (Sequence[str] | None): Parameters that accept several values, causing
            one call to `func` per value. Defaults to any parameter named in
            :data:`LOOP_PARAMETER_NAMES`.

    Returns:
        Callable[..., None]: A callable ready to be registered with ``app.command()``.

    Raises:
        TypeError: If a parameter that has no default cannot be exposed on the command
            line, which would make the command impossible to invoke.
    """
    defaults = dict(defaults or {})
    signature = inspect.signature(func)
    hints = typing.get_type_hints(func)
    summary, arg_help = parse_docstring(func.__doc__)

    loop_names = set(loop_over) if loop_over is not None else {
        name for name in signature.parameters if name in LOOP_PARAMETER_NAMES
    }

    specs: list[_ParameterSpec] = []
    skipped: dict[str, Any] = {}

    for name, parameter in signature.parameters.items():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        has_default = name in defaults or parameter.default is not inspect.Parameter.empty
        default = defaults.get(name, parameter.default)

        spec = _build_parameter_spec(
            name,
            hints.get(name, parameter.annotation),
            default,
            arg_help.get(name, ""),
            is_loop=name in loop_names,
        )

        if spec is None:
            if not has_default:
                msg = (
                    f"Cannot expose required parameter {name!r} of {_func_name(func)} on the command line: "
                    f"unsupported annotation {hints.get(name, parameter.annotation)!r}. "
                    f"Give it a default in the signature or in the CLI defaults."
                )
                raise TypeError(msg)
            logger.debug(f"Not exposing parameter {name!r} of {_func_name(func)} on the command line")
            if name in defaults:
                skipped[name] = default
            continue

        specs.append(spec)

    command = _make_command(func, specs, skipped, summary)
    command.__doc__ = summary or func.__doc__
    command.__name__ = _func_name(func)
    return command


def _make_command(
    func: Callable[..., None],
    specs: list[_ParameterSpec],
    skipped: dict[str, Any],
    summary: str,
) -> Any:  # noqa: ANN401
    """Assemble the wrapper callable and attach the synthesised signature."""
    choice_enums = {spec.name: spec.choice_enum for spec in specs if spec.choice_enum is not None}
    loop_names = [spec.name for spec in specs if spec.is_loop]

    def command(**kwargs: Any) -> None:  # noqa: ANN401
        # The universal options belong to the command, not to the recipe, so they are
        # all removed before the remaining keywords are handed over.
        universal = {name: kwargs.pop(name) for name in _UNIVERSAL_PARAMETER_NAMES if name in kwargs}

        _configure_logging(verbose=universal["verbose"], quiet=universal["quiet"])

        if universal["skip_download"]:
            ep.skip_download = True
        if universal["exit_after_download"]:
            ep.exit_after_download = True

        dry_run = universal["dry_run"]

        # Typer hands back Enum members for choice parameters; recipes expect plain strings.
        for name, choice_enum in choice_enums.items():
            value = kwargs.get(name)
            if isinstance(value, choice_enum):
                kwargs[name] = value.value
            elif isinstance(value, list):
                kwargs[name] = [item.value if isinstance(item, choice_enum) else item for item in value]

        kwargs.update(skipped)

        calls = _expand_loops(kwargs, loop_names)

        for index, call_kwargs in enumerate(calls, start=1):
            # Fails loudly if the generated command ever drifts from the recipe signature.
            inspect.signature(func).bind(**call_kwargs)

            if not universal["quiet"]:
                counter = f" ({index}/{len(calls)})" if len(calls) > 1 else ""
                _report_call(func, call_kwargs, title_suffix=counter, dry_run=dry_run)

            if not dry_run:
                func(**call_kwargs)

    parameters = [
        inspect.Parameter(
            spec.name,
            inspect.Parameter.KEYWORD_ONLY,
            default=_typer_default(spec),
            annotation=spec.annotation,
        )
        for spec in specs
    ]
    parameters.extend(_universal_parameters())

    annotations = {parameter.name: parameter.annotation for parameter in parameters}

    # Typer reads parameters from __signature__ but resolves types from __annotations__,
    # so both have to be set and agree.
    command.__signature__ = inspect.Signature(parameters)  # ty:ignore[unresolved-attribute]
    command.__annotations__ = annotations
    command.__doc__ = summary

    return command


def _typer_default(spec: _ParameterSpec) -> Any:  # noqa: ANN401
    """Convert a recipe default into the form Typer expects for that option."""
    if spec.default is inspect.Parameter.empty:
        return ...

    if spec.is_loop:
        values = spec.default if isinstance(spec.default, (list, tuple)) else [spec.default]
        if spec.choice_enum is not None:
            return [spec.choice_enum(value) for value in values]
        return list(values)

    if spec.choice_enum is not None and spec.default is not None:
        return spec.choice_enum(spec.default)

    return spec.default


def _universal_parameters() -> list[inspect.Parameter]:
    """Options every recipe command gets, independent of its signature."""
    definitions: list[tuple[str, Any, Any]] = [
        (
            "verbose",
            Annotated[
                int,
                typer.Option("--verbose", "-v", count=True, help="Increase log verbosity. Repeat for debug output."),
            ],
            0,
        ),
        ("quiet", Annotated[bool, typer.Option("--quiet", "-q", help="Only log warnings and errors.")], False),
        (
            "dry_run",
            Annotated[
                bool,
                typer.Option("--dry-run", help="Print the resolved recipe arguments and exit without processing."),
            ],
            False,
        ),
        (
            "skip_download",
            Annotated[bool, typer.Option("--skip-download", help="Use already downloaded raw files only.")],
            False,
        ),
        (
            "exit_after_download",
            Annotated[bool, typer.Option("--exit-after-download", help="Download the raw files, then stop.")],
            False,
        ),
        (
            "version",
            Annotated[
                bool,
                typer.Option("--version", callback=_version_callback, is_eager=True, help="Show the version and exit."),
            ],
            False,
        ),
    ]

    return [
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=annotation)
        for name, annotation, default in definitions
    ]


def _version_callback(value: bool) -> None:  # noqa: FBT001
    """Print the EL-PASO version and exit, before any other option is processed."""
    if value:
        typer.echo(f"EL-PASO {ep.__version__}")
        raise typer.Exit


def _configure_logging(*, verbose: int, quiet: bool) -> None:
    """Set up logging uniformly for every recipe invocation."""
    if quiet:
        level = logging.WARNING
    elif verbose >= 1:
        level = logging.DEBUG
    else:
        level = logging.INFO

    ep.setup_logging(level)


def _expand_loops(kwargs: dict[str, Any], loop_names: list[str]) -> list[dict[str, Any]]:
    """Expand looped parameters into one keyword set per combination of values."""
    call_kwargs_list = [dict(kwargs)]

    for name in loop_names:
        values = call_kwargs_list[0].get(name)
        if not isinstance(values, list):
            continue

        call_kwargs_list = [
            {**call_kwargs, name: value} for call_kwargs in call_kwargs_list for value in values
        ]

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


def _report_call(
    func: Callable[..., None],
    call_kwargs: dict[str, Any],
    *,
    title_suffix: str = "",
    dry_run: bool = False,
) -> None:
    """Show the settings a run is about to use, so a long job is never a black box."""
    prefix = "Would run" if dry_run else "Running"
    table = Table(
        title=f"{prefix} {_func_name(func)}{title_suffix}",
        title_justify="left",
        show_header=False,
        box=None,
        padding=(0, 2, 0, 0),
    )
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(overflow="fold")

    for name, value in call_kwargs.items():
        table.add_row(name.replace("_", " "), _format_value(name, value))

    console = Console()
    console.print(table)
    # keep the summary visually separate from the log output that follows it
    console.print()


def run_recipe_cli(
    func: Callable[..., None],
    *,
    defaults: dict[str, Any] | None = None,
    loop_over: Sequence[str] | None = None,
    args: Sequence[str] | None = None,
) -> None:
    """Run a recipe as a standalone command line program.

    Intended for the ``if __name__ == "__main__":`` block of a recipe module. The
    command line it produces is identical to the one the ``el-paso`` entry point
    exposes for the same recipe, because both are built by
    :func:`build_recipe_command`.

    Args:
        func (Callable[..., None]): The recipe function to run.
        defaults (dict[str, Any] | None): Overrides for the defaults taken from the
            signature, typically the processing time range.
        loop_over (Sequence[str] | None): Parameters that accept several values, causing
            one call to `func` per value. Defaults to any parameter named in
            :data:`LOOP_PARAMETER_NAMES`.
        args (Sequence[str] | None): Command line arguments to parse. Defaults to
            `sys.argv`; mainly useful for testing.
    """
    app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
    app.command()(build_recipe_command(func, defaults=defaults, loop_over=loop_over))
    app(args=args)
