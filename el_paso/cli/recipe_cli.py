# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
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

import collections
import enum
import inspect
import logging
import types
import typing
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, get_args, get_origin

import typer
from rich.table import Table

# `el_paso` is imported for its module-level flags and helpers, which are only read at
# call time. This module is imported from `el_paso/__init__.py`, so binding the module
# object here (rather than any of its attributes) keeps that import cycle safe.
import el_paso as ep
from el_paso.cache import mark_run_failed
from el_paso.cli._logging import _configure_logging, _log_table
from el_paso.cli._parsing import (
    _expand_loops,
    _format_value,
    _func_name,
    format_cadence,
    parse_cadence,
    parse_datetime,
    parse_docstring,
)
from el_paso.cli.availability import (
    AVAILABILITY_PARAMETER,
    format_interval,
    load_availability_table,
    overlapping_intervals,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from el_paso.cli.availability import AvailabilitySource, AvailabilityTable, DateInterval
    from el_paso.typing import Recipe

logger = logging.getLogger("el_paso.cli")

FACTORY_ONLY_PARAMETERS = frozenset({"legacy_layout"})
"""Strategy factory parameters that are not recipe parameters.

They only matter when loading processed data (``el-paso plot``): processing always writes the new folder layout.
"""

LOOP_PARAMETER_NAMES = frozenset({"satellite"})
"""Parameters that accept several values and make the recipe run once per value.

Every recipe names its spacecraft parameter ``satellite``. Pass ``loop_over`` to
:func:`build_recipe_command` to loop over a differently named parameter.
"""

ALL_VALUES = "all"
"""Value of a looped choice parameter that stands for all of its choices, e.g. ``--satellite all``."""


def _version_callback(value: bool) -> None:  # noqa: FBT001
    """Print the EL-PASO version and exit, before any other option is processed."""
    if value:
        typer.echo(f"EL-PASO {ep.__version__}")
        raise typer.Exit


_UNIVERSAL_OPTIONS: dict[str, tuple[Any, Any]] = {
    "verbose": (
        Annotated[
            int,
            typer.Option("--verbose", "-v", count=True, help="Increase log verbosity. Repeat for debug output."),
        ],
        0,
    ),
    "quiet": (Annotated[bool, typer.Option("--quiet", "-q", help="Only log warnings and errors.")], False),
    "dry_run": (
        Annotated[
            bool,
            typer.Option("--dry-run", help="Print the resolved recipe arguments and exit without processing."),
        ],
        False,
    ),
    "skip_download": (
        Annotated[bool, typer.Option("--skip-download", help="Use already downloaded raw files only.")],
        False,
    ),
    "exit_after_download": (
        Annotated[bool, typer.Option("--exit-after-download", help="Download the raw files, then stop.")],
        False,
    ),
    "version": (
        Annotated[
            bool,
            typer.Option("--version", callback=_version_callback, is_eager=True, help="Show the version and exit."),
        ],
        False,
    ),
    "logs": (
        Annotated[
            Path,
            typer.Option(
                "--logs",
                help="Base directory for log files, or a path to a specific log file "
                "(e.g. 'log.log'). If a directory, a dated <mission>/<satellite> "
                "subdirectory is created under it for each run.",
            ),
        ],
        Path("logs"),
    ),
}
"""Every recipe command option that is not a recipe parameter: its annotation and its default."""

_UNIVERSAL_PARAMETER_NAMES = tuple(_UNIVERSAL_OPTIONS)
"""Options every recipe command carries; they are consumed by the command, not the recipe."""


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


def split_comma_separated(values: Iterable[Any] | None, *, what: str = "value", example: str = "") -> list[str]:
    """Split the values of a repeatable, comma-separated option into its items, stripped of spaces.

    ``["a,b", " c"]`` becomes ``["a", "b", "c"]``, so ``--satellite a,b`` and ``--satellite a --satellite b``
    are equal.

    Args:
        values (Iterable[Any] | None): The values as given, one per occurrence of the option.
        what (str): What an item is, for the error message, e.g. ``"energy"``.
        example (str): An example of a valid value, for the error message, e.g. ``"470,1575"``.

    Returns:
        list[str]: The items, in the order given; empty if `values` is None or empty.

    Raises:
        typer.BadParameter: If the option was given, but holds no item, e.g. ``--energy ,``.
    """
    values = list(values or [])
    items = [part.strip() for text in values for part in str(text).split(",") if part.strip()]
    if values and not items:
        hint = f", e.g. {example}" if example else ""
        msg = f"no {what} given; use a comma-separated list{hint}."
        raise typer.BadParameter(msg)
    return items


def _given_on_command_line(ctx: typer.Context, param: typer.CallbackParam) -> bool:
    """Whether the user gave an option, rather than it taking its default (or environment) value."""
    source = ctx.get_parameter_source(param.name) if param.name else None
    return source is typer._click.core.ParameterSource.COMMANDLINE  # noqa: SLF001


def _comma_separated_choices(
    choices: Sequence[str],
) -> Callable[[typer.Context, typer.CallbackParam, list[str] | None], list[str] | None]:
    """Build an option callback that splits comma-separated values and checks each against `choices`."""

    def callback(ctx: typer.Context, param: typer.CallbackParam, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value and not _given_on_command_line(ctx, param):
            return []  # an empty default
        items = split_comma_separated(value, example=",".join(choices[:2]))
        invalid = [item for item in items if item not in choices]
        if invalid:
            msg = f"invalid choice(s) {', '.join(map(repr, invalid))}; choose from {', '.join(choices)}."
            raise typer.BadParameter(msg)
        return items

    return callback


def _comma_separated_values(
    item_type: type,
) -> Callable[[typer.Context, typer.CallbackParam, list[str] | None], list[Any] | None]:
    """Build an option callback that splits comma-separated values and converts each to `item_type`."""

    def callback(ctx: typer.Context, param: typer.CallbackParam, value: list[str] | None) -> list[Any] | None:
        if value is None:
            return None
        if not value and not _given_on_command_line(ctx, param):
            return []  # an empty default
        items = split_comma_separated(value)
        try:
            return [item_type(item) for item in items]
        except ValueError as error:
            msg = f"expected comma-separated {item_type.__name__} values, got {', '.join(map(repr, items))}."
            raise typer.BadParameter(msg) from error

    return callback


def _complete_comma_separated(choices: Sequence[str]) -> Callable[[str], list[str]]:
    """Build a shell completion for a comma-separated choice option.

    It completes the item after the last comma, and leaves out the items already given, so
    ``--satellite a,<TAB>`` offers ``a,b``, ``a,c``, and so on.
    """

    def complete(incomplete: str) -> list[str]:
        given, _, current = incomplete.rpartition(",")
        # `all` stands alone, so it is only offered first
        taken = {*given.split(","), ALL_VALUES} if given else set()
        prefix = f"{given}," if given else ""
        return [f"{prefix}{choice}" for choice in choices if choice.startswith(current) and choice not in taken]

    return complete


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
    is_list: bool = False
    loop_values: tuple[str, ...] | None = None
    """The values ``all`` expands to, for looped choice parameters."""
    help: str = ""
    """The parameter's description, from the docstring or the caller."""


SHORT_FLAGS = {"satellite": "-s", "raw_data_path": "-r", "processed_data_path": "-o"}
"""One-letter aliases of the options that are given on nearly every call."""

DATA_PATH_ENV_VARS = {
    "raw_data_path": "EL_PASO_RAW_DATA_PATH",
    "processed_data_path": "EL_PASO_PROCESSED_DATA_PATH",
}
"""The environment variables that set the data directories when the option is not given."""


def _option_names(name: str, *, is_flag: bool) -> list[str]:
    """Return the option spelling for a parameter: kebab-case, plus its short flag if it has one."""
    kebab = name.replace("_", "-")

    if is_flag:
        return [f"--{kebab}/--no-{kebab}"]

    return [f"--{kebab}", *([SHORT_FLAGS[name]] if name in SHORT_FLAGS else [])]


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
            metavar="<time>",
            help=help_text,
            show_default=default.isoformat() if isinstance(default, datetime) else True,
        )
        return _ParameterSpec(name, Annotated[datetime, option], default, None, is_loop=False)

    if inner_hint is timedelta:
        option = typer.Option(
            *_option_names(name, is_flag=False),
            parser=parse_cadence,
            metavar="<cadence>",
            help=help_text,
            show_default=format_cadence(default) if isinstance(default, timedelta) else True,
        )
        return _ParameterSpec(name, Annotated[timedelta, option], default, None, is_loop=False)

    choices = _literal_choices(inner_hint)
    if choices is not None:
        loop_values = None
        if is_loop and ALL_VALUES not in choices:
            # a looped choice also accepts `all`, which runs the recipe once per member
            loop_values = choices
            choices = (*choices, ALL_VALUES)
            help_text = f"{help_text} Use '{ALL_VALUES}' for every value.".strip()
        choice_enum = _make_choice_enum(name, choices)
        if is_loop:
            # A looped option takes plain strings, so that `--satellite a,b` can be split before it is checked.
            option = typer.Option(
                *_option_names(name, is_flag=False),
                help=f"{help_text} Comma-separated for several, e.g. a,b.".strip(),
                metavar=f"<{'|'.join(choices)}>",
                callback=_comma_separated_choices(choices),
                autocompletion=_complete_comma_separated(choices),
            )
            annotation = Annotated[list[str], option]
        else:
            option = typer.Option(*_option_names(name, is_flag=False), help=help_text)
            # The enum is created at runtime, so the annotation has to be assembled dynamically.
            annotation = Annotated[choice_enum, option]  # ty:ignore[invalid-type-form]
        return _ParameterSpec(name, annotation, default, choice_enum, is_loop=is_loop, loop_values=loop_values)

    if _is_path_hint(inner_hint):
        option = typer.Option(*_option_names(name, is_flag=False), help=help_text, envvar=DATA_PATH_ENV_VARS.get(name))
        return _ParameterSpec(name, Annotated[Path, option], default, None, is_loop=False)

    if inner_hint is bool:
        option = typer.Option(*_option_names(name, is_flag=True), help=help_text)
        return _ParameterSpec(name, Annotated[bool, option], default, None, is_loop=False)

    if get_origin(inner_hint) is list:
        item_hints = get_args(inner_hint)
        if len(item_hints) == 1 and item_hints[0] in (int, float, str):
            option = typer.Option(
                *_option_names(name, is_flag=False),
                help=f"{help_text} Comma-separated for several, e.g. 10,90.".strip(),
                metavar=f"<{item_hints[0].__name__}>,...",
                callback=_comma_separated_values(item_hints[0]),
            )
            # A list parameter already takes several values, so they are listed rather than looped:
            # `--alpha-eq 10,90` is one call with two pitch angles, not two calls.
            annotation = Annotated[list[str], option]
            return _ParameterSpec(name, annotation, default, None, is_loop=False, is_list=True)

        return None

    if inner_hint in (int, float, str):
        option = typer.Option(*_option_names(name, is_flag=False), help=help_text)
        if is_loop:
            option = typer.Option(
                *_option_names(name, is_flag=False),
                help=f"{help_text} Comma-separated for several.".strip(),
                metavar=f"<{inner_hint.__name__}>,...",
                callback=_comma_separated_values(inner_hint),
            )
            annotation = Annotated[list[str], option]
        elif default is None:
            annotation = Annotated[inner_hint | None, option]
        else:
            annotation = Annotated[inner_hint, option]
        return _ParameterSpec(name, annotation, default, None, is_loop=is_loop)

    return None


def build_recipe_command(
    func: Recipe,
    *,
    defaults: dict[str, Any] | None = None,
    loop_over: Sequence[str] | None = None,
    availability: AvailabilitySource | None = None,
) -> Callable[..., None]:
    """Build a Typer-compatible command callable for a recipe function.

    The returned callable carries a synthesised ``__signature__`` and matching
    ``__annotations__`` that Typer can introspect. Calling it parses nothing: it
    receives already-converted values from Typer, restores the types the recipe
    expects, and invokes `func` — once per value of each looped parameter.

    Parameters whose annotation cannot be represented on a command line (such as a
    prebuilt `SavingStrategy`) are omitted from the command and keep their default.

    Args:
        func (Recipe): The recipe function to expose.
        defaults (dict[str, Any] | None): Overrides for the defaults taken from the
            signature. Use it for values that have no sensible library default, most
            notably the processing time range.
        loop_over (Sequence[str] | None): Parameters that accept several values, causing
            one call to `func` per value. Defaults to any parameter named in
            :data:`LOOP_PARAMETER_NAMES`.
        availability (AvailabilitySource | None): The recipe's satellite availability table (see
            :mod:`el_paso.cli.availability`). With it, ``--satellite all`` only runs the satellites
            that have data in the requested time range, explicitly requested satellites without
            data get a warning, and the run report says whether each satellite has data. It is
            read when the command runs, not when it is built.

    Returns:
        Callable[..., None]: A callable ready to be registered with ``app.command()``.

    Raises:
        TypeError: If a parameter that has no default cannot be exposed on the command
            line, which would make the command impossible to invoke.
    """
    summary, _ = parse_docstring(func.__doc__)
    specs, skipped = _build_parameter_specs(func, defaults=defaults, loop_over=loop_over)

    command = _make_command(func, specs, skipped, summary, availability)
    command.__doc__ = summary or func.__doc__
    command.__name__ = _func_name(func)
    return command


def _build_parameter_specs(
    func: Callable[..., Any],
    *,
    defaults: dict[str, Any] | None = None,
    loop_over: Sequence[str] | None = None,
    arg_help: dict[str, str] | None = None,
    names: Sequence[str] | None = None,
) -> tuple[list[_ParameterSpec], dict[str, Any]]:
    """Map the parameters of `func` onto command line options.

    This is the signature-to-option machinery shared by every generated command: the recipe
    commands expose a recipe's parameters with it, and ``el-paso plot`` exposes a strategy
    factory's parameters.

    Args:
        func (Callable[..., Any]): The function whose signature is exposed.
        defaults (dict[str, Any] | None): Overrides for the defaults taken from the signature.
        loop_over (Sequence[str] | None): Parameters that accept several values. Defaults to any
            parameter named in :data:`LOOP_PARAMETER_NAMES`.
        arg_help (dict[str, str] | None): Help text per parameter. Defaults to the ``Args:``
            section of the docstring of `func`.
        names (Sequence[str] | None): Only expose these parameters. Defaults to all of them.

    Returns:
        tuple[list[_ParameterSpec], dict[str, Any]]: The exposed options, in signature order, and
        the parameters that cannot be exposed but have a value in `defaults`, which the command
        has to pass on itself.

    Raises:
        TypeError: If a parameter that has no default cannot be exposed on the command line.
    """
    defaults = dict(defaults or {})
    signature = inspect.signature(func)
    hints = typing.get_type_hints(func)
    if arg_help is None:
        _, arg_help = parse_docstring(func.__doc__)

    loop_names = (
        set(loop_over)
        if loop_over is not None
        else {name for name in signature.parameters if name in LOOP_PARAMETER_NAMES}
    )

    specs: list[_ParameterSpec] = []
    skipped: dict[str, Any] = {}

    for name, parameter in signature.parameters.items():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if names is not None and name not in names:
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
        if spec is not None:
            spec = spec._replace(help=arg_help.get(name, ""))

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

    return specs, skipped


TIME_PARAMETERS = ("start_time", "end_time")
"""The recipe parameters that also come as the optional positional arguments ``START`` and ``END``."""

_TIME_ARGUMENTS = {
    "start_time": ("start", "START", "Start of the time range."),
    "end_time": ("end", "END", "End of the time range."),
}
"""For each time parameter: the keyword of its positional argument, the argument's metavar and the
help used when the function does not describe the parameter."""


def _has_time_range(specs: Sequence[_ParameterSpec]) -> bool:
    """Whether `specs` contain both time parameters, which then also come as positional arguments."""
    return {spec.name for spec in specs} >= set(TIME_PARAMETERS)


def _time_arguments(specs: Sequence[_ParameterSpec]) -> list[inspect.Parameter]:
    """The optional positional ``START`` and ``END`` arguments, or nothing without a time range in `specs`."""
    if not _has_time_range(specs):
        return []
    descriptions = {spec.name: spec.help for spec in specs}
    return [
        inspect.Parameter(
            keyword,
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=Annotated[
                datetime | None,
                typer.Argument(
                    parser=parse_datetime,
                    metavar=metavar,
                    help=descriptions[name] or generic_help,
                    show_default=False,
                ),
            ],
        )
        for name, (keyword, metavar, generic_help) in _TIME_ARGUMENTS.items()
    ]


def _merge_time_arguments(kwargs: dict[str, Any], specs: Sequence[_ParameterSpec]) -> None:
    """Set ``start_time`` and ``end_time`` from the positional ``START`` and ``END``, else the recipe's defaults.

    `kwargs` is modified in place: the positional keywords are removed and ``start_time`` and ``end_time`` set.

    Raises:
        typer.BadParameter: If a bound is not given and the recipe has no default for it.
    """
    if not _has_time_range(specs):
        return
    defaults = {spec.name: spec.default for spec in specs}
    for name, (keyword, metavar, _) in _TIME_ARGUMENTS.items():
        value = kwargs.pop(keyword, None)
        if value is None:
            if defaults[name] is inspect.Parameter.empty:
                msg = f"missing the {name.replace('_', ' ')}: give it as {metavar}."
                raise typer.BadParameter(msg, param_hint=f"'{metavar}'")
            value = defaults[name]
            if isinstance(value, (str, datetime)):
                # a default is not parsed by Typer, unlike a value given on the command line
                value = parse_datetime(value)
        kwargs[name] = value


def _restore_parsed_values(kwargs: dict[str, Any], specs: Sequence[_ParameterSpec]) -> set[str]:
    """Convert the values Typer parsed for `specs` back into the types the wrapped function expects.

    Choice parameters come back as plain strings instead of Enum members, ``all`` in a looped
    choice parameter expands to every choice, and repeated options come back as lists instead of
    tuples. `kwargs` is modified in place.

    Returns:
        set[str]: The looped parameters given as ``all``, which were expanded to every choice.
    """
    expanded_all: set[str] = set()
    _merge_time_arguments(kwargs, specs)
    for spec in specs:
        value = kwargs.get(spec.name)

        # Typer hands back Enum members for choice parameters; recipes expect plain strings.
        if spec.choice_enum is not None:
            if isinstance(value, spec.choice_enum):
                value = value.value
            elif isinstance(value, list):
                value = [item.value if isinstance(item, spec.choice_enum) else item for item in value]

        # `all` stands for every choice of a looped parameter, e.g. `--satellite all`
        if spec.loop_values is not None and isinstance(value, list) and ALL_VALUES in value:
            value = list(spec.loop_values)
            expanded_all.add(spec.name)

        # Typer hands back a tuple for options that take several values; recipes annotate them as lists.
        if spec.is_list and isinstance(value, tuple):
            value = list(value)

        if spec.name in kwargs:
            kwargs[spec.name] = value

    return expanded_all


def _spec_parameters(specs: Sequence[_ParameterSpec]) -> list[inspect.Parameter]:
    """Turn option specs into the keyword-only parameters of a synthesised command signature.

    A time range in `specs` also gets the optional positional arguments ``START`` and ``END``, first.
    """
    return [
        *_time_arguments(specs),
        *(
            inspect.Parameter(
                spec.name,
                inspect.Parameter.KEYWORD_ONLY,
                default=_typer_default(spec),
                annotation=spec.annotation,
            )
            for spec in specs
            # the time range is given as the positional START and END only
            if not (_has_time_range(specs) and spec.name in TIME_PARAMETERS)
        ),
    ]


def _attach_signature(command: Callable[..., Any], parameters: Sequence[inspect.Parameter]) -> None:
    """Give a ``**kwargs`` wrapper the signature Typer builds its options from.

    Raises:
        TypeError: If two parameters share a name, which would make two options collide.
    """
    counts = collections.Counter(parameter.name for parameter in parameters)
    duplicates = [name for name, count in counts.items() if count > 1]
    if duplicates:
        msg = f"Command {_func_name(command)} defines these options more than once: {', '.join(sorted(duplicates))}"
        raise TypeError(msg)

    # Typer reads parameters from __signature__ but resolves types from __annotations__,
    # so both have to be set and agree.
    command.__signature__ = inspect.Signature(parameters)  # ty:ignore[missing-attribute]
    command.__annotations__ = {parameter.name: parameter.annotation for parameter in parameters}


def format_time_range(start_time: datetime, end_time: datetime) -> str:
    """Render a time range for messages: dates alone at midnight, otherwise date and time."""

    def render(time: datetime) -> str:
        if (time.hour, time.minute, time.second, time.microsecond) == (0, 0, 0, 0):
            return time.date().isoformat()
        return time.strftime("%Y-%m-%dT%H:%M:%S")

    return f"{render(start_time)} and {render(end_time)}"


def _apply_availability(
    kwargs: dict[str, Any],
    table: AvailabilityTable,
    start_time: datetime,
    end_time: datetime,
    *,
    expanded_all: set[str],
    label: str,
) -> tuple[dict[str, list[DateInterval]], list[tuple[int, str]]]:
    """Restrict ``--satellite all`` to the satellites the availability table has data for.

    An explicitly requested satellite is never dropped: the table is a hint, not a gate. Each one
    without data in the range gets a warning instead. `kwargs` is modified in place.

    Args:
        kwargs (dict[str, Any]): The restored command arguments, with ``satellite``.
        table (AvailabilityTable): The recipe's availability table.
        start_time (datetime): Start of the requested time range.
        end_time (datetime): End of the requested time range.
        expanded_all (set[str]): The looped parameters given as ``all``, see `_restore_parsed_values`.
        label (str): Names the recipe in messages.

    Returns:
        tuple[dict[str, list[DateInterval]], list[tuple[int, str]]]: The satellites with data in the
        range and their overlapping intervals, and the messages to log once logging is configured, as
        ``(level, message)`` pairs.

    Raises:
        typer.BadParameter: If ``--satellite all`` leaves no satellite with data in the range.
    """
    coverage = overlapping_intervals(table, start_time, end_time)
    time_range = format_time_range(start_time, end_time)

    value = kwargs[AVAILABILITY_PARAMETER]
    satellites = value if isinstance(value, list) else [value]
    notices: list[tuple[int, str]] = []

    if AVAILABILITY_PARAMETER in expanded_all:
        kept = [satellite for satellite in satellites if satellite in coverage]
        skipped = [satellite for satellite in satellites if satellite not in coverage]
        if not kept:
            msg = (
                f"'{ALL_VALUES}' selects no satellite: according to the availability table of {label}, "
                f"none has data between {time_range}."
            )
            raise typer.BadParameter(msg, param_hint=f"'--{AVAILABILITY_PARAMETER}'")
        if skipped:
            notices.append(
                (
                    logging.INFO,
                    (
                        f"--{AVAILABILITY_PARAMETER} {ALL_VALUES}: skipping {len(skipped)} of {len(satellites)} "
                        f"satellites without data between {time_range} according to the availability table: "
                        f"{', '.join(skipped)}."
                    ),
                )
            )
        kwargs[AVAILABILITY_PARAMETER] = kept if isinstance(value, list) else kept[0]
    else:
        notices.extend(
            (
                logging.WARNING,
                (
                    f"Satellite {satellite!r} has no data between {time_range} according to the availability "
                    f"table of {label}; running it anyway."
                ),
            )
            for satellite in satellites
            if satellite not in coverage
        )

    return coverage, notices


def _coverage_text(coverage: dict[str, list[DateInterval]], satellite: str) -> str:
    """Describe a satellite's coverage of the requested range for the run report."""
    intervals = coverage.get(satellite)
    if not intervals:
        return "no data in range (availability table)"
    return f"data {'; '.join(format_interval(interval) for interval in intervals)}"


def _print_satellite_listing(label: str, satellites: Sequence[str], table: AvailabilityTable | None) -> None:
    """Print the satellites of a recipe, with the intervals the availability table has data for, if it has one.

    Without a `table` the satellites are listed as they are, since which of them have data is unknown.
    """
    from rich.console import Console  # noqa: PLC0415

    console = Console()
    if table is None:
        console.print(f"{label}: {len(satellites)} satellites")
        names = Table("Satellite", padding=(0, 2, 0, 0))
        for satellite in satellites:
            names.add_row(satellite)
        console.print(names)
        return

    available = [satellite for satellite in satellites if satellite in table]
    without_data = [satellite for satellite in satellites if satellite not in table]
    console.print(f"{label}: {len(available)} of its {len(satellites)} satellites have data, listed below")
    table_out = Table("Satellite", "Intervals with data", padding=(0, 2, 0, 0))
    for satellite in available:
        table_out.add_row(satellite, "\n".join(format_interval(interval) for interval in table[satellite]))
    console.print(table_out)
    if without_data:
        console.print(f"No data in the table: {', '.join(without_data)}")


def _run_calls(
    func: Recipe,
    calls: list[dict[str, Any]],
    loop_names: list[str],
    universal: dict[str, Any],
    *,
    notices: Sequence[tuple[int, str]] = (),
    coverage: dict[str, list[DateInterval]] | None = None,
) -> None:
    """Run `func` once per set of keyword arguments, reporting each call first.

    A failing call does not stop the remaining ones; the run fails at the end.

    `notices` are logged once logging is configured, before the first call. With `coverage`, the
    report of each call says whether the availability table has data for its satellite.

    Raises:
        typer.Exit: With code 1 if any call failed.
    """
    dry_run = universal["dry_run"]
    failed_calls: list[str] = []

    for index, call_kwargs in enumerate(calls, start=1):
        # Fails loudly if the generated command ever drifts from the recipe signature.
        inspect.signature(func).bind(**call_kwargs)

        _configure_logging(
            verbose=universal["verbose"],
            quiet=universal["quiet"],
            logs_path=universal["logs"],
            func=func,
            satellite=call_kwargs.get("satellite"),
            configure_console=index == 1,
        )
        if index == 1:
            for level, message in notices:
                logger.log(level, message)

        if not universal["quiet"]:
            counter = f" ({index}/{len(calls)})" if len(calls) > 1 else ""
            availability = (
                _coverage_text(coverage, call_kwargs[AVAILABILITY_PARAMETER])
                if coverage is not None and AVAILABILITY_PARAMETER in call_kwargs
                else None
            )
            _report_call(func, call_kwargs, title_suffix=counter, dry_run=dry_run, availability=availability)

        if dry_run:
            continue

        call_label = ", ".join(f"{name}={call_kwargs[name]}" for name in loop_names) or _func_name(func)
        try:
            func(**call_kwargs)
        except SystemExit as exit_signal:
            # `ep.exit_if_download_only` ends a call with code 0 once its downloads are done
            if exit_signal.code not in (0, None):
                logger.exception(f"{_func_name(func)} ({call_label}) exited with code {exit_signal.code}.")
                failed_calls.append(call_label)
        except Exception:
            logger.exception(f"{_func_name(func)} ({call_label}) failed.")
            failed_calls.append(call_label)

    if failed_calls:
        mark_run_failed()
        if len(calls) > 1:
            logger.error(f"{len(failed_calls)} of {len(calls)} calls failed: {'; '.join(failed_calls)}")
        raise typer.Exit(code=1)


def _has_availability_parameters(specs: Sequence[_ParameterSpec]) -> bool:
    """Return True if a command has the time range and the ``satellite`` choice an availability table needs."""
    names = {spec.name for spec in specs if spec.choice_enum is not None or spec.name in ("start_time", "end_time")}
    return {"start_time", "end_time", AVAILABILITY_PARAMETER} <= names


def _make_command(
    func: Recipe,
    specs: list[_ParameterSpec],
    skipped: dict[str, Any],
    summary: str,
    availability: AvailabilitySource | None = None,
) -> Any:  # noqa: ANN401
    """Assemble the wrapper callable and attach the synthesised signature."""
    loop_names = [spec.name for spec in specs if spec.is_loop]
    with_availability = _has_availability_parameters(specs)

    def command(**kwargs: Any) -> None:  # noqa: ANN401
        # The universal options belong to the command, not to the recipe, so they are
        # all removed before the remaining keywords are handed over.
        universal = {name: kwargs.pop(name) for name in _UNIVERSAL_PARAMETER_NAMES if name in kwargs}

        if universal["skip_download"]:
            ep.skip_download = True
        if universal["exit_after_download"]:
            ep.exit_after_download = True

        expanded_all = _restore_parsed_values(kwargs, specs)
        kwargs.update(skipped)

        # loaded per invocation, not when the command is built, so that `--help` never reads it
        table = load_availability_table(availability) if availability is not None and with_availability else None

        notices: list[tuple[int, str]] = []
        coverage = None
        if table is not None:
            coverage, notices = _apply_availability(
                kwargs,
                table,
                kwargs["start_time"],
                kwargs["end_time"],
                expanded_all=expanded_all,
                label=_func_name(func),
            )

        _run_calls(func, _expand_loops(kwargs, loop_names), loop_names, universal, notices=notices, coverage=coverage)

    command.__name__ = _func_name(func)
    _attach_signature(command, [*_spec_parameters(specs), *_universal_parameters()])
    command.__doc__ = summary

    return command


def _typer_default(spec: _ParameterSpec) -> Any:  # noqa: ANN401
    """Convert a recipe default into the form Typer expects for that option."""
    if spec.default is inspect.Parameter.empty:
        return ...

    if spec.is_loop:
        values = spec.default if isinstance(spec.default, (list, tuple)) else [spec.default]
        return [str(value) for value in values]

    if spec.is_list and isinstance(spec.default, (list, tuple)):
        return [str(value) for value in spec.default]

    if spec.choice_enum is not None and spec.default is not None:
        return spec.choice_enum(spec.default)

    return spec.default


def _report_call(
    func: Recipe,
    call_kwargs: dict[str, Any],
    *,
    title_suffix: str = "",
    dry_run: bool = False,
    availability: str | None = None,
) -> None:
    """Show the settings a run is about to use, so a long job is never a black box.

    `availability` describes what the recipe's availability table says about the call's satellite.
    """
    prefix = "Would run" if dry_run else "Running"
    table = Table(
        "Parameter",
        "Value",
        title=f"{prefix} {_func_name(func)}{title_suffix}",
        padding=(0, 2, 0, 0),
    )

    for name, value in call_kwargs.items():
        table.add_row(name.replace("_", " "), _format_value(name, value))
    if availability is not None:
        table.add_row("availability", availability)
    _log_table(table)


def _universal_parameters(names: Sequence[str] = _UNIVERSAL_PARAMETER_NAMES) -> list[inspect.Parameter]:
    """Options every recipe command gets, independent of its signature.

    Args:
        names (Sequence[str]): Only return these options, for commands that share only some of
            them. Defaults to all of :data:`_UNIVERSAL_PARAMETER_NAMES`.

    Returns:
        list[inspect.Parameter]: The options, as keyword-only parameters.
    """
    return [
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=annotation)
        for name, (annotation, default) in _UNIVERSAL_OPTIONS.items()
        if name in names
    ]


def check_strategy_factory(
    recipe: Recipe, factory: Callable[..., Any], *, loop_over: Sequence[str] | None = None
) -> None:
    """Check that a strategy factory can be exposed on the command line next to its recipe.

    The ``el-paso plot`` command takes the factory's parameters as its data options, so each of them has
    to mean the same thing as in the recipe. Every parameter of `factory` must

    - exist in the signature of `recipe` (except those in :data:`FACTORY_ONLY_PARAMETERS`),
    - have the same type annotation there, with the same allowed values for a ``Literal``, and
    - have a type the command line builder supports.

    Args:
        recipe (Recipe): The recipe function.
        factory (Callable[..., Any]): The strategy factory the recipe uses to build its saving strategy.
        loop_over (Sequence[str] | None): Parameters of the recipe that accept several values. Defaults to
            any parameter named in :data:`LOOP_PARAMETER_NAMES`.

    Raises:
        TypeError: If any parameter violates the rules above. The message lists all violations.
    """
    recipe_parameters = inspect.signature(recipe).parameters
    recipe_hints = typing.get_type_hints(recipe)
    factory_hints = typing.get_type_hints(factory)
    loop_names = set(loop_over) if loop_over is not None else set(LOOP_PARAMETER_NAMES)

    problems: list[str] = []
    for name, parameter in inspect.signature(factory).parameters.items():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            problems.append(f"{name!r}: variadic parameters cannot be exposed on the command line")
            continue

        factory_hint = factory_hints.get(name, parameter.annotation)

        if name in FACTORY_ONLY_PARAMETERS:
            pass
        elif name not in recipe_parameters:
            problems.append(f"{name!r}: not a parameter of {_func_name(recipe)}")
        elif recipe_hints.get(name, recipe_parameters[name].annotation) != factory_hint:
            problems.append(
                f"{name!r}: annotated {factory_hint!r} in the factory but "
                f"{recipe_hints.get(name, recipe_parameters[name].annotation)!r} in {_func_name(recipe)}"
            )

        if _build_parameter_spec(name, factory_hint, parameter.default, "", is_loop=name in loop_names) is None:
            problems.append(f"{name!r}: unsupported annotation {factory_hint!r}")

    if problems:
        details = "\n  - ".join(problems)
        msg = f"Strategy factory {_func_name(factory)} does not match recipe {_func_name(recipe)}:\n  - {details}"
        raise TypeError(msg)


def build_list_satellites_command(
    func: Recipe,
    *,
    defaults: dict[str, Any] | None = None,
    availability: AvailabilitySource | None = None,
    label: str | None = None,
) -> Callable[..., None]:
    """Build the command that lists the satellites of a recipe.

    The command takes no arguments. It prints the satellites of the recipe with the intervals the
    recipe's availability table has data for, then exits without processing anything. The table is
    optional: without one, the command lists the satellites as they are.

    Args:
        func (Recipe): The recipe whose satellites are listed.
        defaults (dict[str, Any] | None): The recipe's CLI defaults, which override the defaults of its
            signature exactly as in the processing command.
        availability (AvailabilitySource | None): The recipe's satellite availability table. It is read
            when the command runs, not when it is built.
        label (str | None): How the recipe is named in the output, e.g. ``poes meped``. Defaults to the
            recipe's function name.

    Returns:
        Callable[..., None]: A callable ready to be registered with ``app.command()``.
    """
    satellite_specs, _ = _build_parameter_specs(func, defaults=defaults, names=(AVAILABILITY_PARAMETER,))
    satellite_spec = next(iter(satellite_specs), None)
    satellite_choices: list[str] = []
    if satellite_spec is not None and satellite_spec.choice_enum is not None:
        satellite_choices = list(satellite_spec.loop_values or [member.value for member in satellite_spec.choice_enum])

    def command() -> None:
        if not satellite_choices:
            from rich.console import Console  # noqa: PLC0415

            Console().print(f"{_func_name(func)} has no {AVAILABILITY_PARAMETER} option, so there is nothing to list.")
            raise typer.Exit(1)

        # loaded per invocation, not when the command is built, so that `--help` never reads it
        table = load_availability_table(availability) if availability is not None else None
        _print_satellite_listing(label or _func_name(func), satellite_choices, table)

    command.__name__ = f"list_satellites_{_func_name(func)}"
    command.__doc__ = (
        "List the satellites of the recipe, with the intervals the availability table has data for, "
        "then exit without processing. A recipe without a table lists its satellites only."
    )
    return command


def run_recipe_cli(
    func: Recipe,
    *,
    defaults: dict[str, Any] | None = None,
    loop_over: Sequence[str] | None = None,
    strategy: Callable[..., Any] | None = None,
    plot_defaults: dict[str, Any] | None = None,
    availability: AvailabilitySource | None = None,
    args: Sequence[str] | None = None,
) -> None:
    """Run a recipe as a standalone command line program.

    Intended for the ``if __name__ == "__main__":`` block of a recipe module. The
    command line it produces is identical to the one the ``el-paso`` entry point
    exposes for the same recipe, because both are built by
    :func:`build_recipe_command`.

    Args:
        func (Recipe): The recipe function to run.
        defaults (dict[str, Any] | None): Overrides for the defaults taken from the
            signature, typically the processing time range.
        loop_over (Sequence[str] | None): Parameters that accept several values, causing
            one call to `func` per value. Defaults to any parameter named in
            :data:`LOOP_PARAMETER_NAMES`.
        strategy (Callable[..., Any] | None): The strategy factory the recipe builds its saving strategy
            with. It is checked against the recipe with :func:`check_strategy_factory`, so a mismatch
            fails at start-up. The ``el-paso plot`` command uses it to locate the processed data.
        plot_defaults (dict[str, Any] | None): Recipe-specific plot settings for ``el-paso plot``,
            checked with :func:`el_paso.cli.plot_cli.check_plot_defaults`. Requires `strategy`.
        availability (AvailabilitySource | None): The recipe's ``SATELLITE_AVAILABILITY``, see
            :func:`build_recipe_command`.
        args (Sequence[str] | None): Command line arguments to parse. Defaults to
            `sys.argv`; mainly useful for testing.

    Raises:
        ValueError: If `plot_defaults` is given without `strategy`, or holds a setting that is out
            of range.
        TypeError: If `strategy` does not match the recipe, or `plot_defaults` is malformed.
    """
    if plot_defaults is not None and strategy is None:
        msg = "plot_defaults requires a strategy factory."
        raise ValueError(msg)
    if strategy is not None:
        check_strategy_factory(func, strategy, loop_over=loop_over)
    if plot_defaults is not None:
        from el_paso.cli.plot_cli import check_plot_defaults  # noqa: PLC0415  (plot_cli imports this module)

        check_plot_defaults(plot_defaults)

    app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
    app.command()(build_recipe_command(func, defaults=defaults, loop_over=loop_over, availability=availability))
    app(args=args)
