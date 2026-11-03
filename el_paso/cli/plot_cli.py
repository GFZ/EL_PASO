# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""The ``el-paso plot`` command: quick-look plots of data a recipe has already processed.

``el-paso plot <mission> <recipe>`` takes the same arguments as ``el-paso <mission> <recipe>``.
It never processes or downloads anything; it finds the processed files through the strategy
factory the recipe registers, so file names are worked out in one place. Its options come from
three sources:

- shared with processing: ``--start-time``/``--end-time`` (from the recipe signature),
  ``-v``/``--quiet`` and ``--dry-run``;
- the strategy factory's signature, e.g. ``--satellite``, ``--mag-field``,
  ``--processed-data-path``, with the recipe's defaults, so that the same arguments find the
  files a processing run wrote;
- plot-only options: ``--variable``, ``--output`` and ``--show``.

Processing-only options such as ``--num-cores`` or ``--raw-data-path`` are not part of the
factory's signature and so are rejected.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import typer
from rich.table import Table

import el_paso as ep
from el_paso.cli.recipe_cli import (
    _attach_signature,
    _build_parameter_specs,
    _expand_loops,
    _format_value,
    _func_name,
    _log_level,
    _log_table,
    _restore_parsed_values,
    _spec_parameters,
    _universal_parameters,
    check_strategy_factory,
    parse_docstring,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from datetime import datetime

    from matplotlib.figure import Figure

    from el_paso.dataset import DataSet
    from el_paso.typing import InternalName, Recipe, SavingStrategy

logger = logging.getLogger("el_paso.cli")

_PANEL_HEIGHT_INCHES = 2.0
"""Height of one panel in the multi-satellite figure."""

TIME_PARAMETER_NAMES = ("start_time", "end_time")
"""Recipe parameters the plot command takes over from the recipe, next to the factory's parameters."""

_SHARED_PARAMETER_NAMES = ("verbose", "quiet")
"""Universal recipe-command options the plot command shares; see `recipe_cli._universal_parameters`."""

_PLOT_PARAMETER_NAMES = ("dry_run", "variables", "output", "show")
"""Options defined here: ``--dry-run`` is shared with processing in name only, the rest are plot-only."""


@dataclasses.dataclass(frozen=True)
class PlotRequest:
    """Everything a plot run needs, resolved from the command line.

    Attributes:
        recipe: The recipe that processed the data.
        factory: The recipe's strategy factory.
        start_time: Start of the time range to plot.
        end_time: End of the time range to plot.
        strategy_kwargs: Arguments for `factory`, one set per value of a looped parameter
            (one per satellite).
        variables: Internal names of the variables to plot, or ``None`` for the defaults.
        output: File to write the figure to, or ``None`` for the default file name.
        show: Open an interactive window.
        dry_run: Only report what would be plotted.
        verbose: Log every file that is loaded.
        plot_defaults: The recipe's ``PLOT_DEFAULTS``.
        processing_command: The command that runs the recipe, e.g. ``el-paso rbsp mageis-electrons``,
            suggested when processed files are missing. ``None`` for ``python -m <recipe module>``.
    """

    recipe: Recipe
    factory: Callable[..., SavingStrategy]
    start_time: datetime
    end_time: datetime
    strategy_kwargs: list[dict[str, Any]]
    variables: list[str] | None
    output: Path | None
    show: bool
    dry_run: bool
    verbose: bool
    plot_defaults: dict[str, Any]
    processing_command: str | None = None


def _plot_parameters() -> list[inspect.Parameter]:
    """Options defined by the plot command itself, independent of the recipe."""
    definitions: list[tuple[str, Any, Any]] = [
        (
            "dry_run",
            Annotated[
                bool,
                typer.Option(
                    "--dry-run",
                    help="Print the saving strategy and the expected files, and which of them exist, "
                    "then exit without plotting.",
                ),
            ],
            False,
        ),
        (
            "variables",
            Annotated[
                list[str] | None,
                typer.Option(
                    "--variable",
                    help="Internal name of a variable to plot in its own panel (e.g. FEDU). Repeat for "
                    "several panels: --variable FEDU --variable L_star. Required unless the recipe defines "
                    "quick-look panels.",
                ),
            ],
            None,
        ),
        (
            "output",
            Annotated[
                Path | None,
                typer.Option(
                    "--output",
                    "-o",
                    help="File to write the figure to; the format follows its suffix. "
                    "Defaults to <recipe>_<satellite>_<start>_<end>.png unless --show is given.",
                ),
            ],
            None,
        ),
        (
            "show",
            Annotated[bool, typer.Option("--show", help="Open the figure in an interactive window.")],
            False,
        ),
    ]

    return [
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=annotation)
        for name, annotation, default in definitions
    ]


def _factory_defaults(recipe: Recipe, factory: Callable[..., Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """Defaults for the factory's options: the recipe's, so that plot finds what processing wrote.

    The recipe's CLI defaults take precedence over its signature, as in the processing command.
    """
    recipe_parameters = inspect.signature(recipe).parameters
    factory_defaults: dict[str, Any] = {}

    for name in inspect.signature(factory).parameters:
        if name in defaults:
            factory_defaults[name] = defaults[name]
        elif name in recipe_parameters and recipe_parameters[name].default is not inspect.Parameter.empty:
            factory_defaults[name] = recipe_parameters[name].default

    return factory_defaults


def build_plot_command(
    recipe: Recipe,
    factory: Callable[..., SavingStrategy],
    *,
    defaults: dict[str, Any] | None = None,
    loop_over: Sequence[str] | None = None,
    plot_defaults: dict[str, Any] | None = None,
    processing_command: str | None = None,
) -> Callable[..., None]:
    """Build the Typer-compatible ``el-paso plot`` command for one recipe.

    The command's options are the recipe's time range, the strategy factory's parameters and
    the plot-only options, built by the same machinery as the processing command (see
    :func:`el_paso.cli.recipe_cli.build_recipe_command`).

    Args:
        recipe (Recipe): The recipe whose output is plotted.
        factory (Callable[..., SavingStrategy]): The recipe's strategy factory. It is checked against
            the recipe with :func:`el_paso.cli.recipe_cli.check_strategy_factory`.
        defaults (dict[str, Any] | None): The recipe's CLI defaults, which override the defaults of
            its signature exactly as in the processing command.
        loop_over (Sequence[str] | None): Parameters that accept several values. Defaults to any
            parameter named in :data:`el_paso.cli.recipe_cli.LOOP_PARAMETER_NAMES`.
        plot_defaults (dict[str, Any] | None): The recipe's ``PLOT_DEFAULTS``.
        processing_command (str | None): The command that runs the recipe, e.g.
            ``el-paso rbsp mageis-electrons``, suggested when processed files are missing.
            Defaults to ``python -m <recipe module>``.

    Returns:
        Callable[..., None]: A callable ready to be registered with ``app.command()``.

    Raises:
        TypeError: If the factory does not match the recipe, or if a factory parameter has the
            same name as a shared or plot-only option.
    """
    defaults = dict(defaults or {})
    plot_defaults = dict(plot_defaults or {})
    check_strategy_factory(recipe, factory, loop_over=loop_over)

    _, recipe_help = parse_docstring(recipe.__doc__)
    _, factory_help = parse_docstring(factory.__doc__)

    time_specs, _ = _build_parameter_specs(
        recipe,
        defaults=defaults,
        names=TIME_PARAMETER_NAMES,
        arg_help={"start_time": "Start of the time range to plot.", "end_time": "End of the time range to plot."},
    )
    factory_specs, _ = _build_parameter_specs(
        factory,
        defaults=_factory_defaults(recipe, factory, defaults),
        loop_over=loop_over,
        # the factory describes its parameters as the data to find, the recipe as the data to write
        arg_help={**recipe_help, **{name: text for name, text in factory_help.items() if text}},
    )
    specs = [*time_specs, *factory_specs]
    loop_names = [spec.name for spec in factory_specs if spec.is_loop]

    def command(**kwargs: Any) -> None:  # noqa: ANN401
        shared = {name: kwargs.pop(name) for name in _SHARED_PARAMETER_NAMES}
        plot_options = {name: kwargs.pop(name) for name in _PLOT_PARAMETER_NAMES}

        _restore_parsed_values(kwargs, specs)
        start_time = kwargs.pop("start_time")
        end_time = kwargs.pop("end_time")
        if end_time <= start_time:
            msg = f"must be after --start-time ({start_time.isoformat()}), got {end_time.isoformat()}."
            raise typer.BadParameter(msg, param_hint="'--end-time'")

        strategy_kwargs = _expand_loops(kwargs, loop_names)
        for call_kwargs in strategy_kwargs:
            # Fails loudly if the generated command ever drifts from the factory signature.
            inspect.signature(factory).bind(**call_kwargs)

        # The dry-run report is logged at INFO, and printing it is the whole point of --dry-run.
        quiet = shared["quiet"] and not plot_options["dry_run"]
        ep.setup_logging(_log_level(verbose=shared["verbose"], quiet=quiet))

        _run_plot(
            PlotRequest(
                recipe=recipe,
                factory=factory,
                start_time=start_time,
                end_time=end_time,
                strategy_kwargs=strategy_kwargs,
                variables=list(plot_options["variables"]) if plot_options["variables"] else None,
                output=plot_options["output"],
                show=plot_options["show"],
                dry_run=plot_options["dry_run"],
                verbose=shared["verbose"] >= 1,
                plot_defaults=plot_defaults,
                processing_command=processing_command,
            )
        )

    command.__name__ = f"plot_{_func_name(recipe)}"
    _attach_signature(
        command,
        [*_spec_parameters(specs), *_universal_parameters(_SHARED_PARAMETER_NAMES), *_plot_parameters()],
    )
    command.__doc__ = (
        f"Plot data processed by {_func_name(recipe)}.\n\n"
        "Takes the same arguments as the processing command; nothing is processed or downloaded."
    )
    return command


def _report_settings(request: PlotRequest) -> None:
    """Show the resolved settings of a plot run, as processing runs do for their arguments."""
    prefix = "Would plot" if request.dry_run else "Plotting"
    table = Table("Parameter", "Value", title=f"{prefix} {_func_name(request.recipe)}", padding=(0, 2, 0, 0))

    table.add_row("start time", _format_value("start_time", request.start_time))
    table.add_row("end time", _format_value("end_time", request.end_time))
    loop_values: dict[str, list[Any]] = {}
    for kwargs in request.strategy_kwargs:
        for name, value in kwargs.items():
            values = loop_values.setdefault(name, [])
            if value not in values:
                values.append(value)
    for name, values in loop_values.items():
        table.add_row(name.replace("_", " "), ", ".join(_format_value(name, value) for value in values))

    table.add_row("variables", ", ".join(request.variables) if request.variables else "recipe defaults")
    table.add_row("output", _format_value("output", request.output) if request.output else "default file name")
    table.add_row("show", str(request.show))
    _log_table(table)


def _call_labels(strategy_kwargs: list[dict[str, Any]]) -> list[str]:
    """Label each set of factory arguments by the values that differ between the sets (e.g. ``satellite=a``).

    The settings table already shows everything else, so a single set gets an empty label.
    """
    varying = [
        name
        for name in strategy_kwargs[0]
        if len({_format_value(name, kwargs[name]) for kwargs in strategy_kwargs}) > 1
    ]
    return [", ".join(f"{name}={_format_value(name, kwargs[name])}" for name in varying) for kwargs in strategy_kwargs]


def _report_expected_files(request: PlotRequest) -> bool:
    """Print the saving strategy and expected files of every set of factory arguments.

    Returns:
        bool: ``True`` if the files could be determined for every set of arguments.
    """
    # Not checked against the data standard here; get_file_coverage rejects unknown names.
    names = cast("list[InternalName] | None", request.variables)

    ok = True
    for kwargs, label in zip(request.strategy_kwargs, _call_labels(request.strategy_kwargs), strict=True):
        suffix = f" ({label})" if label else ""
        try:
            strategy = request.factory(**kwargs)
            dataset = ep.DataSet(strategy, request.start_time, request.end_time, verbose=False)
            coverage = dataset.get_file_coverage(names)
        except ValueError as error:
            logger.error(f"{error}{suffix}")  # noqa: TRY400
            ok = False
            continue

        logger.info(f"Saving strategy{suffix}: {strategy!r}")
        logger.info("Expected files:")
        for entry in coverage:
            logger.info(f"  [{'found' if entry.exists else 'missing'}] {entry.path}")
        found = sum(entry.exists for entry in coverage)
        logger.info(f"{found} of {len(coverage)} expected files exist.")

    return ok


def _format_time_for_file_name(time: datetime) -> str:
    """Compact time stamp for file names: the date alone at midnight, otherwise date and time."""
    if (time.hour, time.minute, time.second, time.microsecond) == (0, 0, 0, 0):
        return time.strftime("%Y%m%d")
    return time.strftime("%Y%m%dT%H%M%S")


def default_output_path(request: PlotRequest) -> Path:
    """File the figure is written to when neither ``--output`` nor ``--show`` is given.

    The name is ``<recipe>_<satellite>_<start>_<end>.png`` in the current directory, where
    ``<recipe>`` is the recipe function's name without its ``process_`` prefix and ``<satellite>``
    joins all plotted satellites with ``-``. Recipes without a ``satellite`` parameter leave it out.

    Args:
        request (PlotRequest): The resolved plot request.

    Returns:
        Path: The default output file, relative to the current directory.
    """
    recipe_name = _func_name(request.recipe).removeprefix("process_")
    satellites: list[str] = []
    for kwargs in request.strategy_kwargs:
        if "satellite" in kwargs:
            satellite = _format_value("satellite", kwargs["satellite"])
            if satellite not in satellites:
                satellites.append(satellite)

    parts = [recipe_name]
    if satellites:
        parts.append("-".join(satellites))
    parts += [_format_time_for_file_name(request.start_time), _format_time_for_file_name(request.end_time)]
    return Path("_".join(parts) + ".png")


def _interactive_backend_error() -> str | None:
    """Explain why ``--show`` cannot open a window, or return ``None`` if it can.

    Importing pyplot resolves matplotlib's backend; without a display it falls back to Agg,
    which cannot show figures.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.backends import BackendFilter, backend_registry  # noqa: PLC0415

    backend = plt.get_backend()
    if backend.lower() in backend_registry.list_builtin(BackendFilter.NON_INTERACTIVE):
        return (
            f"--show needs an interactive matplotlib backend, but '{backend}' is in use "
            "(no display available?). Use --output to write the figure to a file instead, "
            "or set MPLBACKEND to an interactive backend."
        )
    return None


def _format_time_for_cli(time: datetime) -> str:
    """A time as the command line accepts it: the date alone at midnight, otherwise date and time (UTC)."""
    if (time.hour, time.minute, time.second, time.microsecond) == (0, 0, 0, 0):
        return time.strftime("%Y-%m-%d")
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def processing_command_line(request: PlotRequest, kwargs: dict[str, Any]) -> str:
    """The processing command that writes the files one set of factory arguments expects.

    It runs the recipe for the plotted time range with every factory argument spelled out, so it
    writes exactly the files the plot command looks for.

    Args:
        request (PlotRequest): The resolved plot request.
        kwargs (dict[str, Any]): One set of factory arguments from ``request.strategy_kwargs``.

    Returns:
        str: The command line, quoted for a POSIX shell.
    """
    import shlex  # noqa: PLC0415

    command = request.processing_command or f"python -m {request.recipe.__module__}"
    arguments = [
        "--start-time",
        _format_time_for_cli(request.start_time),
        "--end-time",
        _format_time_for_cli(request.end_time),
    ]
    for name, value in kwargs.items():
        option = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            arguments.append(option if value else f"--no-{option.removeprefix('--')}")
            continue
        for item in value if isinstance(value, list) else [value]:
            arguments += [option, _format_value(name, item)]
    return " ".join([command, *(shlex.quote(argument) for argument in arguments)])


def _format_interval(start: datetime, end: datetime) -> str:
    """Name a time interval as briefly as possible: a month (``2017-10``), a day, or a range."""
    from datetime import timedelta  # noqa: PLC0415

    def is_midnight(time: datetime) -> bool:
        return (time.hour, time.minute, time.second, time.microsecond) == (0, 0, 0, 0)

    at_midnight = is_midnight(start)
    # interval ends are inclusive (23:59:59) or exclusive (the next midnight); compare as exclusive
    stop = end + timedelta(seconds=1) if is_midnight(end + timedelta(seconds=1)) else end
    next_month = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    if at_midnight and start.day == 1 and stop == next_month:
        return start.strftime("%Y-%m")
    if at_midnight and stop == start + timedelta(days=1):
        return start.strftime("%Y-%m-%d")
    return f"{_format_time_for_cli(start)} to {_format_time_for_cli(end)}"


def _format_intervals(intervals: Sequence[tuple[datetime, datetime]]) -> str:
    """Name a list of intervals, merging consecutive ones into ranges (``2017-10 to 2017-12``)."""
    from datetime import timedelta  # noqa: PLC0415

    runs: list[list[tuple[datetime, datetime]]] = []
    for interval in intervals:
        if runs and interval[0] - runs[-1][-1][1] <= timedelta(seconds=1):
            runs[-1].append(interval)
        else:
            runs.append([interval])

    names: list[str] = []
    for run in runs:
        first, last = _format_interval(*run[0]), _format_interval(*run[-1])
        names.append(first if len(run) == 1 else f"{first} to {last}")
    return ", ".join(names)


def _data_label(dataset: DataSet) -> str:
    """Name the data of one dataset for messages, e.g. ``rbspa mageis (TS04)``."""
    mag_field = getattr(dataset.saving_strategy, "mag_field", None)
    name = dataset.get_print_name()
    return f"{name} ({mag_field})" if mag_field else name


def _check_coverage(request: PlotRequest, kwargs: dict[str, Any], dataset: DataSet) -> bool:
    """Report missing processed files for one dataset.

    Partial coverage is a warning naming the missing intervals; the gaps stay blank in the plot.
    No files at all is an error with the expected path of the first file and the processing
    command that would create them.

    Returns:
        bool: ``True`` if at least one expected file exists, so that there is something to plot.
    """
    names = cast("list[InternalName]", request.variables)
    coverage = dataset.get_file_coverage(names)

    found: list[tuple[datetime, datetime]] = []
    missing: list[tuple[datetime, datetime]] = []
    # an interval with several output files counts as covered only if all of them exist
    for entry in coverage:
        interval = (entry.interval_start, entry.interval_end)
        if not entry.exists:
            if interval in found:
                found.remove(interval)
            if interval not in missing:
                missing.append(interval)
        elif interval not in found and interval not in missing:
            found.append(interval)

    if not missing:
        return True

    label = _data_label(dataset)
    if any(entry.exists for entry in coverage):
        covered = f"plotting {_format_intervals(found)} only" if found else "plotting the files that exist"
        logger.warning(f"No processed files for {label} in {_format_intervals(missing)}; {covered}.")
        return True

    first_missing = next(entry.path for entry in coverage if not entry.exists)
    logger.error(
        f"No processed files for {label} between {_format_time_for_cli(request.start_time)} "
        f"and {_format_time_for_cli(request.end_time)}.\n"
        f"Expected: {first_missing}\n"
        f"To create them: {processing_command_line(request, kwargs)}"
    )
    return False


def _load_datasets(request: PlotRequest) -> list[DataSet]:
    """Build one `DataSet` per set of factory arguments (one per satellite) that has processed files.

    A satellite without any files, or whose files cannot be determined, is reported and left out,
    so that the others are still plotted.
    """
    datasets: list[DataSet] = []
    for kwargs, label in zip(request.strategy_kwargs, _call_labels(request.strategy_kwargs), strict=True):
        suffix = f" ({label})" if label else ""
        try:
            strategy = request.factory(**kwargs)
            dataset = ep.DataSet(strategy, request.start_time, request.end_time, verbose=request.verbose)
            if _check_coverage(request, kwargs, dataset):
                datasets.append(dataset)
        except ValueError as error:
            logger.error(f"{error}{suffix}")  # noqa: TRY400
    return datasets


def _new_figure(num_panels: int, *, show: bool) -> Figure:
    """Create the figure for `num_panels` stacked panels.

    Only a figure that is shown is created through pyplot (and so with its backend); otherwise
    the figure is not tied to any GUI backend, and saving renders it with the backend that
    matches the file format (Agg for PNG), which also works without a display.
    """
    size = (10, _PANEL_HEIGHT_INCHES * num_panels + 0.8)
    if show:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        return plt.figure(figsize=size, layout="constrained")

    from matplotlib.figure import Figure  # noqa: PLC0415

    return Figure(figsize=size, layout="constrained")


def build_figure(
    datasets: Sequence[DataSet],
    variables: Sequence[InternalName],
    start_time: datetime,
    end_time: datetime,
    *,
    show: bool = False,
) -> Figure:
    """Draw the variables of several datasets into one column of panels with a shared time axis.

    Panels are stacked in one block of rows per dataset (satellite), one panel per variable, in
    the given order; nothing is overlaid. Each block is drawn by `DataSet.plot` and titled with
    its dataset's name. Only the bottom panel shows time tick labels, covering
    ``[start_time, end_time]``.

    Args:
        datasets (Sequence[DataSet]): The datasets to plot, one block of panels each.
        variables (Sequence[InternalName]): Internal names of the variables to plot per dataset.
        start_time (datetime): Start of the time axis.
        end_time (datetime): End of the time axis.
        show (bool): Create the figure through pyplot, so that it can be shown in a window.

    Returns:
        Figure: The figure.
    """
    import matplotlib.dates as mdates  # noqa: PLC0415

    figure = _new_figure(len(datasets) * len(variables), show=show)
    axes = list(figure.subplots(len(datasets) * len(variables), 1, sharex=True, squeeze=False)[:, 0])

    for index, dataset in enumerate(datasets):
        block = axes[index * len(variables) : (index + 1) * len(variables)]
        dataset.plot(variables, axes=block)

    bottom = axes[-1]
    bottom.xaxis_date()
    locator = mdates.AutoDateLocator()
    bottom.xaxis.set_major_locator(locator)
    bottom.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    bottom.set_xlim(start_time, end_time)
    bottom.set_xlabel("Time (UTC)")
    return figure


def _run_plot(request: PlotRequest) -> None:
    """Plot the processed data a request describes.

    The figure is written to ``--output``, or shown with ``--show``, or both. With neither, it is
    written to :func:`default_output_path`.

    Satellites with only some of the expected files are plotted with gaps; satellites without
    any are left out (see `_check_coverage`).

    Raises:
        typer.BadParameter: If no variables are given.
        typer.Exit: With code 1 if the expected files cannot be determined, if ``--show`` is
            given without an interactive backend, or if no satellite has any processed files.
    """
    _report_settings(request)

    if request.dry_run:
        if not _report_expected_files(request):
            raise typer.Exit(code=1)
        return

    if not request.variables:
        msg = f"is required: {_func_name(request.recipe)} defines no default panels."
        raise typer.BadParameter(msg, param_hint="'--variable'")
    variables = cast("list[InternalName]", request.variables)

    if request.show and (error := _interactive_backend_error()):
        logger.error(error)
        raise typer.Exit(code=1)

    datasets = _load_datasets(request)
    if not datasets:
        raise typer.Exit(code=1)

    try:
        figure = build_figure(datasets, variables, request.start_time, request.end_time, show=request.show)
    except ValueError as error:
        logger.error(str(error))  # noqa: TRY400
        raise typer.Exit(code=1) from error

    output = request.output
    if output is None and not request.show:
        output = default_output_path(request)
    if output is not None:
        figure.savefig(output)
        logger.info(f"Saved the figure to {output}.")

    if request.show:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        plt.show()
