# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
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
- plot-only options: ``--variable``, ``--energy``, ``--pitch-angle``, ``--vmin``, ``--vmax``,
  ``--along-L-star``/``--along-L-m``/``--along-R-eq`` (at most one), ``--output-dir``, ``--filename``,
  ``--show`` and ``--export-script``.

Processing-only options such as ``--num-cores`` or ``--raw-data-path`` are not part of the
factory's signature and so are rejected.

Plot settings come in layers: the defaults of each variable's plot type (in `DataSet.plot`), then
the recipe's ``PLOT_DEFAULTS``, then the plot-only options. ``PLOT_DEFAULTS`` is a dict with an
optional ``"panels"`` list of internal names, used when ``--variable`` is not given, and settings per
internal name with keys from :data:`el_paso.dataset.plot.PANEL_SETTING_NAMES`::

    PLOT_DEFAULTS = {
        "panels": ["FEDU", "L_star", "MLT"],
        "FEDU": {"pitch_angle": 90 * u.deg, "energies": [470, 1575] * u.keV, "vmin": 1e0, "vmax": 1e6},
    }

``--variable`` replaces the panel list and ignores ``PLOT_DEFAULTS`` entirely (the variable-type defaults
still apply); ``--energy``, ``--pitch-angle``, ``--vmin``, ``--vmax`` and ``--along-*`` override the
matching setting of every panel. A variable's ``"along"`` setting (``"L_star"``, ``"L_m"`` or ``"R_Eq"``)
draws it as scatter panels over time with that coordinate on the y axis.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import typer
from rich.table import Table

import el_paso as ep
from el_paso.cli._logging import _log_level, _log_table
from el_paso.cli._parsing import _expand_loops, _format_value, _func_name, parse_docstring
from el_paso.cli.availability import AVAILABILITY_PARAMETER, load_availability_table
from el_paso.cli.plot_figure import DEFAULT_COLORMAP, INDEX_PANELS, build_figure
from el_paso.cli.plot_script import write_plot_script
from el_paso.cli.recipe_cli import (
    _apply_availability,
    _attach_signature,
    _build_parameter_specs,
    _coverage_text,
    _has_availability_parameters,
    _restore_parsed_values,
    _spec_parameters,
    _universal_parameters,
    check_strategy_factory,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from datetime import datetime

    from astropy import units as u

    from el_paso.cli.availability import AvailabilitySource, DateInterval
    from el_paso.dataset import DataSet
    from el_paso.typing import InternalName, Recipe, SavingStrategy

logger = logging.getLogger("el_paso.cli")

TIME_PARAMETER_NAMES = ("start_time", "end_time")
"""Recipe parameters the plot command takes over from the recipe, next to the factory's parameters."""

_SHARED_PARAMETER_NAMES = ("verbose", "quiet")
"""Universal recipe-command options the plot command shares; see `recipe_cli._universal_parameters`."""

PANELS_KEY = "panels"
"""Key of the default panel list in a recipe's ``PLOT_DEFAULTS``; every other key is a variable's settings."""

ALONG_OPTIONS: dict[str, tuple[str, InternalName]] = {
    "along_l_star": ("--along-L-star", "L_star"),
    "along_l_m": ("--along-L-m", "L_m"),
    "along_r_eq": ("--along-R-eq", "R_Eq"),
}
"""Parameter name of each ``--along-*`` option, its flag, and the coordinate it puts on the y axis of the panels.

The coordinates are those of :data:`el_paso.dataset.plot.ALONG_COORDINATES`."""

_PLOT_PARAMETER_NAMES = (
    "dry_run",
    "variables",
    "energies",
    "pitch_angle",
    "vmin",
    "vmax",
    "colormap",
    *ALONG_OPTIONS,
    "output_dir",
    "filename",
    "show",
    "export_script",
)
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
        variables: Internal names of the variables given with ``--variable``, or ``None`` for the
            recipe's default panels (see `resolve_panels`).
        output_dir: Directory to write the figure to, or ``None`` for the current directory.
        filename: Name of the figure's file, or ``None`` for the default name (see `default_filename`).
        show: Open an interactive window.
        dry_run: Only report what would be plotted.
        verbose: Log every file that is loaded.
        plot_defaults: The recipe's ``PLOT_DEFAULTS``.
        processing_command: The command that runs the recipe, e.g. ``el-paso rbsp mageis-electrons``,
            suggested when processed files are missing. ``None`` for ``python -m <recipe module>``.
        energies: Energies to show for variables with an energy dimension (``--energy``), or
            ``None`` for the defaults.
        pitch_angle: Pitch angle at which variables with an energy and a pitch-angle dimension are
            cut (``--pitch-angle``), or ``None`` for the defaults.
        vmin: Lower colour limit of spectrograms (``--vmin``), in the plotted variable's unit.
        vmax: Upper colour limit of spectrograms (``--vmax``), in the plotted variable's unit.
        along: Coordinate on the y axis of scatter panels (``--along-L-star``, ``--along-L-m`` or
            ``--along-R-eq``), or ``None`` for the usual plot types.
        colormap: Colormap of spectrograms and scatter panels (``--colormap``).
        export_script: File to write a standalone Python script that makes the figure to, instead of
            plotting (``--export-script``), or ``None`` to plot.
        availability: The satellites the recipe's availability table has data for in the time range,
            with their overlapping intervals, or ``None`` if the recipe has no table.
    """

    recipe: Recipe
    factory: Callable[..., SavingStrategy]
    start_time: datetime
    end_time: datetime
    strategy_kwargs: list[dict[str, Any]]
    variables: list[str] | None
    output_dir: Path | None
    filename: str | None
    show: bool
    dry_run: bool
    verbose: bool
    plot_defaults: dict[str, Any]
    processing_command: str | None = None
    energies: u.Quantity | None = None
    pitch_angle: u.Quantity | None = None
    vmin: float | None = None
    vmax: float | None = None
    along: InternalName | None = None
    colormap: str = DEFAULT_COLORMAP
    export_script: Path | None = None
    availability: dict[str, list[DateInterval]] | None = None


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
            "energies",
            Annotated[
                list[str] | None,
                typer.Option(
                    "--energy",
                    metavar="ENERGY",
                    help="Energy to show for variables with an energy dimension (e.g. FEDU), with an optional "
                    "unit; a bare number is in keV. Repeat for several: --energy 470 --energy 1MeV.",
                ),
            ],
            None,
        ),
        (
            "pitch_angle",
            Annotated[
                float | None,
                typer.Option(
                    "--pitch-angle",
                    metavar="DEG",
                    min=0.0,
                    max=180.0,
                    help="Pitch angle in degrees at which variables with an energy and a pitch-angle "
                    "dimension (e.g. FEDU) are shown, defaulting to 90. If given, variables that only depend on "
                    "pitch angle (e.g. L_star) are drawn as a line at this pitch angle.",
                ),
            ],
            None,
        ),
        *(
            (
                bound,
                Annotated[
                    float | None,
                    typer.Option(
                        f"--{bound}",
                        help=f"{limit} colour limit of spectrograms and --along-* scatter panels (y-limit of "
                        "--energy line panels), in the unit of the plotted variable. "
                        f"Defaults to the {percentile}th percentile of the data.",
                    ),
                ],
                None,
            )
            for bound, limit, percentile in (("vmin", "Lower", 5), ("vmax", "Upper", 95))
        ),
        (
            "colormap",
            Annotated[
                str,
                typer.Option(
                    "--colormap",
                    help="Colormap of spectrograms and --along-* scatter panels: a colorcet name "
                    "(e.g. rainbow4, fire, CET_L17) or a matplotlib colormap name (e.g. viridis).",
                ),
            ],
            DEFAULT_COLORMAP,
        ),
        *(
            (
                parameter,
                Annotated[
                    bool,
                    typer.Option(
                        flag,
                        help=f"Draw every variable as a scatter plot over time with {coordinate} on the y axis "
                        "and the value as colour. Variables with an energy dimension get one panel per --energy, "
                        "or per channel without --energy; pitch angles are cut at --pitch-angle (default 90). "
                        "At most one --along-* option.",
                    ),
                ],
                False,
            )
            for parameter, (flag, coordinate) in ALONG_OPTIONS.items()
        ),
        (
            "output_dir",
            Annotated[
                Path | None,
                typer.Option(
                    "--output-dir",
                    help="Directory to write the figure to, created if missing. Defaults to the current directory.",
                ),
            ],
            None,
        ),
        (
            "filename",
            Annotated[
                str | None,
                typer.Option(
                    "--filename",
                    help="Name of the figure's file; the format follows its suffix. "
                    "Defaults to <recipe>_<satellite>_<start>_<end>.png. "
                    "The figure is only written if this or --output-dir is given, or if --show is not.",
                ),
            ],
            None,
        ),
        (
            "show",
            Annotated[bool, typer.Option("--show", help="Open the figure in an interactive window.")],
            False,
        ),
        (
            "export_script",
            Annotated[
                Path | None,
                typer.Option(
                    "--export-script",
                    help="Write a standalone Python script that makes this figure to this file, instead of "
                    "plotting. Edit the script to customise the figure.",
                ),
            ],
            None,
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


def panel_settings(plot_defaults: Mapping[str, Any]) -> dict[str, Any]:
    """The per-variable settings of a recipe's ``PLOT_DEFAULTS``: everything but the panel list."""
    return {name: settings for name, settings in plot_defaults.items() if name != PANELS_KEY}


def check_plot_defaults(plot_defaults: Mapping[str, Any]) -> None:
    """Validate a recipe's ``PLOT_DEFAULTS``, so that a mistake fails when the command is built.

    Args:
        plot_defaults (Mapping[str, Any]): The recipe's ``PLOT_DEFAULTS``: an optional ``"panels"``
            list of internal names and settings per internal name (see
            :func:`el_paso.dataset.plot.check_panel_settings`).

    Raises:
        TypeError: If ``PLOT_DEFAULTS`` is not a dict, its panel list is not a non-empty list of names,
            or a setting is unknown or has the wrong type.
        ValueError: If a setting is out of range, a variable's `vmin` is not below its `vmax`, or settings
            are given for a variable that is not in the panel list (they would never be used, since
            ``--variable`` switches ``PLOT_DEFAULTS`` off).
    """
    if not isinstance(plot_defaults, dict):
        msg = f"PLOT_DEFAULTS must be a dict, got {type(plot_defaults).__name__}."
        raise TypeError(msg)

    panels = plot_defaults.get(PANELS_KEY)
    if panels is not None and (
        not isinstance(panels, (list, tuple))
        or not panels
        or not all(isinstance(name, str) and name for name in panels)
    ):
        msg = f"PLOT_DEFAULTS[{PANELS_KEY!r}] must be a non-empty list of internal variable names, got {panels!r}."
        raise TypeError(msg)

    settings = panel_settings(plot_defaults)
    try:
        from el_paso.dataset.plot import check_panel_settings  # noqa: PLC0415

        check_panel_settings(settings)
    except (TypeError, ValueError) as error:
        msg = f"Invalid PLOT_DEFAULTS: {error}"
        raise type(error)(msg) from error

    unused = [name for name in settings if name not in (panels or ())]
    if unused:
        import difflib  # noqa: PLC0415

        hints = [
            f"{name!r} (did you mean {matches[0]!r}?)"
            if (matches := difflib.get_close_matches(name, panels or (), n=1))
            else repr(name)
            for name in unused
        ]
        msg = (
            f"Invalid PLOT_DEFAULTS: settings for {', '.join(hints)}, which is not in "
            f"PLOT_DEFAULTS[{PANELS_KEY!r}] = {list(panels or ())!r}; they would never be used."
        )
        raise ValueError(msg)


def build_plot_command(
    recipe: Recipe,
    factory: Callable[..., SavingStrategy],
    *,
    defaults: dict[str, Any] | None = None,
    loop_over: Sequence[str] | None = None,
    plot_defaults: dict[str, Any] | None = None,
    processing_command: str | None = None,
    availability: AvailabilitySource | None = None,
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
        plot_defaults (dict[str, Any] | None): The recipe's ``PLOT_DEFAULTS``, checked with
            :func:`check_plot_defaults`.
        processing_command (str | None): The command that runs the recipe, e.g.
            ``el-paso rbsp mageis-electrons``, suggested when processed files are missing.
            Defaults to ``python -m <recipe module>``.
        availability (AvailabilitySource | None): The recipe's satellite availability table. With it,
            ``--satellite all`` only plots the satellites that have data in the time range, exactly as
            in the processing command, and ``--dry-run`` says whether each satellite has data.

    Returns:
        Callable[..., None]: A callable ready to be registered with ``app.command()``.

    Raises:
        TypeError: If the factory does not match the recipe, if a factory parameter has the
            same name as a shared or plot-only option, or if `plot_defaults` is malformed.
        ValueError: If a setting in `plot_defaults` is out of range.
    """
    defaults = dict(defaults or {})
    plot_defaults = dict(plot_defaults or {})
    check_strategy_factory(recipe, factory, loop_over=loop_over)
    check_plot_defaults(plot_defaults)

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
    with_availability = availability is not None and _has_availability_parameters(specs)

    def command(**kwargs: Any) -> None:  # noqa: ANN401
        shared = {name: kwargs.pop(name) for name in _SHARED_PARAMETER_NAMES}
        plot_options = {name: kwargs.pop(name) for name in _PLOT_PARAMETER_NAMES}

        expanded_all = _restore_parsed_values(kwargs, specs)
        start_time = kwargs.pop("start_time")
        end_time = kwargs.pop("end_time")
        if end_time <= start_time:
            msg = f"must be after --start-time ({start_time.isoformat()}), got {end_time.isoformat()}."
            raise typer.BadParameter(msg, param_hint="'--end-time'")

        coverage = None
        notices: list[tuple[int, str]] = []
        if with_availability:
            coverage, notices = _apply_availability(
                kwargs,
                load_availability_table(availability),  # ty: ignore[invalid-argument-type]
                start_time,
                end_time,
                expanded_all=expanded_all,
                label=_func_name(recipe),
            )

        plot_options = _parse_plot_options(plot_options)

        strategy_kwargs = _expand_loops(kwargs, loop_names)
        for call_kwargs in strategy_kwargs:
            # Fails loudly if the generated command ever drifts from the factory signature.
            inspect.signature(factory).bind(**call_kwargs)

        # The dry-run report is logged at INFO, and printing it is the whole point of --dry-run.
        quiet = shared["quiet"] and not plot_options["dry_run"]
        ep.setup_logging(_log_level(verbose=shared["verbose"], quiet=quiet))
        for level, message in notices:
            logger.log(level, message)

        _run_plot(
            PlotRequest(
                recipe=recipe,
                factory=factory,
                start_time=start_time,
                end_time=end_time,
                strategy_kwargs=strategy_kwargs,
                verbose=shared["verbose"] >= 1,
                plot_defaults=plot_defaults,
                processing_command=processing_command,
                availability=coverage,
                **plot_options,
            )
        )

    command.__name__ = f"plot_{_func_name(recipe)}"
    _attach_signature(
        command,
        [*_spec_parameters(specs), *_universal_parameters(_SHARED_PARAMETER_NAMES), *_plot_parameters()],
    )
    command.__doc__ = (
        f"Plot data processed by {_func_name(recipe)}.\n\n"
        "Takes the same arguments as the processing command; nothing is processed. Only space-weather\n"
        "indices and solar wind parameters (Kp, Dst, Vsw, ...) are downloaded if missing."
    )
    return command


def _parse_energy(text: str) -> float:
    """Parse an ``--energy`` value such as ``470``, ``100keV`` or ``1 MeV`` to keV; a bare number is in keV.

    Raises:
        typer.BadParameter: If the text is not a number with an optional energy unit.
    """
    match = re.fullmatch(r"\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([^\d\s].*?)?\s*", text)
    if match is None:
        msg = f"{text!r} is not an energy; use a number with an optional unit, e.g. 470 or 1MeV."
        raise typer.BadParameter(msg, param_hint="'--energy'")
    number, unit_text = match.groups()
    from astropy import units as u  # noqa: PLC0415

    try:
        unit = u.Unit(unit_text) if unit_text else u.keV
    except ValueError as error:
        msg = f"unknown unit {unit_text!r} in {text!r}."
        raise typer.BadParameter(msg, param_hint="'--energy'") from error
    if not unit.is_equivalent(u.keV):
        msg = f"{text!r} is not an energy (unit {unit_text!r})."
        raise typer.BadParameter(msg, param_hint="'--energy'")
    return float((float(number) * unit).to_value(u.keV))


def _parse_plot_options(plot_options: dict[str, Any]) -> dict[str, Any]:
    """Validate the plot options and resolve them into the corresponding `PlotRequest` fields.

    Checks ``--energy``, ``--pitch-angle``, ``--vmin``, ``--vmax`` and the ``--along-*`` options.

    Returns:
        dict[str, Any]: The options as `PlotRequest` fields: the energies in keV, the pitch angle in
            degrees and the coordinate of the ``--along-*`` option given (``None`` where not given),
            with the ``--along-*`` flags themselves removed.

    Raises:
        typer.BadParameter: If an energy is not positive, ``--vmin`` is not below ``--vmax``, or more
            than one ``--along-*`` option is given.
    """
    from astropy import units as u  # noqa: PLC0415

    energies = None
    if plot_options["energies"]:
        values = [_parse_energy(text) for text in plot_options["energies"]]
        if any(value <= 0 for value in values):
            msg = f"must be positive, got {', '.join(plot_options['energies'])}."
            raise typer.BadParameter(msg, param_hint="'--energy'")
        energies = u.Quantity(values, u.keV)

    vmin, vmax = plot_options["vmin"], plot_options["vmax"]
    if vmin is not None and vmax is not None and not vmin < vmax:
        msg = f"must be below --vmax ({vmax:g}), got {vmin:g}."
        raise typer.BadParameter(msg, param_hint="'--vmin'")

    given = [(flag, coordinate) for parameter, (flag, coordinate) in ALONG_OPTIONS.items() if plot_options[parameter]]
    if len(given) > 1:
        flags = [flag for flag, _ in given]
        msg = f"only one of {', '.join(flags)} can be given."
        raise typer.BadParameter(msg, param_hint=" / ".join(f"'{flag}'" for flag in flags))

    filename = plot_options["filename"]
    if filename is not None and Path(filename).name != filename:
        msg = f"must be a file name without a directory, got {filename!r}. Use --output-dir for the directory."
        raise typer.BadParameter(msg, param_hint="'--filename'")

    pitch_angle = None if plot_options["pitch_angle"] is None else plot_options["pitch_angle"] * u.deg
    return {
        **{name: value for name, value in plot_options.items() if name not in ALONG_OPTIONS},
        "variables": list(plot_options["variables"]) if plot_options["variables"] else None,
        "energies": energies,
        "pitch_angle": pitch_angle,
        "along": given[0][1] if given else None,
    }


def _format_quantity(value: u.Quantity) -> str:
    """A quantity for the settings table, e.g. ``470, 1000 keV``."""
    return f"{', '.join(f'{item:g}' for item in value.value.ravel())} {value.unit}"


def resolve_panels(request: PlotRequest) -> list[InternalName] | None:
    """The panel list: ``--variable`` if given, otherwise the recipe's default panels, otherwise ``None``."""
    panels = request.variables or request.plot_defaults.get(PANELS_KEY)
    return cast("list[InternalName]", list(panels)) if panels else None


def split_panels(panels: Sequence[str] | None) -> tuple[list[str], list[InternalName]]:
    """Split a panel list into space-weather indices (`INDEX_PANELS`) and dataset variables, keeping the order."""
    indices = [name for name in panels or [] if name in INDEX_PANELS]
    variables = [name for name in panels or [] if name not in INDEX_PANELS]
    return indices, cast("list[InternalName]", variables)


def _dataset_names(request: PlotRequest) -> list[InternalName] | None:
    """Dataset variables a request reads: the non-index panels and the ``--along-*`` coordinate.

    ``None`` if there are none, which `DataSet.get_file_coverage` takes as every output file.
    """
    names = split_panels(resolve_panels(request))[1]
    recipe_settings = _recipe_settings(request)
    coordinates = {request.along} if request.along else {recipe_settings.get(name, {}).get("along") for name in names}
    names.extend(sorted(c for c in coordinates if c and c not in names))
    return names or None


def _recipe_settings(request: PlotRequest) -> dict[str, Any]:
    """The recipe's per-variable settings: none if ``--variable`` is given, which ignores ``PLOT_DEFAULTS``."""
    return {} if request.variables else panel_settings(request.plot_defaults)


def _setting_overrides(request: PlotRequest) -> dict[str, Any]:
    """The settings given on the command line, which override those of every panel (layer 3)."""
    return {
        "energies": request.energies,
        "pitch_angle": request.pitch_angle,
        "vmin": request.vmin,
        "vmax": request.vmax,
    }


def _check_panel_settings(request: PlotRequest, panels: Sequence[str]) -> None:
    """Fail early if the command-line settings clash with the recipe's (e.g. ``--vmin`` above its vmax).

    Raises:
        typer.BadParameter: If the layered settings of a panel are invalid.
    """
    from el_paso.dataset.plot import resolve_panel_settings  # noqa: PLC0415

    try:
        resolve_panel_settings(panels, _recipe_settings(request), _setting_overrides(request))
    except (TypeError, ValueError) as error:
        msg = f"conflicts with the recipe's plot defaults: {error}"
        raise typer.BadParameter(msg, param_hint="'--vmin' / '--vmax'") from error


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
    if request.dry_run and request.availability is not None:
        for satellite in loop_values.get(AVAILABILITY_PARAMETER, []):
            table.add_row(f"availability {satellite}", _coverage_text(request.availability, satellite))

    panels = resolve_panels(request)
    if request.variables:
        table.add_row("variables", ", ".join(request.variables))
    elif panels:
        table.add_row("variables", f"{', '.join(panels)} (recipe defaults)")
    else:
        table.add_row("variables", "none given")

    # a setting not given on the command line comes from the recipe if it sets it for any panel
    recipe_settings = _recipe_settings(request)
    for key, value in _setting_overrides(request).items():
        if value is not None:
            text = _format_quantity(value) if hasattr(value, "unit") else f"{value:g}"
        elif any(key in recipe_settings.get(name, {}) for name in panels or []):
            text = "recipe defaults"
        else:
            text = "default"
        table.add_row(key.replace("_", " "), text)
    if request.along:
        along_text = request.along
    elif any("along" in recipe_settings.get(name, {}) for name in panels or []):
        along_text = "recipe defaults"
    else:
        along_text = "none (plot types by dimensions)"
    table.add_row("along", along_text)
    output_path = _output_path(request)
    if request.export_script:
        table.add_row("export script", str(request.export_script))
    if output_path is None:
        table.add_row("output", "none (shown only)")
    else:
        table.add_row("output dir", str(output_path.parent.resolve()))
        table.add_row("filename", output_path.name)
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
    names = _dataset_names(request)

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


def default_filename(request: PlotRequest) -> str:
    """File name of the figure when ``--filename`` is not given.

    The name is ``<recipe>_<satellite>_<start>_<end>.png``, where
    ``<recipe>`` is the recipe function's name without its ``process_`` prefix and ``<satellite>``
    joins all plotted satellites with ``-``. Recipes without a ``satellite`` parameter leave it out.

    Args:
        request (PlotRequest): The resolved plot request.

    Returns:
        str: The default file name.
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
    return "_".join(parts) + ".png"


def _output_path(request: PlotRequest) -> Path | None:
    """File the figure is written to, or ``None`` if it is only shown.

    The figure is written if ``--filename`` or ``--output-dir`` is given, or if ``--show`` is not.
    """
    if request.filename is None and request.output_dir is None and request.show:
        return None
    return (request.output_dir or Path()) / (request.filename or default_filename(request))


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
            "(no display available?). Use --output-dir or --filename to write the figure to a file instead, "
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
    coverage = dataset.get_file_coverage(_dataset_names(request))

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


def _run_plot(request: PlotRequest) -> None:
    """Plot the processed data a request describes.

    The figure is written to ``--output-dir``/``--filename``, or shown with ``--show``, or both. With
    neither, it is written to the current directory under :func:`default_filename`.

    Satellites with only some of the expected files are plotted with gaps; satellites without
    any are left out (see `_check_coverage`).

    With ``--export-script``, a script that makes the figure is written instead, and nothing is loaded.

    Raises:
        typer.BadParameter: If no variables are given and the recipe has no default panels, or if
            the command-line settings clash with the recipe's.
        typer.Exit: With code 1 if the expected files cannot be determined, if ``--show`` is
            given without an interactive backend, or if no satellite has any processed files.
    """
    _report_settings(request)
    panels = resolve_panels(request)
    indices, variables = split_panels(panels)
    if panels:
        _check_panel_settings(request, variables)

    if request.dry_run:
        if not _report_expected_files(request):
            raise typer.Exit(code=1)
        return

    if not panels:
        msg = f"is required: {_func_name(request.recipe)} defines no default panels."
        raise typer.BadParameter(msg, param_hint="'--variable'")

    if request.export_script:
        request.export_script.parent.mkdir(parents=True, exist_ok=True)
        request.export_script.write_text(write_plot_script(request), encoding="utf-8")
        logger.info(f"Wrote the plot script to {request.export_script}. Run it with python.")
        return

    if request.show and (error := _interactive_backend_error()):
        logger.error(error)
        raise typer.Exit(code=1)

    datasets = _load_datasets(request) if variables else []
    if variables and not datasets:
        raise typer.Exit(code=1)

    try:
        figure = build_figure(
            datasets,
            variables,
            request.start_time,
            request.end_time,
            indices=indices,
            show=request.show,
            panel_settings=_recipe_settings(request),
            **_setting_overrides(request),
            along=request.along,
            colormap=request.colormap,
        )
    except ValueError as error:
        logger.error(str(error))  # noqa: TRY400
        raise typer.Exit(code=1) from error

    output = _output_path(request)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output)
        logger.info(f"Saved the figure to {output}.")

    if request.show:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        plt.show()
