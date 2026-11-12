# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``el-paso plot`` command (`el_paso.cli.plot_cli`)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import patch

import astropy.units as u
import click
import el_paso as ep
import pytest
import typer
import typer.testing
from el_paso.cli import plot_cli, plot_figure
from el_paso.cli.app import RECIPES, RecipeEntry, app, load_strategy_factory
from el_paso.cli.plot_cli import PlotRequest, build_plot_command
from typer.testing import CliRunner, Result

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

pytestmark = pytest.mark.usefixtures("no_live_logging")

runner = CliRunner()

PLOTTABLE_RECIPES = tuple(entry for entry in RECIPES if entry.strategy is not None)

PROCESSING_ONLY_OPTIONS = (
    ["--num-cores", "4"],
    ["--raw-data-path", "raw"],
    ["--bin-cadence", "5min"],
    ["--skip-download"],
    ["--exit-after-download"],
    ["--skip-existing"],
    ["--add-B-obs"],
    ["--logs", "logs"],
)


def invoke_app(args: Sequence[str], **kwargs: Any) -> Result:  # noqa: ANN401
    """Invoke the real ``app`` (see `el_paso.cli.app`)."""
    with patch.object(typer.testing, "_get_command", side_effect=lambda instance: instance):
        return runner.invoke(app, args, env={"COLUMNS": "200"}, **kwargs)


@pytest.fixture(autouse=True)
def log_levels(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the log levels the plot command configures instead of configuring console logging.

    Console logging set up inside a command would outlive CliRunner's captured streams.
    """
    levels: list[int] = []
    monkeypatch.setattr(ep, "setup_logging", levels.append)
    return levels


@pytest.fixture
def requests(monkeypatch: pytest.MonkeyPatch) -> list[PlotRequest]:
    """Record the requests the plot command resolves instead of plotting them."""
    recorded: list[PlotRequest] = []
    monkeypatch.setattr(plot_cli, "_run_plot", recorded.append)
    return recorded


def stub_strategy(
    processed_data_path: str | Path,
    mag_field: Literal["T89", "TS04"],
    satellite: Literal["a", "b"],
) -> ep.SavingStrategy:
    """Saving strategy of the stub recipe.

    Args:
        processed_data_path (str | Path): Base directory of the processed data.
        mag_field (Literal["T89", "TS04"]): Magnetic field model the data was processed with.
        satellite (Literal["a", "b"]): Satellite.

    Returns:
        ep.SavingStrategy: Never; the stub is not called.
    """
    raise NotImplementedError


def stub_recipe(
    start_time: datetime,
    end_time: datetime,
    satellite: Literal["a", "b"] = "a",
    mag_field: Literal["T89", "TS04"] = "T89",
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = "processed",
    num_cores: int = 16,
) -> None:
    """Process stub data.

    Args:
        start_time (datetime): Start of the time range to process.
        end_time (datetime): End of the time range to process.
        satellite (Literal["a", "b"]): Satellite.
        mag_field (Literal["T89", "TS04"]): Magnetic field model.
        raw_data_path (str | Path): Raw data directory.
        processed_data_path (str | Path): Directory the processed files are written to.
        num_cores (int): Number of cores.
    """


def stub_app(**kwargs: Any) -> typer.Typer:  # noqa: ANN401
    """A single-command app exposing the plot command of the stub recipe."""
    built = typer.Typer(add_completion=False)
    built.command()(build_plot_command(stub_recipe, stub_strategy, **kwargs))
    return built


TIME_ARGS = ["--start-time", "2017-10-15", "--end-time", "2017-10-16"]


@pytest.mark.minimal
@pytest.mark.basic
def test_plot_command_takes_the_recipe_defaults(requests: list[PlotRequest]) -> None:
    """Without data options, the factory gets the values a processing run with the same arguments used."""
    result = runner.invoke(stub_app(defaults={"mag_field": "TS04"}), TIME_ARGS)
    assert result.exit_code == 0, result.output

    (request,) = requests
    assert request.recipe is stub_recipe
    assert request.factory is stub_strategy
    assert request.start_time == datetime(2017, 10, 15, tzinfo=timezone.utc)
    assert request.end_time == datetime(2017, 10, 16, tzinfo=timezone.utc)
    # the CLI defaults win over the recipe signature, which wins over the factory having no default
    assert request.strategy_kwargs == [
        {"processed_data_path": Path("processed"), "mag_field": "TS04", "satellite": "a"}
    ]
    assert request.variables is None
    assert request.output_dir is None
    assert request.filename is None
    assert not request.show
    assert not request.dry_run
    assert not request.verbose
    assert request.plot_defaults == {}
    assert request.energies is None
    assert request.pitch_angle is None
    assert request.vmin is None
    assert request.vmax is None


@pytest.mark.minimal
@pytest.mark.basic
def test_plot_command_passes_plot_options_through(requests: list[PlotRequest]) -> None:
    result = runner.invoke(
        stub_app(plot_defaults={"panels": ["FEDU"]}),
        [
            *TIME_ARGS,
            "--variable",
            "FEDU",
            "--variable",
            "L_star",
            "--output-dir",
            "figures",
            "--filename",
            "out.pdf",
            "--show",
            "--dry-run",
            "-v",
        ],
    )
    assert result.exit_code == 0, result.output

    (request,) = requests
    assert request.variables == ["FEDU", "L_star"]
    assert request.output_dir == Path("figures")
    assert request.filename == "out.pdf"
    assert request.show
    assert request.dry_run
    assert request.verbose
    assert request.plot_defaults == {"panels": ["FEDU"]}


@pytest.mark.minimal
@pytest.mark.basic
def test_plot_command_passes_selection_options_through(requests: list[PlotRequest]) -> None:
    from astropy import units as u  # noqa: PLC0415

    result = runner.invoke(
        stub_app(),
        [*TIME_ARGS, "--energy", "470", "--energy", "1000", "--pitch-angle", "45", "--vmin", "1", "--vmax", "1e6"],
    )
    assert result.exit_code == 0, result.output

    (request,) = requests
    assert request.energies is not None
    assert request.energies.unit == u.keV
    assert list(request.energies.value) == [470.0, 1000.0]
    assert request.pitch_angle == 45 * u.deg
    assert (request.vmin, request.vmax) == (1.0, 1e6)


@pytest.mark.minimal
@pytest.mark.basic
@pytest.mark.parametrize(
    ("args", "option"),
    [
        (["--energy", "0"], "--energy"),
        (["--energy", "abc"], "--energy"),
        (["--energy", "100deg"], "--energy"),
        (["--energy", "100foo"], "--energy"),
        (["--energy", "470", "--energy", "-1"], "--energy"),
        (["--pitch-angle", "181"], "--pitch-angle"),
        (["--pitch-angle", "-1"], "--pitch-angle"),
        (["--vmin", "10", "--vmax", "10"], "--vmin"),
        (["--vmin", "100", "--vmax", "10"], "--vmin"),
    ],
)
def test_plot_command_rejects_invalid_selection_options(
    requests: list[PlotRequest], args: list[str], option: str
) -> None:
    result = runner.invoke(stub_app(), [*TIME_ARGS, *args])
    assert result.exit_code == 2, result.output
    assert option in click.unstyle(result.output)
    assert not requests


@pytest.mark.minimal
@pytest.mark.basic
@pytest.mark.parametrize(
    ("flag", "coordinate"),
    [("--along-L-star", "L_star"), ("--along-L-m", "L_m"), ("--along-R-eq", "R_Eq"), (None, None)],
)
def test_plot_command_passes_the_along_option_through(
    requests: list[PlotRequest], flag: str | None, coordinate: str | None
) -> None:
    result = runner.invoke(stub_app(), [*TIME_ARGS, *([flag] if flag else [])])
    assert result.exit_code == 0, result.output

    (request,) = requests
    assert request.along == coordinate


@pytest.mark.minimal
@pytest.mark.basic
def test_along_options_are_mutually_exclusive(requests: list[PlotRequest]) -> None:
    result = runner.invoke(stub_app(), [*TIME_ARGS, "--along-L-star", "--along-R-eq"])

    assert result.exit_code == 2, result.output
    output = click.unstyle(result.output)
    assert "only one of" in output
    assert "'--along-L-star' / '--along-R-eq'" in output
    assert not requests


@pytest.mark.minimal
@pytest.mark.basic
def test_along_options_match_the_along_coordinates() -> None:
    from el_paso.dataset.plot import ALONG_COORDINATES  # noqa: PLC0415

    assert tuple(coordinate for _, coordinate in plot_cli.ALONG_OPTIONS.values()) == ALONG_COORDINATES


@pytest.mark.minimal
@pytest.mark.basic
@pytest.mark.parametrize(
    ("satellite_args", "expected"),
    [
        (["--satellite", "b"], ["b"]),
        (["--satellite", "a", "--satellite", "b"], ["a", "b"]),
        (["--satellite", "all"], ["a", "b"]),
    ],
)
def test_plot_command_calls_the_factory_once_per_satellite(
    requests: list[PlotRequest], satellite_args: list[str], expected: list[str]
) -> None:
    result = runner.invoke(stub_app(), [*TIME_ARGS, "--mag-field", "TS04", *satellite_args])
    assert result.exit_code == 0, result.output

    (request,) = requests
    assert [kwargs["satellite"] for kwargs in request.strategy_kwargs] == expected
    assert all(kwargs["mag_field"] == "TS04" for kwargs in request.strategy_kwargs)


@pytest.mark.minimal
@pytest.mark.basic
def test_plot_command_requires_the_time_range(requests: list[PlotRequest]) -> None:
    result = runner.invoke(stub_app(), [])
    assert result.exit_code != 0
    assert "--start-time" in click.unstyle(result.output)
    assert not requests


@pytest.mark.minimal
@pytest.mark.basic
@pytest.mark.parametrize("option", [["--num-cores", "4"], ["--raw-data-path", "raw"]], ids=lambda o: o[0])
def test_plot_command_rejects_recipe_only_parameters(requests: list[PlotRequest], option: list[str]) -> None:
    result = runner.invoke(stub_app(), [*TIME_ARGS, *option])
    assert result.exit_code == 2
    assert "No such option" in click.unstyle(result.output)
    assert not requests


@pytest.mark.minimal
@pytest.mark.basic
def test_plot_command_rejects_mismatching_factory() -> None:
    def other_strategy(processed_data_path: str | Path, mag_field: Literal["T89"]) -> ep.SavingStrategy:
        raise NotImplementedError

    with pytest.raises(TypeError, match="mag_field"):
        build_plot_command(stub_recipe, other_strategy)


@pytest.mark.minimal
@pytest.mark.basic
def test_plot_command_rejects_factory_parameters_named_like_plot_options() -> None:
    def recipe(start_time: datetime, end_time: datetime, filename: str = "x") -> None:
        """Process.

        Args:
            start_time (datetime): Start.
            end_time (datetime): End.
            filename (str): File name.
        """

    def strategy(filename: str) -> ep.SavingStrategy:
        raise NotImplementedError

    with pytest.raises(TypeError, match="filename"):
        build_plot_command(recipe, strategy)


@pytest.mark.minimal
@pytest.mark.basic
def test_plot_group_lists_only_recipes_with_a_strategy_factory() -> None:
    result = invoke_app(["plot", "--help"])
    assert result.exit_code == 0, result.output
    output = click.unstyle(result.output)

    plottable_missions = {entry.mission for entry in PLOTTABLE_RECIPES}
    for mission in {entry.mission for entry in RECIPES}:
        assert (f" {mission} " in output) == (mission in plottable_missions), mission

    for mission in plottable_missions:
        mission_result = invoke_app(["plot", mission, "--help"])
        assert mission_result.exit_code == 0, mission_result.output
        mission_output = click.unstyle(mission_result.output)
        for entry in RECIPES:
            if entry.mission == mission:
                assert (f" {entry.command} " in mission_output) == (entry.strategy is not None), entry.command


@pytest.mark.minimal
@pytest.mark.basic
@pytest.mark.parametrize("entry", PLOTTABLE_RECIPES, ids=lambda e: f"{e.mission}-{e.command}")
def test_every_recipe_declares_valid_plot_defaults(entry: RecipeEntry) -> None:
    from el_paso.cli.app import load_plot_defaults  # noqa: PLC0415

    plot_defaults = load_plot_defaults(entry)

    plot_cli.check_plot_defaults(plot_defaults)


@pytest.mark.minimal
@pytest.mark.basic
@pytest.mark.parametrize("entry", RECIPES, ids=lambda e: f"{e.mission}-{e.command}")
def test_every_recipe_is_plottable_and_has_plot_defaults(entry: RecipeEntry) -> None:
    from el_paso.cli.app import load_plot_defaults  # noqa: PLC0415

    assert entry.strategy is not None, f"{entry.mission} {entry.command} registers no strategy factory"
    assert load_plot_defaults(entry), f"{entry.mission} {entry.command} defines no PLOT_DEFAULTS"


@pytest.mark.minimal
@pytest.mark.basic
@pytest.mark.parametrize("entry", PLOTTABLE_RECIPES, ids=lambda e: f"{e.mission}-{e.command}")
def test_every_plottable_recipe_builds_a_plot_command(entry: RecipeEntry, requests: list[PlotRequest]) -> None:
    help_result = invoke_app(["plot", entry.mission, entry.command, "--help"])
    assert help_result.exit_code == 0, help_result.output
    help_text = click.unstyle(help_result.output)
    for option in (
        "--start-time",
        "--end-time",
        "--dry-run",
        "--verbose",
        "--quiet",
        "--variable",
        "--energy",
        "--pitch-angle",
        "--vmin",
        "--vmax",
        "--output-dir",
        "--filename",
    ):
        assert option in help_text

    result = invoke_app(["plot", entry.mission, entry.command, *TIME_ARGS])
    assert result.exit_code == 0, result.output
    assert len(requests) == 1


@pytest.mark.minimal
@pytest.mark.basic
def test_mageis_plot_command_exposes_the_factory_options() -> None:
    result = invoke_app(["plot", "rbsp", "mageis-electrons", "--help"])
    assert result.exit_code == 0, result.output
    help_text = click.unstyle(result.output)

    for option in ("--satellite", "--mag-field", "--processed-data-path", "--save-strategy", "--show"):
        assert option in help_text
    for option, *_ in PROCESSING_ONLY_OPTIONS:
        assert option not in help_text


@pytest.mark.minimal
@pytest.mark.basic
@pytest.mark.parametrize("option", PROCESSING_ONLY_OPTIONS, ids=lambda o: o[0])
def test_mageis_plot_command_rejects_processing_only_options(requests: list[PlotRequest], option: list[str]) -> None:
    result = invoke_app(["plot", "rbsp", "mageis-electrons", *TIME_ARGS, *option])
    assert result.exit_code == 2, result.output
    assert "No such option" in click.unstyle(result.output)
    assert not requests


@pytest.fixture
def plot_log(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Capture the plot command's log at INFO level."""
    caplog.set_level(logging.INFO, logger="el_paso.cli")
    return caplog


def mageis_dry_run(tmp_path: Path, *args: str) -> Result:
    """Run ``el-paso plot rbsp mageis-electrons --dry-run`` for October and November 2017."""
    return invoke_app(
        [
            "plot",
            "rbsp",
            "mageis-electrons",
            "--start-time",
            "2017-10-15",
            "--end-time",
            "2017-11-02",
            "--mag-field",
            "TS04",
            "--processed-data-path",
            str(tmp_path),
            "--dry-run",
            *args,
        ]
    )


def mageis_file_path(tmp_path: Path, satellite: str, interval_start: datetime) -> Path:
    """Path at which the MagEIS strategy expects the file of the interval starting at `interval_start`."""
    entry = next(e for e in PLOTTABLE_RECIPES if (e.mission, e.command) == ("rbsp", "mageis-electrons"))
    factory = load_strategy_factory(entry)
    assert factory is not None
    strategy = factory(processed_data_path=tmp_path, mag_field="TS04", satellite=satellite, save_strategy="netcdf")
    dataset = ep.DataSet(strategy, interval_start, interval_start + timedelta(days=1), verbose=False)
    (coverage,) = dataset.get_file_coverage()
    return coverage.path


@pytest.mark.minimal
@pytest.mark.basic
def test_dry_run_reports_settings_and_expected_files(tmp_path: Path, plot_log: pytest.LogCaptureFixture) -> None:
    existing = mageis_file_path(tmp_path, "a", datetime(2017, 10, 1, tzinfo=timezone.utc))
    existing.parent.mkdir(parents=True)
    existing.touch()

    result = mageis_dry_run(tmp_path, "--satellite", "all")
    assert result.exit_code == 0, result.output

    text = "\n".join(record.getMessage() for record in plot_log.records)
    assert "Would plot" in text
    assert "2017-10-15" in text
    assert "(recipe defaults)" in text  # MagEIS defines default panels
    assert "energies" in text
    # labelled by the looped parameter only; the settings table shows the rest
    assert "Saving strategy (satellite=a): MonthlyRBStrategy(" in text
    assert "Saving strategy (satellite=b): MonthlyRBStrategy(" in text
    assert f"[found] {existing}" in text
    assert text.count("[missing]") == 3
    assert "1 of 2 expected files exist." in text
    assert "0 of 2 expected files exist." in text


@pytest.mark.minimal
@pytest.mark.basic
def test_dry_run_is_not_silenced_by_quiet(tmp_path: Path, log_levels: list[int]) -> None:
    result = mageis_dry_run(tmp_path, "--quiet")
    assert result.exit_code == 0, result.output
    assert log_levels == [logging.INFO]


@pytest.mark.minimal
@pytest.mark.basic
def test_dry_run_fails_for_variables_outside_the_strategy(tmp_path: Path, plot_log: pytest.LogCaptureFixture) -> None:
    result = mageis_dry_run(tmp_path, "--variable", "not_a_variable")

    assert result.exit_code == 1, result.output
    assert "not_a_variable is not part of the chosen saving strategy" in plot_log.text


PLOT_START = datetime(2017, 10, 15, tzinfo=timezone.utc)
PLOT_END = datetime(2017, 10, 16, tzinfo=timezone.utc)


def save_mageis_shaped_data(tmp_path: Path, satellite: str) -> None:
    """Write one day of small synthetic data with the MagEIS strategy, as a processing run would."""
    import numpy as np  # noqa: PLC0415
    from astropy import units as u  # noqa: PLC0415
    from el_paso.dataset.utils import python2matlab  # noqa: PLC0415

    time_size = 48
    datetimes = [PLOT_START + i * timedelta(minutes=30) for i in range(time_size)]
    flux_unit = (u.cm**2 * u.s * u.sr * u.keV) ** (-1)
    variables = {
        "Epoch": ep.Variable(original_unit=ep.units.datenum, data=np.array([python2matlab(t) for t in datetimes])),
        "FEDU": ep.Variable(original_unit=flux_unit, data=np.random.default_rng(0).uniform(1, 100, (time_size, 3, 4))),
        "Energy_FEDU": ep.Variable(original_unit=u.keV, data=np.tile([100.0, 400.0, 1600.0], (time_size, 1))),
        "Alpha": ep.Variable(original_unit=u.deg, data=np.tile([10.0, 30.0, 60.0, 90.0], (time_size, 1))),
        "B_Eq": ep.Variable(original_unit=u.nT, data=np.linspace(40.0, 60.0, time_size)),
        "L_star": ep.Variable(
            original_unit=u.dimensionless_unscaled, data=np.tile(np.linspace(2.0, 6.0, time_size)[:, None], (1, 4))
        ),
        "MLT": ep.Variable(original_unit=u.hour, data=np.linspace(0.0, 23.5, time_size)),
    }

    entry = next(e for e in PLOTTABLE_RECIPES if (e.mission, e.command) == ("rbsp", "mageis-electrons"))
    factory = load_strategy_factory(entry)
    assert factory is not None
    strategy = factory(processed_data_path=tmp_path, mag_field="TS04", satellite=satellite, save_strategy="netcdf")
    ep.save(variables, strategy, start_time=PLOT_START, end_time=PLOT_END, time_var=variables["Epoch"])  # ty:ignore[invalid-argument-type]


def output_args(path: Path) -> list[str]:
    """The ``--output-dir`` and ``--filename`` options that write a figure to `path`."""
    return ["--output-dir", str(path.parent), "--filename", path.name]


def mageis_plot(tmp_path: Path, *args: str, end_time: datetime = PLOT_END) -> Result:
    """Run ``el-paso plot rbsp mageis-electrons`` from the day of `save_mageis_shaped_data` to `end_time`."""
    return invoke_app(
        [
            "plot",
            "rbsp",
            "mageis-electrons",
            "--start-time",
            PLOT_START.isoformat(),
            "--end-time",
            end_time.isoformat(),
            "--mag-field",
            "TS04",
            "--processed-data-path",
            str(tmp_path),
            *args,
        ]
    )


@pytest.mark.minimal
@pytest.mark.basic
def test_partial_coverage_plots_what_exists_and_names_the_gap(
    tmp_path: Path, plot_log: pytest.LogCaptureFixture
) -> None:
    save_mageis_shaped_data(tmp_path, "a")  # October only
    output = tmp_path / "out.png"

    result = mageis_plot(
        tmp_path,
        "--satellite",
        "a",
        "--variable",
        "B_Eq",
        *output_args(output),
        end_time=datetime(2017, 11, 2, tzinfo=timezone.utc),
    )

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert "No processed files for rbspa mageis (TS04) in 2017-11; plotting 2017-10 only." in plot_log.text


@pytest.mark.minimal
@pytest.mark.basic
def test_no_coverage_fails_with_the_expected_path_and_processing_command(
    tmp_path: Path, plot_log: pytest.LogCaptureFixture
) -> None:
    output = tmp_path / "out.png"

    result = mageis_plot(tmp_path, "--satellite", "a", "--variable", "B_Eq", *output_args(output))

    assert result.exit_code == 1, result.output
    assert not output.exists()
    expected_path = mageis_file_path(tmp_path, "a", PLOT_START)
    assert (
        "No processed files for rbspa mageis (TS04) between 2017-10-15 and 2017-10-16.\n"
        f"Expected: {expected_path}\n"
        "To create them: el-paso rbsp mageis-electrons --start-time 2017-10-15 --end-time 2017-10-16 "
        f"--processed-data-path {tmp_path} --mag-field TS04 --satellite a --save-strategy netcdf"
    ) in plot_log.text


@pytest.mark.minimal
@pytest.mark.basic
def test_satellite_without_files_is_left_out(tmp_path: Path, plot_log: pytest.LogCaptureFixture) -> None:
    save_mageis_shaped_data(tmp_path, "a")
    output = tmp_path / "out.png"

    with patch.object(plot_cli, "build_figure", wraps=plot_cli.build_figure) as build_figure:
        result = mageis_plot(tmp_path, "--satellite", "all", "--variable", "B_Eq", *output_args(output))

    # the exit code for a missing satellite is settled in Phase 5, so it is not checked here
    assert output.exists(), result.output
    (datasets, *_), _ = build_figure.call_args
    assert [dataset.get_print_name() for dataset in datasets] == ["rbspa mageis"]
    assert "No processed files for rbspb mageis (TS04)" in plot_log.text
    assert "--satellite b" in plot_log.text


@pytest.mark.minimal
@pytest.mark.basic
@pytest.mark.parametrize(
    ("intervals", "expected"),
    [
        ([("2017-10-01T00:00:00", "2017-10-31T23:59:59")], "2017-10"),
        (
            [
                ("2017-10-01T00:00:00", "2017-10-31T23:59:59"),
                ("2017-11-01T00:00:00", "2017-11-30T23:59:59"),
                ("2017-12-01T00:00:00", "2017-12-31T23:59:59"),
                ("2018-02-01T00:00:00", "2018-02-28T23:59:59"),
            ],
            "2017-10 to 2017-12, 2018-02",
        ),
        ([("2017-10-15T00:00:00", "2017-10-16T00:00:00")], "2017-10-15"),
        ([("2017-10-15T06:00:00", "2017-10-15T12:00:00")], "2017-10-15T06:00:00 to 2017-10-15T12:00:00"),
    ],
    ids=["month", "merged-months", "day", "irregular"],
)
def test_format_intervals(intervals: list[tuple[str, str]], expected: str) -> None:
    parsed = [
        (
            datetime.fromisoformat(start).replace(tzinfo=timezone.utc),
            datetime.fromisoformat(end).replace(tzinfo=timezone.utc),
        )
        for start, end in intervals
    ]
    assert plot_cli._format_intervals(parsed) == expected


@pytest.mark.minimal
@pytest.mark.basic
def test_processing_command_line_spells_out_every_argument() -> None:
    request = PlotRequest(
        recipe=stub_recipe,
        factory=stub_strategy,
        start_time=PLOT_START,
        end_time=PLOT_START + timedelta(hours=12),
        strategy_kwargs=[],
        variables=None,
        output_dir=None,
        filename=None,
        show=False,
        dry_run=False,
        verbose=False,
        plot_defaults={},
    )
    kwargs = {"processed_data_path": Path("my data"), "add_b": False, "fit": True, "energies": [1.0, 2.0]}

    assert plot_cli.processing_command_line(request, kwargs) == (
        f"python -m {__name__} --start-time 2017-10-15 --end-time 2017-10-15T12:00:00 "
        "--processed-data-path 'my data' --no-add-b --fit --energies 1.0 --energies 2.0"
    )


@pytest.fixture
def mageis_data(tmp_path: Path) -> Path:
    """Synthetic MagEIS output for both satellites; returns the processed-data path."""
    save_mageis_shaped_data(tmp_path, "a")
    save_mageis_shaped_data(tmp_path, "b")
    return tmp_path


@pytest.fixture
def pyplot_figures() -> Iterator[None]:
    """Close every pyplot figure a test leaves open (``--show`` creates them through pyplot)."""
    yield
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.close("all")


@pytest.mark.minimal
@pytest.mark.basic
def test_plot_writes_the_figure_to_output(mageis_data: Path, plot_log: pytest.LogCaptureFixture) -> None:
    output = mageis_data / "quicklook.png"
    result = mageis_plot(
        mageis_data, "--satellite", "all", "--variable", "FEDU", "--variable", "B_Eq", *output_args(output)
    )

    assert result.exit_code == 0, result.output
    assert output.read_bytes().startswith(b"\x89PNG")
    assert f"Saved the figure to {output}." in plot_log.text


@pytest.mark.minimal
@pytest.mark.basic
def test_plot_applies_pitch_angle_and_colour_limits(mageis_data: Path) -> None:
    from astropy import units as u  # noqa: PLC0415

    datasets = [mageis_dataset(mageis_data, "a")]
    figure = plot_cli.build_figure(
        datasets, ["FEDU", "B_Eq"], PLOT_START, PLOT_END, pitch_angle=30 * u.deg, vmin=2.0, vmax=50.0
    )

    spectrogram, line_panel = (ax for ax in figure.axes if ax.get_label() != "<colorbar>")
    (mesh,) = spectrogram.collections
    assert (mesh.norm.vmin, mesh.norm.vmax) == (2.0, 50.0)
    assert [text.get_text() for text in spectrogram.texts] == ["pitch angle 30°"]
    assert not line_panel.collections  # the colour limits do not touch the line panel


@pytest.mark.minimal
@pytest.mark.basic
def test_plot_with_selection_options_writes_the_figure(mageis_data: Path, plot_log: pytest.LogCaptureFixture) -> None:
    output = mageis_data / "quicklook.png"
    result = mageis_plot(
        mageis_data,
        "--satellite",
        "a",
        "--variable",
        "FEDU",
        "--pitch-angle",
        "60",
        "--vmin",
        "5",
        "--vmax",
        "80",
        *output_args(output),
    )

    assert result.exit_code == 0, result.output
    assert output.read_bytes().startswith(b"\x89PNG")
    assert "Could not plot" not in plot_log.text


@pytest.fixture
def mageis_plot_defaults(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Give the registered MagEIS recipe these ``PLOT_DEFAULTS``; fill the dict in the test."""
    from el_paso.cli import app as app_module  # noqa: PLC0415

    plot_defaults: dict[str, Any] = {}
    monkeypatch.setattr(app_module, "load_plot_defaults", lambda entry: plot_defaults)
    return plot_defaults


@pytest.mark.minimal
@pytest.mark.basic
def test_plot_defaults_make_variable_optional(
    mageis_data: Path, mageis_plot_defaults: dict[str, Any], plot_log: pytest.LogCaptureFixture
) -> None:
    mageis_plot_defaults["panels"] = ["FEDU", "B_Eq"]
    output = mageis_data / "defaults.png"

    result = mageis_plot(mageis_data, "--satellite", "a", *output_args(output))

    assert result.exit_code == 0, result.output
    assert output.read_bytes().startswith(b"\x89PNG")
    assert "FEDU, B_Eq (recipe defaults)" in plot_log.text


@pytest.mark.minimal
@pytest.mark.basic
def test_mageis_plot_defaults_draw_fedu_along_l_star(
    mageis_data: Path, monkeypatch: pytest.MonkeyPatch, plot_log: pytest.LogCaptureFixture
) -> None:
    figures: list[Any] = []
    original = plot_cli.build_figure
    monkeypatch.setattr(plot_cli, "build_figure", lambda *a, **k: figures.append(original(*a, **k)) or figures[-1])
    output = mageis_data / "defaults.png"
    # the synthetic data has no Kp, which would be downloaded
    monkeypatch.setattr(plot_figure, "_plot_index", lambda *a, **k: None)

    result = mageis_plot(mageis_data, "--satellite", "a", *output_args(output))

    assert result.exit_code == 0, result.output
    assert output.read_bytes().startswith(b"\x89PNG")
    assert "along" in plot_log.text
    (figure,) = figures
    panels = [ax for ax in figure.axes if ax.get_label() != "<colorbar>"]
    # Kp, then one scatter panel per default energy with L_star on the y axis
    assert len(panels) == 4  # noqa: PLR2004
    assert all("L_star" in ax.get_ylabel() for ax in panels[1:])
    # the synthetic channels (100, 400, 1600 keV) are within 10% of 108 and 1575 keV only; labels show the matches
    assert [len(ax.collections) for ax in panels[1:]] == [1, 0, 1]
    assert [ax.texts[0].get_text().split(",")[0] for ax in panels[1:]][::2] == ["0.1 MeV", "1.6 MeV"]


@pytest.mark.minimal
@pytest.mark.basic
def test_variable_replaces_the_default_panels(
    mageis_data: Path, mageis_plot_defaults: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    mageis_plot_defaults["panels"] = ["FEDU", "B_Eq"]
    drawn: list[list[str]] = []
    original = plot_cli.build_figure

    def build_figure(datasets: Any, variables: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        drawn.append(variables)
        return original(datasets, variables, *args, **kwargs)

    monkeypatch.setattr(plot_cli, "build_figure", build_figure)

    result = mageis_plot(mageis_data, "--satellite", "a", "--variable", "B_Eq", *output_args(mageis_data / "out.png"))

    assert result.exit_code == 0, result.output
    assert drawn == [["B_Eq"]]


@pytest.mark.minimal
@pytest.mark.basic
def test_cli_flags_override_plot_defaults_which_override_type_defaults(
    mageis_data: Path, mageis_plot_defaults: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from astropy import units as u  # noqa: PLC0415

    mageis_plot_defaults.update({"panels": ["FEDU"], "FEDU": {"pitch_angle": 30 * u.deg, "vmin": 2.0, "vmax": 50.0}})
    figures: list[Any] = []
    original = plot_cli.build_figure
    monkeypatch.setattr(plot_cli, "build_figure", lambda *a, **k: figures.append(original(*a, **k)) or figures[-1])

    result = mageis_plot(mageis_data, "--satellite", "a", "--vmax", "80", *output_args(mageis_data / "out.png"))

    assert result.exit_code == 0, result.output
    (figure,) = figures
    (mesh,) = figure.axes[0].collections
    # vmin and the pitch angle from PLOT_DEFAULTS, vmax from the command line
    assert (mesh.norm.vmin, mesh.norm.vmax) == (2.0, 80.0)
    assert [text.get_text() for text in figure.axes[0].texts] == ["pitch angle 30°"]


@pytest.mark.minimal
@pytest.mark.basic
def test_settings_report_names_recipe_defaults(
    tmp_path: Path, mageis_plot_defaults: dict[str, Any], plot_log: pytest.LogCaptureFixture
) -> None:
    mageis_plot_defaults.update({"panels": ["FEDU"], "FEDU": {"vmin": 2.0}})

    result = mageis_dry_run(tmp_path, "--vmax", "80")

    assert result.exit_code == 0, result.output
    rows = {
        cells[1].strip(): cells[2].strip()
        for line in plot_log.text.splitlines()
        if len(cells := line.split("│")) == 4  # noqa: PLR2004
    }
    assert rows["variables"] == "FEDU (recipe defaults)"
    assert rows["vmin"] == "recipe defaults"
    assert rows["vmax"] == "80"
    assert rows["pitch angle"] == "default"


@pytest.mark.minimal
@pytest.mark.basic
def test_cli_flag_conflicting_with_plot_defaults_is_rejected(
    mageis_data: Path, mageis_plot_defaults: dict[str, Any]
) -> None:
    mageis_plot_defaults.update({"panels": ["FEDU"], "FEDU": {"vmax": 50.0}})

    result = mageis_plot(mageis_data, "--satellite", "a", "--vmin", "60")

    assert result.exit_code == 2, result.output
    output = click.unstyle(result.output)
    assert "conflicts with the recipe's plot defaults" in output
    assert "FEDU: vmin (60) must be below vmax (50)" in output


@pytest.mark.minimal
@pytest.mark.basic
@pytest.mark.parametrize(
    ("plot_defaults", "error", "match"),
    [
        ({"panels": "FEDU"}, TypeError, "non-empty list"),
        ({"panels": []}, TypeError, "non-empty list"),
        ({"FEDU": {"colour": 1}}, TypeError, "FEDU: unknown setting"),
        ({"FEDU": 1.0}, TypeError, "FEDU: settings must be a mapping"),
        ({"FEDU": {"pitch_angle": 90}}, TypeError, "pitch_angle needs a unit"),
        ({"FEDU": {"vmin": 10.0, "vmax": 1.0}}, ValueError, "FEDU: vmin"),
        ({"panels": ["L_star"], "Lstar": {"vmin": 1.0}}, ValueError, "'Lstar' \\(did you mean 'L_star'\\?\\)"),
        ({"FEDU": {"vmin": 1.0}}, ValueError, "settings for 'FEDU', which is not in"),
    ],
)
def test_malformed_plot_defaults_fail_when_the_command_is_built(
    plot_defaults: dict[str, Any], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        build_plot_command(stub_recipe, stub_strategy, plot_defaults=plot_defaults)  # ty:ignore[invalid-argument-type]
    with pytest.raises(error, match=match):
        ep.run_recipe_cli(
            stub_recipe,  # ty:ignore[invalid-argument-type]
            strategy=stub_strategy,
            plot_defaults=plot_defaults,
            args=["--help"],
        )


@pytest.mark.minimal
@pytest.mark.basic
def test_plot_writes_the_default_file_name(mageis_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    result = mageis_plot(mageis_data, "--satellite", "a", "--variable", "B_Eq")

    assert result.exit_code == 0, result.output
    assert [path.name for path in workdir.iterdir()] == ["rbsp_mageis_electrons_a_20171015_20171016.png"]


@pytest.mark.minimal
@pytest.mark.basic
def test_plot_requires_variables_without_defaults(mageis_data: Path, mageis_plot_defaults: dict[str, Any]) -> None:
    assert not mageis_plot_defaults  # MagEIS stripped of its PLOT_DEFAULTS
    result = mageis_plot(mageis_data, "--satellite", "a")

    assert result.exit_code == 2, result.output
    assert "--variable" in click.unstyle(result.output)


@pytest.mark.minimal
@pytest.mark.basic
@pytest.mark.usefixtures("pyplot_figures")
def test_show_fails_clearly_without_a_display(
    mageis_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plot_log: pytest.LogCaptureFixture
) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "out.png"

    result = mageis_plot(mageis_data, "--satellite", "a", "--variable", "B_Eq", "--show", *output_args(output))

    assert result.exit_code == 1, result.output
    assert "--show needs an interactive matplotlib backend, but 'Agg' is in use" in plot_log.text
    assert not output.exists()  # fails before loading or drawing anything


@pytest.mark.minimal
@pytest.mark.basic
@pytest.mark.usefixtures("pyplot_figures")
@pytest.mark.parametrize("with_output", [False, True], ids=["show-only", "show-and-output"])
def test_show_opens_a_window_and_writes_only_a_requested_file(
    mageis_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_output: bool
) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(plot_cli, "_interactive_backend_error", lambda: None)
    shown: list[list[int]] = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.get_fignums()))

    file_args = ["--filename", "requested.png"] if with_output else []
    result = mageis_plot(mageis_data, "--satellite", "a", "--variable", "B_Eq", "--show", *file_args)

    assert result.exit_code == 0, result.output
    assert len(shown) == 1
    assert len(shown[0]) == 1  # the figure is managed by pyplot, so show() displays it
    assert [path.name for path in workdir.iterdir()] == (["requested.png"] if with_output else [])


def mageis_dataset(processed_data_path: Path, satellite: str) -> ep.DataSet:
    """The MagEIS `DataSet` of one satellite for the day of `save_mageis_shaped_data`."""
    entry = next(e for e in PLOTTABLE_RECIPES if (e.mission, e.command) == ("rbsp", "mageis-electrons"))
    factory = load_strategy_factory(entry)
    assert factory is not None
    strategy = factory(
        processed_data_path=processed_data_path, mag_field="TS04", satellite=satellite, save_strategy="netcdf"
    )
    return ep.DataSet(strategy, PLOT_START, PLOT_END, verbose=False)


@pytest.mark.minimal
@pytest.mark.basic
def test_build_figure_stacks_one_block_per_satellite(mageis_data: Path) -> None:
    import matplotlib.dates as mdates  # noqa: PLC0415

    datasets = [mageis_dataset(mageis_data, satellite) for satellite in ("a", "b")]

    figure = plot_cli.build_figure(datasets, ["FEDU", "B_Eq"], PLOT_START, PLOT_END)

    # four panels plus the spectrograms' colour bars, which are not part of the shared column
    panels = [ax for ax in figure.axes if ax.get_label() != "<colorbar>"]
    assert len(panels) == 4
    assert [ax.get_title() for ax in panels] == ["rbspa mageis", "", "rbspb mageis", ""]
    assert ["Energy_FEDU" in panels[0].get_ylabel(), "B_Eq" in panels[1].get_ylabel()] == [True, True]
    assert ["Energy_FEDU" in panels[2].get_ylabel(), "B_Eq" in panels[3].get_ylabel()] == [True, True]

    # one time axis: shared by every panel, labelled only at the bottom, spanning the requested range
    assert all(ax.get_shared_x_axes().joined(ax, panels[-1]) for ax in panels)
    assert [ax.xaxis.get_tick_params()["labelbottom"] for ax in panels] == [False, False, False, True]
    assert panels[-1].get_xlim() == pytest.approx((mdates.date2num(PLOT_START), mdates.date2num(PLOT_END)))


@pytest.mark.minimal
@pytest.mark.basic
def test_build_figure_along_l_star_has_one_panel_per_energy_and_satellite(mageis_data: Path) -> None:
    datasets = [mageis_dataset(mageis_data, satellite) for satellite in ("a", "b")]

    figure = plot_cli.build_figure(datasets, ["FEDU", "B_Eq"], PLOT_START, PLOT_END, along="L_star")

    panels = [ax for ax in figure.axes if ax.get_label() != "<colorbar>"]
    assert len(panels) == 8  # per satellite: three energy channels and B_Eq
    assert [ax.get_title() for ax in panels][::4] == ["rbspa mageis", "rbspb mageis"]
    assert all(ax.get_ylabel() == "L_star (TS04)" for ax in panels)
    assert all(len(ax.collections) == 1 for ax in panels)


@pytest.mark.minimal
@pytest.mark.basic
def test_plot_along_l_star_writes_the_figure(mageis_data: Path, plot_log: pytest.LogCaptureFixture) -> None:
    output = mageis_data / "along.png"
    result = mageis_plot(
        mageis_data, "--satellite", "a", "--variable", "FEDU", "--energy", "400", "--along-L-star", *output_args(output)
    )

    assert result.exit_code == 0, result.output
    assert output.read_bytes().startswith(b"\x89PNG")
    assert "Could not plot" not in plot_log.text
    assert "L_star" in plot_log.text  # the settings table names the coordinate


@pytest.mark.minimal
@pytest.mark.basic
def test_plot_along_a_coordinate_without_data_fails(mageis_data: Path, plot_log: pytest.LogCaptureFixture) -> None:
    output = mageis_data / "along.png"
    result = mageis_plot(mageis_data, "--satellite", "a", "--variable", "FEDU", "--along-R-eq", *output_args(output))

    assert result.exit_code == 1, result.output
    assert not output.exists()
    assert "'R_Eq' has no data" in plot_log.text


FAKE_INDEX_UNITS = {"Dst": u.nT, "Vsw": u.km / u.s}


def fake_indices(start_time: datetime, end_time: datetime, requested: list[str]) -> dict[str, Any]:
    """Stand-in for `ep.load_indices_solar_wind_parameters` without a target time: (data, time) per index."""
    import numpy as np  # noqa: PLC0415

    posix = np.linspace(start_time.timestamp(), end_time.timestamp(), 9)
    result = {}
    for name in requested:
        data = np.arange(9.0 * 6).reshape(9, 6) if name == "W_params" else np.arange(9.0)
        result[name] = (
            ep.Variable(data=data, original_unit=FAKE_INDEX_UNITS.get(name, u.dimensionless_unscaled)),
            ep.Variable(data=posix, original_unit=ep.units.posixtime),
        )
    return result


@pytest.mark.minimal
@pytest.mark.basic
def test_build_figure_draws_indices_once_at_the_top(mageis_data: Path) -> None:
    datasets = [mageis_dataset(mageis_data, satellite) for satellite in ("a", "b")]

    with patch.object(ep, "load_indices_solar_wind_parameters", side_effect=fake_indices) as load:
        figure = plot_cli.build_figure(datasets, ["B_Eq"], PLOT_START, PLOT_END, indices=["Kp", "Dst"])

    panels = [ax for ax in figure.axes if ax.get_label() != "<colorbar>"]
    assert [ax.get_ylabel() for ax in panels][:2] == ["Kp", "Dst [nT]"]
    assert len(panels) == 4  # two indices once, plus one panel per satellite
    assert [call.args[2] for call in load.call_args_list] == [["Kp"], ["Dst"]]
    assert all(len(ax.lines) == 1 for ax in panels[:2])


@pytest.mark.minimal
@pytest.mark.basic
def test_index_panels_match_sw_index() -> None:
    from typing import get_args  # noqa: PLC0415

    from el_paso.load_indices_solar_wind_parameters import SW_Index  # noqa: PLC0415

    assert set(plot_cli.INDEX_PANELS) == set(get_args(SW_Index))


@pytest.mark.minimal
@pytest.mark.basic
def test_build_figure_draws_every_solar_wind_parameter() -> None:
    with patch.object(ep, "load_indices_solar_wind_parameters", side_effect=fake_indices):
        figure = plot_cli.build_figure([], [], PLOT_START, PLOT_END, indices=list(plot_cli.INDEX_PANELS))

    panels = {ax.get_ylabel(): ax for ax in figure.axes}
    assert len(panels) == len(plot_cli.INDEX_PANELS)
    assert {"Kp", "Dst [nT]", "Vsw [km / s]", "G1", "W_params"} <= set(panels)
    assert all(not ax.texts for ax in panels.values())  # every panel has data
    assert [line.get_label() for line in panels["W_params"].lines] == [f"W{i}" for i in range(1, 7)]


@pytest.mark.minimal
@pytest.mark.basic
def test_index_download_failure_warns_and_leaves_the_panel_empty(
    mageis_data: Path, plot_log: pytest.LogCaptureFixture
) -> None:
    with patch.object(ep, "load_indices_solar_wind_parameters", side_effect=OSError("offline")):
        figure = plot_cli.build_figure(
            [mageis_dataset(mageis_data, "a")], ["B_Eq"], PLOT_START, PLOT_END, indices=["Kp"]
        )

    assert "Could not load Kp: offline" in plot_log.text
    assert [text.get_text() for text in figure.axes[0].texts] == ["Kp: no data"]


@pytest.mark.minimal
@pytest.mark.basic
def test_split_panels_separates_indices_from_variables() -> None:
    assert plot_cli.split_panels(["FEDU", "Kp", "B_Eq", "Dst"]) == (["Kp", "Dst"], ["FEDU", "B_Eq"])
    assert plot_cli.split_panels(None) == ([], [])


@pytest.mark.minimal
@pytest.mark.basic
@pytest.mark.parametrize(
    ("strategy_kwargs", "start_time", "end_time", "expected"),
    [
        ([{"satellite": "a"}], PLOT_START, PLOT_END, "rbsp_mageis_electrons_a_20171015_20171016.png"),
        (
            [{"satellite": "a"}, {"satellite": "b"}],
            PLOT_START,
            PLOT_END + timedelta(hours=12, seconds=5),
            "rbsp_mageis_electrons_a-b_20171015_20171016T120005.png",
        ),
        ([{"mag_field": "T89"}], PLOT_START, PLOT_END, "rbsp_mageis_electrons_20171015_20171016.png"),
    ],
    ids=["one-satellite", "two-satellites-with-time", "no-satellite-parameter"],
)
def test_default_filename(
    strategy_kwargs: list[dict[str, Any]], start_time: datetime, end_time: datetime, expected: str
) -> None:
    from el_paso.recipes.rbsp.process_rbsp_mageis_electrons import process_rbsp_mageis_electrons  # noqa: PLC0415

    request = PlotRequest(
        recipe=process_rbsp_mageis_electrons,
        factory=stub_strategy,
        start_time=start_time,
        end_time=end_time,
        strategy_kwargs=strategy_kwargs,
        variables=None,
        output_dir=None,
        filename=None,
        show=False,
        dry_run=False,
        verbose=False,
        plot_defaults={},
    )

    assert plot_cli.default_filename(request) == expected


@pytest.mark.minimal
@pytest.mark.basic
@pytest.mark.parametrize("end_time", ["2017-10-14", "2017-10-15"])
def test_plot_command_rejects_end_time_not_after_start_time(requests: list[PlotRequest], end_time: str) -> None:
    result = runner.invoke(stub_app(), ["--start-time", "2017-10-15", "--end-time", end_time])
    assert result.exit_code == 2, result.output
    assert "must be after --start-time" in click.unstyle(result.output)
    assert not requests


@pytest.mark.minimal
@pytest.mark.parametrize(
    ("text", "kev"),
    [("470", 470.0), ("100keV", 100.0), ("10 keV", 10.0), ("1MeV", 1000.0), ("1.5e2eV", 0.15), (".5MeV", 500.0)],
)
def test_energy_option_accepts_units(text: str, kev: float) -> None:
    from el_paso.cli.plot_cli import _parse_energy

    assert _parse_energy(text) == pytest.approx(kev)


@pytest.mark.minimal
@pytest.mark.basic
def test_plot_creates_the_output_dir_and_uses_the_default_filename(
    mageis_data: Path, plot_log: pytest.LogCaptureFixture
) -> None:
    output_dir = mageis_data / "figures" / "nested"

    result = mageis_plot(mageis_data, "--satellite", "a", "--variable", "B_Eq", "--output-dir", str(output_dir))

    assert result.exit_code == 0, result.output
    [figure] = output_dir.iterdir()
    assert figure.name.startswith("rbsp_mageis_electrons_a_")
    assert figure.read_bytes().startswith(b"\x89PNG")


@pytest.mark.minimal
@pytest.mark.basic
def test_plot_filename_must_not_contain_a_directory(requests: list[PlotRequest]) -> None:
    result = invoke_app(["plot", "rbsp", "mageis-electrons", *TIME_ARGS, "--filename", "sub/out.png"])

    assert result.exit_code == 2, result.output
    assert "Use --output-dir" in click.unstyle(result.output)


@pytest.mark.minimal
@pytest.mark.basic
def test_plot_script_values_render_as_python() -> None:
    from el_paso.cli.plot_script import _py  # noqa: PLC0415

    assert _py(datetime(2017, 10, 15, tzinfo=timezone.utc)) == "datetime(2017, 10, 15, 0, 0, 0, tzinfo=timezone.utc)"
    assert _py(Path("/data")) == "Path('/data')"
    assert _py([2.1, 3.4] * u.MeV) == 'u.Quantity([2.1, 3.4], "MeV")'
    assert _py(90 * u.deg) == 'u.Quantity(90.0, "deg")'
    assert _py({"FEDU": {"vmin": 1e-2, "along": None}}) == "{'FEDU': {'vmin': 0.01, 'along': None}}"
    assert _py(timedelta(minutes=5)) == "timedelta(seconds=300.0)"


@pytest.mark.minimal
@pytest.mark.basic
def test_exported_script_makes_the_same_figure(
    mageis_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import runpy  # noqa: PLC0415

    monkeypatch.setattr(ep, "setup_logging", lambda *_args, **_kwargs: None)  # the script sets up its own logging

    script = tmp_path / "scripts" / "my_plot.py"
    figure_path = tmp_path / "figures" / "from_script.png"

    result = mageis_plot(
        mageis_data,
        "--satellite",
        "all",
        "--variable",
        "FEDU",
        "--variable",
        "B_Eq",
        "--energy",
        "400",
        "--along-L-star",
        *output_args(figure_path),
        "--export-script",
        str(script),
    )

    assert result.exit_code == 0, result.output
    assert not figure_path.exists()  # exporting does not plot
    source = script.read_text()
    assert "plt.subplots(" in source
    assert "build_figure" not in source
    assert 'ENERGIES = u.Quantity([400.0], "keV")' in source
    assert "ALONG = 'L_star'" in source
    compile(source, str(script), "exec")

    runpy.run_path(str(script))

    assert figure_path.read_bytes().startswith(b"\x89PNG")


@pytest.mark.minimal
@pytest.mark.basic
def test_exported_script_leaves_a_panel_empty_when_an_index_cannot_be_loaded(
    mageis_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import runpy  # noqa: PLC0415

    monkeypatch.setattr(ep, "setup_logging", lambda *_args, **_kwargs: None)  # the script sets up its own logging

    def failing_download(*args: object, **kwargs: object) -> None:
        msg = "no network"
        raise OSError(msg)

    monkeypatch.setattr(ep, "load_indices_solar_wind_parameters", failing_download)
    script = tmp_path / "my_plot.py"
    figure_path = tmp_path / "with_kp.png"

    result = mageis_plot(
        mageis_data,
        "--satellite",
        "a",
        "--variable",
        "Kp",
        "--variable",
        "B_Eq",
        *output_args(figure_path),
        "--export-script",
        str(script),
    )
    assert result.exit_code == 0, result.output

    runpy.run_path(str(script))

    assert "Could not load Kp: no network" in caplog.text
    assert figure_path.read_bytes().startswith(b"\x89PNG")
