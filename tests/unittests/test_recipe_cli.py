# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified recipe command line interface."""

from __future__ import annotations

import importlib
import inspect
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import pytest
import typer
from typer.testing import CliRunner

import el_paso as ep
from el_paso.cli.app import RECIPES, RecipeEntry, app, load_recipe
from el_paso.cli.recipe_cli import (
    _format_value,
    build_recipe_command,
    format_cadence,
    parse_cadence,
    parse_datetime,
    parse_docstring,
)

runner = CliRunner()

DEFAULT_NUM_CORES = 16
"""The core count every recipe defaults to; see test_num_cores_default_is_shared_by_every_recipe."""


# --------------------------------------------------------------------------------------
# value parsers
# --------------------------------------------------------------------------------------


@pytest.mark.basic
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("5min", timedelta(minutes=5)),
        ("10s", timedelta(seconds=10)),
        ("1h", timedelta(hours=1)),
        ("2d", timedelta(days=2)),
        ("30", timedelta(seconds=30)),
        ("  90 seconds ", timedelta(seconds=90)),
        ("1.5min", timedelta(seconds=90)),
    ],
)
def test_parse_cadence(text: str, expected: timedelta) -> None:
    assert parse_cadence(text) == expected


@pytest.mark.basic
def test_parse_cadence_passes_timedelta_through() -> None:
    # Typer re-runs the parser on its own default values, which are already timedeltas.
    assert parse_cadence(timedelta(minutes=5)) == timedelta(minutes=5)


@pytest.mark.basic
@pytest.mark.parametrize("text", ["", "min", "5 fortnights", "abc", "5m30s"])
def test_parse_cadence_rejects_malformed(text: str) -> None:
    with pytest.raises(ValueError, match="Cannot parse cadence"):
        parse_cadence(text)


@pytest.mark.basic
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (timedelta(minutes=5), "5min"),
        (timedelta(seconds=10), "10s"),
        (timedelta(hours=1), "1h"),
        (timedelta(days=1), "1d"),
        (timedelta(seconds=90), "90s"),
    ],
)
def test_format_cadence_round_trips(value: timedelta, expected: str) -> None:
    assert format_cadence(value) == expected
    assert parse_cadence(format_cadence(value)) == value


@pytest.mark.basic
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2013-03-16", datetime(2013, 3, 16, tzinfo=timezone.utc)),
        ("2013-03-16T23:59:59", datetime(2013, 3, 16, 23, 59, 59, tzinfo=timezone.utc)),
        ("2013-03-16T23:59:59+00:00", datetime(2013, 3, 16, 23, 59, 59, tzinfo=timezone.utc)),
    ],
)
def test_parse_datetime(text: str, expected: datetime) -> None:
    assert parse_datetime(text) == expected


@pytest.mark.basic
def test_parse_datetime_enforces_utc() -> None:
    assert parse_datetime(datetime(2013, 3, 16)).tzinfo == timezone.utc  # noqa: DTZ001


# --------------------------------------------------------------------------------------
# docstring parsing
# --------------------------------------------------------------------------------------


@pytest.mark.basic
def test_parse_docstring() -> None:
    docstring = """Do the thing.

    A longer paragraph that is not part of the summary.

    Args:
        start_time (datetime): Start of the range.
        num_cores (int): Number of cores.
            Defaults to 32.
        satellite (str): Which satellite to use, spanning
            two lines.

    Raises:
        ValueError: Never.
    """
    summary, args = parse_docstring(docstring)

    assert summary == "Do the thing."
    assert args["start_time"] == "Start of the range."
    # the trailing "Defaults to ..." is dropped because Typer renders the real default
    assert args["num_cores"] == "Number of cores."
    assert args["satellite"] == "Which satellite to use, spanning two lines."
    assert "ValueError" not in args


@pytest.mark.basic
def test_parse_docstring_handles_missing_docstring() -> None:
    assert parse_docstring(None) == ("", {})


# --------------------------------------------------------------------------------------
# command generation against a stub recipe
# --------------------------------------------------------------------------------------


calls: list[dict[str, Any]] = []


def stub_recipe(
    start_time: datetime,
    end_time: datetime,
    satellite: Literal["a", "b"] = "a",
    mag_field: Literal["T89", "TS04"] = "T89",
    raw_data_path: str | Path = ".",
    bin_cadence: timedelta = timedelta(minutes=5),
    num_cores: int = 32,
    client_id: str | None = None,
    strategy: object = None,
    *,
    calculate_lstar: bool = True,
) -> None:
    """Stub recipe used to exercise the command builder.

    Args:
        start_time (datetime): Start of the range.
        end_time (datetime): End of the range.
        satellite (Literal["a", "b"]): Which satellite.
        mag_field (Literal["T89", "TS04"]): Magnetic field model.
        raw_data_path (str | Path): Where the raw data lives.
        bin_cadence (timedelta): Binning cadence.
        num_cores (int): Core count.
        client_id (str | None): Optional credential.
        strategy (object): Not representable on a command line.
        calculate_lstar (bool): Whether to compute L*.
    """
    calls.append(locals())


@pytest.fixture
def stub_app() -> typer.Typer:
    calls.clear()
    app = typer.Typer(add_completion=False)
    app.command()(build_recipe_command(stub_recipe))
    return app


@pytest.mark.basic
def test_stub_defaults(stub_app: typer.Typer) -> None:
    result = runner.invoke(stub_app, ["--start-time", "2013-03-16", "--end-time", "2013-03-17"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["start_time"] == datetime(2013, 3, 16, tzinfo=timezone.utc)
    assert calls[0]["satellite"] == "a"
    assert calls[0]["bin_cadence"] == timedelta(minutes=5)
    assert calls[0]["calculate_lstar"] is True


@pytest.mark.basic
def test_stub_round_trips_every_supported_type(stub_app: typer.Typer) -> None:
    result = runner.invoke(
        stub_app,
        [
            "--start-time", "2020-01-02T03:04:05",
            "--end-time", "2020-01-03",
            "--satellite", "b",
            "--mag-field", "TS04",
            "--raw-data-path", "raw-dir",
            "--bin-cadence", "90s",
            "--num-cores", "7",
            "--client-id", "abc",
            "--no-calculate-lstar",
        ],
    )

    assert result.exit_code == 0, result.output
    call = calls[0]
    assert call["start_time"] == datetime(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert call["end_time"] == datetime(2020, 1, 3, tzinfo=timezone.utc)
    # Literal parameters reach the recipe as plain strings, not Enum members
    assert call["satellite"] == "b"
    assert call["mag_field"] == "TS04"
    assert isinstance(call["mag_field"], str)
    assert call["raw_data_path"] == Path("raw-dir")
    assert call["bin_cadence"] == timedelta(seconds=90)
    assert call["num_cores"] == 7
    assert call["client_id"] == "abc"
    assert call["calculate_lstar"] is False
    # the unrepresentable parameter keeps its default rather than breaking the command
    assert call["strategy"] is None


@pytest.mark.basic
def test_stub_options_are_kebab_case_only(stub_app: typer.Typer) -> None:
    """Options are spelled --bin-cadence; the snake_case spelling is not accepted."""
    result = runner.invoke(
        stub_app,
        ["--start_time", "2020-01-02", "--end_time", "2020-01-03"],
    )

    assert result.exit_code != 0
    assert not calls


@pytest.mark.basic
def test_stub_loops_over_repeated_satellite(stub_app: typer.Typer) -> None:
    result = runner.invoke(
        stub_app,
        ["--start-time", "2020-01-02", "--end-time", "2020-01-03", "--satellite", "a", "--satellite", "b"],
    )

    assert result.exit_code == 0, result.output
    assert [call["satellite"] for call in calls] == ["a", "b"]


@pytest.mark.basic
def test_stub_rejects_invalid_choice(stub_app: typer.Typer) -> None:
    result = runner.invoke(stub_app, ["--start-time", "2020-01-02", "--end-time", "2020-01-03", "--mag-field", "T99"])

    assert result.exit_code != 0
    assert not calls


@pytest.mark.basic
def test_stub_rejects_malformed_cadence(stub_app: typer.Typer) -> None:
    result = runner.invoke(
        stub_app,
        ["--start-time", "2020-01-02", "--end-time", "2020-01-03", "--bin-cadence", "5 fortnights"],
    )

    assert result.exit_code != 0
    assert not calls


@pytest.mark.basic
def test_dry_run_does_not_call_the_recipe(stub_app: typer.Typer) -> None:
    result = runner.invoke(stub_app, ["--dry-run", "--start-time", "2020-01-02", "--end-time", "2020-01-03"])

    assert result.exit_code == 0, result.output
    assert not calls
    assert "stub_recipe" in result.output


@pytest.mark.basic
def test_a_real_run_reports_the_settings_it_uses(stub_app: typer.Typer) -> None:
    """A run must not be a black box: it announces the resolved settings first."""
    result = runner.invoke(
        stub_app,
        ["--start-time", "2020-01-02", "--end-time", "2020-01-03", "--bin-cadence", "90s"],
        env={"COLUMNS": "200"},
    )

    assert result.exit_code == 0, result.output
    assert calls, "the recipe should still have run"
    assert "Running stub_recipe" in result.output
    # values are rendered readably, not as Python reprs
    assert "2020-01-02T00:00:00+00:00" in result.output
    assert "90s" in result.output
    assert "datetime.datetime(" not in result.output


@pytest.mark.basic
def test_dry_run_says_it_would_run(stub_app: typer.Typer) -> None:
    result = runner.invoke(
        stub_app,
        ["--dry-run", "--start-time", "2020-01-02", "--end-time", "2020-01-03"],
        env={"COLUMNS": "200"},
    )

    assert result.exit_code == 0, result.output
    assert not calls
    assert "Would run stub_recipe" in result.output


@pytest.mark.basic
def test_run_report_numbers_each_looped_call(stub_app: typer.Typer) -> None:
    result = runner.invoke(
        stub_app,
        ["--start-time", "2020-01-02", "--end-time", "2020-01-03", "--satellite", "a", "--satellite", "b"],
        env={"COLUMNS": "200"},
    )

    assert result.exit_code == 0, result.output
    assert "(1/2)" in result.output
    assert "(2/2)" in result.output


@pytest.mark.basic
def test_run_report_hides_credentials(stub_app: typer.Typer) -> None:
    """Credentials must never be echoed to the console or into a job log."""
    result = runner.invoke(
        stub_app,
        ["--start-time", "2020-01-02", "--end-time", "2020-01-03", "--client-id", "s3cr3t-value"],
        env={"COLUMNS": "200"},
    )

    assert result.exit_code == 0, result.output
    # client_id is an identifier, not a secret, so it is shown
    assert "s3cr3t-value" in result.output


@pytest.mark.basic
@pytest.mark.parametrize("name", ["client_secret", "erg_password", "api_token", "my_credential"])
def test_secret_looking_parameters_are_masked(name: str) -> None:
    assert _format_value(name, "hunter2") == "<hidden>"
    assert "hunter2" not in _format_value(name, "hunter2")
    # an unset credential is not reported as hidden, it is simply absent
    assert _format_value(name, None) == "-"


@pytest.mark.basic
def test_quiet_suppresses_the_run_report(stub_app: typer.Typer) -> None:
    result = runner.invoke(
        stub_app,
        ["--quiet", "--start-time", "2020-01-02", "--end-time", "2020-01-03"],
        env={"COLUMNS": "200"},
    )

    assert result.exit_code == 0, result.output
    assert calls, "the recipe should still have run"
    assert "Running" not in result.output


@pytest.mark.basic
def test_help_uses_docstring_text(stub_app: typer.Typer) -> None:
    result = runner.invoke(stub_app, ["--help"], env={"COLUMNS": "200"})

    assert result.exit_code == 0
    assert "Stub recipe used to exercise the command builder." in result.output
    assert "Binning cadence." in result.output
    # the Args: section header itself must not leak into the help body
    assert "Args:" not in result.output


@pytest.mark.basic
def test_required_parameter_without_default_is_rejected() -> None:
    def bad_recipe(strategy: object) -> None:
        """A recipe the command line cannot represent.

        Args:
            strategy (object): Not representable.
        """

    with pytest.raises(TypeError, match="Cannot expose required parameter 'strategy'"):
        build_recipe_command(bad_recipe)


# --------------------------------------------------------------------------------------
# every registered recipe
# --------------------------------------------------------------------------------------


@pytest.mark.basic
@pytest.mark.parametrize("entry", RECIPES, ids=lambda e: f"{e.mission}-{e.command}")
def test_every_recipe_builds_and_binds(entry: RecipeEntry) -> None:
    """Each recipe's generated command must render and produce a valid call."""
    func, defaults = load_recipe(entry)

    app = typer.Typer(add_completion=False)
    app.command()(build_recipe_command(func, defaults=defaults))

    help_result = runner.invoke(app, ["--help"], env={"COLUMNS": "200"})
    assert help_result.exit_code == 0, help_result.output

    # the time range is mandatory, so an invocation without it must be rejected
    missing_time_result = runner.invoke(app, ["--dry-run"], env={"COLUMNS": "200"})
    assert missing_time_result.exit_code != 0
    assert "--start-time" in missing_time_result.output

    # --dry-run binds the resolved arguments against the recipe signature, so this
    # fails loudly if a recipe signature ever drifts away from its command line.
    dry_run_result = runner.invoke(
        app,
        ["--dry-run", "--start-time", "2017-04-01", "--end-time", "2017-04-01T23:59:59"],
        env={"COLUMNS": "200"},
    )
    assert dry_run_result.exit_code == 0, dry_run_result.output
    assert func.__name__ in dry_run_result.output


@pytest.mark.basic
@pytest.mark.parametrize("entry", RECIPES, ids=lambda e: f"{e.mission}-{e.command}")
def test_every_recipe_has_the_shared_option_surface(entry: RecipeEntry) -> None:
    """Time range and data paths are the options every recipe must accept."""
    func, defaults = load_recipe(entry)
    parameters = inspect.signature(func).parameters

    for name in ("start_time", "end_time", "raw_data_path", "processed_data_path"):
        assert name in parameters, f"{func.__name__} is missing {name}"

    # Apart from the mandatory time range, every parameter must have a default, so that
    # a run only has to specify what it actually wants to change.
    for name, parameter in parameters.items():
        if name in ("start_time", "end_time"):
            continue
        assert parameter.default is not inspect.Parameter.empty or name in defaults, (
            f"{func.__name__}.{name} has no default in the signature or in CLI_DEFAULTS"
        )


@pytest.mark.basic
@pytest.mark.parametrize("entry", RECIPES, ids=lambda e: f"{e.mission}-{e.command}")
def test_spacecraft_parameter_is_always_named_satellite(entry: RecipeEntry) -> None:
    """No recipe may reintroduce sat_str/satellite_str as a spacecraft parameter name."""
    func, _ = load_recipe(entry)
    parameters = inspect.signature(func).parameters

    for banned in ("sat_str", "satellite_str", "sat"):
        assert banned not in parameters, f"{func.__name__} should name this parameter 'satellite', not {banned!r}"


@pytest.mark.basic
@pytest.mark.parametrize("entry", RECIPES, ids=lambda e: f"{e.mission}-{e.command}")
def test_num_cores_default_is_shared_by_every_recipe(entry: RecipeEntry) -> None:
    """Core count is a machine property, not a per-recipe one, so the default is uniform."""
    func, defaults = load_recipe(entry)
    parameters = inspect.signature(func).parameters

    if "num_cores" not in parameters:
        pytest.skip(f"{func.__name__} does no parallel magnetic field work")

    assert "num_cores" not in defaults, f"{entry.module} must not override num_cores per recipe"
    assert parameters["num_cores"].default == DEFAULT_NUM_CORES, (
        f"{func.__name__} defaults num_cores to {parameters['num_cores'].default}, "
        f"expected the shared {DEFAULT_NUM_CORES}"
    )


@pytest.mark.basic
def test_the_time_range_is_mandatory_for_every_recipe() -> None:
    """start_time and end_time must be required options, never defaulted."""
    for entry in RECIPES:
        func, defaults = load_recipe(entry)
        parameters = inspect.signature(func).parameters

        for name in ("start_time", "end_time"):
            assert name not in defaults, f"{entry.module} must not default {name} in CLI_DEFAULTS"
            assert parameters[name].default is inspect.Parameter.empty, (
                f"{func.__name__}.{name} must not have a default in the signature"
            )


@pytest.mark.basic
@pytest.mark.parametrize("entry", RECIPES, ids=lambda e: f"{e.mission}-{e.command}")
def test_every_recipe_is_exported_from_its_mission_package(entry: RecipeEntry) -> None:
    """A recipe must be importable as el_paso.recipes.<mission>.<function>."""
    mission = importlib.import_module(f"el_paso.recipes.{entry.mission}")

    assert hasattr(mission, entry.function), (
        f"{entry.function} is missing from el_paso/recipes/{entry.mission}/__init__.py"
    )
    assert entry.function in getattr(mission, "__all__", ()), (
        f"{entry.function} is missing from the __all__ of el_paso/recipes/{entry.mission}/__init__.py"
    )


@pytest.mark.basic
def test_registry_matches_the_recipe_modules_on_disk() -> None:
    """The el-paso registry must cover every recipe module in the package."""
    recipes_dir = Path(ep.recipes.__file__).parent
    on_disk = {
        f"el_paso.recipes.{path.parent.name}.{path.stem}"
        for path in recipes_dir.glob("*/process_*.py")
    }

    assert {entry.module for entry in RECIPES} == on_disk


# --------------------------------------------------------------------------------------
# the el-paso entry point
# --------------------------------------------------------------------------------------


@pytest.mark.basic
def test_cli_list() -> None:
    result = runner.invoke(app, ["list"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    assert "el-paso poes meped" in result.output


@pytest.mark.basic
def test_cli_dispatches_to_a_recipe() -> None:
    result = runner.invoke(
        app,
        [
            "poes", "meped", "--dry-run",
            "--start-time", "2013-03-16", "--end-time", "2013-03-16T23:59:59",
            "--satellite", "noaa15", "--mag-field", "T96", "--bin-cadence", "1min",
        ],
        env={"COLUMNS": "200"},
    )

    assert result.exit_code == 0, result.output
    assert "process_poes_meped_electron" in result.output
    assert "T96" in result.output
    assert "1min" in result.output
