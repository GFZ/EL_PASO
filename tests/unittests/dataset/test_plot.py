# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for `DataSet.plot` (`el_paso.dataset.plot`)."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import el_paso as ep
import matplotlib
import numpy as np
import pytest
from astropy import units as u
from el_paso.dataset import DataSet
from el_paso.dataset.plot import PanelSettings, match_nearest, resolve_panel_settings
from el_paso.dataset.utils import python2matlab

matplotlib.use("Agg")

if TYPE_CHECKING:
    from pathlib import Path

    from el_paso.typing import InternalName

TIME_SIZE = 48
START = datetime(2013, 1, 1, tzinfo=timezone.utc)
END = datetime(2013, 1, 2, tzinfo=timezone.utc)


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> DataSet:
    """A small dataset saved with a field-model strategy and loaded back."""
    energy_size, alpha_size = 3, 4
    datetimes = [START + i * np.timedelta64(1800, "s") for i in range(TIME_SIZE)]
    flux_unit = (u.cm**2 * u.s * u.sr * u.keV) ** (-1)

    variables: dict[InternalName, ep.Variable] = {
        "Epoch": ep.Variable(original_unit=ep.units.datenum, data=np.array([python2matlab(t) for t in datetimes])),
        "FEDU": ep.Variable(
            original_unit=flux_unit,
            data=np.random.default_rng(0).uniform(1, 100, (TIME_SIZE, energy_size, alpha_size)),
        ),
        "Energy_FEDU": ep.Variable(original_unit=u.MeV, data=np.tile([0.5, 1.0, 2.0], (TIME_SIZE, 1))),
        "Alpha": ep.Variable(original_unit=u.deg, data=np.tile([10.0, 30.0, 60.0, 90.0], (TIME_SIZE, 1))),
        "L_m": ep.Variable(original_unit=u.dimensionless_unscaled, data=np.full((TIME_SIZE, alpha_size), 4.5)),
        "Position": ep.Variable(original_unit=ep.units.RE, data=np.ones((TIME_SIZE, 3))),
        "B_Eq": ep.Variable(original_unit=u.nT, data=np.linspace(40.0, 60.0, TIME_SIZE)),
    }
    strategy = ep.saving_strategies.MonthlyRBStrategy(
        base_data_path=tmp_path,
        mission="GOES",
        satellite="primary",
        instrument="MAGED",
        mag_field="T89",
        file_format="nc",
        data_standard=ep.data_standards.GFZStandard(),
    )
    ep.save(variables, strategy, start_time=START, end_time=END, time_var=variables["Epoch"])
    return DataSet(saving_strategy=strategy, start_time=START, end_time=END, verbose=False)


@pytest.mark.minimal
def test_plot_draws_one_panel_per_variable(synthetic_dataset: DataSet) -> None:
    figure = synthetic_dataset.plot(["FEDU", "B_Eq", "Position"])

    assert len(figure.axes) >= 3  # plus a colour bar for the spectrogram
    labels = [ax.get_ylabel() for ax in figure.axes]
    assert any("Energy_FEDU" in label for label in labels)
    assert any("B_Eq (T89)" in label for label in labels)
    figure.savefig(synthetic_dataset.saving_strategy.base_data_path / "plot.png")


@pytest.mark.minimal
def test_plot_into_given_axes(synthetic_dataset: DataSet) -> None:
    from matplotlib.figure import Figure  # noqa: PLC0415

    figure = Figure()
    axes = list(figure.subplots(2, 1, sharex=True))

    returned = synthetic_dataset.plot(["B_Eq", "L_m"], axes=axes)

    assert returned is figure
    assert "B_Eq" in axes[0].get_ylabel()
    assert "Alpha" in axes[1].get_ylabel()  # L_m is a spectrogram over pitch angle (in degrees)
    assert "[deg]" in axes[1].get_ylabel()


@pytest.mark.minimal
def test_plot_rejects_bad_input(synthetic_dataset: DataSet) -> None:
    with pytest.raises(ValueError, match="At least one"):
        synthetic_dataset.plot([])
    with pytest.raises(ValueError, match="not a variable"):
        synthetic_dataset.plot(["nonsense"])  # ty:ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="axes"):
        synthetic_dataset.plot(["B_Eq", "L_m"], axes=[])


@pytest.mark.minimal
def test_spectrogram_colour_limits_are_the_5th_and_95th_percentiles(synthetic_dataset: DataSet) -> None:
    figure = synthetic_dataset.plot(["FEDU"])

    (mesh,) = figure.axes[0].collections
    flux = np.asarray(synthetic_dataset.get_by_internal_name("FEDU"))[:, :, -1]  # the 90° pitch angle
    assert (mesh.norm.vmin, mesh.norm.vmax) == pytest.approx(tuple(np.percentile(flux, [5, 95])))
    assert type(mesh.norm).__name__ == "LogNorm"


@pytest.mark.minimal
def test_vmin_and_vmax_set_spectrogram_colour_limits(synthetic_dataset: DataSet) -> None:
    figure = synthetic_dataset.plot(["FEDU", "B_Eq"], vmin=2.0, vmax=50.0)

    (mesh,) = figure.axes[0].collections
    assert (mesh.norm.vmin, mesh.norm.vmax) == (2.0, 50.0)
    assert not figure.axes[1].collections  # line panels are not affected


@pytest.mark.minimal
def test_one_colour_limit_keeps_the_percentile_for_the_other(synthetic_dataset: DataSet) -> None:
    figure = synthetic_dataset.plot(["FEDU"], vmin=2.0)

    (mesh,) = figure.axes[0].collections
    flux = np.asarray(synthetic_dataset.get_by_internal_name("FEDU"))[:, :, -1]
    assert (mesh.norm.vmin, mesh.norm.vmax) == pytest.approx((2.0, np.percentile(flux, 95)))


@pytest.mark.minimal
def test_invalid_colour_limits(synthetic_dataset: DataSet, caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(ValueError, match="vmin"):
        synthetic_dataset.plot(["FEDU"], vmin=10.0, vmax=1.0)

    # a log colour scale cannot start at zero; the panel is reported and left empty
    figure = synthetic_dataset.plot(["FEDU"], vmin=0.0)
    assert "positive" in caplog.text
    assert any("could not be plotted" in text.get_text() for text in figure.axes[0].texts)


@pytest.mark.minimal
def test_pitch_angle_selects_the_nearest_channel(synthetic_dataset: DataSet) -> None:
    figure = synthetic_dataset.plot(["FEDU"], pitch_angle=np.radians(35.0) * u.rad)

    (mesh,) = figure.axes[0].collections
    flux = np.asarray(synthetic_dataset.get_by_internal_name("FEDU"))[:, :, 1]  # the 30° channel
    assert np.asarray(mesh.get_array()).reshape(-1, TIME_SIZE) == pytest.approx(flux.T)
    assert [text.get_text() for text in figure.axes[0].texts] == ["pitch angle 30°"]


@pytest.mark.minimal
def test_selection_arguments_need_units(synthetic_dataset: DataSet) -> None:
    with pytest.raises(TypeError, match="pitch_angle needs a unit"):
        synthetic_dataset.plot(["FEDU"], pitch_angle=90.0)  # ty:ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="energies must be convertible to keV"):
        synthetic_dataset.plot(["FEDU"], energies=[1.0] * u.deg)


@pytest.mark.minimal
def test_settings_layering_order() -> None:
    resolved = resolve_panel_settings(
        ["FEDU", "L_star"],
        {"FEDU": {"pitch_angle": 30 * u.deg, "vmin": 2.0, "vmax": 50.0}, "MLT": {"vmin": 0.0}},
        {"vmax": 80.0, "energies": None},
    )

    # per-variable settings, then the overrides for every panel; entries for other variables are ignored
    assert resolved == {
        "FEDU": PanelSettings(pitch_angle=30 * u.deg, vmin=2.0, vmax=80.0),
        "L_star": PanelSettings(vmax=80.0),
    }


@pytest.mark.minimal
def test_settings_conflicting_across_layers_name_the_variable() -> None:
    with pytest.raises(ValueError, match=r"FEDU: vmin \(60\) must be below vmax \(50\)"):
        resolve_panel_settings(["FEDU"], {"FEDU": {"vmax": 50.0}}, {"vmin": 60.0})


@pytest.mark.minimal
@pytest.mark.parametrize(
    ("settings", "error", "match"),
    [
        ({"FEDU": {"colour": 1}}, TypeError, "FEDU: unknown setting"),
        ({"FEDU": [1.0]}, TypeError, "FEDU: settings must be a mapping"),
        ({"FEDU": {"energies": [470.0]}}, TypeError, "energies needs a unit"),
        ({"FEDU": {"energies": [470.0, -1.0] * u.keV}}, ValueError, "positive"),
        ({"FEDU": {"pitch_angle": [30.0, 60.0] * u.deg}}, ValueError, "single angle"),
        ({"FEDU": {"pitch_angle": 200 * u.deg}}, ValueError, "between 0 and 180"),
        ({"FEDU": {"vmin": 1.0 * u.keV}}, TypeError, "plain number"),
    ],
)
def test_invalid_panel_settings(settings: dict[str, Any], error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        resolve_panel_settings(["FEDU"], settings)


@pytest.mark.minimal
def test_plot_applies_panel_settings_per_variable(synthetic_dataset: DataSet) -> None:
    figure = synthetic_dataset.plot(
        ["FEDU", "L_m"],
        panel_settings={"FEDU": {"pitch_angle": 30 * u.deg, "vmin": 2.0, "vmax": 50.0}, "L_m": {"vmin": 4.0}},
        vmax=80.0,
    )

    fedu_panel, l_m_panel = (ax for ax in figure.axes if ax.get_label() != "<colorbar>")
    (fedu_mesh,) = fedu_panel.collections
    (l_m_mesh,) = l_m_panel.collections
    assert (fedu_mesh.norm.vmin, fedu_mesh.norm.vmax) == (2.0, 80.0)
    assert (l_m_mesh.norm.vmin, l_m_mesh.norm.vmax) == (4.0, 80.0)
    assert [text.get_text() for text in fedu_panel.texts] == ["pitch angle 30°"]


@pytest.mark.minimal
def test_spectrogram_label_fits_its_panel(synthetic_dataset: DataSet) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: PLC0415

    figure = synthetic_dataset.plot(["FEDU", "B_Eq"])
    renderer = FigureCanvasAgg(figure).get_renderer()
    figure.draw(renderer)
    panel, line_panel, colorbar = figure.axes

    label = colorbar.yaxis.label
    assert label.get_text().startswith("FEDU\n[")
    assert label.get_window_extent(renderer).height <= panel.get_window_extent(renderer).height
    # the pitch-angle cut is noted inside the panel rather than in the colour-bar label
    assert [text.get_text() for text in panel.texts] == ["pitch angle 90°"]
    # the colour bar takes its space from the margin, so the panels stay aligned
    assert panel.get_position().x1 == pytest.approx(line_panel.get_position().x1)


@pytest.mark.minimal
def test_plot_raises_for_variable_without_data(synthetic_dataset: DataSet) -> None:
    with pytest.raises(ValueError, match="'R_Eq' has no data"):
        synthetic_dataset.plot(["R_Eq"])


@pytest.mark.minimal
def test_import_el_paso_does_not_import_matplotlib() -> None:
    code = "import sys, el_paso; print('matplotlib' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)  # noqa: S603

    assert result.stdout.strip() == "False"


def test_match_nearest_fixed_channels() -> None:
    indices, values = match_nearest(np.array([100.0, 200.0, 400.0]), np.array([190.0, 1000.0]), n_times=2)

    assert indices.tolist() == [[1, 2], [1, 2]]
    assert values.tolist() == [[200.0, 400.0], [200.0, 400.0]]


def test_match_nearest_follows_shifted_channels_and_applies_tolerance() -> None:
    coordinate = np.array([[100.0, 470.0, 900.0], [100.0, 500.0, 900.0], [100.0, 600.0, 900.0]])

    indices, values = match_nearest(coordinate, np.array([470.0]), n_times=3, tolerance=0.1)

    assert indices[:, 0].tolist() == [1, 1, 0]
    np.testing.assert_array_equal(values[:, 0], [470.0, 500.0, np.nan])  # 600 is >10% away -> blank


def test_match_nearest_ignores_nan_channels_and_checks_time_steps() -> None:
    _, values = match_nearest(np.array([[np.nan, 300.0]]), np.array([100.0]), n_times=1)
    assert values[0, 0] == 300.0

    with pytest.raises(ValueError, match="time steps"):
        match_nearest(np.ones((2, 3)), np.array([1.0]), n_times=3)


@pytest.mark.minimal
def test_selected_energies_are_drawn_as_lines(synthetic_dataset: DataSet) -> None:
    figure = synthetic_dataset.plot(["FEDU", "B_Eq"], energies=[1000.0, 2100.0] * u.keV)

    ax = figure.axes[0]
    assert not ax.collections  # no spectrogram
    assert ax.get_yscale() == "log"
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["1 MeV", "2 MeV"]
    flux = np.asarray(synthetic_dataset.get_by_internal_name("FEDU"))[:, 1, 3]  # 90° channel
    assert ax.lines[0].get_ydata() == pytest.approx(flux)
    assert [text.get_text() for text in ax.texts] == ["pitch angle 90°"]


@pytest.mark.minimal
def test_energy_lines_use_vmin_vmax_as_y_limits_and_skip_far_channels(
    synthetic_dataset: DataSet, caplog: pytest.LogCaptureFixture
) -> None:
    figure = synthetic_dataset.plot(["FEDU"], energies=[5000.0] * u.keV, vmin=1.0, vmax=50.0)

    ax = figure.axes[0]
    assert ax.get_ylim() == pytest.approx((1.0, 50.0))
    assert np.all(np.isnan(ax.lines[0].get_ydata()))
    assert "no channel within 10%" in caplog.text
    assert ax.get_legend().get_texts()[0].get_text() == "5 MeV: no match"


@pytest.mark.minimal
def test_energies_are_ignored_by_variables_without_energy(synthetic_dataset: DataSet) -> None:
    figure = synthetic_dataset.plot(["B_Eq"], energies=[1000.0] * u.keV)

    assert len(figure.axes[0].lines) == 1


@pytest.mark.minimal
def test_pitch_angle_collapses_pitch_angle_dependent_variables(synthetic_dataset: DataSet) -> None:
    spectrogram = synthetic_dataset.plot(["L_m"])
    assert len(spectrogram.axes[0].collections) == 1  # without a pitch angle: all channels

    figure = synthetic_dataset.plot(["L_m"], pitch_angle=35.0 * u.deg)

    ax = figure.axes[0]
    assert not ax.collections
    assert ax.lines[0].get_ydata() == pytest.approx(np.full(TIME_SIZE, 4.5))
    assert [text.get_text() for text in ax.texts] == ["pitch angle 30°"]


def panels_of(figure: Any) -> list[Any]:  # noqa: ANN401
    """The panels of a figure, without its colour bars."""
    return [ax for ax in figure.axes if ax.get_label() != "<colorbar>"]


@pytest.mark.minimal
def test_along_draws_one_scatter_panel_per_energy_channel(synthetic_dataset: DataSet) -> None:
    figure = synthetic_dataset.plot(["FEDU", "B_Eq"], along="L_m")

    panels = panels_of(figure)
    assert len(panels) == 4  # three FEDU channels, one B_Eq panel
    assert all(ax.get_ylabel() == "L_m (T89)" for ax in panels)
    assert [ax.texts[0].get_text() for ax in panels] == [
        "0.5 MeV, pitch angle 90°",
        "1 MeV, pitch angle 90°",
        "2 MeV, pitch angle 90°",
        "L_m at pitch angle 90°",
    ]

    (points,) = panels[1].collections
    flux = np.asarray(synthetic_dataset.get_by_internal_name("FEDU"))[:, 1, 3]
    assert np.asarray(points.get_array()) == pytest.approx(flux)
    assert np.asarray(points.get_offsets())[:, 1] == pytest.approx(np.full(TIME_SIZE, 4.5))
    assert type(points.norm).__name__ == "LogNorm"
    assert (points.norm.vmin, points.norm.vmax) == pytest.approx(tuple(np.percentile(flux, [5, 95])))

    (b_eq_points,) = panels[3].collections
    assert type(b_eq_points.norm).__name__ == "Normalize"
    figure.savefig(synthetic_dataset.saving_strategy.base_data_path / "along.png")


@pytest.mark.minimal
def test_along_with_energies_and_pitch_angle(synthetic_dataset: DataSet) -> None:
    figure = synthetic_dataset.plot(
        ["FEDU"], along="L_m", energies=[1000.0, 2100.0] * u.keV, pitch_angle=35 * u.deg, vmin=2.0, vmax=50.0
    )

    panels = panels_of(figure)
    assert [ax.texts[0].get_text() for ax in panels] == ["1 MeV, pitch angle 30°", "2 MeV, pitch angle 30°"]
    (points,) = panels[0].collections
    flux = np.asarray(synthetic_dataset.get_by_internal_name("FEDU"))[:, 1, 1]
    assert np.asarray(points.get_array()) == pytest.approx(flux)
    assert (points.norm.vmin, points.norm.vmax) == (2.0, 50.0)


@pytest.mark.minimal
def test_count_panels(synthetic_dataset: DataSet) -> None:
    from el_paso.dataset.plot import count_panels  # noqa: PLC0415

    variables: list[InternalName] = ["FEDU", "B_Eq", "Position"]
    assert count_panels(synthetic_dataset, variables) == [1, 1, 1]
    assert count_panels(synthetic_dataset, variables, along="L_m") == [3, 1, 1]
    assert count_panels(synthetic_dataset, variables, along="L_m", energies=[1.0] * u.MeV) == [1, 1, 1]
    settings = {"FEDU": {"energies": [0.5, 1.0] * u.MeV}}
    assert count_panels(synthetic_dataset, variables, along="L_m", panel_settings=settings) == [2, 1, 1]


@pytest.mark.minimal
def test_along_setting_applies_to_its_variable_only_and_the_argument_overrides_it(
    synthetic_dataset: DataSet,
) -> None:
    from el_paso.dataset.plot import count_panels  # noqa: PLC0415

    variables: list[InternalName] = ["FEDU", "B_Eq"]
    settings = {"FEDU": {"along": "L_m"}}

    assert count_panels(synthetic_dataset, variables, panel_settings=settings) == [3, 1]
    panels = panels_of(synthetic_dataset.plot(variables, panel_settings=settings))
    assert len(panels) == 4
    assert [ax.get_ylabel() for ax in panels[:3]] == ["L_m (T89)"] * 3
    assert "L_m" not in panels[3].get_ylabel()  # B_Eq stays a line

    # the keyword argument is the top layer and applies to every variable
    assert count_panels(synthetic_dataset, variables, along="L_m", panel_settings={"FEDU": {"along": "L_m"}}) == [3, 1]
    figure = synthetic_dataset.plot(["B_Eq"], panel_settings={"B_Eq": {"along": "L_m"}}, along="L_m")
    assert panels_of(figure)[0].get_ylabel() == "L_m (T89)"


@pytest.mark.minimal
def test_along_setting_is_validated() -> None:
    from el_paso.dataset.plot import check_panel_settings  # noqa: PLC0415

    check_panel_settings({"FEDU": {"along": "L_star"}})
    with pytest.raises(ValueError, match="FEDU: along must be one of"):
        check_panel_settings({"FEDU": {"along": "MLT"}})


@pytest.mark.minimal
def test_along_cannot_draw_vectors(synthetic_dataset: DataSet, caplog: pytest.LogCaptureFixture) -> None:
    figure = synthetic_dataset.plot(["Position", "B_Eq"], along="L_m")

    position, b_eq = panels_of(figure)
    assert "only variables of time, energy and pitch angle" in caplog.text
    assert any("could not be plotted" in text.get_text() for text in position.texts)
    assert len(b_eq.collections) == 1


@pytest.mark.minimal
def test_along_rejects_bad_coordinates(synthetic_dataset: DataSet) -> None:
    with pytest.raises(ValueError, match="along must be one of"):
        synthetic_dataset.plot(["B_Eq"], along="MLT")
    with pytest.raises(ValueError, match="'R_Eq' has no data"):
        synthetic_dataset.plot(["B_Eq"], along="R_Eq")
    with pytest.raises(ValueError, match="axes"):
        synthetic_dataset.plot(["FEDU"], along="L_m", axes=[None])  # ty:ignore[invalid-argument-type]


@pytest.mark.minimal
def test_colormap_defaults_to_colorcet_rainbow4(synthetic_dataset: DataSet) -> None:
    import colorcet as cc  # noqa: PLC0415

    figure = synthetic_dataset.plot(["FEDU"])
    mesh = figure.axes[0].collections[0]
    assert mesh.get_cmap().name == cc.cm["rainbow4"].name

    figure = synthetic_dataset.plot(["FEDU"], colormap="viridis")
    mesh = figure.axes[0].collections[0]
    assert mesh.get_cmap().name == "viridis"

    figure = synthetic_dataset.plot(["B_Eq"], along="L_m", colormap="fire")
    points = figure.axes[0].collections[0]
    assert points.get_cmap().name == "fire"


@pytest.mark.minimal
def test_unknown_colormap_is_rejected(synthetic_dataset: DataSet) -> None:
    with pytest.raises(ValueError, match="unknown colormap 'nope'"):
        synthetic_dataset.plot(["FEDU"], colormap="nope")
