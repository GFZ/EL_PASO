# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for `DataSet.plot` (`el_paso.dataset.plot`)."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import el_paso as ep
import matplotlib
import numpy as np
import pytest
from astropy import units as u
from el_paso.dataset import DataSet
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
def test_plot_shows_missing_variable_as_no_data(synthetic_dataset: DataSet) -> None:
    figure = synthetic_dataset.plot(["R_Eq"])

    assert any("no data" in text.get_text() for text in figure.axes[0].texts)


@pytest.mark.minimal
def test_import_el_paso_does_not_import_matplotlib() -> None:
    code = "import sys, el_paso; print('matplotlib' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)  # noqa: S603

    assert result.stdout.strip() == "False"
