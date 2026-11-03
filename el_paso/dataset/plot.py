# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""Quick-look plotting for `DataSet`, attached as `DataSet.plot`.

The plot type follows from the dimensions of a variable in the data standard (layer 1 of the plot
defaults): a line for time series, one line per component for vectors, and a spectrogram for
variables that depend on energy and/or pitch angle. matplotlib is imported when `plot` is called.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from astropy import units as u

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from el_paso.dataset import DataSet
    from el_paso.typing import InternalName

logger = logging.getLogger(__name__)

_FIELD_MODEL_VARIABLES = frozenset({"L_star", "L_m", "MLT", "MLT_Eq", "B_Eq", "R_Eq", "InvMu", "InvK", "MLat", "PSD"})
"""Variables computed with a magnetic field model, whose labels name the strategy's ``mag_field``."""

_LOG_COLOR_VARIABLES = frozenset({"FEDU", "FEIU", "FEDO", "FPDU", "PSD"})
"""Flux-like variables that get a logarithmic colour scale."""

_PITCH_ANGLE_TARGET_DEG = 90.0
"""Pitch angle nearest to which spectrograms with an energy and a pitch-angle dimension are cut."""

_COLOR_LIMIT_PERCENTILES = (5.0, 95.0)
"""Percentiles of a spectrogram's finite values that set its colour limits."""

_PANEL_HEIGHT_INCHES = 2.5


def _values(array: object) -> NDArray[np.float64]:
    """Plain float array of `array`, dropping any astropy unit (units are shown in labels instead)."""
    return np.asarray(getattr(array, "value", array), dtype=np.float64)


def _label(dataset: DataSet, name: InternalName) -> str:
    """Axis label ``name [unit]``, naming the field model for model-dependent variables."""
    info = dataset.saving_strategy.data_standard.variable_infos[name]
    label = name
    mag_field = getattr(dataset.saving_strategy, "mag_field", None)
    if mag_field and name in _FIELD_MODEL_VARIABLES:
        label += f" ({mag_field})"
    unit = "deg" if info.unit == u.rad else info.unit.to_string()
    return f"{label} [{unit}]" if unit else label


def _coordinate(dataset: DataSet, dependency: str | tuple[str, ...]) -> tuple[str, NDArray[np.float64]]:
    """Name and values of the coordinate variable of a dimension.

    A dimension with alternatives (e.g. electron or proton energy) uses the first that has data.
    Time-dependent coordinates are reduced to their median over time.
    """
    alternatives = dependency if isinstance(dependency, tuple) else (dependency,)
    values = np.asarray([])
    name = alternatives[0]
    for name in alternatives:
        values = _values(dataset.get_by_internal_name(name))  # ty:ignore[invalid-argument-type]
        if values.size > 0:
            break
    if values.ndim > 1:
        values = np.nanmedian(values, axis=0)
    if dataset.saving_strategy.data_standard.variable_infos[name].unit == u.rad:  # ty:ignore[invalid-argument-type]
        values = np.degrees(values)  # angles are stored in radians, but shown in degrees
    return name, values


def _plot_variable(dataset: DataSet, ax: Axes, name: InternalName, time: NDArray[Any]) -> None:
    """Draw one variable into `ax`, choosing the plot type from its dimensions."""
    import matplotlib.dates as mdates  # noqa: PLC0415

    standard = dataset.saving_strategy.data_standard
    data = _values(dataset.get_by_internal_name(name))
    dependencies = standard.get_dependencies(name)[1:]
    mag_field = getattr(dataset.saving_strategy, "mag_field", None)
    title = f"{name} ({mag_field})" if mag_field and name in _FIELD_MODEL_VARIABLES else name

    if data.size == 0 or time.size == 0:
        ax.text(0.5, 0.5, f"{title}: no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks([])
        return

    ax.xaxis_date()
    if not dependencies:
        ax.plot(time, data, ".-", markersize=2, linewidth=0.8)
        ax.set_ylabel(_label(dataset, name))
        return

    coordinates = [_coordinate(dataset, dependency) for dependency in dependencies]

    if data.ndim - 1 != len(dependencies):
        msg = f"{name} has {data.ndim - 1} non-time dimensions, but its data standard lists {len(dependencies)}."
        raise ValueError(msg)

    # (time, energy, pitch angle) -> (time, energy), cut at the pitch angle nearest 90 degrees
    selection_note = ""
    if data.ndim == 3:
        pitch_angles = coordinates[1][1]
        index = int(np.nanargmin(np.abs(pitch_angles - _PITCH_ANGLE_TARGET_DEG)))
        data = data[:, :, index]
        selection_note = f"pitch angle {pitch_angles[index]:.0f}°"
        coordinates = coordinates[:1]

    first_dependency = dependencies[0]
    is_component = isinstance(first_dependency, str) and first_dependency not in standard.variable_infos

    if is_component:
        for component in range(data.shape[1]):
            ax.plot(time, data[:, component], linewidth=0.8, label=f"{component}")
        ax.legend(loc="upper right", fontsize="x-small", ncols=data.shape[1])
        ax.set_ylabel(_label(dataset, name))
        return

    coordinate_name, coordinate = coordinates[0]
    if coordinate.size != data.shape[1]:
        msg = f"{name} has {data.shape[1]} {coordinate_name} channels, but the coordinate has {coordinate.size} values."
        raise ValueError(msg)

    use_log = name in _LOG_COLOR_VARIABLES
    if use_log:
        data = np.where(data > 0, data, np.nan)

    mesh = ax.pcolormesh(
        mdates.date2num(time),
        coordinate,
        data.T,
        shading="nearest",
        norm=_color_norm(data, log=use_log),
        rasterized=True,
    )
    if coordinate_name.startswith("Energy"):
        ax.set_yscale("log")
    ax.set_ylabel(_label(dataset, coordinate_name))  # ty:ignore[invalid-argument-type]
    # name and unit on separate lines, so that the label fits the height of one panel
    ax.figure.colorbar(mesh, ax=ax, pad=0.01, extend="both", label=_label(dataset, name).replace(" [", "\n[", 1))
    if selection_note:
        ax.text(
            0.005,
            0.96,
            selection_note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize="small",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )


def _color_norm(data: NDArray[np.float64], *, log: bool) -> Normalize | None:
    """Colour limits from the 5th and 95th percentiles of the finite data, so outliers do not dominate.

    Returns ``None`` (matplotlib's own autoscaling) if the data has no spread.
    """
    from matplotlib.colors import LogNorm, Normalize  # noqa: PLC0415

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return None
    vmin, vmax = np.percentile(finite, _COLOR_LIMIT_PERCENTILES)
    if not vmin < vmax:
        return None
    return LogNorm(vmin=vmin, vmax=vmax) if log else Normalize(vmin=vmin, vmax=vmax)


def plot(self: DataSet, variables: Sequence[InternalName], *, axes: Sequence[Axes] | None = None) -> Figure:
    """Draw a quick-look plot of the given variables, one panel per variable.

    The plot type follows from the variable's dimensions in the data standard: a line for time
    series, one line per component for vectors, and a spectrogram (energy on a log y axis, flux on
    a log colour scale) for variables with an energy or pitch-angle dimension. Colour limits are
    the 5th and 95th percentiles of the data. For variables with both an energy and a pitch-angle
    dimension, the pitch angle nearest 90° is shown and noted in the panel. Labels name the strategy's ``mag_field`` for
    variables that depend on a magnetic field model. Empty variables get a "no data" panel, and a
    variable that cannot be drawn is reported with a warning and leaves its panel empty.

    Args:
        self (DataSet): The dataset to plot.
        variables (Sequence[InternalName]): Internal names of the variables to plot, in panel order.
        axes (Sequence[Axes] | None): Axes to draw into, one per variable (e.g. one satellite's
            block of a multi-satellite figure). If ``None``, a new figure is created.

    Returns:
        Figure: The figure that contains the panels.

    Raises:
        ValueError: If no variables are given, if the number of axes does not match the number of
            variables, or if a variable is not part of the data standard.
    """
    from matplotlib.figure import Figure  # noqa: PLC0415

    if len(variables) == 0:
        msg = "At least one variable is required to plot."
        raise ValueError(msg)

    standard = self.saving_strategy.data_standard
    for name in variables:
        if name not in standard.variable_infos:
            msg = f"'{name}' is not a variable of {type(standard).__name__}."
            raise ValueError(msg)

    if axes is None:
        figure = Figure(figsize=(10, _PANEL_HEIGHT_INCHES * len(variables) + 0.8), layout="constrained")
        axes = list(figure.subplots(len(variables), 1, sharex=True, squeeze=False)[:, 0])
    else:
        axes = list(axes)
        if len(axes) != len(variables):
            msg = f"Got {len(axes)} axes for {len(variables)} variables."
            raise ValueError(msg)
        figure = axes[0].get_figure(root=True)
        assert figure is not None

    time = np.asarray(self.datetime)
    for ax, name in zip(axes, variables, strict=True):
        try:
            _plot_variable(self, ax, name, time)
        except ValueError as error:
            logger.warning(f"Could not plot {name}: {error}")
            ax.text(0.5, 0.5, f"{name}: could not be plotted", ha="center", va="center", transform=ax.transAxes)

    axes[0].set_title(self.get_print_name(), fontsize="medium")
    return figure
