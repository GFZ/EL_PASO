# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""Write the resolved settings of ``el-paso plot`` as a standalone Python script.

The script rebuilds the datasets and draws the figure with matplotlib, with every setting as a
variable at the top, so that it can be edited and extended into a custom figure.
"""

from __future__ import annotations

import importlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from el_paso.cli._parsing import _func_name

if TYPE_CHECKING:
    from collections.abc import Callable

    from el_paso.cli.plot_cli import PlotRequest


def _py(value: Any) -> str:  # noqa: ANN401, PLR0911
    """Render a setting as Python source that evaluates to an equal value."""
    if hasattr(value, "unit") and hasattr(value, "value"):  # an astropy quantity
        magnitude = getattr(value.value, "tolist", lambda: value.value)()
        return f'u.Quantity({magnitude!r}, "{value.unit}")'
    if isinstance(value, datetime):
        fields = [value.year, value.month, value.day, value.hour, value.minute, value.second]
        if value.microsecond:
            fields.append(value.microsecond)
        return f"datetime({', '.join(map(str, fields))}, tzinfo=timezone.utc)"
    if isinstance(value, timedelta):
        return f"timedelta(seconds={value.total_seconds()!r})"
    if isinstance(value, Path):
        return f"Path({str(value)!r})"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_py(key)}: {_py(item)}" for key, item in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_py(item) for item in value) + "]"
    return repr(value)


def _factory_import(factory: Callable[..., Any]) -> str:
    """The import statement for a strategy factory, from its mission package if that exports it."""
    name = getattr(factory, "__name__", "")
    package = factory.__module__.rpartition(".")[0]
    if package and getattr(importlib.import_module(package), name, None) is factory:
        return f"from {package} import {name}"
    return f"from {factory.__module__} import {name}"


def write_plot_script(request: PlotRequest) -> str:
    """The Python source of a script that makes the figure a plot request describes.

    The script does what ``el-paso plot`` does after loading, so it draws the same figure, but it
    does not check that processed files exist: satellites without files give an empty block of panels.

    Args:
        request (PlotRequest): The resolved plot request.

    Returns:
        str: The source of the script.
    """
    from el_paso.cli.plot_cli import (  # noqa: PLC0415  (plot_cli imports this module)
        _output_path,
        _recipe_settings,
        _setting_overrides,
        resolve_panels,
        split_panels,
    )

    indices, variables = split_panels(resolve_panels(request))
    output = _output_path(request)
    overrides = _setting_overrides(request)

    settings = {
        "START_TIME": request.start_time,
        "END_TIME": request.end_time,
        "STRATEGY_KWARGS": request.strategy_kwargs,
        "INDICES": indices,
        "VARIABLES": variables,
        "PANEL_SETTINGS": _recipe_settings(request),
        "ENERGIES": overrides["energies"],
        "PITCH_ANGLE": overrides["pitch_angle"],
        "VMIN": overrides["vmin"],
        "VMAX": overrides["vmax"],
        "ALONG": request.along,
        "COLORMAP": request.colormap,
        "OUTPUT_PATH": output,
        "SHOW": request.show,
    }
    comments = {
        "STRATEGY_KWARGS": "Arguments of the strategy factory, one set per satellite.",
        "INDICES": "Space weather indices drawn at the top, once for all satellites.",
        "VARIABLES": "One panel per variable and satellite.",
        "PANEL_SETTINGS": "Settings per variable, from the recipe's plot defaults.",
        "ENERGIES": "The settings below override PANEL_SETTINGS for every variable; None keeps them.",
        "ALONG": "Coordinate on the y axis of scatter panels, e.g. 'L_star', or None.",
        "COLORMAP": "Colormap of spectrograms and scatter panels.",
        "OUTPUT_PATH": "File to write the figure to, or None to not save it.",
        "SHOW": "Open the figure in an interactive window.",
    }
    body = ""
    for name, value in settings.items():
        if name in comments:
            body += f"\n# {comments[name]}\n"
        body += f"{name} = {_py(value)}\n"

    factory = getattr(request.factory, "__name__", "")
    imports = [
        "from datetime import datetime, timezone",
        *(["from datetime import timedelta"] if "timedelta(" in body else []),
        *(["from pathlib import Path"] if "Path(" in body else []),
        "",
        *(["import astropy.units as u"] if "u.Quantity(" in body else []),
        "import logging",
        "import matplotlib",
        "import numpy as np",
        "import el_paso as ep",
        "from el_paso.dataset.plot import count_panels",
        _factory_import(request.factory),
    ]

    return (
        f'"""Plot of {_func_name(request.recipe)}, generated by `el-paso plot`.\n\n'
        "Edit the settings below, then run this file with python. The figure is drawn with matplotlib,\n"
        'so every step can be changed or extended.\n"""\n\n'
        + "\n".join(imports)
        + "\nep.setup_logging()"
        + f'\nlogger=logging.getLogger("{request.export_script.stem!s}")'  # ty: ignore[unresolved-attribute]
        + "\n\n# Settings\n"
        + body
        + f"""
STEP_INDICES = {{"Kp", "Dst"}}  # averages over fixed intervals, drawn as steps

if not SHOW:
    matplotlib.use("Agg")  # saving a figure needs no display
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# One dataset per satellite, and the number of panels each one needs.
datasets = [ep.DataSet({factory}(**kwargs), START_TIME, END_TIME) for kwargs in STRATEGY_KWARGS]
if not VARIABLES:
    datasets = []
panel_counts = [
    sum(count_panels(dataset, VARIABLES, along=ALONG, panel_settings=PANEL_SETTINGS, energies=ENERGIES))
    for dataset in datasets
]
num_panels = len(INDICES) + sum(panel_counts)

# One column of panels with a shared time axis: the indices first, then one block per satellite.
fig, axes = plt.subplots(
    num_panels, 1, sharex=True, squeeze=False, figsize=(10, 2.0 * num_panels + 0.8), layout="constrained"
)
axes = list(axes[:, 0])

for ax, name in zip(axes, INDICES):
    try:
        data, time = ep.load_indices_solar_wind_parameters(START_TIME, END_TIME, [name])[name]
        times = [datetime.fromtimestamp(t, tz=timezone.utc) for t in time.get_data(ep.units.posixtime)]
        values = np.asarray(data.get_data(), dtype=float)
        unit = data.metadata.unit
    except Exception as error:  # a failed download or read leaves the panel empty
        logger.warning(f"Could not load {{name}}: {{error}}")
        times, values, unit = [], np.empty(0), None
    ax.set_ylabel(f"{{name}} [{{unit}}]" if unit is not None and str(unit) else name)
    if len(times) == 0 or not np.any(np.isfinite(values)):
        ax.text(0.5, 0.5, f"{{name}}: no data", ha="center", va="center", transform=ax.transAxes)
        continue
    if name in STEP_INDICES:
        ax.step(times, values, where="post", linewidth=0.8)
    elif values.ndim == 2:  # vector parameters have one line per component
        for component in range(values.shape[-1]):
            ax.plot(times, values[:, component], linewidth=0.8, label=f"W{{component + 1}}")
        ax.legend(loc="upper right", fontsize="x-small", ncols=values.shape[-1])
    else:
        ax.plot(times, values, linewidth=0.8)
if INDICES:
    axes[0].set_title("Space weather", fontsize="medium")

first = len(INDICES)
for dataset, count in zip(datasets, panel_counts):
    dataset.plot(
        VARIABLES,
        axes=axes[first : first + count],
        panel_settings=PANEL_SETTINGS,
        energies=ENERGIES,
        pitch_angle=PITCH_ANGLE,
        vmin=VMIN,
        vmax=VMAX,
        along=ALONG,
        colormap=COLORMAP,
    )
    first += count

# Time axis, shown on the bottom panel only.
bottom = axes[-1]
locator = mdates.AutoDateLocator()
bottom.xaxis_date()
bottom.xaxis.set_major_locator(locator)
bottom.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
bottom.set_xlim(START_TIME, END_TIME)
bottom.set_xlabel("Time (UTC)")

if OUTPUT_PATH is not None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    logger.info(f"Saved the figure to {{OUTPUT_PATH}}.")

if SHOW:
    plt.show()
"""
    )
