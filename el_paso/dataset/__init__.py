# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import lazy_loader as lazy

# Resolved on first access (SPEC 1), so that `DataSet` does not also pay for the
# model-grid binning helpers and their richpool/tqdm/icecream stack.
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "bin_and_interpolate_to_model_grid": [
            "DebugPlotSettings",
            "bin_and_interpolate_to_model_grid",
            "plot_debug_figures",
            "plot_debug_figures_plasmasphere",
        ],
        "dataset": ["DataSet"],
        "dataset_implementations": ["GFZDataSet", "PRBEMDataSet"],
        "metadata": ["DatasetMetadata", "GFZMetaData", "PRBEMMetaData"],
    },
)

if TYPE_CHECKING:
    from el_paso.dataset.bin_and_interpolate_to_model_grid import (
        DebugPlotSettings,
        bin_and_interpolate_to_model_grid,
        plot_debug_figures,
        plot_debug_figures_plasmasphere,
    )
    from el_paso.dataset.dataset import DataSet
    from el_paso.dataset.dataset_implementations import GFZDataSet, PRBEMDataSet
    from el_paso.dataset.metadata import DatasetMetadata, GFZMetaData, PRBEMMetaData

    __all__ = [
        "DataSet",
        "DatasetMetadata",
        "DebugPlotSettings",
        "GFZDataSet",
        "GFZMetaData",
        "PRBEMDataSet",
        "PRBEMMetaData",
        "bin_and_interpolate_to_model_grid",
        "plot_debug_figures",
        "plot_debug_figures_plasmasphere",
    ]
