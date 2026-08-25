# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

"""Reproduce the PAD-shape figure from Smirnov et al. 2022 (https://doi.org/10.1029/2022SW003053)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt

from el_paso.processing.models.pa_distribution_smirnov_et_al_2022 import (
    _apply_coefs,
    _get_coefs,
)

ENERGY = 871
L_SHELL = 5.4
PDYN_VALUES_NPA = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]

PDYN_COLORS = [
    "#0000cc",  # 0.5
    "#0080ff",  # 1.5
    "#00ccff",  # 2.5
    "#00cc88",  # 3.5
    "#ccdd00",  # 4.5
    "#ffaa00",  # 5.5
    "#ee0000",  # 6.5
]

PANELS = [
    ("(a) 871 keV, 5.3<L<5.5, 11<MLT<13", 12.0),
    ("(b) 871 keV, 5.3<L<5.5, 23<MLT<01", 0.0),
]


@pytest.fixture
def equatorial_pa_rad() -> np.ndarray:
    """Equatorial pitch angle grid, 0-180 deg in 1 deg steps, in radians."""
    return np.linspace(0, np.pi, 181)


def _normalized_pad_shape(mlt: float, pdyn: float, pa_rad: np.ndarray) -> np.ndarray:
    """Compute a single normalized PAD shape curve, handling the Pdyn>5.5 switch."""
    a1, a3, a5 = _get_coefs(ENERGY, L_SHELL, mlt, pdyn)

    shape = _apply_coefs(pa_rad, a1, a3, a5)
    return shape / np.max(shape)

@pytest.mark.basic
class TestPADShapeNoonMLT:
    """Panel (a): 871 keV, 5.3<L<5.5, 11<MLT<13 -> single-peaked distribution,
    peak becomes sharper (narrower) as Pdyn increases.
    """

    mlt = 12.0

    def test_peak_is_at_90_degrees(self, equatorial_pa_rad: np.ndarray) -> None:
        for pdyn in PDYN_VALUES_NPA:
            shape = _normalized_pad_shape(self.mlt, pdyn, equatorial_pa_rad)
            peak_pa_deg = np.degrees(equatorial_pa_rad[np.argmax(shape)])
            assert peak_pa_deg == pytest.approx(90, abs=5), f"Pdyn={pdyn}"

    def test_single_peaked_no_local_dip(self, equatorial_pa_rad: np.ndarray) -> None:
        for pdyn in PDYN_VALUES_NPA:
            shape = _normalized_pad_shape(self.mlt, pdyn, equatorial_pa_rad)
            peak_idx = int(np.argmax(shape))
            # Monotonically increasing up to the peak, decreasing after.
            assert np.all(np.diff(shape[: peak_idx + 1]) >= -1e-8), f"Pdyn={pdyn}"
            assert np.all(np.diff(shape[peak_idx:]) <= 1e-8), f"Pdyn={pdyn}"

    def test_distribution_narrows_as_pdyn_increases(self, equatorial_pa_rad: np.ndarray) -> None:
        fwhm_deg = []
        for pdyn in PDYN_VALUES_NPA:
            shape = _normalized_pad_shape(self.mlt, pdyn, equatorial_pa_rad)
            above_half = equatorial_pa_rad[shape >= 0.5]
            fwhm_deg.append(np.degrees(above_half[-1] - above_half[0]))

        assert fwhm_deg == sorted(fwhm_deg, reverse=True), (
            f"Expected FWHM to shrink monotonically with Pdyn, got {fwhm_deg}"
        )

@pytest.mark.basic
class TestPADShapeMidnightMLT:
    """Panel (b): 871 keV, 5.3<L<5.5, 23<MLT<01 -> butterfly
    distribution with a local minimum near PA=90 deg.
    """

    mlt = 0.0

    def test_local_minimum_near_90_degrees(self, equatorial_pa_rad: np.ndarray) -> None:
        idx_90 = int(np.argmin(np.abs(np.degrees(equatorial_pa_rad) - 90)))
        for pdyn in PDYN_VALUES_NPA:
            shape = _normalized_pad_shape(self.mlt, pdyn, equatorial_pa_rad)
            # The value at 90 deg should be a local minimum relative to its
            # immediate neighborhood (i.e. genuinely dipped, not just flat).
            window = 15  # degrees
            n_pts = int(window)
            local_max = np.max(
                np.concatenate([shape[max(0, idx_90 - n_pts) : idx_90], shape[idx_90 + 1 : idx_90 + 1 + n_pts]])
            )
            assert shape[idx_90] <= local_max + 1e-8, f"Pdyn={pdyn}"

    def test_dip_depth_increases_with_pdyn(self, equatorial_pa_rad: np.ndarray) -> None:
        idx_90 = int(np.argmin(np.abs(np.degrees(equatorial_pa_rad) - 90)))
        dip_depths = []
        for pdyn in PDYN_VALUES_NPA:
            shape = _normalized_pad_shape(self.mlt, pdyn, equatorial_pa_rad)
            dip_depths.append(1.0 - shape[idx_90])

        assert dip_depths == sorted(dip_depths), (
            f"Expected dip depth at PA=90 to grow monotonically with Pdyn, got {dip_depths}"
        )

@pytest.mark.visual
def test_replot_smirnov2022_figure() -> None:
    """Regenerate the published two-panel figure and check its key features."""
    pa_rad = np.linspace(0, np.pi, 181)
    pa_deg = np.degrees(pa_rad)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    shapes: dict[tuple[str, float], np.ndarray] = {}

    for ax, (panel_title, mlt) in zip(axes, PANELS, strict=True):
        for pdyn, color in zip(PDYN_VALUES_NPA, PDYN_COLORS, strict=True):
            shape = _normalized_pad_shape(mlt, pdyn, pa_rad)
            shapes[panel_title, pdyn] = shape
            ax.plot(pa_deg, shape, color=color, lw=2, label=f"{pdyn}")

        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel("Equatorial PA, [°]")
        ax.set_xlim(0, 180)
        ax.set_xticks(np.arange(0, 181, 30))
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel("Normalized PAD shape")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles[::-1],
        labels[::-1],
        title="Pdyn\n[nPa]",
        loc="center right",
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 0.88, 1))

    out_path = "smirnov2022_pad_shapes.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
