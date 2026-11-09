# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""Visual validation: the energy spectra of Arase MEP-e, HEP-L, HEP-H and XEP should join up smoothly.

The instruments have very different data coverage (e.g. HEP has long gaps, while XEP covers most of the
orbit, including long stretches outside the radiation belts). Statistics over time are therefore only
computed over the time bins in which all channels of all instruments provide valid data; otherwise the
spectra would be medians over different parts of the orbit and would not be comparable.
"""

import functools
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from el_paso.dataset import DataSet
from el_paso.recipes.arase import (
    arase_hep_netcdf_strategy,
    arase_mepe_netcdf_strategy,
    arase_xep_strategy,
    process_arase_hep_h,
    process_arase_hep_l,
    process_arase_mepe,
    process_arase_xep,
)
from matplotlib import pyplot as plt
from numpy.typing import NDArray

_DATES = [
    datetime(2017, 9, 8, tzinfo=timezone.utc),  # geomagnetic storm
    datetime(2018, 4, 10, tzinfo=timezone.utc),
]
_MAG_FIELD = "T89"

# kept outside of the repository, and fixed so that downloads are reused between runs
_WORK_DIR = Path(tempfile.gettempdir()) / "el_paso_visual_arase_spectrum"
_RAW_DATA_PATH = _WORK_DIR / "raw"
_PROCESSED_DATA_PATH = _WORK_DIR / "processed"

_INSTRUMENTS = {
    "MEP-e": {"color": "tab:blue", "marker": "o"},
    "HEP-L": {"color": "tab:green", "marker": "s"},
    "HEP-H": {"color": "tab:red", "marker": "^"},
    "XEP": {"color": "tab:purple", "marker": "v"},
}


@functools.cache
def _load_instrument_datasets(start_time: datetime) -> dict[str, DataSet]:
    end_time = start_time + timedelta(days=1, seconds=-1)

    for path in (_RAW_DATA_PATH, _PROCESSED_DATA_PATH):
        path.mkdir(parents=True, exist_ok=True)

    common_kwargs = {
        "start_time": start_time,
        "end_time": end_time,
        "mag_field": _MAG_FIELD,
        "raw_data_path": _RAW_DATA_PATH,
        "processed_data_path": _PROCESSED_DATA_PATH,
        "save_strategy": "netcdf",
    }

    process_arase_mepe(**common_kwargs)
    process_arase_hep_l(**common_kwargs)
    process_arase_hep_h(**common_kwargs)
    process_arase_xep(**common_kwargs)

    strategies = {
        "MEP-e": arase_mepe_netcdf_strategy(_PROCESSED_DATA_PATH, _MAG_FIELD),
        "HEP-L": arase_hep_netcdf_strategy(_PROCESSED_DATA_PATH, _MAG_FIELD, "L"),
        "HEP-H": arase_hep_netcdf_strategy(_PROCESSED_DATA_PATH, _MAG_FIELD, "H"),
        "XEP": arase_xep_strategy(_PROCESSED_DATA_PATH, _MAG_FIELD),
    }

    return {name: DataSet(strategy, start_time, end_time, verbose=False) for name, strategy in strategies.items()}


def _flux_at_pitch_angle(
    dataset: DataSet, target_pitch_angle: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Return energy [MeV], FEDU (time, energy) at the closest local pitch angle, and that pitch angle [deg]."""
    energy = dataset.get_by_internal_name("Energy_FEDU").astype(np.float64)
    flux = dataset.get_by_internal_name("FEDU").astype(np.float64)
    alpha = dataset.get_by_internal_name("Alpha").astype(np.float64)

    # energies (MeV) and pitch angles (rad) are repeated in time
    energy_mev = np.nanmedian(energy, axis=0)
    alpha_deg = np.rad2deg(np.nanmedian(alpha, axis=0))

    pa_idx = int(np.argmin(np.abs(alpha_deg - target_pitch_angle)))

    return energy_mev, flux[:, :, pa_idx], float(alpha_deg[pa_idx])


def _spectrum_statistics(
    flux_at_pa: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return median, 25th and 75th percentile of the flux over time."""
    with np.errstate(all="ignore"):
        median = np.nanmedian(flux_at_pa, axis=0)
        q25, q75 = np.nanpercentile(flux_at_pa, [25, 75], axis=0)

    return median, q25, q75


@pytest.mark.visual
@pytest.mark.network
@pytest.mark.parametrize("start_time", _DATES, ids=[f"{date:%Y%m%d}" for date in _DATES])
@pytest.mark.parametrize("target_pitch_angle", [30, 70])
def test_arase_energy_spectrum_combined(start_time: datetime, target_pitch_angle: float) -> None:
    datasets = _load_instrument_datasets(start_time)

    fluxes = {name: _flux_at_pitch_angle(dataset, target_pitch_angle) for name, dataset in datasets.items()}

    # Only compare time bins in which every channel of every instrument has valid data. Requiring just
    # any valid channel is not enough: HEP fluxes at or below background are set to NaN per channel,
    # so the high-energy HEP channels would still only sample the radiation belts while XEP also
    # samples the low fluxes outside of them.
    num_times = {flux_at_pa.shape[0] for _, flux_at_pa, _ in fluxes.values()}
    assert len(num_times) == 1, f"instruments are not on a common time grid: {num_times}"

    common_times = np.logical_and.reduce(
        [np.all(np.isfinite(flux_at_pa), axis=1) for _, flux_at_pa, _ in fluxes.values()]
    )
    num_common = int(common_times.sum())
    assert num_common > 0, "no time bins in which all instruments have valid data"

    _, ax = plt.subplots(figsize=(9, 6))

    for name, (energy_mev, flux_at_pa, used_alpha) in fluxes.items():
        median, q25, q75 = _spectrum_statistics(flux_at_pa[common_times])

        assert energy_mev.size > 0, f"{name}: no energies loaded"
        assert np.any(np.isfinite(median)), f"{name}: flux is entirely NaN"

        style = _INSTRUMENTS[name]
        ax.fill_between(energy_mev, q25, q75, color=style["color"], alpha=0.15, linewidth=0)
        ax.plot(
            energy_mev,
            median,
            color=style["color"],
            marker=style["marker"],
            label=f"{name} ({used_alpha:.0f}° local PA)",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel(r"Electron flux (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ keV$^{-1}$)")
    ax.set_title(
        f"Arase energy spectra, median and IQR on {start_time:%Y-%m-%d} (target PA {target_pitch_angle}°)\n"
        f"over the {num_common} of {common_times.size} time bins with data in all channels of all instruments"
    )
    ax.grid(visible=True, which="both", alpha=0.3)
    ax.legend()

    plt.savefig(
        Path(__file__).parent / f"arase_energy_spectrum_{start_time:%Y%m%d}_pa{target_pitch_angle}.png", dpi=150
    )
    plt.close()
