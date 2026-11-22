# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Jaskirat Singh
#
# SPDX-License-Identifier: Apache-2.0

"""Visual validation: the perp energy spectrogram of ELFIN EPD electrons processed by EL-PASO should look
like the one made directly from the raw CDF file.

Perp is the mean flux over the pitch-angle bins outside the loss cone and the anti-loss cone.
"""

import warnings
from datetime import datetime, timezone
from pathlib import Path

import cdflib
import el_paso as ep
import matplotlib.dates as mdates
import numpy as np
import pytest
from astropy import units as u
from el_paso.dataset import DataSet
from el_paso.recipes.elfin import elfin_epd_electron_strategy, process_elfin_epd_electrons
from matplotlib import pyplot as plt

SATELLITE = "elb"
START = datetime(2020, 9, 29, 6, 56, tzinfo=timezone.utc)
END = datetime(2020, 9, 29, 7, 2, tzinfo=timezone.utc)


def _perp(flux, pa, lc):
    """Mean flux (time, energy, pa) over the pitch angles between lc and 180 - lc -> (time, energy)."""
    mask = np.isfinite(pa) & (pa > lc[:, None]) & (pa < 180 - lc[:, None])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.where(mask[:, None, :], flux, np.nan), axis=2)


def _timenum(posix):
    return mdates.date2num([datetime.fromtimestamp(t, tz=timezone.utc) for t in posix])


@pytest.mark.visual
@pytest.mark.network
@pytest.mark.usefixtures("enable_downloads")
def test_elfin_epd_perp_comparison(tmp_path: Path) -> None:
    process_elfin_epd_electrons(
        start_time=START,
        end_time=END,
        satellite=SATELLITE,
        raw_data_path=tmp_path,
        processed_data_path=tmp_path,
        num_cores=32,
    )

    ds = DataSet(
        saving_strategy=elfin_epd_electron_strategy(tmp_path, "Dip", SATELLITE),
        start_time=START,
        end_time=END,
        preferred_extension="nc",
    )
    pa = np.rad2deg(ds.alpha_local)
    ok = ~np.all(np.isnan(pa), axis=1)
    lc = np.rad2deg(ds.alpha_lc)
    flux = np.where(ds.Flux < 0, np.nan, ds.Flux)  # zeros are kept
    ep_energy = ds.energy_channels * 1000  # MeV -> keV
    if ep_energy.ndim == 2:
        ep_energy = np.nanmedian(ep_energy, axis=0)
    ep_time = _timenum(u.Quantity(ds.time, ep.units.datenum).to_value(ep.units.posixtime)[ok])
    ep_flux = _perp(flux[ok], pa[ok], lc[ok])

    path = sorted((tmp_path / "ELFIN" / SATELLITE / f"{START:%Y}" / f"{START:%m}").glob("*_l2_epdef_*.cdf"))[-1]
    cdf = cdflib.CDF(str(path))
    t = ep.Variable(data=np.asarray(cdf.varget(f"{SATELLITE}_pef_hs_time")), original_unit=ep.units.tt2000)
    raw_time = _timenum(t.get_data(ep.units.posixtime))
    raw_energy = np.asarray(cdf.varget(f"{SATELLITE}_pef_energies_mean"), dtype=float)
    raw_pa = np.asarray(cdf.varget(f"{SATELLITE}_pef_hs_epa_spec"), dtype=float)
    raw_lc = np.asarray(cdf.varget(f"{SATELLITE}_pef_hs_LCdeg"), dtype=float)
    raw_flux = np.asarray(cdf.varget(f"{SATELLITE}_pef_hs_Epat_eflux"), dtype=float)
    raw_flux = np.swapaxes(raw_flux, 1, 2)  # file: (time, pa, energy) -> (time, energy, pa)
    keep = (raw_time >= mdates.date2num(START)) & (raw_time <= mdates.date2num(END)) & ~np.all(np.isnan(raw_pa), axis=1)
    raw_flux = _perp(np.where(raw_flux[keep] < 0, np.nan, raw_flux[keep]), raw_pa[keep], raw_lc[keep])
    raw_time = raw_time[keep]

    log = lambda v: np.log10(np.where(v > 0, v, np.nan))  # noqa: E731

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, sharey=True)
    fig.suptitle(f"ELFIN-{SATELLITE[-1].upper()} EPD, perp, {START:%Y-%m-%d}")
    for ax, x, y, v, title in (
        (axes[0], raw_time, raw_energy, raw_flux / 1000, "raw CDF (Epat_eflux)"),
        (axes[1], ep_time, ep_energy, ep_flux / 1000, "EL-PASO (FEDU)"),
    ):
        mesh = ax.pcolormesh(x, y, np.ma.masked_invalid(log(v).T), shading="nearest", cmap="turbo", vmin=1, vmax=7)
        ax.set_yscale("log")
        ax.set_ylabel("Energy (keV)")
        ax.set_title(title, fontsize=10)
        fig.colorbar(mesh, ax=ax, pad=0.02).set_label(
            r"log$_{10}$(Flux [cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$])", fontsize=8
        )
    axes[1].xaxis_date()
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[1].set_xlabel("Time (UTC)")

    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "elfin_epd_perp_comparison.png", dpi=150)
