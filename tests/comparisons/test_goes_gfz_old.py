# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta, timezone
from pathlib import Path

import el_paso as ep
import matplotlib.dates as mdates
import numpy as np
import pytest
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from swvo.io.RBMDataSet import InstrumentEnum, MfmEnum, RBMDataSet, Satellite

DATA_PATH = Path(__file__).parents[1] / "system" / "data"

# the old GFZ processing labels its 5 minute bins 150 s later than EL-PASO
OLD_TIME_OFFSET = timedelta(seconds=150)
TIME_MATCH_TOLERANCE_S = 10

TARGET_ENERGIES_MEV = [0.075, 0.275, 0.8, 2.0]


def _match_old_to_new(new_dt: list[datetime], old_dt: list[datetime]) -> NDArray[np.intp]:
    """Return, for every new time step, the index of the matching old time step (-1 if there is none)."""
    new_t = np.array([t.timestamp() for t in new_dt])
    old_t = np.array([(t - OLD_TIME_OFFSET).timestamp() for t in old_dt])

    idx = np.clip(np.searchsorted(old_t, new_t), 1, len(old_t) - 1)
    idx = np.where(np.abs(new_t - old_t[idx - 1]) < np.abs(new_t - old_t[idx]), idx - 1, idx)

    return np.where(np.abs(new_t - old_t[idx]) <= TIME_MATCH_TOLERANCE_S, idx, -1)


def _old_at_new_pitch_angle(
    old_var: NDArray[np.float64],
    old_alpha: NDArray[np.float64],
    new_alpha: NDArray[np.float64],
    idx_old: NDArray[np.intp],
    *,
    log: bool,
) -> NDArray[np.float64]:
    """Interpolate the old (time, alpha) data linearly in pitch angle onto the new pitch angle of each time step."""
    values = np.log10(np.where(old_var > 0, old_var, np.nan)) if log else old_var

    out = np.full(len(idx_old), np.nan)
    for it, io in enumerate(idx_old):
        if io >= 0 and np.isfinite(new_alpha[it]):
            out[it] = np.interp(new_alpha[it], old_alpha[io], values[io], left=np.nan, right=np.nan)

    return 10**out if log else out


@pytest.mark.visual
def test_goes15_maged_and_epead_gfz_old(tmp_path: Path) -> None:
    start_time = datetime(2017, 9, 6, tzinfo=timezone.utc)
    end_time = start_time + timedelta(days=3, seconds=-1)

    ep.recipes.goes.process_goes_eps_maged_and_epead(
        start_time,
        end_time,
        satellite="goes15",
        mag_field="T89",
        raw_data_path=DATA_PATH / "raw" / "GOES",
        processed_data_path=tmp_path,
        num_cores=64,
    )

    goes_new = ep.dataset.GFZDataSet(
        ep.recipes.goes.goes_eps_maged_and_epead_netcdf_strategy(tmp_path, "T89", "goes15"),
        start_time=start_time,
        end_time=end_time,
    )

    goes_old = RBMDataSet(
        start_time=start_time,
        end_time=end_time,
        folder_path=DATA_PATH / "GFZ_old",
        satellite=Satellite("GOES15", "GOES"),
        instrument=InstrumentEnum.MAGEDandEPEAD,
        mfm=MfmEnum.T89,
        verbose=False,
        preferred_extension="mat",
    )

    idx_old = _match_old_to_new(goes_new.datetime, goes_old.datetime)

    # the comparison is done at the largest local pitch angle measured by EL-PASO
    new_alpha = np.rad2deg(np.asarray(goes_new.alpha_local, dtype=np.float64)[:, -1])
    old_alpha = np.rad2deg(np.asarray(goes_old.alpha_local, dtype=np.float64))
    new_energies = np.asarray(goes_new.energy_channels, dtype=np.float64)[0]
    old_energies = np.asarray(goes_old.energy_channels, dtype=np.float64)

    kp, kp_time = ep.load_indices_solar_wind_parameters(start_time, end_time, ["Kp"])["Kp"]
    kp_datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in kp_time.get_data(ep.units.posixtime)]

    plt.style.use("seaborn-v0_8-bright")
    plt.rc("font", size=14)

    f, axs = plt.subplots(len(TARGET_ENERGIES_MEV) + 2, 1, figsize=(19 / 1.5, 18), dpi=300, sharex=True)
    f.suptitle(
        f"GOES-15 MAGED and EPEAD, T89, local pitch angle = largest measured ({np.nanmedian(new_alpha):.0f}° median)"
    )

    axs[0].plot(kp_datetimes, kp.get_data(), "k", drawstyle="steps-post")
    axs[0].set_ylim(0, 9)
    axs[0].set_ylabel("Kp")

    for ax, target_energy in zip(axs[1:-1], TARGET_ENERGIES_MEV, strict=True):
        ie_new = int(np.argmin(np.abs(new_energies - target_energy)))
        ie_old = int(np.argmin(np.abs(old_energies - target_energy)))

        old_flux = _old_at_new_pitch_angle(
            np.asarray(goes_old.Flux, dtype=np.float64)[:, ie_old, :], old_alpha, new_alpha, idx_old, log=True
        )
        new_flux = np.asarray(goes_new.Flux, dtype=np.float64)[:, ie_new, -1]

        ax.semilogy(goes_new.datetime, old_flux, "k")
        ax.semilogy(goes_new.datetime, new_flux, "r--")
        ax.set_title(f"Energy: old GFZ {old_energies[ie_old]:.3f} MeV, EL-PASO {new_energies[ie_new]:.3f} MeV")
        ax.set_ylabel("Flux\n[1/(sr cm^2 s keV)]")
        ax.legend(["Old GFZ", "EL-PASO"])

    old_lstar = _old_at_new_pitch_angle(
        np.asarray(goes_old.Lstar, dtype=np.float64), old_alpha, new_alpha, idx_old, log=False
    )
    axs[-1].plot(goes_new.datetime, old_lstar, "k")
    axs[-1].plot(goes_new.datetime, np.asarray(goes_new.Lstar, dtype=np.float64)[:, -1], "r--")
    axs[-1].set_title("L*")
    axs[-1].set_ylabel("L*")
    axs[-1].legend(["Old GFZ", "EL-PASO"])

    for ax in axs:
        ax.grid()
        ax.set_xlim(start_time, end_time)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "goes15_maged_and_epead_gfz_old_T89.png")
