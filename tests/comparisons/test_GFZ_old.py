# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import matplotlib.dates as mdates
import numpy as np
import pytest
from matplotlib import pyplot as plt
from swvo.io.dst import DSTOMNI
from swvo.io.kp import KpOMNI
from swvo.io.RBMDataSet import InstrumentEnum, MfmEnum, RBMDataSet

import el_paso as ep
from el_paso.recipes.rbsp import process_rbsp_hope_electrons

sat_str_list = ["a", "b"]
mag_field_list = ["TS04", "T89"]


@pytest.mark.parametrize("sat_str", sat_str_list)
@pytest.mark.parametrize("mag_field", mag_field_list)
@pytest.mark.visual
def test_gfz_old(sat_str: Literal["a", "b"], mag_field: Literal["T89", "TS04"]):
    start_time = datetime(2013, 3, 17, tzinfo=timezone.utc)
    end_time = start_time + timedelta(days=0, hours=23, minutes=59)

    Path("tests/comparisons/raw_data").mkdir(exist_ok=True)
    Path("tests/comparisons/processed_data").mkdir(exist_ok=True)

    # process_rbsp_hope_electrons(
    #     start_time,
    #     end_time,
    #     sat_str,
    #     mag_field,
    #     raw_data_path="tests/comparisons/raw_data",
    #     processed_data_path="tests/comparisons/processed_data",
    #     num_cores=64,
    # )

    match mag_field:
        case "T89":
            mfm_enum = MfmEnum.T89
        case "TS04":
            mfm_enum = MfmEnum.T04s

    rbsp_data = ep.dataset.GFZDataSet(
        start_time=start_time,
        end_time=end_time,
        saving_strategy=ep.saving_strategies.MonthlyRBStrategy(
            Path("tests/comparisons/processed_data/"), "RBSP", "rbspb", "hope", mag_field, ep.data_standards.GFZStandard(),
        ),
        verbose=True,
    )

    rbsp_data_server = RBMDataSet(
        start_time=start_time,
        end_time=end_time,
        folder_path=Path("/export/rbm6/data/data-dev/"),
        satellite="RBSPA",
        instrument=InstrumentEnum.HOPE,
        mfm=mfm_enum,
        verbose=True,
    )

    home_path = Path(os.environ["HOME"])
    kp_data = KpOMNI(home_path / ".el_paso/KpOmni").read(start_time, end_time, download=True)
    dst_data = DSTOMNI(home_path / ".el_paso/KpOmni").read(start_time, end_time, download=True)

    plt.style.use("seaborn-v0_8-bright")
    plt.rc("font", size=14)

    f, (ax_kp, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(19 / 1.5, 14), dpi=300)

    f.suptitle("Fluxes")

    ax_kp.stairs(kp_data["kp"][:-1], kp_data.index, color="k", linewidth=1.5)
    ax_kp.set_ylim(0, 9)
    ax_kp.set_ylabel("Kp")
    ax_kp.grid()
    ax_kp.set_xlim(start_time, end_time)
    ax_kp.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_kp.xaxis.get_major_locator()))

    ax_dst = ax_kp.twinx()
    ax_dst.plot(dst_data.index, dst_data["dst"], "b")
    ax_dst.set_ylabel("Dst [nT]", color="b")

    energy_idx = np.argmin(np.abs(rbsp_data.energy_channels[0, :] - 1e-3))

    ax1.plot(rbsp_data_server.datetime, np.log10(rbsp_data_server.Flux[:, energy_idx, -1]), "k")
    ax1.plot(rbsp_data.datetime, np.log10(rbsp_data.Flux[:, energy_idx, -1]), "r--")
    ax1.legend(["Old GFZ", "EL-PASO"])
    ax1.set_title("Energy = 1 keV")
    ax1.set_ylim(4, 9)
    ax1.set_xlim(start_time, end_time)
    ax1.grid()
    ax1.set_ylabel("log10 Flux\n[1/(sr cm^2 s keV)]")
    ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))

    energy_idx = np.argmin(np.abs(rbsp_data.energy_channels[0, :] - 10e-3))

    ax2.plot(rbsp_data_server.datetime, np.log10(rbsp_data_server.Flux[:, energy_idx, -1]), "k")
    ax2.plot(rbsp_data.datetime, np.log10(rbsp_data.Flux[:, energy_idx, -1]), "r--")
    ax2.legend(["Old GFZ", "EL-PASO"])
    ax2.set_title("Energy = 10 keV")
    ax2.set_ylim(4, 9)
    ax2.set_xlim(start_time, end_time)
    ax2.grid()
    ax2.set_ylabel("log10 Flux\n[1/(sr cm^2 s keV)]")
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))

    energy_idx = np.argmin(np.abs(rbsp_data.energy_channels[0, :] - 30e-3))

    ax3.plot(rbsp_data_server.datetime, np.log10(rbsp_data_server.Flux[:, energy_idx, -1]), "k")
    ax3.plot(rbsp_data.datetime, np.log10(rbsp_data.Flux[:, energy_idx, -1]), "r--")
    ax3.legend(["Old GFZ", "EL-PASO"])
    ax3.set_title("Energy = 30 keV")
    ax3.set_ylim(4, 8)
    ax3.set_xlim(start_time, end_time)
    ax3.grid()
    ax3.set_ylabel("log10 Flux\n[1/(sr cm^2 s keV)]")
    ax3.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax3.xaxis.get_major_locator()))

    plt.tight_layout()

    plt.savefig(f"{Path(__file__).parent / f'old_GFZ_test_{mag_field}_flux.png'}")

    f, (ax_kp, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(19 / 1.5, 14), dpi=300)

    f.suptitle("Adiabatic invariants and PSD")

    ax_kp.stairs(kp_data["kp"][:-1], kp_data.index, color="k", linewidth=1.5)
    ax_kp.set_ylim(0, 9)
    ax_kp.set_ylabel("Kp")
    ax_kp.grid()
    ax_kp.set_xlim(start_time, end_time)
    ax_kp.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_kp.xaxis.get_major_locator()))

    ax_dst = ax_kp.twinx()
    ax_dst.plot(dst_data.index, dst_data["dst"], "b")
    ax_dst.set_ylabel("Dst [nT]", color="b")

    mu_idx = 10
    k_idx = 3

    ax1.plot(rbsp_data_server.datetime, np.log10(rbsp_data_server.InvMu[:, mu_idx, k_idx]), "k")
    ax1.plot(rbsp_data.datetime, np.log10(rbsp_data.InvMu[:, mu_idx, k_idx]), "r--")
    ax1.legend(["Old GFZ", "EL-PASO"])
    ax1.set_title(r"Invariant $\mu$")
    ax1.set_xlim(start_time, end_time)
    ax1.grid()
    ax1.set_ylabel(r"log10 $\mu$ [MeV/G]")
    ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))

    ax2.plot(rbsp_data_server.datetime, np.log10(rbsp_data_server.InvK[:, k_idx]), "k")
    ax2.plot(rbsp_data.datetime, np.log10(rbsp_data.InvK[:, k_idx]), "r--")
    ax2.legend(["Old GFZ", "EL-PASO"])
    ax2.set_title("Invariant K")
    ax2.set_xlim(start_time, end_time)
    ax2.grid()
    ax2.set_ylabel("log10 K [G^0.5 R_E]")
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))

    target_mu = 0.1
    target_k = 0.3

    psd_data_server = rbsp_data_server.interp_psd(target_mu, target_k, "TargetPairs")
    psd_new = rbsp_data_server.interp_psd(target_mu, target_k, "TargetPairs")

    ax3.plot(rbsp_data_server.datetime, np.log10(psd_data_server), "k")
    ax3.plot(rbsp_data.datetime, np.log10(psd_new), "r--")

    ax3.legend(["Old GFZ", "EL-PASO"])
    ax3.set_title(r"Phase space density for $\mu = 0.1$ MeV/G and $K = 0.3$ G^0.5 R_E")
    ax3.set_xlim(start_time, end_time)
    ax3.grid()
    ax3.set_ylabel("log10 PSD\n[s^3/(m^6 kg^3)]")
    ax3.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax3.xaxis.get_major_locator()))

    plt.tight_layout()

    plt.savefig(f"{Path(__file__).parent / f'old_GFZ_test_{mag_field}_psd.png'}")
