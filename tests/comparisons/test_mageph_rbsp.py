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
from astropy import units as u
from matplotlib import pyplot as plt
from swvo.io.dst import DSTOMNI
from swvo.io.kp import KpOMNI

import el_paso as ep
from el_paso import InstrumentEnum, MfmEnum, RBMDataSet
from examples.VanAllenProbes.process_hope_electrons import process_hope_electrons

# ruff: noqa: PLR2004

sat_str_list = ["a", "b"]
mag_field_list = ["TS04", "T89"]


@pytest.mark.parametrize("sat_str", sat_str_list)
@pytest.mark.parametrize("mag_field", mag_field_list)
@pytest.mark.visual
def test_mageph_rbsp(sat_str: Literal["a", "b"], mag_field: Literal["T89", "TS04"]):  # noqa: PLR0915
    # process Lstar using el paso
    start_time = datetime(2013, 3, 17, tzinfo=timezone.utc)
    end_time = start_time + timedelta(days=0, hours=23, minutes=59)

    Path("tests/comparisons/raw_data").mkdir(exist_ok=True)
    Path("tests/comparisons/processed_data").mkdir(exist_ok=True)

    process_hope_electrons(
        start_time,
        end_time,
        sat_str,
        "IRBEM/libirbem.so",
        mag_field,
        raw_data_path="tests/comparisons/raw_data",
        processed_data_path="tests/comparisons/processed_data",
        num_cores=12,
    )

    match mag_field:
        case "T89":
            mfm_enum = MfmEnum.T89
        case "TS04":
            mfm_enum = MfmEnum.T04s

    rbsp_data = RBMDataSet(
        start_time=start_time,
        end_time=end_time,
        folder_path=Path("tests/comparisons/processed_data/"),
        satellite="RBSPA",
        instrument=InstrumentEnum.HOPE,
        mfm=mfm_enum,
        verbose=True,
    )

    # load from mageph data
    match mag_field:
        case "T89":
            mag_field_str_data = "T89D"
        case "TS04":
            mag_field_str_data = "TS04D"
    file_name_stem = "rbsp" + sat_str + "_def_MagEphem_" + mag_field_str_data + "_YYYYMMDD_.{6}.h5"

    ep.download(
        start_time,
        end_time,
        save_path="tests/comparisons/raw_data",
        download_url=f"https://rbsp-ect.newmexicoconsortium.org/data_pub/rbsp{sat_str}/MagEphem/definitive/YYYY/",
        file_name_stem=file_name_stem,
        file_cadence="daily",
        method="request",
        skip_existing=True,
    )

    extraction_infos = [
        ep.ExtractionInfo(
            result_key="Epoch",
            name_or_column="IsoTime",
            unit=u.dimensionless_unscaled,
        ),
        ep.ExtractionInfo(
            result_key="Lstar",
            name_or_column="Lstar",
            unit=u.dimensionless_unscaled,
        ),
        ep.ExtractionInfo(
            result_key="Alpha_eq",
            name_or_column="Alpha",
            unit=u.dimensionless_unscaled,
        ),
        ep.ExtractionInfo(
            result_key="Kp",
            name_or_column="Kp",
            unit=u.dimensionless_unscaled,
        ),
    ]

    variables = ep.extract_variables_from_files(
        start_time, end_time, "daily", "tests/comparisons/raw_data", file_name_stem, extraction_infos
    )

    timestamps = [
        datetime.fromisoformat(str(t)[2:-2]).replace(tzinfo=timezone.utc) for t in variables["Epoch"].get_data()
    ]

    el_paso_timestamps = rbsp_data.datetime

    variables["Lstar"].apply_thresholds_on_data(lower_threshold=0.0)

    home_path = Path(os.environ["HOME"])
    kp_data = KpOMNI(home_path / ".el_paso/KpOmni").read(start_time, end_time, download=True)
    dst_data = DSTOMNI(home_path / ".el_paso/KpOmni").read(start_time, end_time, download=True)

    plt.style.use("seaborn-v0_8-bright")
    plt.rc("font", size=14)

    f, (ax_kp, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(19 / 1.5, 14), dpi=300)

    f.suptitle(f"Magnetic field: {mag_field}")

    ax_kp.stairs(kp_data["kp"][:-1], kp_data.index, color="k", linewidth=1.5)
    ax_kp.set_ylim(0, 9)
    ax_kp.set_ylabel("Kp")
    ax_kp.grid()
    ax_kp.set_xlim(timestamps[0], timestamps[-1])
    ax_kp.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_kp.xaxis.get_major_locator()))

    ax_dst = ax_kp.twinx()
    ax_dst.plot(dst_data.index, dst_data["dst"], "b")
    ax_dst.set_ylabel("Dst [nT]", color="b")

    el_paso_Lstar = []
    for it, _ in enumerate(el_paso_timestamps):
        el_paso_Lstar.append(np.interp(20, np.rad2deg(rbsp_data.alpha_local[it, :]), rbsp_data.Lstar[it, :]))

    deg_idx = np.argwhere(variables["Alpha_eq"].get_data() == 20)

    ax1.plot(timestamps, variables["Lstar"].get_data()[:, deg_idx[0]], "k")
    ax1.plot(el_paso_timestamps, el_paso_Lstar, "r--")
    ax1.legend(["ECT Team", "EL-PASO"])
    ax1.set_title("Local pitch angle = 20°")
    ax1.set_ylim(1, 7)
    ax1.set_xlim(timestamps[0], timestamps[-1])
    ax1.grid()
    ax1.set_ylabel("L*")
    ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))

    variables["Lstar"].apply_thresholds_on_data(lower_threshold=0.0)

    el_paso_Lstar = []
    for it, _ in enumerate(el_paso_timestamps):
        el_paso_Lstar.append(np.interp(50, np.rad2deg(rbsp_data.alpha_local[it, :]), rbsp_data.Lstar[it, :]))

    deg_idx = np.argwhere(variables["Alpha_eq"].get_data() == 50)

    ax2.plot(timestamps, variables["Lstar"].get_data()[:, deg_idx[0]], "k")
    ax2.plot(el_paso_timestamps, el_paso_Lstar, "r--")
    ax2.legend(["ECT Team", "EL-PASO"])
    ax2.set_title("Local pitch angle = 50°")
    ax2.set_ylim(1, 7)
    ax2.set_xlim(timestamps[0], timestamps[-1])
    ax2.grid()
    ax2.set_ylabel("L*")
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))

    variables["Lstar"].apply_thresholds_on_data(lower_threshold=0.0)

    el_paso_Lstar: list[float] = []
    for it, _ in enumerate(el_paso_timestamps):
        el_paso_Lstar.append(np.interp(70, np.rad2deg(rbsp_data.alpha_local[it, :]), rbsp_data.Lstar[it, :]))

    deg_idx = np.argwhere(variables["Alpha_eq"].get_data() == 70)

    ax3.plot(timestamps, variables["Lstar"].get_data()[:, deg_idx[0]], "k")
    ax3.plot(el_paso_timestamps, el_paso_Lstar, "r--")
    ax3.legend(["ECT Team", "EL-PASO"])
    ax3.set_title("Local pitch angle = 70°")
    ax3.set_ylim(1, 7)
    ax3.set_xlim(timestamps[0], timestamps[-1])
    ax3.grid()
    ax3.set_ylabel("L*")
    ax3.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax3.xaxis.get_major_locator()))

    plt.tight_layout()

    plt.savefig(f"{Path(__file__).parent / f'mag_eph_test_{mag_field}.png'}")
