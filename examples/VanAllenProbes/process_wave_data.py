# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Alwin Roy
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.constants import e, m_e

sys.path.append(str(Path(__file__).resolve().parents[2]))

import el_paso as ep
from el_paso.units import RE
from el_paso.variable import Variable

SATELLITE = "a"
START_TIME = datetime(2018, 4, 12, 0, 0, 0, tzinfo=timezone.utc)
END_TIME = datetime(2018, 4, 12, 23, 59, 59, tzinfo=timezone.utc)

RAW_DATA_PATH = Path("./wave_data/raw")
PROCESSED_DATA_PATH = Path("./wave_data/processed")
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

WFR_DOWNLOAD_URL = "https://cdaweb.gsfc.nasa.gov/pub/data/rbsp/rbspa/l2/emfisis/wfr/spectral-matrix-diagonal/YYYY/"
WFR_FILE_STEM = "rbsp-a_wfr-spectral-matrix-diagonal_emfisis-l2_YYYYMMDD_.{6}.cdf"

DENSITY_DOWNLOAD_URL = "https://cdaweb.gsfc.nasa.gov/pub/data/rbsp/rbspa/l4/emfisis/density/YYYY/"
DENSITY_FILE_STEM = "rbsp-a_density_emfisis-l4_YYYYMMDD_.{7}.cdf"

MAG_DOWNLOAD_URL = "https://cdaweb.gsfc.nasa.gov/pub/data/rbsp/rbspa/l3/emfisis/magnetometer/4sec/sm/YYYY/"
MAG_FILE_STEM = "rbsp-a_magnetometer_4sec-sm_emfisis-l3_YYYYMMDD_.{6}.cdf"

WNA_DOWNLOAD_URL = "https://cdaweb.gsfc.nasa.gov/pub/data/rbsp/rbspa/l4/emfisis/wna-survey-sheath-corrected-e/YYYY/"
WNA_FILE_STEM = "rbsp-a_wna-survey-sheath-corrected-e_emfisis-l4_YYYYMMDD_.{6}.cdf"


def download_day_data(start_time: datetime, end_time: datetime) -> None:
    ep.download(
        start_time,
        end_time,
        save_path=RAW_DATA_PATH,
        download_url=WFR_DOWNLOAD_URL,
        file_name_stem=WFR_FILE_STEM,
        file_cadence="daily",
        method="request",
        skip_existing=True,
    )
    ep.download(
        start_time,
        end_time,
        save_path=RAW_DATA_PATH,
        download_url=DENSITY_DOWNLOAD_URL,
        file_name_stem=DENSITY_FILE_STEM,
        file_cadence="daily",
        method="request",
        skip_existing=True,
    )
    ep.download(
        start_time,
        end_time,
        save_path=RAW_DATA_PATH,
        download_url=MAG_DOWNLOAD_URL,
        file_name_stem=MAG_FILE_STEM,
        file_cadence="daily",
        method="request",
        skip_existing=True,
    )
    ep.download(
        start_time,
        end_time,
        save_path=RAW_DATA_PATH,
        download_url=WNA_DOWNLOAD_URL,
        file_name_stem=WNA_FILE_STEM,
        file_cadence="daily",
        method="request",
        skip_existing=True,
    )


def load_density_data(start_time: datetime, end_time: datetime) -> dict[str, ep.Variable]:
    extraction_infos = [
        ep.ExtractionInfo(result_key="Epoch", name_or_column="Epoch", unit=ep.units.tt2000),
        ep.ExtractionInfo(result_key="Density", name_or_column="density", unit=u.cm ** (-3)),
    ]
    return ep.extract_variables_from_files(
        start_time=start_time,
        end_time=end_time,
        file_cadence="daily",
        data_path=RAW_DATA_PATH,
        file_name_stem=DENSITY_FILE_STEM,
        extraction_infos=extraction_infos,
    )


def load_magnetometer_data(start_time: datetime, end_time: datetime) -> dict[str, ep.Variable]:
    extraction_infos = [
        ep.ExtractionInfo(result_key="Epoch", name_or_column="Epoch", unit=ep.units.tt2000),
        ep.ExtractionInfo(result_key="Bt", name_or_column="Magnitude", unit=u.nT),
        ep.ExtractionInfo(result_key="Coordinates", name_or_column="coordinates", unit=u.km),
    ]
    return ep.extract_variables_from_files(
        start_time=start_time,
        end_time=end_time,
        file_cadence="daily",
        data_path=RAW_DATA_PATH,
        file_name_stem=MAG_FILE_STEM,
        extraction_infos=extraction_infos,
    )


def clean_magnetometer_data(mag_vars: dict[str, ep.Variable]) -> dict[str, ep.Variable]:
    mask = ep.processing.magnetometer_quality_flags(mag_vars)
    good = mask.get_data()

    for var in mag_vars.values():
        data = var.get_data()
        if data.shape[0] != good.shape[0]:
            error_msg = f"Data length mismatch for variable '{var.name}'. \
                         Expected {good.shape[0]}, got {data.shape[0]}."
            raise ValueError(error_msg)
        var.set_data(data[good], unit="same")

    return mag_vars


def add_orbital_variables(mag_vars: dict[str, ep.Variable]) -> dict[str, ep.Variable]:
    bt = np.asarray(mag_vars["Bt"].get_data(u.T))
    coords = np.asarray(mag_vars["Coordinates"].get_data())

    x = coords[:, 0] / RE.si.in_units(u.km)
    y = coords[:, 1] / RE.si.in_units(u.km)
    z = coords[:, 2] / RE.si.in_units(u.km)

    r_xy = np.hypot(x, y)
    r = np.sqrt(x**2 + y**2 + z**2)
    mlat_rad = np.arctan2(z, r_xy)
    mlat = np.degrees(mlat_rad)

    l_shell = r / np.cos(mlat_rad) ** 2
    mlt = np.degrees(np.arctan2(y, x)) / 15.0 + 12.0
    mlt = np.mod(mlt, 24.0)

    fce = (e.si * bt) / (2 * np.pi * m_e.si)
    fce_eq = fce * (np.cos(mlat_rad) ** 6) / np.sqrt(1 + 3 * np.sin(mlat_rad) ** 2)

    mag_vars["L"] = Variable(u.dimensionless_unscaled, data=l_shell)
    mag_vars["mlat"] = Variable(u.deg, data=mlat)
    mag_vars["mlt"] = Variable(u.hourangle, data=mlt)
    mag_vars["fce"] = Variable(u.Hz, data=fce)
    mag_vars["fce_eq"] = Variable(u.Hz, data=fce_eq)

    return mag_vars


def load_wfr_data(start_time: datetime, end_time: datetime) -> dict[str, ep.Variable]:
    extraction_infos = [
        ep.ExtractionInfo(result_key="Epoch", name_or_column="Epoch", unit=ep.units.tt2000),
        ep.ExtractionInfo(result_key="freq", name_or_column="WFR_frequencies", unit=u.Hz),
        ep.ExtractionInfo(result_key="bandwidth", name_or_column="WFR_bandwidth", unit=u.Hz),
        ep.ExtractionInfo(result_key="BuBu", name_or_column="BuBu", unit=(u.nT) ** 2 / u.Hz),
        ep.ExtractionInfo(result_key="BvBv", name_or_column="BvBv", unit=(u.nT) ** 2 / u.Hz),
        ep.ExtractionInfo(result_key="BwBw", name_or_column="BwBw", unit=(u.nT) ** 2 / u.Hz),
    ]
    return ep.extract_variables_from_files(
        start_time=start_time,
        end_time=end_time,
        file_cadence="daily",
        data_path=RAW_DATA_PATH,
        file_name_stem=WFR_FILE_STEM,
        extraction_infos=extraction_infos,
    )


def compute_total_psd(wfr_vars: dict[str, ep.Variable]) -> dict[str, ep.Variable]:
    bb = wfr_vars["BuBu"].get_data() + wfr_vars["BvBv"].get_data() + wfr_vars["BwBw"].get_data()
    wfr_vars["BB"] = Variable((u.nT) ** 2 / u.Hz, data=bb)
    return wfr_vars


def load_wna_data(start_time: datetime, end_time: datetime) -> dict[str, ep.Variable]:
    extraction_infos = [
        ep.ExtractionInfo(result_key="Epoch", name_or_column="Epoch", unit=ep.units.tt2000),
        ep.ExtractionInfo(result_key="freq", name_or_column="WFR_frequencies", unit=u.Hz),
        ep.ExtractionInfo(result_key="WNA", name_or_column="thsvd", unit=u.deg),
        ep.ExtractionInfo(result_key="ellipticity", name_or_column="ellsvd", unit=u.dimensionless_unscaled),
        ep.ExtractionInfo(result_key="planarity", name_or_column="plansvd", unit=u.dimensionless_unscaled),
    ]
    return ep.extract_variables_from_files(
        start_time=start_time,
        end_time=end_time,
        file_cadence="daily",
        data_path=RAW_DATA_PATH,
        file_name_stem=WNA_FILE_STEM,
        extraction_infos=extraction_infos,
    )


def save_single_files(
    start_time: datetime,
    end_time: datetime,
    density_vars: dict[str, ep.Variable],
    mag_vars: dict[str, ep.Variable],
    wfr_vars: dict[str, ep.Variable],
    wna_vars: dict[str, ep.Variable],
) -> None:
    density_output = PROCESSED_DATA_PATH / f"rbspa_density_{start_time:%Y%m%d}_to_{end_time:%Y%m%d}.pickle"
    ep.save(
        variables_dict=density_vars,
        saving_strategy=ep.saving_strategies.SingleFileStrategy(density_output),
        start_time=start_time,
        end_time=end_time,
        time_var=density_vars["Epoch"],
    )

    mag_output = PROCESSED_DATA_PATH / f"rbspa_magnetometer_{start_time:%Y%m%d}_to_{end_time:%Y%m%d}.pickle"
    ep.save(
        variables_dict=mag_vars,
        saving_strategy=ep.saving_strategies.SingleFileStrategy(mag_output),
        start_time=start_time,
        end_time=end_time,
        time_var=mag_vars["Epoch"],
    )

    wave_output = PROCESSED_DATA_PATH / f"rbspa_wave_spectrum_{start_time:%Y%m%d}_to_{end_time:%Y%m%d}.pickle"

    ep.save(
        variables_dict={k: wfr_vars[k] for k in ["Epoch", "BuBu", "BvBv", "BwBw", "BB"]},
        saving_strategy=ep.saving_strategies.SingleFileStrategy(wave_output),
        start_time=start_time,
        end_time=end_time,
        time_var=wfr_vars["Epoch"],
    )

    wna_output = PROCESSED_DATA_PATH / f"rbspa_wave_properties_{start_time:%Y%m%d}_to_{end_time:%Y%m%d}.pickle"
    ep.save(
        variables_dict={k: wna_vars[k] for k in ["Epoch", "WNA", "ellipticity", "planarity"]},
        saving_strategy=ep.saving_strategies.SingleFileStrategy(wna_output),
        start_time=start_time,
        end_time=end_time,
        time_var=wna_vars["Epoch"],
    )


def plot_density(density_vars: dict[str, ep.Variable]) -> None:
    density_vars["Epoch"].convert_to_unit(ep.units.posixtime)
    times = np.array([datetime.fromtimestamp(ts, timezone.utc) for ts in density_vars["Epoch"].get_data()])
    density_data = density_vars["Density"].get_data()

    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, density_data)
    ax.set_ylabel(r"Electron Density ($n_e$) [cm$^{-3}$]")
    ax.set_yscale("log")
    ax.set_xlabel("Time [UTC]")
    ax.set_title(f"Density - {times[0].strftime('%Y-%m-%d %H:%M')} to {times[-1].strftime('%Y-%m-%d %H:%M')}")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.show()


def plot_orbit(mag_vars: dict[str, ep.Variable]) -> None:
    mag_vars["Epoch"].convert_to_unit(ep.units.posixtime)
    times = np.array([datetime.fromtimestamp(ts, timezone.utc) for ts in mag_vars["Epoch"].get_data()])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("RBSP-A Orbit Parameters")

    ax1.plot(times, mag_vars["L"].get_data(), "k-", linewidth=1)
    ax1.set_ylabel("L-shell")
    ax1.grid(alpha=0.3)

    ax2.plot(times, mag_vars["mlat"].get_data(), "k-", linewidth=1)
    ax2.set_ylabel(r"MLAT [°]")
    ax2.grid(alpha=0.3)

    ax3.plot(times, mag_vars["mlt"].get_data(), "k-", linewidth=1)
    ax3.set_ylabel("MLT [h]")
    ax3.set_xlabel(f"UT {times[0].strftime('%Y-%m-%d')}")
    ax3.grid(alpha=0.3)

    hours = mdates.HourLocator(interval=4)
    hours_fmt = mdates.DateFormatter("%H:%M")
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_locator(hours)
        ax.xaxis.set_major_formatter(hours_fmt)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()


def plot_magnetometer(mag_vars: dict[str, ep.Variable]) -> None:
    mag_vars["Epoch"].convert_to_unit(ep.units.posixtime)
    times = np.array([datetime.fromtimestamp(ts, timezone.utc) for ts in mag_vars["Epoch"].get_data()])
    bt = mag_vars["Bt"].get_data()

    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, bt, "k-", linewidth=1)
    ax.set_ylabel("Bt [nT]")
    ax.set_xlabel(f"UT {times[0].strftime('%Y-%m-%d')}")
    ax.set_title("Cleaned Magnetometer Data (Bt)")
    ax.grid(alpha=0.3)

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.show()


def plot_wfr(wfr_vars: dict[str, ep.Variable]) -> None:
    wfr_vars["Epoch"].convert_to_unit(ep.units.posixtime)
    times = np.array([datetime.fromtimestamp(ts, timezone.utc) for ts in wfr_vars["Epoch"].get_data()])
    bb = wfr_vars["BB"].get_data()

    fig, ax = plt.subplots(figsize=(12, 8))

    img = ax.imshow(
        np.log10(bb.T),
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )

    ax.set_ylabel("Frequency bin")
    ax.set_xlabel(f"Time UT ({times[0].strftime('%Y-%m-%d')})")

    n_time = len(times)
    tick_idx = np.linspace(0, n_time - 1, 6, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([times[i].strftime("%H:%M") for i in tick_idx], rotation=45)

    ax.set_yticks([])

    fig.colorbar(img, ax=ax, shrink=0.8, label=r"log$_{10}$(B$^2$) [nT$^2$/Hz]")
    ax.set_title("RBSP-A Total Magnetic Wave Power Spectral Density")
    plt.tight_layout()
    plt.show()


def plot_wna(wna_vars: dict[str, ep.Variable]) -> None:
    wna_vars["Epoch"].convert_to_unit(ep.units.posixtime)
    times = np.array([datetime.fromtimestamp(ts, timezone.utc) for ts in wna_vars["Epoch"].get_data()])
    freq = wna_vars["freq"].get_data()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("RBSP-A Wave Properties")

    cax1 = ax1.pcolormesh(times, freq, wna_vars["WNA"].get_data().T, cmap="RdBu_r", shading="auto")
    ax1.set_ylabel("Frequency [Hz]")
    ax1.set_yscale("log")
    fig.colorbar(cax1, ax=ax1, label="WNA [°]")

    cax2 = ax2.pcolormesh(
        times, freq, wna_vars["ellipticity"].get_data().T, vmin=0, vmax=1, cmap="viridis", shading="auto"
    )
    ax2.set_ylabel("Frequency [Hz]")
    ax2.set_yscale("log")
    fig.colorbar(cax2, ax=ax2, label="Ellipticity")

    cax3 = ax3.pcolormesh(
        times, freq, wna_vars["planarity"].get_data().T, vmin=0, vmax=1, cmap="plasma", shading="auto"
    )
    ax3.set_xlabel(f"Time UT ({times[0].strftime('%Y-%m-%d')})")
    ax3.set_ylabel("Frequency [Hz]")
    ax3.set_yscale("log")
    fig.colorbar(cax3, ax=ax3, label="Planarity")

    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.show()


def main() -> None:
    download_day_data(START_TIME, END_TIME)

    density_vars = load_density_data(START_TIME, END_TIME)
    plot_density(density_vars)

    mag_vars = load_magnetometer_data(START_TIME, END_TIME)
    mag_vars = clean_magnetometer_data(mag_vars)
    mag_vars = add_orbital_variables(mag_vars)
    plot_orbit(mag_vars)
    plot_magnetometer(mag_vars)

    wfr_vars = load_wfr_data(START_TIME, END_TIME)
    wfr_vars = compute_total_psd(wfr_vars)
    plot_wfr(wfr_vars)

    wna_vars = load_wna_data(START_TIME, END_TIME)
    plot_wna(wna_vars)

    save_single_files(START_TIME, END_TIME, density_vars, mag_vars, wfr_vars, wna_vars)


if __name__ == "__main__":
    main()
