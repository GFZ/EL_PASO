# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Alwin Roy
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import dateutil
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.constants import e, m_e  # ty:ignore[unresolved-import]

import el_paso as ep
from el_paso.units import RE
from el_paso.variable import Variable

if TYPE_CHECKING:
    from el_paso.processing.interpolate_in_time import InterpolationMethod


def process_rbsp_emfisis_waves(
    start_time: datetime,
    end_time: datetime,
    sat_str: Literal["a", "b"],
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
) -> None:
    """Process RBSP EMFISIS wave, density, and magnetometer data and save derived wave properties.

    Downloads and extracts the EMFISIS WFR spectral-matrix-diagonal data, the EMFISIS density
    data, the EMFISIS magnetometer data, and the EMFISIS wave-normal-angle (WNA) survey data for
    the given time range and satellite. The density and magnetometer data are interpolated onto
    the WFR time grid, the magnetometer data is cleaned using a quality flag, orbital quantities
    (L-shell, MLAT, MLT, electron cyclotron frequency) are derived from the cleaned magnetometer
    data, and the total magnetic wave power spectral density is computed from the WFR
    spectral-matrix components. The wave frequency, frequency bandwidth, wave normal angle,
    planarity, and ellipticity are saved using `DailyWaveStrategy`. Finally, diagnostic plots of
    the density, orbit, magnetometer, WFR spectrogram, and WNA properties are generated and shown
    or saved to disk.

    Args:
        start_time (datetime): Start of the time range to process.
        end_time (datetime): End of the time range to process.
        sat_str (Literal["a", "b"]): RBSP satellite identifier ("a" or "b").
        raw_data_path (str | Path, optional): Base directory where raw CDF files are downloaded to
            and read from. Defaults to ".".
        processed_data_path (str | Path, optional): Directory where the processed output files are
            written to. Defaults to ".".
    """
    wfr_vars = _get_wfr_data(start_time, end_time, Path(raw_data_path), sat_str)

    target_time_var = wfr_vars["Epoch"]
    density_vars = _get_density_data(start_time, end_time, Path(raw_data_path), sat_str, target_time_var)
    mag_vars = _get_magnetometer_data(start_time, end_time, Path(raw_data_path), sat_str, target_time_var)
    wna_vars = _get_wna_data(start_time, end_time, Path(raw_data_path), sat_str)

    mag_vars = _clean_magnetometer_data(mag_vars)

    orbit_vars = _calculate_orbital_vars(mag_vars)
    psd_var = _compute_total_psd(wfr_vars)

    vars_to_save: dict[ep.typing.InternalName, ep.Variable] = {
        "Epoch": target_time_var,
        "Wave_frequency": wfr_vars["freq"],
        "Wave_frequency_bandwidth": wfr_vars["bandwidth"],
        "Wave_normal_angle": wna_vars["WNA"],
        "Wave_planarity": wna_vars["planarity"],
        "Wave_ellipticity": wna_vars["ellipticity"],
    }

    saving_strat = ep.saving_strategies.DailyWaveStrategy(
        processed_data_path, "RBSP", f"rbsp{sat_str}", "EMFISIS", ep.data_standards.GFZStandard()
    )

    ep.save(vars_to_save, saving_strat, start_time, end_time)

    wfr_vars["BB"] = psd_var

    _plot_density(density_vars)
    orbit_vars["Epoch"] = mag_vars["Epoch"]
    _plot_orbit(orbit_vars)
    _plot_magnetometer(mag_vars)
    _plot_wfr(wfr_vars)
    _plot_wna(wna_vars)


def _calculate_orbital_vars(mag_vars: dict[str, ep.Variable]) -> dict[str, ep.Variable]:
    bt = np.asarray(mag_vars["Bt"].get_data(u.T))
    coords = np.asarray(mag_vars["Coordinates"].get_data(u.km))

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    r_xy = np.hypot(x, y)
    r = np.sqrt(x**2 + y**2 + z**2)
    mlat_rad = np.arctan2(z, r_xy)
    mlat = np.degrees(mlat_rad)

    l_shell = r / np.cos(mlat_rad) ** 2
    mlt = np.degrees(np.arctan2(y, x)) / 15.0 + 12.0
    mlt = np.mod(mlt, 24.0)

    fce = (e.si * bt) / (2 * np.pi * m_e.si)
    fce_eq = fce * (np.cos(mlat_rad) ** 6) / np.sqrt(1 + 3 * np.sin(mlat_rad) ** 2)

    orbit_vars = {
        "L": Variable(u.dimensionless_unscaled, data=l_shell),
        "mlat": Variable(u.deg, data=mlat),
        "mlt": Variable(u.hour, data=mlt),
        "fce": Variable(u.Hz, data=fce),
        "fce_eq": Variable(u.Hz, data=fce_eq),
    }

    return orbit_vars


def _get_wfr_data(
    start_time: datetime, end_time: datetime, raw_data_path: Path, sat_str: Literal["a", "b"]
) -> dict[str, ep.Variable]:
    url = f"https://cdaweb.gsfc.nasa.gov/pub/data/rbsp/rbsp{sat_str}/l2/emfisis/wfr/spectral-matrix-diagonal/YYYY/"
    file_name_stem = "rbsp-" + sat_str + r"_wfr-spectral-matrix-diagonal_emfisis-l2_YYYYMMDD_.{6}.cdf"

    raw_data_path = raw_data_path / "YYYY" / "MM" / "wfr"

    ep.download(
        start_time,
        end_time,
        save_path=raw_data_path,
        download_url=url,
        file_name_stem=file_name_stem,
        file_cadence="daily",
        method="request",
        skip_existing=True,
    )

    extraction_infos = [
        ep.ExtractionInfo(result_key="Epoch", name_or_column="Epoch", unit=ep.units.tt2000),
        ep.ExtractionInfo(result_key="freq", name_or_column="WFR_frequencies", unit=u.Hz),
        ep.ExtractionInfo(result_key="bandwidth", name_or_column="WFR_bandwidth", unit=u.Hz),
        ep.ExtractionInfo(result_key="BuBu", name_or_column="BuBu", unit=(u.nT) ** 2 / u.Hz),
        ep.ExtractionInfo(result_key="BvBv", name_or_column="BvBv", unit=(u.nT) ** 2 / u.Hz),
        ep.ExtractionInfo(result_key="BwBw", name_or_column="BwBw", unit=(u.nT) ** 2 / u.Hz),
    ]
    variables = ep.extract_variables_from_files(
        start_time=start_time,
        end_time=end_time,
        file_cadence="daily",
        data_path=raw_data_path,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )

    freq = variables["freq"].get_data()
    variables["freq"].set_data(np.squeeze(freq), unit="same")
    freq_bw = variables["bandwidth"].get_data()
    variables["bandwidth"].set_data(np.squeeze(freq_bw), unit="same")

    return variables


def _get_wna_data(
    start_time: datetime,
    end_time: datetime,
    raw_data_path: Path,
    sat_str: Literal["a", "b"],
) -> dict[str, ep.Variable]:
    url = f"https://cdaweb.gsfc.nasa.gov/pub/data/rbsp/rbsp{sat_str}/l4/emfisis/wna-survey-sheath-corrected-e/YYYY/"
    file_name_stem = "rbsp-" + sat_str + r"_wna-survey-sheath-corrected-e_emfisis-l4_YYYYMMDD_.{6}.cdf"

    raw_data_path = raw_data_path / "YYYY" / "MM" / "sna"

    ep.download(
        start_time,
        end_time,
        save_path=raw_data_path,
        download_url=url,
        file_name_stem=file_name_stem,
        file_cadence="daily",
        method="request",
        skip_existing=True,
    )

    extraction_infos = [
        ep.ExtractionInfo(result_key="Epoch", name_or_column="Epoch", unit=ep.units.tt2000),
        ep.ExtractionInfo(result_key="freq", name_or_column="WFR_frequencies", unit=u.Hz, is_time_dependent=False),
        ep.ExtractionInfo(result_key="WNA", name_or_column="thsvd", unit=u.deg),
        ep.ExtractionInfo(result_key="ellipticity", name_or_column="ellsvd", unit=u.dimensionless_unscaled),
        ep.ExtractionInfo(result_key="planarity", name_or_column="plansvd", unit=u.dimensionless_unscaled),
    ]
    return ep.extract_variables_from_files(
        start_time=start_time,
        end_time=end_time,
        file_cadence="daily",
        data_path=raw_data_path,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )


def _get_density_data(
    start_time: datetime,
    end_time: datetime,
    raw_data_path: Path,
    sat_str: Literal["a", "b"],
    target_time_var: ep.Variable,
) -> dict[str, ep.Variable]:
    url = f"https://cdaweb.gsfc.nasa.gov/pub/data/rbsp/rbsp{sat_str}/l4/emfisis/density/YYYY/"
    file_name_stem = "rbsp-" + sat_str + r"_density_emfisis-l4_YYYYMMDD_.{7}.cdf"

    raw_data_path = raw_data_path / "YYYY" / "MM" / "density"

    ep.download(
        start_time,
        end_time,
        save_path=raw_data_path,
        download_url=url,
        file_name_stem=file_name_stem,
        file_cadence="daily",
        method="request",
        skip_existing=True,
    )

    extraction_infos = [
        ep.ExtractionInfo(result_key="Epoch", name_or_column="Epoch", unit=ep.units.tt2000),
        ep.ExtractionInfo(result_key="Density", name_or_column="density", unit=u.cm ** (-3)),
    ]
    variables = ep.extract_variables_from_files(
        start_time=start_time,
        end_time=end_time,
        file_cadence="daily",
        data_path=raw_data_path,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )

    interp_methods: dict[str, InterpolationMethod] = {"Density": "linear"}

    _ = ep.processing.interpolate_in_time(
        variables["Epoch"],
        variables,
        interp_methods,
        target_time_variable=target_time_var,
    )

    return variables


def _get_magnetometer_data(
    start_time: datetime,
    end_time: datetime,
    raw_data_path: Path,
    sat_str: Literal["a", "b"],
    target_time_var: ep.Variable,
) -> dict[str, ep.Variable]:
    url = f"https://cdaweb.gsfc.nasa.gov/pub/data/rbsp/rbsp{sat_str}/l3/emfisis/magnetometer/4sec/sm/YYYY/"
    file_name_stem = "rbsp-" + sat_str + r"_magnetometer_4sec-sm_emfisis-l3_YYYYMMDD_.{6}.cdf"

    raw_data_path = raw_data_path / "YYYY" / "MM" / "magnetometer"

    ep.download(
        start_time,
        end_time,
        save_path=raw_data_path,
        download_url=url,
        file_name_stem=file_name_stem,
        file_cadence="daily",
        method="request",
        skip_existing=True,
    )

    extraction_infos = [
        ep.ExtractionInfo(result_key="Epoch", name_or_column="Epoch", unit=ep.units.tt2000),
        ep.ExtractionInfo(result_key="Bt", name_or_column="Magnitude", unit=u.nT),
        ep.ExtractionInfo(result_key="Coordinates", name_or_column="coordinates", unit=u.km),
    ]

    variables = ep.extract_variables_from_files(
        start_time=start_time,
        end_time=end_time,
        file_cadence="daily",
        data_path=raw_data_path,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )

    interp_methods: dict[str, InterpolationMethod] = {"Bt": "nearest", "Coordinates": "nearest"}

    _ = ep.processing.interpolate_in_time(
        variables["Epoch"],
        variables,
        interp_methods,
        target_time_variable=target_time_var,
    )

    del variables["Epoch"]

    return variables


def _clean_magnetometer_data(mag_vars: dict[str, ep.Variable]) -> dict[str, ep.Variable]:
    mask = ep.processing.create_quality_flag_from_magnetometer(mag_vars["Bt"])
    good = mask.get_data()

    for var in mag_vars.values():
        data = var.get_data()
        if data.shape[0] != good.shape[0]:
            error_msg = f"Data length mismatch for variable'. \
                         Expected {good.shape[0]}, got {data.shape[0]}."
            raise ValueError(error_msg)
        var.set_data(data[good], unit="same")  # ty:ignore[invalid-argument-type]

    return mag_vars


def _compute_total_psd(wfr_vars: dict[str, ep.Variable]) -> ep.Variable:
    bb = wfr_vars["BuBu"].get_data().astype(np.float64) + wfr_vars["BvBv"].get_data() + wfr_vars["BwBw"].get_data()  # ty: ignore[unsupported-operator]
    return Variable((u.nT) ** 2 / u.Hz, data=bb)


def _plot_density(density_vars: dict[str, ep.Variable]) -> None:
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


def _plot_orbit(mag_vars: dict[str, ep.Variable]) -> None:
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
    plt.savefig("orbit.png")


def _plot_magnetometer(mag_vars: dict[str, ep.Variable]) -> None:
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
    plt.savefig("mag.png")


def _plot_wfr(wfr_vars: dict[str, ep.Variable]) -> None:
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
    plt.savefig("wfr.png")


def _plot_wna(wna_vars: dict[str, ep.Variable]) -> None:
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
    plt.savefig("wna.png")


if __name__ == "__main__":
    ep.setup_logging()

    parser = argparse.ArgumentParser(
        description="Process density data from EFW and EMFISIS instrument on VanAllenProbes."
    )
    parser.add_argument(
        "--start_time",
        type=str,
        help="Start time in valid dateparse format. Example: YYYY-MM-DDTHH:MM:SS.",
        default=datetime(2017, 4, 1, tzinfo=timezone.utc).isoformat(),
        required=False,
    )
    parser.add_argument(
        "--end_time",
        type=str,
        help="End time in valid dateparse format. Example: YYYY-MM-DDTHH:MM:SS.",
        default=datetime(2017, 4, 1, 0, 5, 59, tzinfo=timezone.utc).isoformat(),
        required=False,
    )

    args = parser.parse_args()

    dt_start = dateutil.parser.parse(args.start_time)
    dt_end = dateutil.parser.parse(args.end_time)

    for sat_str in ["a", "b"]:
        process_rbsp_emfisis_waves(
            dt_start,
            dt_end,
            sat_str=sat_str,
            raw_data_path=".",
            processed_data_path=".",
        )
