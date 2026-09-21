# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.constants import e, m_e  # ty:ignore[unresolved-import]
from pyspedas.projects import themis

import el_paso as ep
from el_paso.recipes.themis import ThemisProbe
from el_paso.recipes.themis.get_themis_pyspedas_variables import (
    build_trange,
    get_themis_position_geo,
    get_themis_scpot_density,
    set_pyspedas_data_dir,
    tplot_to_bin_variable,
    tplot_to_time_variable,
    tplot_to_variable,
)

if TYPE_CHECKING:
    from el_paso.processing.interpolate_in_time import InterpolationMethod

logger = logging.getLogger(__name__)

_HISS_BAND = (40 * u.Hz, 2000 * u.Hz)
"""Frequency band integrated to obtain the plasmaspheric hiss RMS amplitude."""

_PLASMASPHERE_DENSITY_THRESHOLD = 100 * u.cm ** (-3)
"""Density above which the spacecraft is taken to be inside the plasmasphere."""


def themis_fft_waves_strategy(
    base_data_path: str | Path,
    satellite: ThemisProbe,
    data_standard: ep.typing.DataStandard[ep.typing.StandardName] | None = None,
) -> ep.SavingStrategy:
    """Daily NetCDF wave saving strategy for THEMIS FFT."""
    return ep.saving_strategies.DailyWaveStrategy(
        Path(base_data_path),
        "THEMIS",
        "th" + satellite,
        "FFT",
        data_standard or ep.data_standards.GFZStandard(),
    )


def process_themis_fft_waves(
    start_time: datetime,
    end_time: datetime,
    satellite: ThemisProbe = "a",
    mag_field: Literal["T89", "T96", "TS04"] = "T89",
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    bin_cadence: timedelta = timedelta(minutes=5),
    num_cores: int = 16,
    save_strategy: Literal["netcdf"] = "netcdf",
    *,
    skip_existing: bool = True,
) -> None:
    """Process THEMIS FFT wave data and save the magnetic power spectral density.

    Downloads the THEMIS Level 2 FFT, FGM, ESA and state data for the given probe and time range
    via pyspedas. The magnetic power spectral density is formed by summing the three search-coil
    magnetometer axes of the 32-bin FFT spectrum, which sets the master time cadence for this
    recipe. The electron density derived from the spacecraft potential, the total magnetic field,
    and the spacecraft position are all interpolated onto that cadence. Magnetic local time and
    the mapped equatorial radial distance are computed with IRBEM for the given `mag_field`, while
    the magnetic latitude is derived directly from the position. The results are written with
    `DailyWaveStrategy`, one NetCDF file per day, and a summary plot of the density, the wave
    spectrogram, and the plasmaspheric hiss amplitude is produced.

    Args:
        start_time (datetime): Start of the time range to process.
        end_time (datetime): End of the time range to process.
        satellite (Literal["a", "b", "c", "d", "e"]): THEMIS probe identifier.
        mag_field (Literal["T89", "T96", "TS04"]): The magnetic field model used to compute the
            magnetic local time and the mapped equatorial radial distance.
        raw_data_path (str | Path): Base directory pyspedas downloads the raw THEMIS files into
            and reads them from. Defaults to ".".
        processed_data_path (str | Path): Directory where the processed output files are
            written to. Defaults to ".".
        bin_cadence (timedelta): Unused by this recipe; accepted only for interface
            consistency with other EL-PASO recipes, since the wave processing uses the
            instrument's own FFT time grid instead of a configurable binning cadence.
        num_cores (int): Number of CPU cores used for the IRBEM magnetic field computations.
            Defaults to 16.
        save_strategy (Literal["netcdf"]): Unused by this recipe; accepted only for
            interface consistency with other EL-PASO recipes, since the THEMIS wave saving
            strategy factory only supports a single output format. Defaults to "netcdf".
        skip_existing (bool): If True, let pyspedas reuse raw files that already exist locally
            instead of re-downloading them. Defaults to True.
    """
    del bin_cadence
    del save_strategy

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel(logging.INFO)

    set_pyspedas_data_dir(raw_data_path)

    fft_vars = _get_fft_data(start_time, end_time, satellite)
    target_time_var = fft_vars["Epoch"]

    mag_vars = _get_magnetometer_data(start_time, end_time, satellite, target_time_var)
    density_vars = _get_density_data(start_time, end_time, satellite, target_time_var)
    orbit_vars = _get_orbit_vars(start_time, end_time, satellite, target_time_var, mag_field, num_cores)

    vars_to_save: dict[ep.typing.InternalName, ep.Variable] = {
        "Epoch": target_time_var,
        "Wave_frequency": fft_vars["freq"],
        "Magnetic_Power_Spectral_Density": fft_vars["BB"],
        "Number_density": density_vars["Density"],
        "B_total_obs": mag_vars["Bt"],
        "MLat": orbit_vars["MLat"],
        "MLT": orbit_vars["MLT_" + mag_field],
        "R_Eq": orbit_vars["R_Eq_" + mag_field],
    }

    saving_strat = themis_fft_waves_strategy(processed_data_path, satellite)

    ep.save(vars_to_save, saving_strat, start_time, end_time, time_var=target_time_var)

    _plot_wave_summary(satellite, target_time_var, fft_vars, density_vars, mag_vars, orbit_vars)


def _get_fft_data(
    start_time: datetime,
    end_time: datetime,
    satellite: ThemisProbe,
) -> dict[str, ep.Variable]:
    """Load the 32-bin FFT search-coil spectrum and sum it over the three SCM axes."""
    probe = str(satellite)
    axis_names = [f"th{probe}_fff_32_scm{axis}" for axis in (2,3)]

    themis.fft(
        trange=build_trange(start_time, end_time),
        probe=probe,
        level="l2",
        varnames=axis_names,
    )

    psd_unit = (u.nT) ** 2 / u.Hz

    total_psd = None
    for axis_name in axis_names:
        axis_psd = np.asarray(tplot_to_variable(axis_name, psd_unit).get_data(psd_unit)).astype(np.float64)
        total_psd = axis_psd if total_psd is None else total_psd + axis_psd

    bb_var = ep.Variable(
        original_unit=psd_unit,
        data=total_psd,
        description="Total magnetic wave power spectral density.",
        processing_notes="Sum of the three search-coil axes of the THEMIS 32-bin FFT spectrum.",
    )

    return {
        "Epoch": tplot_to_time_variable(axis_names[0]),
        "freq": tplot_to_bin_variable(axis_names[0], u.Hz),
        "BB": bb_var,
    }


def _get_magnetometer_data(
    start_time: datetime,
    end_time: datetime,
    satellite: ThemisProbe,
    target_time_var: ep.Variable,
) -> dict[str, ep.Variable]:
    """Load the total magnetic field from FGM and interpolate it onto the FFT cadence."""
    probe = str(satellite)

    themis.fgm(
        trange=build_trange(start_time, end_time),
        probe=probe,
        level="l2",
    )

    btotal_name = f"th{probe}_fgs_btotal"

    try:
        time_var = tplot_to_time_variable(btotal_name)
        bt_data = np.asarray(tplot_to_variable(btotal_name, u.nT).get_data(u.nT)).astype(np.float64)
    except ValueError:
        # Some intervals only carry the field vector, not the precomputed magnitude.
        vector_name = f"th{probe}_fgs_gse"
        logger.info(f"'{btotal_name}' unavailable; deriving the magnitude from '{vector_name}'.")
        time_var = tplot_to_time_variable(vector_name)
        b_vec = np.asarray(tplot_to_variable(vector_name, u.nT).get_data(u.nT)).astype(np.float64)
        bt_data = np.linalg.norm(b_vec, axis=1)

    # FGM occasionally repeats timestamps across file boundaries, which the interpolation
    # below cannot handle.
    times = np.asarray(time_var.get_data(ep.units.posixtime))
    unique_mask = ~pd.Index(times).duplicated(keep="first")

    variables = {
        "Epoch": ep.Variable(original_unit=ep.units.posixtime, data=times[unique_mask]),
        "Bt": ep.Variable(
            original_unit=u.nT,
            data=bt_data[unique_mask],
            description="Observed total magnetic field at the satellite location.",
        ),
    }

    interp_methods: dict[str, InterpolationMethod] = {"Bt": "nearest"}

    _ = ep.processing.interpolate_in_time(
        variables["Epoch"],
        variables,
        interp_methods,
        target_time_variable=target_time_var,
    )

    del variables["Epoch"]

    return variables


def _get_density_data(
    start_time: datetime,
    end_time: datetime,
    satellite: ThemisProbe,
    target_time_var: ep.Variable,
) -> dict[str, ep.Variable]:
    """Derive the spacecraft-potential density and interpolate it onto the FFT cadence."""
    variables = get_themis_scpot_density(satellite, start_time, end_time)

    interp_methods: dict[str, InterpolationMethod] = {"Density": "linear"}

    _ = ep.processing.interpolate_in_time(
        variables["Epoch"],
        variables,
        interp_methods,
        target_time_variable=target_time_var,
    )

    del variables["Epoch"]

    return variables


def _get_orbit_vars(
    start_time: datetime,
    end_time: datetime,
    satellite: ThemisProbe,
    target_time_var: ep.Variable,
    mag_field: Literal["T89", "T96", "TS04"],
    num_cores: int,
) -> dict[str, ep.Variable]:
    """Interpolate the position onto the FFT cadence and derive the orbital quantities."""
    variables = get_themis_position_geo(satellite, start_time, end_time)

    interp_methods: dict[str, InterpolationMethod] = {"xGEO": "linear"}

    _ = ep.processing.interpolate_in_time(
        variables["Epoch"],
        variables,
        interp_methods,
        target_time_variable=target_time_var,
    )

    del variables["Epoch"]

    # MLat is not among the quantities IRBEM computes here, so it is derived from the position.
    pos = np.asarray(variables["xGEO"].get_data(ep.units.RE)).astype(np.float64)
    mlat = np.degrees(np.arctan2(pos[:, 2], np.hypot(pos[:, 0], pos[:, 1])))
    variables["MLat"] = ep.Variable(
        original_unit=u.deg,
        data=mlat,
        description="Magnetic latitude of the satellite location.",
    )

    variables_to_compute: ep.processing.VariableRequest = [
        ("MLT", mag_field),
        ("R_Eq", mag_field),
    ]

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=target_time_var,
        xgeo_var=variables["xGEO"],
        variables_to_compute=variables_to_compute,
        irbem_options=ep.processing.magnetic_field_utils.IrbemOptions(),
        num_cores=num_cores,
    )

    return variables | magnetic_field_variables


def _equatorial_electron_gyrofrequency(mag_vars: dict[str, ep.Variable], mlat_var: ep.Variable) -> np.ndarray:
    """Map the local electron gyrofrequency down to the magnetic equator."""
    bt = np.asarray(mag_vars["Bt"].get_data(u.T)).astype(np.float64)
    mlat_rad = np.radians(np.asarray(mlat_var.get_data(u.deg)).astype(np.float64))

    fce = (e.si.value * bt) / (2 * np.pi * m_e.si.value)

    return fce * np.cos(mlat_rad) ** 6 / np.sqrt(1 + 3 * np.sin(mlat_rad) ** 2)


def _hiss_rms_amplitude(
    fft_vars: dict[str, ep.Variable],
    density_var: ep.Variable,
) -> np.ndarray:
    """Integrate the wave power over the hiss band to get an RMS amplitude in pT.

    Only samples taken inside the plasmasphere are retained; everything else is returned as
    NaN, since the same frequency band outside the plasmasphere is chorus rather than hiss.
    """
    psd_unit = (u.nT) ** 2 / u.Hz
    psd = np.asarray(fft_vars["BB"].get_data(psd_unit)).astype(np.float64)
    freq = np.asarray(fft_vars["freq"].get_data(u.Hz)).astype(np.float64)

    # Integrate over frequency using the bin widths implied by the bin centres.
    edges = np.zeros(len(freq) + 1)
    edges[1:-1] = 0.5 * (freq[:-1] + freq[1:])
    edges[0] = freq[0] - 0.5 * (freq[1] - freq[0])
    edges[-1] = freq[-1] + 0.5 * (freq[-1] - freq[-2])
    bin_widths = np.diff(edges)

    band_mask = (freq >= _HISS_BAND[0].to_value(u.Hz)) & (freq <= _HISS_BAND[1].to_value(u.Hz))

    # nT^2 -> pT^2 is a factor of 1e6.
    integrated = np.nansum(psd[:, band_mask] * bin_widths[band_mask], axis=1) * 1e6
    amplitude = np.sqrt(integrated)

    density = np.asarray(density_var.get_data(u.cm ** (-3))).astype(np.float64)
    inside_plasmasphere = density > _PLASMASPHERE_DENSITY_THRESHOLD.to_value(u.cm ** (-3))

    return np.where(inside_plasmasphere, amplitude, np.nan)


def _plot_wave_summary(
    satellite: ThemisProbe,
    time_var: ep.Variable,
    fft_vars: dict[str, ep.Variable],
    density_vars: dict[str, ep.Variable],
    mag_vars: dict[str, ep.Variable],
    orbit_vars: dict[str, ep.Variable],
) -> None:
    """Plot density, the wave spectrogram, and the hiss amplitude on a shared time axis."""
    times = np.array([datetime.fromtimestamp(ts, timezone.utc) for ts in time_var.get_data(ep.units.posixtime)])
    freq = np.asarray(fft_vars["freq"].get_data(u.Hz)).astype(np.float64)
    psd = np.asarray(fft_vars["BB"].get_data((u.nT) ** 2 / u.Hz)).astype(np.float64)
    density = np.asarray(density_vars["Density"].get_data(u.cm ** (-3))).astype(np.float64)

    fce_eq = _equatorial_electron_gyrofrequency(mag_vars, orbit_vars["MLat"])
    hiss_amplitude = _hiss_rms_amplitude(fft_vars, density_vars["Density"])

    fig, (ax_density, ax_spectrum, ax_hiss) = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True, height_ratios=[1, 3, 1.5]
    )
    fig.suptitle(f"THEMIS-{str(satellite).upper()} wave observations")

    ax_density.plot(times, density, color="darkgreen", lw=1.5)
    ax_density.axhline(_PLASMASPHERE_DENSITY_THRESHOLD.to_value(u.cm ** (-3)), color="red", ls="--", lw=1)
    ax_density.set_yscale("log")
    ax_density.set_ylabel(r"$n_e$" + "\n(cm$^{-3}$)")
    ax_density.grid(alpha=0.3)

    with np.errstate(divide="ignore"):
        log_psd = np.log10(np.where(psd > 0, psd, np.nan))

    mesh = ax_spectrum.pcolormesh(times, freq, log_psd.T, shading="auto", cmap="turbo", vmin=-7, vmax=-4)
    fig.colorbar(mesh, ax=ax_spectrum, label=r"log$_{10}$(nT$^2$/Hz)")

    for fraction in (1.0, 0.5, 0.05):
        ax_spectrum.plot(times, fraction * fce_eq, color="white", lw=1.2, ls=":")

    ax_spectrum.set_yscale("log")
    ax_spectrum.set_ylim(10, 3000)
    ax_spectrum.set_ylabel("Frequency\n(Hz)")
    ax_spectrum.grid(alpha=0.25)

    ax_hiss.scatter(times, hiss_amplitude, color="black", s=4)
    ax_hiss.set_ylabel("Hiss RMS\nBw (pT)")
    ax_hiss.set_xlabel(f"UT ({times[0].strftime('%Y-%m-%d')})")
    ax_hiss.grid(alpha=0.3)

    ax_hiss.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax_hiss.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax_hiss.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(f"themis_{satellite}_fft_waves.png")


CLI_DEFAULTS = {
    "raw_data_path": "./data",
    "processed_data_path": "./data",
}

if __name__ == "__main__":
    ep.run_recipe_cli(process_themis_fft_waves, defaults=CLI_DEFAULTS)
