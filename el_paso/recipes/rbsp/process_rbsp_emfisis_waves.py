# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Alwin Roy
#
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from astropy import units as u

import el_paso as ep
from el_paso.recipes.rbsp import RBSPSatellite
from el_paso.variable import Variable

if TYPE_CHECKING:
    from el_paso.processing.interpolate_in_time import InterpolationMethod


def rbsp_emfisis_waves_strategy(
    base_data_path: str | Path,
    satellite: RBSPSatellite,
    data_standard: ep.typing.DataStandard[ep.typing.StandardName] | None = None,
) -> ep.SavingStrategy:
    """Daily NetCDF wave saving strategy for RBSP EMFISIS."""
    return ep.saving_strategies.DailyWaveStrategy(
        Path(base_data_path),
        "RBSP",
        "rbsp" + satellite,
        "EMFISIS",
        data_standard or ep.data_standards.GFZStandard(),
    )


def process_rbsp_emfisis_waves(
    start_time: datetime,
    end_time: datetime,
    satellite: RBSPSatellite = "a",
    mag_field: Literal["T89", "T96", "TS04"] = "T89",
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    bin_cadence: timedelta = timedelta(minutes=5),
    num_cores: int = 16,
    save_strategy: Literal["netcdf"] = "netcdf",
    *,
    skip_existing: bool = True,
) -> None:
    """Process RBSP EMFISIS wave, density, and magnetometer data and save derived wave properties.

    Downloads and extracts the EMFISIS WFR spectral-matrix-diagonal data, the EMFISIS density
    data, the EMFISIS magnetometer data, and the EMFISIS wave-normal-angle (WNA) survey data for
    the given time range and satellite. The density and magnetometer data are interpolated onto
    the WFR time grid, the magnetometer data is cleaned using a quality flag, orbital quantities
    (L-shell, MLAT, MLT, electron cyclotron frequency) are derived from the cleaned magnetometer
    data, and the total magnetic wave power spectral density is computed from the WFR
    spectral-matrix components. The wave frequency, frequency bandwidth, wave normal angle,
    planarity, ellipticity, power spectral density, density, magnetic field, magnetic latitude,
    magnetic local time and electron gyrofrequency are saved using `DailyWaveStrategy`.

    Args:
        start_time (datetime): Start of the time range to process.
        end_time (datetime): End of the time range to process.
        satellite (Literal["a", "b"]): RBSP satellite identifier ("a" or "b").
        mag_field (Literal["T89", "T96", "TS04"]): Unused by this recipe; accepted only for
            interface consistency with other EL-PASO recipes, since EMFISIS wave processing
            derives its orbital quantities (L-shell, MLAT, MLT, electron cyclotron frequency)
            directly from the magnetometer data without using a magnetic field model.
        raw_data_path (str | Path): Base directory where raw CDF files are downloaded to
            and read from. Defaults to ".".
        processed_data_path (str | Path): Directory where the processed output files are
            written to. Defaults to ".".
        bin_cadence (timedelta): Unused by this recipe; accepted only for interface
            consistency with other EL-PASO recipes, since EMFISIS wave processing uses the
            instrument's own WFR time grid instead of a configurable binning cadence.
        num_cores (int): Unused by this recipe; accepted only for interface consistency
            with other EL-PASO recipes, since EMFISIS wave processing does not run parallel
            IRBEM computations.
        save_strategy (Literal["netcdf"]): Unused by this recipe; accepted only for
            interface consistency with other EL-PASO recipes, since the EMFISIS wave saving
            strategy factory only supports a single output format. Defaults to "netcdf".
        skip_existing (bool): If True, skip downloading files that already exist locally.
            Defaults to True.
    """
    del mag_field
    del bin_cadence
    del num_cores
    del save_strategy

    wfr_vars = _get_wfr_data(start_time, end_time, Path(raw_data_path), satellite, skip_existing=skip_existing)

    target_time_var = wfr_vars["Epoch"]
    density_vars = _get_density_data(
        start_time, end_time, Path(raw_data_path), satellite, target_time_var, skip_existing=skip_existing
    )
    mag_vars = _get_magnetometer_data(
        start_time, end_time, Path(raw_data_path), satellite, target_time_var, skip_existing=skip_existing
    )
    wna_vars = _get_wna_data(start_time, end_time, Path(raw_data_path), satellite, skip_existing=skip_existing)

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
        "Magnetic_Power_Spectral_Density": psd_var,
        "Number_density": density_vars["Density"],
        "B_total_obs": mag_vars["Bt"],
        "MLat": orbit_vars["MLat"],
        "MLT": orbit_vars["MLT"],
        "f_ce": orbit_vars["f_ce"],
        "f_ce_Eq": orbit_vars["f_ce_Eq"],
    }

    saving_strat = rbsp_emfisis_waves_strategy(processed_data_path, satellite)

    ep.save(vars_to_save, saving_strat, start_time, end_time)


def _calculate_orbital_vars(mag_vars: dict[str, ep.Variable]) -> dict[str, ep.Variable]:
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

    # The EMFISIS magnetometer product is already in SM coordinates, whose z-axis is the
    # geomagnetic dipole axis, so the latitude in this frame is the magnetic latitude.
    mlat_var = Variable(u.deg, data=mlat)

    # Computed from the measured field rather than a model, so this must not be paired with
    # the "f_ce_Eq" that compute_magnetic_field_variables derives from a modelled B.
    f_ce_var = ep.processing.compute_electron_gyrofrequency(mag_vars["Bt"])

    return {
        "L": Variable(u.dimensionless_unscaled, data=l_shell),
        "MLat": mlat_var,
        "MLT": Variable(u.hour, data=mlt),
        "f_ce": f_ce_var,
        "f_ce_Eq": ep.processing.map_to_dipole_equator(f_ce_var, mlat_var),
    }


def _get_wfr_data(
    start_time: datetime,
    end_time: datetime,
    raw_data_path: Path,
    satellite: RBSPSatellite,
    *,
    skip_existing: bool = True,
) -> dict[str, ep.Variable]:
    url = f"https://cdaweb.gsfc.nasa.gov/pub/data/rbsp/rbsp{satellite}/l2/emfisis/wfr/spectral-matrix-diagonal/YYYY/"
    file_name_stem = "rbsp-" + satellite + r"_wfr-spectral-matrix-diagonal_emfisis-l2_YYYYMMDD_.{6}.cdf"

    raw_data_path = raw_data_path / "YYYY" / "MM" / "wfr"

    ep.download(
        start_time,
        end_time,
        save_path=raw_data_path,
        download_url=url,
        file_name_stem=file_name_stem,
        file_cadence="daily",
        method="request",
        skip_existing=skip_existing,
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
    satellite: RBSPSatellite,
    *,
    skip_existing: bool = True,
) -> dict[str, ep.Variable]:
    url = f"https://cdaweb.gsfc.nasa.gov/pub/data/rbsp/rbsp{satellite}/l4/emfisis/wna-survey-sheath-corrected-e/YYYY/"
    file_name_stem = "rbsp-" + satellite + r"_wna-survey-sheath-corrected-e_emfisis-l4_YYYYMMDD_.{6}.cdf"

    raw_data_path = raw_data_path / "YYYY" / "MM" / "sna"

    ep.download(
        start_time,
        end_time,
        save_path=raw_data_path,
        download_url=url,
        file_name_stem=file_name_stem,
        file_cadence="daily",
        method="request",
        skip_existing=skip_existing,
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
    satellite: RBSPSatellite,
    target_time_var: ep.Variable,
    *,
    skip_existing: bool = True,
) -> dict[str, ep.Variable]:
    url = f"https://cdaweb.gsfc.nasa.gov/pub/data/rbsp/rbsp{satellite}/l4/emfisis/density/YYYY/"
    file_name_stem = "rbsp-" + satellite + r"_density_emfisis-l4_YYYYMMDD_.{7}.cdf"

    raw_data_path = raw_data_path / "YYYY" / "MM" / "density"

    ep.download(
        start_time,
        end_time,
        save_path=raw_data_path,
        download_url=url,
        file_name_stem=file_name_stem,
        file_cadence="daily",
        method="request",
        skip_existing=skip_existing,
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
    satellite: RBSPSatellite,
    target_time_var: ep.Variable,
    *,
    skip_existing: bool = True,
) -> dict[str, ep.Variable]:
    url = f"https://cdaweb.gsfc.nasa.gov/pub/data/rbsp/rbsp{satellite}/l3/emfisis/magnetometer/4sec/sm/YYYY/"
    file_name_stem = "rbsp-" + satellite + r"_magnetometer_4sec-sm_emfisis-l3_YYYYMMDD_.{6}.cdf"

    raw_data_path = raw_data_path / "YYYY" / "MM" / "magnetometer"

    ep.download(
        start_time,
        end_time,
        save_path=raw_data_path,
        download_url=url,
        file_name_stem=file_name_stem,
        file_cadence="daily",
        method="request",
        skip_existing=skip_existing,
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


if __name__ == "__main__":
    ep.run_recipe_cli(process_rbsp_emfisis_waves)
