# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, cast

import numpy as np
from astropy import units as u

import el_paso as ep
from el_paso.processing.magnetic_field_utils.irbem import Coords
from el_paso.recipes.arase import (
    get_arase_orbit_level_2_variables,
    get_arase_orbit_level_3_variables,
)
from el_paso.recipes.arase._mgf_magnetometer import compute_observed_B_variables, download_mgf_magnetometer
from el_paso.recipes.arase.get_arase_orbit_variables import interpolate_level_3_to_pitch_angles
from el_paso.recipes.solar_wind_indices.process_solar_wind_indices import process_solar_wind_indices

HepSensor = Literal["L", "H"]

_FILE_NAME_STEM = "erg_hep_l3_pa_YYYYMMDD_.{6}.cdf"
_URL = "https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/hep/l3/pa/YYYY/MM/"


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


@dataclass(frozen=True)
class _HepSensorConfig:
    """Static configuration of one HEP sensor inside the shared Level-3 CDF files."""

    instrument: str
    # number of leading energy channels that are kept (the remaining ones are contaminated or unusable)
    n_energy_channels: int
    # (start, end) intervals in which the pitch-angle distribution is not usable (end is exclusive)
    bad_periods: tuple[tuple[datetime, datetime], ...]


_HEP_SENSORS: dict[HepSensor, _HepSensorConfig] = {
    "L": _HepSensorConfig(
        instrument="hepl",
        n_energy_channels=14,
        bad_periods=(
            (_utc(2017, 6, 8), _utc(2017, 6, 22)),
            (_utc(2019, 4, 12), _utc(2019, 5, 18)),
        ),
    ),
    "H": _HepSensorConfig(
        instrument="heph",
        n_energy_channels=5,
        bad_periods=((_utc(2019, 4, 12), _utc(2019, 6, 29)),),
    ),
}


def arase_hep_gfz_strategy(
    base_data_path: str | Path,
    mag_field: ep.typing.MagneticFieldLiteral,
    sensor: HepSensor,
    data_standard: ep.typing.DataStandard[ep.typing.StandardName] | None = None,
) -> ep.SavingStrategy:
    """Legacy GFZ .mat saving strategy for Arase HEP-L / HEP-H."""
    return ep.saving_strategies.GFZStrategy(
        Path(base_data_path),
        "ARASE",
        "arase",
        _HEP_SENSORS[sensor].instrument,
        mag_field,
        data_standard=data_standard,
    )


def arase_hep_h5_strategy(
    base_data_path: str | Path,
    mag_field: ep.typing.MagneticFieldLiteral,
    sensor: HepSensor,
    data_standard: ep.typing.DataStandard[ep.typing.StandardName] | None = None,
    *,
    legacy_layout: bool = False,
) -> ep.SavingStrategy:
    """Monthly HDF5 saving strategy for Arase HEP-L / HEP-H."""
    return ep.saving_strategies.MonthlyRBStrategy(
        Path(base_data_path),
        "Arase",
        "arase",
        _HEP_SENSORS[sensor].instrument,
        mag_field,
        data_standard=data_standard or ep.data_standards.GFZStandard(),
        file_format="h5",
        legacy_layout=legacy_layout,
    )


def arase_hep_netcdf_strategy(
    base_data_path: str | Path,
    mag_field: ep.typing.MagneticFieldLiteral,
    sensor: HepSensor,
    data_standard: ep.typing.DataStandard[ep.typing.StandardName] | None = None,
    *,
    legacy_layout: bool = False,
) -> ep.SavingStrategy:
    """Monthly NetCDF saving strategy for Arase HEP-L / HEP-H."""
    return ep.saving_strategies.MonthlyRBStrategy(
        Path(base_data_path),
        "Arase",
        "arase",
        _HEP_SENSORS[sensor].instrument,
        mag_field,
        data_standard=data_standard or ep.data_standards.GFZStandard(),
        file_format="nc",
        legacy_layout=legacy_layout,
    )


def _select_hep_saver(
    sensor: HepSensor,
    processed_data_path: str | Path,
    mag_field: ep.typing.MagneticFieldLiteral,
    save_strategy: Literal["gfz", "h5", "netcdf"],
    data_standard: Literal["gfz", "prbem"],
    *,
    legacy_layout: bool = False,
) -> ep.SavingStrategy:
    """Pick the saving strategy of a HEP sensor for the requested format and data standard."""
    data_standard_instance = (
        ep.data_standards.GFZStandard() if data_standard == "gfz" else ep.data_standards.PRBEMStandard()
    )
    match save_strategy:
        case "gfz":
            return arase_hep_gfz_strategy(processed_data_path, mag_field, sensor, data_standard_instance)
        case "h5":
            return arase_hep_h5_strategy(
                processed_data_path, mag_field, sensor, data_standard_instance, legacy_layout=legacy_layout
            )
        case "netcdf":
            return arase_hep_netcdf_strategy(
                processed_data_path, mag_field, sensor, data_standard_instance, legacy_layout=legacy_layout
            )


def arase_hep_l_strategy(
    processed_data_path: str | Path,
    mag_field: ep.typing.MagneticFieldLiteral,
    save_strategy: Literal["gfz", "h5", "netcdf"] = "gfz",
    data_standard: Literal["gfz", "prbem"] = "gfz",
    *,
    legacy_layout: bool = False,
) -> ep.SavingStrategy:
    """Saving strategy of the Arase HEP-L recipe.

    Args:
        processed_data_path (str | Path): Base directory of the processed data.
        mag_field (ep.typing.MagneticFieldLiteral): Magnetic field model the data was processed with.
        save_strategy (Literal["gfz", "h5", "netcdf"]): Saving strategy of the output files. Defaults to "gfz".
        data_standard (Literal["gfz", "prbem"]): Data standard of the output files. Defaults to "gfz".
        legacy_layout (bool): Deprecated. Set to True to load data saved without instrument folders. Defaults to False.

    Returns:
        ep.SavingStrategy: Saving strategy for the given arguments.
    """
    return _select_hep_saver(
        "L", processed_data_path, mag_field, save_strategy, data_standard, legacy_layout=legacy_layout
    )


def arase_hep_h_strategy(
    processed_data_path: str | Path,
    mag_field: ep.typing.MagneticFieldLiteral,
    save_strategy: Literal["gfz", "h5", "netcdf"] = "gfz",
    data_standard: Literal["gfz", "prbem"] = "gfz",
    *,
    legacy_layout: bool = False,
) -> ep.SavingStrategy:
    """Saving strategy of the Arase HEP-H recipe.

    Args:
        processed_data_path (str | Path): Base directory of the processed data.
        mag_field (ep.typing.MagneticFieldLiteral): Magnetic field model the data was processed with.
        save_strategy (Literal["gfz", "h5", "netcdf"]): Saving strategy of the output files. Defaults to "gfz".
        data_standard (Literal["gfz", "prbem"]): Data standard of the output files. Defaults to "gfz".
        legacy_layout (bool): Deprecated. Set to True to load data saved without instrument folders. Defaults to False.

    Returns:
        ep.SavingStrategy: Saving strategy for the given arguments.
    """
    return _select_hep_saver(
        "H", processed_data_path, mag_field, save_strategy, data_standard, legacy_layout=legacy_layout
    )


def _download_hep(start_time: datetime, end_time: datetime, raw_data_path: Path, *, skip_existing: bool) -> None:
    ep.download(
        start_time,
        end_time,
        save_path=raw_data_path,
        download_url=_URL,
        file_name_stem=_FILE_NAME_STEM,
        file_cadence="daily",
        method="request",
        skip_existing=skip_existing,
    )


def _mask_bad_periods(
    flux: ep.Variable, epoch: ep.Variable, bad_periods: tuple[tuple[datetime, datetime], ...]
) -> None:
    """Set the flux to NaN inside the periods in which its pitch-angle distribution is unusable."""
    posix_time = epoch.get_data(ep.units.posixtime)
    mask = np.zeros(posix_time.shape, dtype=bool)
    for bad_start, bad_end in bad_periods:
        mask |= (posix_time >= bad_start.timestamp()) & (posix_time < bad_end.timestamp())

    if np.any(mask):
        logging.getLogger(__name__).warning(
            "The requested range overlaps periods in which HEP data is not suitable for pitch-angle "
            "analyses. Setting the flux in these periods to NaN."
        )
        data = flux.get_data().astype(np.float64)
        data[mask] = np.nan
        flux.set_data(data, "same")


def _run_arase_hep(
    sensor: HepSensor,
    start_time: datetime,
    end_time: datetime,
    mag_field: ep.typing.MagneticFieldLiteral,
    raw_data_path: str | Path,
    processed_data_path: str | Path,
    bin_cadence: timedelta,
    num_cores: int,
    save_strategy: Literal["gfz", "h5", "netcdf"],
    data_standard: Literal["gfz", "prbem"],
    *,
    use_level_3_orbit_data: bool,
    skip_existing: bool,
    add_B_obs: bool,
    calculate_Lstar: bool,
    save_indices_sw: bool,
    append: bool,
) -> None:
    if add_B_obs and data_standard == "prbem":
        msg = "add_B_obs requires the gfz data standard; the prbem standard does not define B_total_obs and InvMu_obs."
        raise ValueError(msg)

    if use_level_3_orbit_data and mag_field not in ("OP77Q", "T89", "TS04"):
        msg = (
            f"Invalid mag_field '{mag_field}' for Arase Level 3 orbit data. "
            "Valid options are OP77Q, T89 and TS04, or set use_level_3_orbit_data=False."
        )
        raise ValueError(msg)

    config = _HEP_SENSORS[sensor]
    raw_data_path = Path(raw_data_path)
    processed_data_path = Path(processed_data_path)

    _download_hep(start_time, end_time, raw_data_path, skip_existing=skip_existing)

    # downloaded before the orbit data, since the orbit helpers exit in download-only mode
    if add_B_obs:
        download_mgf_magnetometer(start_time, end_time, raw_data_path, skip_existing=skip_existing)

    if use_level_3_orbit_data:
        orb_variables = get_arase_orbit_level_3_variables(
            start_time,
            end_time,
            cast("Literal['OP77Q', 'T89', 'TS04']", mag_field),
            raw_data_path=raw_data_path,
            skip_existing=skip_existing,
        )
    else:
        orb_variables = get_arase_orbit_level_2_variables(
            start_time, end_time, raw_data_path=raw_data_path, skip_existing=skip_existing
        )

    extraction_infos = [
        ep.ExtractionInfo(
            result_key="Epoch",
            name_or_column=f"Epoch_{sensor}",
            unit=ep.units.tt2000,
        ),
        ep.ExtractionInfo(
            result_key="Energy",
            name_or_column=f"FEDU_{sensor}_Energy",
            unit=u.keV,
            is_time_dependent=False,
        ),
        ep.ExtractionInfo(
            result_key="Pitch_angle",
            name_or_column=f"FEDU_{sensor}_Alpha",
            unit=u.deg,
            is_time_dependent=False,
        ),
        ep.ExtractionInfo(
            result_key="FEDU",
            name_or_column=f"FEDU_{sensor}",
            unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
        ),
    ]

    hep_variables = ep.extract_variables_from_files(
        start_time,
        end_time,
        "daily",
        data_path=raw_data_path,
        file_name_stem=_FILE_NAME_STEM,
        extraction_infos=extraction_infos,
    )

    hep_variables["FEDU"].truncate(hep_variables["Epoch"], start_time, end_time)
    hep_variables["Epoch"].truncate(hep_variables["Epoch"], start_time, end_time)

    # energies are given as (lower, upper) bin edges; use the geometric mean (as in the CDF labels)
    # and keep only the usable channels
    energy_edges = hep_variables["Energy"].get_data().astype(np.float64)
    n_keep = config.n_energy_channels
    hep_variables["Energy"].set_data(np.sqrt(energy_edges[0, :n_keep] * energy_edges[1, :n_keep]), "same")
    hep_variables["FEDU"].set_data(hep_variables["FEDU"].get_data()[:, :n_keep, :], "same")

    _mask_bad_periods(hep_variables["FEDU"], hep_variables["Epoch"], config.bad_periods)

    time_bin_methods = {
        "Energy": ep.TimeBinMethod.Repeat,
        "FEDU": ep.TimeBinMethod.NanMedian,
        "Pitch_angle": ep.TimeBinMethod.Repeat,
    }

    _ = ep.processing.bin_by_time(
        hep_variables["Epoch"],
        variables=hep_variables,
        time_bin_method_dict=time_bin_methods,
        time_binning_cadence=bin_cadence,
        start_time=start_time,
        end_time=end_time,
    )

    ep.processing.fold_pitch_angles_and_flux(hep_variables["FEDU"], hep_variables["Pitch_angle"])

    # negative fluxes (background subtraction) were included in the median above
    hep_variables["FEDU"].apply_thresholds_on_data(lower_threshold=1e-21)

    if use_level_3_orbit_data:
        time_bin_methods = {
            "B_local": ep.TimeBinMethod.NanMedian,
            "B_eq": ep.TimeBinMethod.NanMedian,
            "Lm": ep.TimeBinMethod.NanMean,
            "MLT": ep.TimeBinMethod.NanMean,
            "R0": ep.TimeBinMethod.NanMean,
        }
        if calculate_Lstar:
            time_bin_methods["Lstar"] = ep.TimeBinMethod.NanMean

        binned_time_variable = ep.processing.bin_by_time(
            orb_variables["Epoch"],
            variables=orb_variables,
            time_bin_method_dict=time_bin_methods,
            time_binning_cadence=bin_cadence,
            start_time=start_time,
            end_time=end_time,
        )

        pa_local = hep_variables["Pitch_angle"].get_data(u.radian)
        pa_eq = np.asin(
            np.sin(pa_local)
            * np.sqrt(orb_variables["B_eq"].get_data(u.nT) / orb_variables["B_local"].get_data(u.nT))[:, np.newaxis]
        )
        hep_variables["Pa_eq"] = ep.Variable(data=pa_eq, original_unit=u.radian)

        # same as IRBEM's InvMu, which uses the model field at the satellite
        hep_variables["InvMu"] = ep.processing.compute_invariant_mu(
            hep_variables["Energy"], hep_variables["Pitch_angle"], orb_variables["B_local"], "electron"
        )

        orb_variables["Lm"] = interpolate_level_3_to_pitch_angles(
            orb_variables["Lm"], hep_variables["Pitch_angle"].get_data(u.deg)
        )
        if calculate_Lstar:
            orb_variables["Lstar"] = interpolate_level_3_to_pitch_angles(
                orb_variables["Lstar"], hep_variables["Pitch_angle"].get_data(u.deg)
            )

    else:
        time_bin_methods = {
            "pos_sm": ep.TimeBinMethod.NanMean,
        }

        binned_time_variable = ep.processing.bin_by_time(
            orb_variables["Epoch"],
            variables=orb_variables,
            time_bin_method_dict=time_bin_methods,
            time_binning_cadence=bin_cadence,
            start_time=start_time,
            end_time=end_time,
        )

        datetimes = [
            datetime.fromtimestamp(t, tz=timezone.utc) for t in binned_time_variable.get_data(ep.units.posixtime)
        ]

        geo_data = Coords().transform(
            time=datetimes,
            pos=orb_variables["pos_sm"].get_data().astype(np.float64),
            sysaxes_in=ep.IRBEM_SYSAXIS_SM,
            sysaxes_out=ep.IRBEM_SYSAXIS_GEO,
        )
        pos_geo_var = ep.Variable(data=geo_data, original_unit=ep.units.RE)

        irbem_options = ep.processing.magnetic_field_utils.IrbemOptions(
            lstar_quantity=ep.processing.magnetic_field_utils.LstarQuantity.LSTAR
            if calculate_Lstar
            else ep.processing.magnetic_field_utils.LstarQuantity.NONE,
        )

        variables_to_compute: ep.processing.VariableRequest = [
            ("B_Calc", mag_field),
            ("MLT", mag_field),
            ("B_Eq", mag_field),
            ("R_Eq", mag_field),
            ("Alpha_Eq", mag_field),
            ("L_m", mag_field),
            ("InvK", mag_field),
            ("InvMu", mag_field),
        ]

        if calculate_Lstar:
            variables_to_compute.append(("L_star", mag_field))

        magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
            time_var=binned_time_variable,
            xgeo_var=pos_geo_var,
            variables_to_compute=variables_to_compute,
            irbem_options=irbem_options,
            num_cores=num_cores,
            pa_local_var=hep_variables["Pitch_angle"],
            energy_var=hep_variables["Energy"],
            particle_species="electron",
        )

        orb_variables["R0"] = magnetic_field_variables["R_Eq_" + mag_field]
        orb_variables["MLT"] = magnetic_field_variables["MLT_" + mag_field]
        hep_variables["Pa_eq"] = magnetic_field_variables["Alpha_Eq_" + mag_field]
        orb_variables["Lm"] = magnetic_field_variables["L_m_" + mag_field]
        if calculate_Lstar:
            orb_variables["Lstar"] = magnetic_field_variables["L_star_" + mag_field]
        hep_variables["InvMu"] = magnetic_field_variables["InvMu_" + mag_field]
        hep_variables["InvK"] = magnetic_field_variables["InvK_" + mag_field]

    psd_var = ep.processing.compute_phase_space_density(
        hep_variables["FEDU"], hep_variables["Energy"], particle_species="electron"
    )

    variables_to_save: ep.typing.VariablesDict = {
        "Epoch": binned_time_variable,
        "FEDU": hep_variables["FEDU"],
        "Energy_FEDU": hep_variables["Energy"],
        "Alpha": hep_variables["Pitch_angle"],
        "Alpha_Eq": hep_variables["Pa_eq"],
        "R_Eq": orb_variables["R0"],
        "MLT": orb_variables["MLT"],
        "L_m": orb_variables["Lm"],
        "InvMu": hep_variables["InvMu"],
        "PSD": psd_var,
    }

    if calculate_Lstar:
        variables_to_save["L_star"] = orb_variables["Lstar"]

    # Level 3 orbit data does not provide the second adiabatic invariant
    if not use_level_3_orbit_data:
        variables_to_save["InvK"] = hep_variables["InvK"]

    if add_B_obs:
        variables_to_save |= compute_observed_B_variables(
            start_time,
            end_time,
            raw_data_path,
            binned_time_variable,
            bin_cadence,
            hep_variables["Energy"],
            hep_variables["Pitch_angle"],
            "electron",
        )

    strategy_factory = arase_hep_l_strategy if sensor == "L" else arase_hep_h_strategy
    saving_strategy = strategy_factory(processed_data_path, mag_field, save_strategy, data_standard)

    ep.save(variables_to_save, saving_strategy, start_time, end_time, binned_time_variable, append=append)

    if save_indices_sw:
        process_solar_wind_indices(
            start_time,
            end_time,
            processed_data_path=processed_data_path,
            bin_cadence=bin_cadence,
        )


def process_arase_hep_l(
    start_time: datetime,
    end_time: datetime,
    satellite: Literal["arase"] = "arase",
    mag_field: ep.typing.MagneticFieldLiteral = "T89",
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    bin_cadence: timedelta = timedelta(minutes=5),
    num_cores: int = 16,
    save_strategy: Literal["gfz", "h5", "netcdf"] = "gfz",
    data_standard: Literal["gfz", "prbem"] = "gfz",
    *,
    use_level_3_orbit_data: bool = True,
    skip_existing: bool = True,
    add_B_obs: bool = False,
    calculate_Lstar: bool = True,
    save_indices_sw: bool = False,
    append: bool = False,
) -> None:
    """Process Arase HEP-L Level 3 electron flux data and save derived products.

    Downloads the corresponding Arase orbit data (Level 3 if `use_level_3_orbit_data` is True,
    otherwise Level 2) and the daily Arase HEP Level 3 pitch-angle resolved flux (FEDU) CDF files for
    the requested time range, time-bins the flux onto a common cadence (negative fluxes from the
    background subtraction are included in the median), folds the pitch-angle distribution, applies
    a lower flux threshold, and computes the equatorial pitch angle and magnetic-field-related
    quantities (Lm, MLT, R0, InvMu, and InvK when using Level 2 orbit data) either from the Level 3
    orbit data directly or via IRBEM using the SM position from Level 2 orbit data. The phase space
    density (PSD) is computed from the flux. If `add_B_obs` is set, the observed magnetic field
    magnitude from the MGF magnetometer and the first adiabatic invariant computed from it are added.

    Only the 14 lowest energy channels (44 to 994.5 keV) are kept. The two highest channels are
    contaminated or almost entirely non-positive. The flux is set to NaN during 2017-06-08 to
    2017-06-21 and 2019-04-12 to 2019-05-17, which the instrument team marks as unsuitable for
    analyses of the angular distribution. The lowest energy channel may differ from the true flux
    due to a threshold ambiguity. See https://ergsc.isee.nagoya-u.ac.jp/mw/index.php/ErgSat/Hep.

    Args:
        start_time (datetime): Start of the time range to process.
        end_time (datetime): End of the time range to process.
        satellite (Literal["arase"]): Identifier of the satellite to process. Arase is a
            single-satellite mission, so this has only one value.
        mag_field (ep.typing.MagneticFieldLiteral): The magnetic field model used for the
            magnetic-field-related output variables.
        raw_data_path (str | Path): Directory where downloaded raw data files are
            stored. Defaults to ".".
        processed_data_path (str | Path): Base directory where the processed output
            data is saved. Defaults to ".".
        bin_cadence (timedelta): Time binning cadence applied to all variables.
        num_cores (int): Number of CPU cores used for the IRBEM magnetic field
            computations (only used when `use_level_3_orbit_data` is False).
        save_strategy (Literal["gfz", "h5", "netcdf"]): The saving strategy used to
            write the processed data. Defaults to "gfz".
        data_standard (Literal["gfz", "prbem"]): The data standard used when saving
            the processed data. Defaults to "gfz".
        use_level_3_orbit_data (bool): If True, use Arase Level 3 orbit data (which
            already contains precomputed magnetic field quantities for `mag_field`); if False,
            use Level 2 orbit data and compute the magnetic field quantities via IRBEM.
            Defaults to True.
        skip_existing (bool): If True, skip downloading files that already exist locally.
            Defaults to True.
        add_B_obs (bool): If True, also download the MGF Level 2 8-second magnetometer data and save
            the observed magnetic field magnitude (B_total_obs) and InvMu_obs computed
            from it. Only supported with the "gfz" data standard. Defaults to False.
        calculate_Lstar (bool): If True, also compute (or, when `use_level_3_orbit_data` is True,
            interpolate and save) the L* magnetic field quantity. Defaults to True.
        save_indices_sw (bool): If True, also save all solar wind indices and parameters,
            independent of this recipe's own output. Defaults to False.
        append (bool): If True, append to existing output files instead of overwriting them.
            Defaults to False.

    Raises:
        ValueError: If `use_level_3_orbit_data` is set and `mag_field` is not one of "OP77Q", "T89" or "TS04".
        ValueError: If `add_B_obs` is set together with the "prbem" data standard, which does not
            define B_total_obs and InvMu_obs.
    """
    del satellite

    _run_arase_hep(
        "L",
        start_time,
        end_time,
        mag_field,
        raw_data_path,
        processed_data_path,
        bin_cadence,
        num_cores,
        save_strategy,
        data_standard,
        use_level_3_orbit_data=use_level_3_orbit_data,
        skip_existing=skip_existing,
        add_B_obs=add_B_obs,
        calculate_Lstar=calculate_Lstar,
        save_indices_sw=save_indices_sw,
        append=append,
    )


def process_arase_hep_h(
    start_time: datetime,
    end_time: datetime,
    satellite: Literal["arase"] = "arase",
    mag_field: ep.typing.MagneticFieldLiteral = "T89",
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    bin_cadence: timedelta = timedelta(minutes=5),
    num_cores: int = 16,
    save_strategy: Literal["gfz", "h5", "netcdf"] = "gfz",
    data_standard: Literal["gfz", "prbem"] = "gfz",
    *,
    use_level_3_orbit_data: bool = True,
    skip_existing: bool = True,
    add_B_obs: bool = False,
    calculate_Lstar: bool = True,
    save_indices_sw: bool = False,
    append: bool = False,
) -> None:
    """Process Arase HEP-H Level 3 electron flux data and save derived products.

    Downloads the corresponding Arase orbit data (Level 3 if `use_level_3_orbit_data` is True,
    otherwise Level 2) and the daily Arase HEP Level 3 pitch-angle resolved flux (FEDU) CDF files for
    the requested time range, time-bins the flux onto a common cadence (negative fluxes from the
    background subtraction are included in the median), folds the pitch-angle distribution, applies
    a lower flux threshold, and computes the equatorial pitch angle and magnetic-field-related
    quantities (Lm, MLT, R0, InvMu, and InvK when using Level 2 orbit data) either from the Level 3
    orbit data directly or via IRBEM using the SM position from Level 2 orbit data. The phase space
    density (PSD) is computed from the flux. If `add_B_obs` is set, the observed magnetic field
    magnitude from the MGF magnetometer and the first adiabatic invariant computed from it are added.

    Only the 5 lowest energy channels (541 to 1095.5 keV) are kept, because the instrument team marks
    channels 6 and higher as unsuitable for analyses of the angular distribution. The flux is set to
    NaN during 2019-04-12 to 2019-06-28, which is also marked as unsuitable.
    See https://ergsc.isee.nagoya-u.ac.jp/mw/index.php/ErgSat/Hep.

    Args:
        start_time (datetime): Start of the time range to process.
        end_time (datetime): End of the time range to process.
        satellite (Literal["arase"]): Identifier of the satellite to process. Arase is a
            single-satellite mission, so this has only one value.
        mag_field (ep.typing.MagneticFieldLiteral): The magnetic field model used for the
            magnetic-field-related output variables.
        raw_data_path (str | Path): Directory where downloaded raw data files are
            stored. Defaults to ".".
        processed_data_path (str | Path): Base directory where the processed output
            data is saved. Defaults to ".".
        bin_cadence (timedelta): Time binning cadence applied to all variables.
        num_cores (int): Number of CPU cores used for the IRBEM magnetic field
            computations (only used when `use_level_3_orbit_data` is False).
        save_strategy (Literal["gfz", "h5", "netcdf"]): The saving strategy used to
            write the processed data. Defaults to "gfz".
        data_standard (Literal["gfz", "prbem"]): The data standard used when saving
            the processed data. Defaults to "gfz".
        use_level_3_orbit_data (bool): If True, use Arase Level 3 orbit data (which
            already contains precomputed magnetic field quantities for `mag_field`); if False,
            use Level 2 orbit data and compute the magnetic field quantities via IRBEM.
            Defaults to True.
        skip_existing (bool): If True, skip downloading files that already exist locally.
            Defaults to True.
        add_B_obs (bool): If True, also download the MGF Level 2 8-second magnetometer data and save
            the observed magnetic field magnitude (B_total_obs) and InvMu_obs computed
            from it. Only supported with the "gfz" data standard. Defaults to False.
        calculate_Lstar (bool): If True, also compute (or, when `use_level_3_orbit_data` is True,
            interpolate and save) the L* magnetic field quantity. Defaults to True.
        save_indices_sw (bool): If True, also save all solar wind indices and parameters,
            independent of this recipe's own output. Defaults to False.
        append (bool): If True, append to existing output files instead of overwriting them.
            Defaults to False.

    Raises:
        ValueError: If `use_level_3_orbit_data` is set and `mag_field` is not one of "OP77Q", "T89" or "TS04".
        ValueError: If `add_B_obs` is set together with the "prbem" data standard, which does not
            define B_total_obs and InvMu_obs.
    """
    del satellite

    _run_arase_hep(
        "H",
        start_time,
        end_time,
        mag_field,
        raw_data_path,
        processed_data_path,
        bin_cadence,
        num_cores,
        save_strategy,
        data_standard,
        use_level_3_orbit_data=use_level_3_orbit_data,
        skip_existing=skip_existing,
        add_B_obs=add_B_obs,
        calculate_Lstar=calculate_Lstar,
        save_indices_sw=save_indices_sw,
        append=append,
    )


PLOT_DEFAULTS = {
    "panels": ["Kp", "FEDU"],
    "FEDU": {"pitch_angle": 90 * u.deg},
}

if __name__ == "__main__":
    ep.run_recipe_cli(process_arase_hep_l, strategy=arase_hep_l_strategy, plot_defaults=PLOT_DEFAULTS)
