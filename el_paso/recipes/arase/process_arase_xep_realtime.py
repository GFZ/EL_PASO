# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import logging
import os
import sys
import typing
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from astropy import units as u

from el_paso.utils import timed_function

if TYPE_CHECKING:
    from numpy.typing import NDArray

import el_paso as ep


@timed_function("process_arase_xep_real_time")
def process_arase_xep_real_time(  # noqa: D417
    processed_data_path: str | Path,
    download_data_dir: str | Path,
    start_time: datetime,
    end_time: datetime,
    erg_user: str | None = None,
    erg_password: str | None = None,
    num_cores: int = 32,
    save_strategy: Literal["gfz", "netcdf", "both"] = "netcdf",
    *,
    download: bool = True,
    skip_existing: bool = True,
    do_xep_extraction: bool = True,
) -> None:
    """Process Arase XEP real-time electron flux data and save derived products.

    Downloads (unless `download` is False) and extracts the daily Arase real-time XEP
    omnidirectional flux (FEDO) text files and the real-time orbit position text files for the
    requested time range, converts the orbit position from SM to GEO coordinates, time-bins the
    flux and position variables onto a 5-minute cadence, computes magnetic-field-related
    quantities (B_Calc, B_Eq, MLT, R_Eq, Alpha_Eq, L_star, L_m, InvMu, InvK) for the T89 model via
    IRBEM, constructs a pitch-angle distribution (FEDU) from the omnidirectional flux, applies a
    lower flux threshold, computes the electron phase space density, and saves the resulting
    variables using the requested saving strategy/strategies.

    Args:
        processed_data_path (str | Path): Base directory where the processed output data is saved.
        download_data_dir (str | Path): Base directory where downloaded raw data files are stored.
        start_time (datetime): Start of the time range to process.
        end_time (datetime): End of the time range to process.
        erg_user (str | None, optional): Username for the ERG data server. If None, it is read
                                        from the ``ERG_USER`` environment variable.
                                        Defaults to None.
        erg_password (str | None, optional): Password for the ERG data server. If None, it is
                                            read from the ``ERG_PASSWORD`` environment variable.
                                            Defaults to None.
        num_cores (int, optional): Number of CPU cores used for the IRBEM magnetic field
                                computations. Defaults to 32.
        save_strategy (Literal["gfz", "netcdf", "both"], optional): Which saving strategy/strategies
                                                                    to use for writing the processed
                                                                    data. Defaults to "netcdf".
        download (bool, optional): Whether to download the raw data files before processing.
                                Defaults to True.
        skip_existing (bool, optional): Whether to skip downloading files that already exist
                                        locally. Defaults to True.

    Raises:
        ValueError: If `erg_user` is not provided and the ``ERG_USER`` environment variable is
                not set, or if `erg_password` is not provided and the ``ERG_PASSWORD``
                environment variable is not set.
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if erg_user is None:
        erg_user = os.environ.get("ERG_USER")
    if erg_password is None:
        erg_password = os.environ.get("ERG_PASSWORD")

    if erg_user is None:
        msg = "ERG_USER not found! Either load it from environment variables or pass it as an argument."
        raise ValueError(msg)

    if erg_password is None:
        msg = "ERG_PASSWORD not found! Either load it from environment variables or pass it as an argument."
        raise ValueError(msg)

    if do_xep_extraction:
        xep_variables = _get_xep_variables(
            download_data_dir,
            start_time,
            end_time,
            erg_user,
            erg_password,
            download=download,
            skip_existing=skip_existing,
        )
    orb_variables = _get_orb_variables(
        download_data_dir,
        start_time,
        end_time,
        erg_user,
        erg_password,
        download=download,
        skip_existing=skip_existing,
    )

    time_bin_methods_xep = {
        "FEDO": ep.TimeBinMethod.NanMedian,
        "Energy_FEDO": ep.TimeBinMethod.Repeat,
        "PA_local_FEDU": ep.TimeBinMethod.Repeat,
    }
    binned_time_var = ep.processing.bin_by_time(
        xep_variables["Epoch"],
        xep_variables,
        time_bin_methods_xep,
        time_binning_cadence=timedelta(minutes=5),
        start_time=start_time,
        end_time=end_time,
    )

    time_bin_methods_orb = {
        "xGEO": ep.TimeBinMethod.NanMedian,
    }
    _ = ep.processing.bin_by_time(
        orb_variables["Epoch"],
        orb_variables,
        time_bin_methods_orb,
        time_binning_cadence=timedelta(minutes=5),
        start_time=start_time,
        end_time=end_time,
    )

    variables_combined = xep_variables | orb_variables

    variables_to_compute: ep.processing.VariableRequest = [
        ("B_Calc", "T89"),
        ("B_Eq", "T89"),
        ("MLT", "T89"),
        ("R_Eq", "T89"),
        ("Alpha_Eq", "T89"),
        ("L_star", "T89"),
        ("L_m", "T89"),
        ("InvMu", "T89"),
        ("InvK", "T89"),
    ]

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=binned_time_var,
        xgeo_var=variables_combined["xGEO"],
        energy_var=variables_combined["Energy_FEDO"],
        pa_local_var=variables_combined["PA_local_FEDU"],
        particle_species="electron",
        variables_to_compute=variables_to_compute,
        irbem_options=ep.processing.magnetic_field_utils.IrbemOptions(),
        num_cores=num_cores,
    )

    variables_combined |= magnetic_field_variables

    FEDU_var = ep.processing.construct_pitch_angle_distribution(
        variables_combined["FEDO"], variables_combined["PA_local_FEDU"], magnetic_field_variables["Alpha_Eq_T89"]
    )
    FEDU_var.apply_thresholds_on_data(lower_threshold=0)

    psd_var = ep.processing.compute_phase_space_density(
        FEDU_var, variables_combined["Energy_FEDO"], particle_species="electron"
    )

    variables_to_save: dict[ep.typing.InternalName, ep.Variable] = {
        "Epoch": binned_time_var,
        "FEDU": FEDU_var,
        "Energy_FEDU": variables_combined["Energy_FEDO"],
        "Alpha": variables_combined["PA_local_FEDU"],
        "Alpha_Eq": magnetic_field_variables["Alpha_Eq_T89"],
        "R_Eq": magnetic_field_variables["R_Eq_T89"],
        "MLT": magnetic_field_variables["MLT_T89"],
        "L_star": magnetic_field_variables["L_star_T89"],
        "B_Calc": magnetic_field_variables["B_Calc_T89"],
        "B_Eq": magnetic_field_variables["B_Eq_T89"],
        "PSD": psd_var,
        "InvMu": magnetic_field_variables["InvMu_T89"],
        "InvK": magnetic_field_variables["InvK_T89"],
        "Position": variables_combined["xGEO"],
    }

    if save_strategy in ("gfz", "both"):
        saving_strategy = ep.saving_strategies.GFZStrategy(
            processed_data_path,
            mission="Arase",
            satellite="Arase",
            instrument="XEP",
            mag_field="T89",
        )

        ep.save(
            variables_to_save,
            saving_strategy,
            start_time,
            end_time,
            time_var=binned_time_var,
            append=True,
        )

    if save_strategy in ("netcdf", "both"):
        saving_strategy = ep.saving_strategies.MonthlyRBStrategy(
            base_data_path=Path(processed_data_path),
            mission="Arase",
            satellite="arase",
            instrument="xep",
            mag_field="T89",
            data_standard=ep.data_standards.GFZStandard(),
        )

        ep.save(
            variables_to_save,
            saving_strategy,
            start_time,
            end_time,
            time_var=binned_time_var,
            append=True,
        )


def _get_xep_variables(
    download_data_dir: str | Path,
    start_time: datetime,
    end_time: datetime,
    erg_user: str,
    erg_password: str,
    *,
    download: bool,
    skip_existing: bool,
) -> dict[str, ep.Variable]:
    # Energies from the User's guide
    energy_min = np.asarray((400.0, 600.0, 1000.0, 1500.0, 2200.0, 3500.0, 4300.0, 5400.0))
    energy_max = np.asarray((600.0, 1000.0, 1500.0, 2200.0, 3500.0, 4300.0, 5400.0, 9800.0))
    energy_mean = _get_mean_energy(energy_min, energy_max)

    data_path_stem = f"{download_data_dir}/ARASE/YYYY/MM/"
    file_name_stem = "erg_real_xep_YYYYMMDD_v002.txt"
    url = "https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/swx/xep/l2/"

    if download:
        ep.download(
            start_time,
            end_time,
            save_path=data_path_stem,
            file_cadence="daily",
            download_url=url,
            authentication_info=(erg_user, erg_password),
            file_name_stem=file_name_stem,
            skip_existing=skip_existing,
        )

    fedo_unit = typing.cast("u.Unit", (u.cm**2 * u.s * u.keV) ** (-1))

    extraction_infos = [
        ep.ExtractionInfo(name_or_column="time", unit=u.dimensionless_unscaled, result_key="Epoch"),
        ep.ExtractionInfo(name_or_column="ch1", unit=fedo_unit, result_key="FEDO_ch1"),
        ep.ExtractionInfo(name_or_column="ch2", unit=fedo_unit, result_key="FEDO_ch2"),
        ep.ExtractionInfo(name_or_column="ch3", unit=fedo_unit, result_key="FEDO_ch3"),
        ep.ExtractionInfo(name_or_column="ch4", unit=fedo_unit, result_key="FEDO_ch4"),
        ep.ExtractionInfo(name_or_column="ch5", unit=fedo_unit, result_key="FEDO_ch5"),
        ep.ExtractionInfo(name_or_column="ch6", unit=fedo_unit, result_key="FEDO_ch6"),
        ep.ExtractionInfo(name_or_column="ch7", unit=fedo_unit, result_key="FEDO_ch7"),
        ep.ExtractionInfo(name_or_column="ch8", unit=fedo_unit, result_key="FEDO_ch8"),
    ]

    # Bernhard: the header is also in the file, but there is a comment after it, so it cannot be read by pd.read_csv
    xep_header = ("time", "ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8")
    xep_variables = ep.extract_variables_from_files(
        extraction_infos=extraction_infos,
        data_path=data_path_stem,
        file_name_stem=file_name_stem,
        start_time=start_time,
        end_time=end_time,
        file_cadence="daily",
        pd_read_csv_kwargs={"skiprows": 6, "names": xep_header},
    )

    # convert time variable
    # parse time strings
    datetimes = ep.processing.convert_string_to_datetime(xep_variables["Epoch"])
    xep_variables["Epoch"].set_data(np.asarray([t.timestamp() for t in datetimes]), unit=ep.units.posixtime)

    # add energy variable
    energy_var = ep.Variable(original_unit=u.keV, data=energy_mean)
    xep_variables["Energy_FEDO"] = energy_var

    # add local pitch angle variable
    pa_local_data = np.arange(5, 91, 5)
    xep_variables["PA_local_FEDU"] = ep.Variable(data=pa_local_data, original_unit=u.deg)

    # build flux variable from channels
    fedo_data = np.vstack(
        (
            xep_variables["FEDO_ch1"].get_data().astype(np.float64),
            xep_variables["FEDO_ch2"].get_data().astype(np.float64),
            xep_variables["FEDO_ch3"].get_data().astype(np.float64),
            xep_variables["FEDO_ch4"].get_data().astype(np.float64),
            xep_variables["FEDO_ch5"].get_data().astype(np.float64),
            xep_variables["FEDO_ch6"].get_data().astype(np.float64),
            xep_variables["FEDO_ch7"].get_data().astype(np.float64),
            xep_variables["FEDO_ch8"].get_data().astype(np.float64),
        )
    ).T

    fedo_var = ep.Variable(
        original_unit=fedo_unit,
        data=fedo_data,
    )
    fedo_var.apply_thresholds_on_data(lower_threshold=0)
    xep_variables["FEDO"] = fedo_var

    # delete unused variables
    del xep_variables["FEDO_ch1"]
    del xep_variables["FEDO_ch2"]
    del xep_variables["FEDO_ch3"]
    del xep_variables["FEDO_ch4"]
    del xep_variables["FEDO_ch5"]
    del xep_variables["FEDO_ch6"]
    del xep_variables["FEDO_ch7"]
    del xep_variables["FEDO_ch8"]

    return xep_variables


def _get_orb_variables(
    download_data_dir: str | Path,
    start_time: datetime,
    end_time: datetime,
    erg_user: str,
    erg_password: str,
    *,
    download: bool,
    skip_existing: bool,
) -> dict[str, ep.Variable]:
    data_path_stem = f"{download_data_dir}/ARASE/YYYY/MM/"
    file_name_stem = "erg_orb_pre_l2_YYYYMMDD_v01.txt"
    url = "https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/swx/orb/"

    if download:
        ep.download(
            start_time,
            end_time,
            save_path=data_path_stem,
            file_cadence="daily",
            download_url=url,
            authentication_info=(erg_user, erg_password),
            file_name_stem=file_name_stem,
            skip_existing=skip_existing,
        )

    extraction_infos = [
        ep.ExtractionInfo(name_or_column="time", unit=u.dimensionless_unscaled, result_key="Epoch"),
        ep.ExtractionInfo(name_or_column="sm_x", unit=ep.units.RE, result_key="sm_x"),
        ep.ExtractionInfo(name_or_column="sm_y", unit=ep.units.RE, result_key="sm_y"),
        ep.ExtractionInfo(name_or_column="sm_z", unit=ep.units.RE, result_key="sm_z"),
    ]

    orb_variables = ep.extract_variables_from_files(
        extraction_infos=extraction_infos,
        data_path=data_path_stem,
        file_name_stem=file_name_stem,
        start_time=start_time,
        end_time=end_time,
        file_cadence="daily",
    )

    datetimes = ep.processing.convert_string_to_datetime(orb_variables["Epoch"])
    orb_variables["Epoch"].set_data(np.asarray([t.timestamp() for t in datetimes]), unit=ep.units.posixtime)

    # convert SM to GEO
    xsm_arr = np.stack(
        (
            orb_variables["sm_x"].get_data(),
            orb_variables["sm_y"].get_data(),
            orb_variables["sm_z"].get_data(),
        )
    ).T.astype(np.float64)

    model_coord = ep.processing.magnetic_field_utils.Coords()

    xgeo_arr = model_coord.transform(list(datetimes), xsm_arr, ep.IRBEM_SYSAXIS_SM, ep.IRBEM_SYSAXIS_GEO)
    orb_variables["xGEO"] = ep.Variable(data=xgeo_arr, original_unit=ep.units.RE)

    # delete unused variables
    del orb_variables["sm_x"]
    del orb_variables["sm_y"]
    del orb_variables["sm_z"]

    return orb_variables


def _get_mean_energy(e_min: NDArray[np.float64], e_max: NDArray[np.float64]) -> NDArray[np.float64]:
    b = 7.068e-3

    weighted_max = (1 / b) * np.exp(-b * e_max)
    weighted_min = (1 / b) * np.exp(-b * e_min)

    tmp = (weighted_min - weighted_max) / (e_max - e_min)

    return -np.log(tmp) / b
