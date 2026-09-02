# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import logging
import os
import sys
import typing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import dateutil
import numpy as np
from astropy import units as u
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation

import el_paso as ep
from el_paso.utils import timed_function

logger = logging.getLogger(__name__)

CHI2_BAD_QUALITY_THRESHOLD = 2
EPT_ENERGY_LIMITS = [0.5, 0.6, 0.7, 0.8, 1.0, 2.4, 8.0]

SATELLITE_TO_ID = {
    "EDRS-C": "https://swe.ssa.esa.int/hapi/data?id=spase://SSA/NumericalData/D3S/d3s_edrsc_ngrm_spid204030252_science_ep_l1_gc_v3",
    "S6-MF": "https://swe.ssa.esa.int/hapi/data?id=spase://SSA/NumericalData/D3S/d3s_sentinel6mf_ngrm_science_ep_l1_gc_v1",
    "S6-B": "https://swe.ssa.esa.int/hapi/data?id=spase://SSA/NumericalData/D3S/d3s_sentinel6b_ngrm_science_ep_l1_gc_v1",
    "MTG-S1": "https://swe.ssa.esa.int/hapi/data?id=spase://SSA/NumericalData/D3S/d3s_mtgs1_ngrm_science_ep_l1_gc_v1",
    "MTG-I1": "https://swe.ssa.esa.int/hapi/data?id=spase://SSA/NumericalData/D3S/d3s_mtgi1_ngrm_science_ep_l1_gc_v1",
}
NGRM_ENERGIES = [0.18, 0.27, 0.40, 0.60, 0.88, 1.30, 1.93, 2.90, 3.40, 4.00]


@timed_function("process_ngrm_electron_fluxes")
def process_ngrm_electron_fluxes(
    start_time: datetime,
    end_time: datetime,
    satellite: Literal["EDRS-C", "S6-MF", "S6-B", "MTG-S1", "MTG-I1"] = "EDRS-C",
    mag_field: ep.typing.MagneticFieldLiteral = "T89",
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    bin_cadence: timedelta = timedelta(seconds=10),
    num_cores: int = 16,
    client_id: str | None = None,
    client_secret: str | None = None,
    saving_strategy: ep.SavingStrategy | None = None,
    skip_existing: bool = True,  # noqa: FBT001, FBT002,
    *,
    calculate_Lstar: bool = True,
) -> None:
    """Process ESA NGRM electron flux data for the given satellite into omnidirectional fluxes and PSD.

    This downloads the NGRM L1d CSV data for the given satellite from the ESA SWE HAPI service,
    extracts the differential electron flux channels, time, ECI position, and L-shell, converts
    the time stamps to POSIX time and the ECI position to GEO coordinates, and combines the flux
    channels into a single omnidirectional differential flux variable (converted to
    (cm^2 s keV)^-1 units with negative values clipped to zero). The data is then time-binned to
    `bin_cadence`, magnetic field model quantities (B_Calc, B_Eq, MLT_Eq, R_Eq, Alpha_Eq, L_m, and
    optionally L_star) are computed using `mag_field` for a fixed set of local pitch angles, the
    pitch angle distribution (FEDU) and phase space density (PSD) are derived from the omnidirectional
    flux, and all resulting variables are saved to disk (appending to existing files) using either the
    provided `saving_strategy` or a default daily/monthly strategy depending on the satellite.

    Args:
        start_time (datetime): Start of the time range to process.
        end_time (datetime): End of the time range to process.
        satellite (Literal["EDRS-C", "S6-MF", "S6-B", "MTG-S1", "MTG-I1"]): Identifier of the
            NGRM-carrying satellite to process.
        mag_field (MagneticFieldLiteral): Magnetic field model used for the derived quantities.
        raw_data_path (str | Path): Base directory used for downloading and locating the raw NGRM CSV files.
        processed_data_path (str | Path): Base directory in which the processed output files are saved.
        bin_cadence (timedelta): Time binning cadence applied to the extracted variables.
        num_cores (int): Number of CPU cores used for the magnetic field computations.
        client_id (str | None): Client ID for the ESA SWE authentication. If None, it is read
            from the `CLIENT_ID` environment variable.
        client_secret (str | None): Client secret for the ESA SWE authentication. If None, it
            is read from the `CLIENT_SECRET` environment variable.
        saving_strategy (ep.SavingStrategy | None): Strategy used to save the processed
            variables. If None, a `DailyLEORBStrategy` is used for the Sentinel-6 satellites and a
            `MonthlyRBStrategy` for all others. Not available on the command line.
        skip_existing (bool): If True, skip downloading files that already exist locally.
        calculate_Lstar (bool): If True, also compute the L* magnetic field quantity.

    Raises:
        ValueError: If `client_id` or `client_secret` is not provided and not available via the
            `CLIENT_ID`/`CLIENT_SECRET` environment variables.
    """
    data_path_stem = f"{raw_data_path}/NGRM/{satellite}/YYYY/MM/"
    file_name_stem = f"{satellite}_ngrm_YYYYMMDD_L1d.csv"

    if client_id is None:
        client_id = os.environ.get("CLIENT_ID")
    if client_secret is None:
        client_secret = os.environ.get("CLIENT_SECRET")

    if client_id is None:
        msg = "Client ID not found! Either load it from environment variables or pass it as an argument."
        raise ValueError(msg)

    if client_secret is None:
        msg = "Client secret not found! Either load it from environment variables or pass it as an argument."
        raise ValueError(msg)

    ep.download(
        start_time,
        end_time,
        save_path=data_path_stem,
        file_cadence="daily",
        download_url=SATELLITE_TO_ID[satellite],
        file_name_stem="",
        rename_file_name_stem=file_name_stem,
        authentication_info=(client_id, client_secret),
        method="esa_swe",
        skip_existing=skip_existing,
    )

    flux_unit = typing.cast("u.Unit", (u.cm**2 * u.s * u.sr * u.MeV) ** (-1))

    extraction_infos = [
        ep.ExtractionInfo(result_key="Epoch_iso", name_or_column="Time", unit=u.dimensionless_unscaled),
        ep.ExtractionInfo(
            result_key="FEDO_ch1", name_or_column="Differential electron flux (0.18 MeV)", unit=flux_unit
        ),
        ep.ExtractionInfo(
            result_key="FEDO_ch2", name_or_column="Differential electron flux (0.27 MeV)", unit=flux_unit
        ),
        ep.ExtractionInfo(
            result_key="FEDO_ch3", name_or_column="Differential electron flux (0.40 MeV)", unit=flux_unit
        ),
        ep.ExtractionInfo(
            result_key="FEDO_ch4", name_or_column="Differential electron flux (0.60 MeV)", unit=flux_unit
        ),
        ep.ExtractionInfo(
            result_key="FEDO_ch5", name_or_column="Differential electron flux (0.88 MeV)", unit=flux_unit
        ),
        ep.ExtractionInfo(
            result_key="FEDO_ch6", name_or_column="Differential electron flux (1.30 MeV)", unit=flux_unit
        ),
        ep.ExtractionInfo(
            result_key="FEDO_ch7", name_or_column="Differential electron flux (1.93 MeV)", unit=flux_unit
        ),
        ep.ExtractionInfo(
            result_key="FEDO_ch8", name_or_column="Differential electron flux (2.90 MeV)", unit=flux_unit
        ),
        ep.ExtractionInfo(
            result_key="FEDO_ch9", name_or_column="Differential electron flux (3.40 MeV)", unit=flux_unit
        ),
        ep.ExtractionInfo(
            result_key="FEDO_ch10", name_or_column="Differential electron flux (4.00 MeV)", unit=flux_unit
        ),
        ep.ExtractionInfo(result_key="x_ECI", name_or_column="X", unit=u.km),
        ep.ExtractionInfo(result_key="y_ECI", name_or_column="Y", unit=u.km),
        ep.ExtractionInfo(result_key="z_ECI", name_or_column="Z", unit=u.km),
        ep.ExtractionInfo(result_key="L", name_or_column="L", unit=ep.units.RE),
    ]
    try:
        variables = ep.extract_variables_from_files(
            start_time,
            end_time,
            file_cadence="daily",
            data_path=data_path_stem,
            file_name_stem=file_name_stem,
            extraction_infos=extraction_infos,
            pd_read_csv_kwargs={"index_col": False},
        )
    except Exception:
        logger.exception(f"Error extracting variables for {satellite}")
        return

    time_format = "%Y-%m-%dT%H:%M:%S.%fZ" if satellite in ["MTG-I1", "MTG-S1"] else "%Y-%m-%dT%H:%M:%SZ"

    # convert iso strings to posixtime
    datetimes = ep.processing.convert_string_to_datetime(variables["Epoch_iso"], time_format=time_format)
    variables["Epoch"] = ep.Variable(
        data=np.asarray([t.timestamp() for t in datetimes]), original_unit=ep.units.posixtime
    )
    del variables["Epoch_iso"]

    # convert ECI coordinates to GEO using astropy
    coords_ECI = GCRS(
        CartesianRepresentation(
            x=variables["x_ECI"].get_data(), y=variables["y_ECI"].get_data(), z=variables["z_ECI"].get_data()
        ),
        obstime=datetimes,
    )
    coords_ITRS = coords_ECI.transform_to(ITRS(obstime=datetimes))
    xgeo_data = np.stack((coords_ITRS.x, coords_ITRS.y, coords_ITRS.z)).T

    variables["xGEO"] = ep.Variable(data=xgeo_data.value, original_unit=u.km)  # ty:ignore[unresolved-attribute]
    del variables["x_ECI"], variables["y_ECI"], variables["z_ECI"]

    # create flux variable
    flux_data = np.stack(
        [
            variables["FEDO_ch1"].get_data(),
            variables["FEDO_ch2"].get_data(),
            variables["FEDO_ch3"].get_data(),
            variables["FEDO_ch4"].get_data(),
            variables["FEDO_ch5"].get_data(),
            variables["FEDO_ch6"].get_data(),
            variables["FEDO_ch7"].get_data(),
            variables["FEDO_ch8"].get_data(),
            variables["FEDO_ch9"].get_data(),
            variables["FEDO_ch10"].get_data(),
        ]
    ).T.astype(np.float64)

    # convert to proper omnidirectional units
    flux_data = flux_data * 4 * np.pi

    variables["FEDO"] = ep.Variable(data=flux_data, original_unit=flux_unit * u.sr)
    for i in range(1, 11):
        del variables[f"FEDO_ch{i}"]
    variables["FEDO"].apply_thresholds_on_data(lower_threshold=0)
    variables["FEDO"].convert_to_unit((u.cm**2 * u.s * u.keV) ** (-1))

    # get energies
    variables["Energy"] = ep.Variable(data=np.asarray(NGRM_ENERGIES), original_unit=u.MeV)

    time_bin_methods = {
        "Energy": ep.TimeBinMethod.Repeat,
        "FEDO": ep.TimeBinMethod.NanMedian,
        "FEDU": ep.TimeBinMethod.NanMedian,
        "xGEO": ep.TimeBinMethod.NanMean,
        "L": ep.TimeBinMethod.NanMean,
    }

    binned_time_var = ep.processing.bin_by_time(
        variables["Epoch"], variables, time_bin_methods, bin_cadence, start_time=start_time, end_time=end_time
    )

    pa_local_data = np.tile(np.arange(5, 91, 5), (len(binned_time_var.get_data()), 1)).astype(np.float64)
    variables["PA_local_FEDU"] = ep.Variable(data=pa_local_data, original_unit=u.deg)

    variables_to_compute: ep.processing.VariableRequest = [
        ("B_Calc", mag_field),
        ("B_Eq", mag_field),
        ("MLT_Eq", mag_field),
        ("B_Eq", mag_field),
        ("R_Eq", mag_field),
        ("Alpha_Eq", mag_field),
    ]

    if calculate_Lstar:
        variables_to_compute.append(("L_star", mag_field))  # ty:ignore[invalid-argument-type]
        variables_to_compute.append(("L_m", mag_field))  # ty:ignore[invalid-argument-type]

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=binned_time_var,
        xgeo_var=variables["xGEO"],
        energy_var=variables["Energy"],
        pa_local_var=variables["PA_local_FEDU"],
        particle_species="electron",
        variables_to_compute=variables_to_compute,
        irbem_options=ep.processing.magnetic_field_utils.IrbemOptions(
            lstar_quantity=ep.processing.magnetic_field_utils.LstarQuantity.LSTAR
            if calculate_Lstar
            else ep.processing.magnetic_field_utils.LstarQuantity.NONE,
        ),
        num_cores=num_cores,
    )

    variables |= magnetic_field_variables

    FEDU_var = ep.processing.construct_pitch_angle_distribution(
        variables["FEDO"],
        variables["PA_local_FEDU"],
        magnetic_field_variables[f"Alpha_Eq_{mag_field}"],
        flux_type="omni",
    )
    FEDU_var.apply_thresholds_on_data(lower_threshold=0)

    psd_var = ep.processing.compute_phase_space_density(FEDU_var, variables["Energy"], particle_species="electron")

    variables_to_save: dict[ep.typing.InternalName, ep.Variable] = {
        "Epoch": binned_time_var,
        "FEDU": FEDU_var,
        # "FEDO": variables["FEDO"], disabled for now, since Alpha_range is missing
        "Energy_FEDU": variables["Energy"],
        "Alpha": variables["PA_local_FEDU"],
        "Alpha_Eq": magnetic_field_variables[f"Alpha_Eq_{mag_field}"],
        "R_Eq": magnetic_field_variables[f"R_Eq_{mag_field}"],
        "MLT": magnetic_field_variables[f"MLT_Eq_{mag_field}"],
        "B_Calc": magnetic_field_variables[f"B_Calc_{mag_field}"],
        "B_Eq": magnetic_field_variables[f"B_Eq_{mag_field}"],
        "Position": variables["xGEO"],
        "PSD": psd_var,
    }

    if calculate_Lstar:
        variables_to_save["L_star"] = magnetic_field_variables[f"L_star_{mag_field}"]
        variables_to_save["L_m"] = magnetic_field_variables[f"L_m_{mag_field}"]

    if saving_strategy is None:
        save_srat_class = (
            ep.saving_strategies.DailyLEORBStrategy
            if satellite.startswith("S6")
            else ep.saving_strategies.MonthlyRBStrategy
        )

        saving_strategy = save_srat_class(
            base_data_path=Path(processed_data_path),
            mission="ESA",
            satellite=f"{satellite.lower()}",
            instrument="ngrm",
            mag_field=mag_field,
            file_format=".nc",
            data_standard=ep.data_standards.GFZStandard(),
        )

    ep.save(variables_to_save, saving_strategy, start_time, end_time, time_var=binned_time_var, append=True)


CLI_DEFAULTS = {
    "bin_cadence": timedelta(minutes=5),
}

if __name__ == "__main__":
    ep.run_recipe_cli(process_ngrm_electron_fluxes, defaults=CLI_DEFAULTS)
