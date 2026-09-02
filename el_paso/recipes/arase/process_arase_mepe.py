# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from astropy import units as u

import el_paso as ep
from el_paso.processing.magnetic_field_utils.irbem import Coords
from el_paso.recipes.arase import (
    get_arase_orbit_level_2_variables,
    get_arase_orbit_level_3_variables,
)


def process_arase_mepe(
    start_time: datetime,
    end_time: datetime,
    mag_field: Literal["T89", "TS04", "OP77Q"] = "T89",
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    bin_cadence: timedelta = timedelta(minutes=5),
    num_cores: int = 16,
    save_strategy: Literal["gfz", "h5", "netcdf"] = "gfz",
    data_standard: Literal["gfz", "prbem"] = "gfz",
    *,
    use_level_3_orbit_data: bool = True,
) -> None:
    """Process Arase MEP-e Level 3 electron flux data and save derived products.

    Downloads the corresponding Arase orbit data (Level 3 if `use_level_3_orbit_data` is True,
    otherwise Level 2) and the daily Arase MEP-e Level 3 pitch-angle resolved flux (FEDU) CDF
    files for the requested time range, then sorts energies into ascending order, time-bins the
    flux, position and pitch-angle variables onto a common cadence, folds the pitch-angle
    distribution, applies a lower flux threshold, and computes the equatorial pitch angle and
    magnetic-field-related quantities (Lm, MLT, R0) either from the Level 3 orbit data directly
    or via IRBEM using the SM position from Level 2 orbit data. The resulting variables are
    saved using the requested saving strategy.

    Args:
        start_time (datetime): Start of the time range to process.
        end_time (datetime): End of the time range to process.
        mag_field (Literal["T89", "TS04", "OP77Q"]): The magnetic field model used for the
                                                    magnetic-field-related output variables.
        raw_data_path (str | Path): Directory where downloaded raw data files are
                                            stored. Defaults to ".".
        processed_data_path (str | Path): Base directory where the processed output
                                                    data is saved. Defaults to ".".
        num_cores (int): Number of CPU cores used for the IRBEM magnetic field
                                computations (only used when `use_level_3_orbit_data` is
                                False). Defaults to 4.
        bin_cadence (timedelta): Time binning cadence applied to all variables.
        save_strategy (Literal["gfz", "h5", "netcdf"]): The saving strategy used to
                                                                write the processed data.
                                                                Defaults to "gfz".
        data_standard (Literal["gfz", "prbem"]): The data standard used when saving
                                                            the processed data. Defaults to "gfz".
        use_level_3_orbit_data (bool): If True, use Arase Level 3 orbit data (which
                                                already contains precomputed magnetic field
                                                quantities for `mag_field`); if False, use Level 2
                                                orbit data and compute the magnetic field
                                                quantities via IRBEM. Defaults to True.
    """
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel(logging.INFO)

    raw_data_path = Path(raw_data_path)
    processed_data_path = Path(processed_data_path)

    if use_level_3_orbit_data:
        orb_variables = get_arase_orbit_level_3_variables(start_time, end_time, mag_field, raw_data_path=raw_data_path)
    else:
        orb_variables = get_arase_orbit_level_2_variables(start_time, end_time, raw_data_path=raw_data_path)

    file_name_stem = "erg_mepe_l3_pa_YYYYMMDD_.{6}.cdf"
    url = "https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/mepe/l3/pa/YYYY/MM/"

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
        ep.ExtractionInfo(
            result_key="Epoch",
            name_or_column="epoch",
            unit=ep.units.tt2000,
        ),
        ep.ExtractionInfo(
            result_key="Energy",
            name_or_column="FEDU_Energy",
            unit=u.keV,
            is_time_dependent=False,
        ),
        ep.ExtractionInfo(
            result_key="Pitch_angle",
            name_or_column="FEDU_Alpha",
            unit=u.deg,
            is_time_dependent=False,
        ),
        ep.ExtractionInfo(
            result_key="FEDU",
            name_or_column="FEDU",
            unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
        ),
    ]

    mepe_variables = ep.extract_variables_from_files(
        start_time,
        end_time,
        "daily",
        data_path=raw_data_path,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )

    mepe_variables["FEDU"].truncate(mepe_variables["Epoch"], start_time, end_time)
    mepe_variables["Epoch"].truncate(mepe_variables["Epoch"], start_time, end_time)

    # sort energies into ascending order
    idx_sorted = np.argsort(mepe_variables["Energy"].get_data())
    mepe_variables["Energy"].set_data(mepe_variables["Energy"].get_data()[idx_sorted[:-1]], "same")
    mepe_variables["FEDU"].set_data(mepe_variables["FEDU"].get_data()[:, idx_sorted[:-1], :], "same")

    time_bin_methods = {
        "xGEO": ep.TimeBinMethod.NanMean,
        "Energy": ep.TimeBinMethod.Repeat,
        "FEDU": ep.TimeBinMethod.NanMedian,
        "Pitch_angle": ep.TimeBinMethod.Repeat,
    }

    binned_time_variable = ep.processing.bin_by_time(
        mepe_variables["Epoch"],
        variables=mepe_variables,
        time_bin_method_dict=time_bin_methods,
        time_binning_cadence=bin_cadence,
        start_time=start_time,
        end_time=end_time,
    )

    ep.processing.fold_pitch_angles_and_flux(mepe_variables["FEDU"], mepe_variables["Pitch_angle"])

    mepe_variables["FEDU"].apply_thresholds_on_data(lower_threshold=1e-21)

    if use_level_3_orbit_data:
        time_bin_methods = {
            "B_local": ep.TimeBinMethod.NanMedian,
            "B_eq": ep.TimeBinMethod.NanMedian,
            "Lm": ep.TimeBinMethod.NanMean,
            "Lstar": ep.TimeBinMethod.NanMean,
            "MLT": ep.TimeBinMethod.NanMean,
            "R0": ep.TimeBinMethod.NanMean,
        }

        binned_time_variable = ep.processing.bin_by_time(
            orb_variables["Epoch"],
            variables=orb_variables,
            time_bin_method_dict=time_bin_methods,
            time_binning_cadence=bin_cadence,
            start_time=start_time,
            end_time=end_time,
        )

        pa_local = mepe_variables["Pitch_angle"].get_data(u.radian)
        pa_eq = np.asin(
            np.sin(pa_local)
            * np.sqrt(orb_variables["B_eq"].get_data(u.nT) / orb_variables["B_local"].get_data(u.nT))[:, np.newaxis]
        )
        mepe_variables["Pa_eq"] = ep.Variable(data=pa_eq, original_unit=u.radian)

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

        irbem_options = ep.processing.magnetic_field_utils.IrbemOptions()

        variables_to_compute: ep.processing.VariableRequest = [
            ("B_Calc", mag_field),
            ("MLT", mag_field),
            ("B_Eq", mag_field),
            ("R_Eq", mag_field),
            ("Alpha_Eq", mag_field),
            ("L_m", mag_field),
        ]

        magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
            time_var=binned_time_variable,
            xgeo_var=pos_geo_var,
            variables_to_compute=variables_to_compute,
            irbem_options=irbem_options,
            num_cores=num_cores,
            pa_local_var=mepe_variables["Pitch_angle"],
        )

        orb_variables["R0"] = magnetic_field_variables["R_Eq_" + mag_field]
        orb_variables["MLT"] = magnetic_field_variables["MLT_" + mag_field]
        mepe_variables["Pa_eq"] = magnetic_field_variables["Alpha_Eq_" + mag_field]
        orb_variables["Lm"] = magnetic_field_variables["L_m_" + mag_field]

    match mag_field:
        case "T89":
            mag_field_save = "T89"
        case "TS04":
            mag_field_save = "T04s"
        case "OP77Q":
            mag_field_save = "OP77"

    data_standard_instance = (
        ep.data_standards.GFZStandard() if data_standard == "gfz" else ep.data_standards.PRBEMStandard()
    )

    variables_to_save: dict[ep.typing.InternalName, ep.Variable] = {
        "Epoch": binned_time_variable,
        "FEDU": mepe_variables["FEDU"],
        "Energy_FEDU": mepe_variables["Energy"],
        "Alpha": mepe_variables["Pitch_angle"],
        "Alpha_Eq": mepe_variables["Pa_eq"],
        "R_Eq": orb_variables["R0"],
        "MLT": orb_variables["MLT"],
        "L_m": orb_variables["Lm"],
    }

    match save_strategy:
        case "gfz":
            saving_strategy = ep.saving_strategies.GFZStrategy(
                processed_data_path,
                "ARASE",
                "arase",
                "mepe",
                mag_field_save,
                data_standard_instance,
            )

        case "h5":
            saving_strategy = ep.saving_strategies.MonthlyRBStrategy(
                processed_data_path,
                "Arase",
                "arase",
                "mepe",
                mag_field=mag_field,
                file_format="h5",
                data_standard=data_standard_instance,
            )

        case "netcdf":
            saving_strategy = ep.saving_strategies.MonthlyRBStrategy(
                processed_data_path,
                "Arase",
                "arase",
                "mepe",
                mag_field=mag_field,
                file_format="nc",
                data_standard=data_standard_instance,
            )

    ep.save(variables_to_save, saving_strategy, start_time, end_time, binned_time_variable)


if __name__ == "__main__":
    ep.run_recipe_cli(process_arase_mepe)
