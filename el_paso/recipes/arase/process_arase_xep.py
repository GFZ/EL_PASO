# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import dateutil
import numpy as np
from astropy import units as u

import el_paso as ep
from el_paso.processing.magnetic_field_utils.irbem import Coords
from el_paso.recipes.arase import (
    get_arase_orbit_level_2_variables,
    get_arase_orbit_level_3_variables,
)


def process_arase_xep(
    start_time: datetime,
    end_time: datetime,
    mag_field: Literal["T89", "TS04", "OP77Q"] = "T89",
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    bin_cadence: timedelta = timedelta(minutes=5),
    num_cores: int = 16,
    *,
    use_level_3_orbit_data: bool = True,
) -> None:
    """Process Arase XEP Level 2 omnidirectional electron flux data and save derived products.

    Downloads the corresponding Arase orbit data (Level 3 if `use_level_3_orbit_data` is True,
    otherwise Level 2) and the daily Arase XEP Level 2 omnidirectional flux (FEDO) CDF files for
    the requested time range, then averages the energy tables into a single per-channel energy
    array, assigns a fixed local pitch-angle grid (10, 30, 50, 70, 90 degrees) since the omni flux
    itself carries no pitch-angle information, and converts FEDO from a per-steradian to a true
    omnidirectional flux (multiplying by 4*pi sr). The flux, position and pitch-angle variables are
    then time-binned onto a common cadence and the flux is thresholded at a lower bound. The
    equatorial pitch angle and magnetic-field-related quantities (L_m, MLT, R0/R_Eq) are computed
    either directly from the Level 3 orbit data (equatorial pitch angle from the local/equatorial
    B-field ratio) or via IRBEM using the SM position transformed to GEO from Level 2 orbit data.
    Finally, a pitch-angle-resolved directional flux (FEDU) is constructed from the omnidirectional
    flux using a sine-shaped pitch-angle distribution, and the resulting variables are saved using
    the requested saving strategy.

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
        use_level_3_orbit_data (bool): If True, use Arase Level 3 orbit data (which
                                                already contains precomputed magnetic field
                                                quantities for `mag_field`); if False, use Level 2
                                                orbit data and compute the magnetic field
                                                quantities via IRBEM. Defaults to True.
    """
    raw_data_path = Path(raw_data_path)
    processed_data_path = Path(processed_data_path)

    if use_level_3_orbit_data:
        orb_variables = get_arase_orbit_level_3_variables(start_time, end_time, mag_field, raw_data_path=raw_data_path)
    else:
        orb_variables = get_arase_orbit_level_2_variables(start_time, end_time, raw_data_path=raw_data_path)

    file_name_stem = "erg_xep_l2_omniflux_YYYYMMDD_.{6}.cdf"
    url = "https://spdf.gsfc.nasa.gov/pub/data/arase/xep/l2/omniflux/YYYY/"

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
            name_or_column="Epoch",
            unit=ep.units.tt2000,
        ),
        ep.ExtractionInfo(
            result_key="Energy",
            name_or_column="FEDO_SSD_Energy",
            unit=u.keV,
            is_time_dependent=False,
        ),
        ep.ExtractionInfo(
            result_key="FEDO",
            name_or_column="FEDO_SSD",
            unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
        ),
    ]

    xep_variables = ep.extract_variables_from_files(
        start_time,
        end_time,
        "daily",
        data_path=raw_data_path,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )

    # average energy bins
    energies = xep_variables["Energy"].get_data().astype(np.float64)
    xep_variables["Energy"].set_data(np.mean(energies, axis=0), unit="same")

    # add local pitch angles
    xep_variables["AlphaLocal"] = ep.Variable(data=np.array([10, 30, 50, 70, 90]), original_unit=u.degree)

    # put into units of omnidirectional flux
    xep_variables["FEDO"].set_data(
        xep_variables["FEDO"].get_data().astype(np.float64) * 4 * np.pi,
        unit=(u.cm**2 * u.s * u.keV) ** (-1),
    )

    time_bin_methods = {
        "Energy": ep.TimeBinMethod.Repeat,
        "FEDO": ep.TimeBinMethod.NanMedian,
        "AlphaLocal": ep.TimeBinMethod.Repeat,
    }

    _ = ep.processing.bin_by_time(
        xep_variables["Epoch"],
        variables=xep_variables,
        time_bin_method_dict=time_bin_methods,
        time_binning_cadence=bin_cadence,
        start_time=start_time,
        end_time=end_time,
    )

    xep_variables["FEDO"].apply_thresholds_on_data(lower_threshold=1e-21)

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

        pa_local = xep_variables["AlphaLocal"].get_data(u.radian)
        pa_eq = np.asin(
            np.sin(pa_local)
            * np.sqrt(orb_variables["B_eq"].get_data(u.nT) / orb_variables["B_local"].get_data(u.nT))[:, np.newaxis]
        )
        xep_variables["Alpha_eq"] = ep.Variable(data=pa_eq, original_unit=u.radian)

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
            ("L_star", mag_field),
            ("InvK", mag_field),
            ("InvMu", mag_field),
        ]

        magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
            time_var=binned_time_variable,
            xgeo_var=pos_geo_var,
            variables_to_compute=variables_to_compute,
            irbem_options=irbem_options,
            num_cores=num_cores,
            pa_local_var=xep_variables["AlphaLocal"],
            energy_var=xep_variables["Energy"],
            particle_species="electron",
        )

        orb_variables["R0"] = magnetic_field_variables["R_Eq_" + mag_field]
        orb_variables["MLT"] = magnetic_field_variables["MLT_" + mag_field]
        xep_variables["Alpha_eq"] = magnetic_field_variables["Alpha_Eq_" + mag_field]
        orb_variables["Lm"] = magnetic_field_variables["L_m_" + mag_field]
        orb_variables["Lstar"] = magnetic_field_variables["L_star_" + mag_field]
        orb_variables["InvK"] = magnetic_field_variables["InvK_" + mag_field]
        orb_variables["InvMu"] = magnetic_field_variables["InvMu_" + mag_field]

    xep_variables["FEDU"] = ep.processing.construct_pitch_angle_distribution(
        xep_variables["FEDO"],
        xep_variables["AlphaLocal"],
        xep_variables["Alpha_eq"],
        method="Smirnov2022",
        flux_type="omni",
        time_var=binned_time_variable,
        L_var=orb_variables["R0"],
        MLT_var=orb_variables["MLT"],
        energy_var=xep_variables["Energy"],
    )

    psd_var = ep.processing.compute_phase_space_density(
        xep_variables["FEDU"], xep_variables["Energy"], particle_species="electron"
    )

    variables_to_save: dict[ep.typing.InternalName, ep.Variable] = {
        "Epoch": binned_time_variable,
        "FEDU": xep_variables["FEDU"],
        "Energy_FEDU": xep_variables["Energy"],
        "Alpha": xep_variables["AlphaLocal"],
        "Alpha_Eq": xep_variables["Alpha_eq"],
        "R_Eq": orb_variables["R0"],
        "MLT": orb_variables["MLT"],
        "L_m": orb_variables["Lm"],
        "PSD": psd_var,
    }

    if not use_level_3_orbit_data:
        variables_to_save |= {"L_star": orb_variables["Lstar"], "InvK": orb_variables["InvK"], "InvMu": orb_variables["InvMu"]}

    saving_strategy = ep.saving_strategies.MonthlyRBStrategy(
        processed_data_path,
        "Arase",
        "arase",
        "xep",
        mag_field=mag_field,
        file_format="nc",
        data_standard=ep.data_standards.GFZStandard(),
    )

    ep.save(variables_to_save, saving_strategy, start_time, end_time, binned_time_variable)


CLI_DEFAULTS = {
    "use_level_3_orbit_data": False,
}

if __name__ == "__main__":
    ep.run_recipe_cli(process_arase_xep, defaults=CLI_DEFAULTS)
