# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from numpy.typing import NDArray
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from astropy import units as u

import el_paso as ep

if TYPE_CHECKING:
    from el_paso.typing import InternalName

logging.captureWarnings(capture=True)
logger = logging.getLogger(__name__)

LONG_FROM_SAT = {
    "goes13": 74.5,
    "goes15": 135.1,
}

def process_goes_eps_maged(
    sat_str: Literal["goes13", "goes15"],
    processed_data_path: str | Path,
    raw_data_path: str | Path,
    start_time: datetime,
    end_time: datetime,
    num_cores: int = 32,
) -> None:

    data_path_stem = f"{raw_data_path}/YYYY/MM/{sat_str}/"

    maged_vars = _get_eps_maged_variables(sat_str, data_path_stem, start_time, end_time)
    pa_vars = _get_eps_pitch_angles_variables(sat_str, data_path_stem, start_time, end_time)

    # calculate xGEO
    xgeo_data = _geo_cartesian_from_west_lon_incl(LONG_FROM_SAT[sat_str], 0)
    maged_vars["xGEO"] = ep.Variable(data=xgeo_data, original_unit=u.km)

    time_bin_methods_maged = {
        "FEDU": ep.TimeBinMethod.NanMedian,
        "Energy": ep.TimeBinMethod.Repeat,
        "xGEO": ep.TimeBinMethod.Repeat,
    }

    _ = ep.processing.bin_by_time(
        time_variable=maged_vars["Epoch"],
        variables=maged_vars,
        time_bin_method_dict=time_bin_methods_maged,
        time_binning_cadence=timedelta(minutes=5),
        start_time=start_time,
        end_time=end_time,
    )

    time_bin_methods_pa = {
        "AlphaLocal": ep.TimeBinMethod.NanMean,
    }

    binned_time_var = ep.processing.bin_by_time(
        time_variable=pa_vars["Epoch"],
        variables=pa_vars,
        time_bin_method_dict=time_bin_methods_pa,
        time_binning_cadence=timedelta(minutes=5),
        start_time=start_time,
        end_time=end_time,
    )

    maged_vars["FEDU"].transpose_data((0, 2, 1))
    maged_vars["FEDU"].apply_thresholds_on_data(lower_threshold=0)

    # sort pitch angles in ascending order and apply to fluxes
    local_pa = pa_vars["AlphaLocal"].get_data(u.degree)
    local_pa[local_pa > 90] = 180 - local_pa[local_pa > 90]

    idx_sorted = np.argsort(local_pa, axis=1)
    sorted_local_pa = np.take_along_axis(local_pa, idx_sorted, axis=1)
    n_energy = maged_vars["Energy"].get_data().shape[1]
    sorted_diff_flux = np.take_along_axis(
        maged_vars["FEDU"].get_data(), np.tile(idx_sorted[:, np.newaxis, :], [1, n_energy, 1]), axis=2
    )

    pa_vars["AlphaLocal"].set_data(sorted_local_pa, unit="same")
    pa_vars["AlphaLocal"].metadata.add_processing_note("Folded around 90 deg and sorted in ascending order")

    maged_vars["FEDU"].set_data(sorted_diff_flux, unit="same")

    # Calculate magnetic field variables
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
        xgeo_var=maged_vars["xGEO"],
        energy_var=maged_vars["Energy"],
        pa_local_var=pa_vars["AlphaLocal"],
        particle_species="electron",
        variables_to_compute=variables_to_compute,
        irbem_options=ep.processing.magnetic_field_utils.IrbemOptions(),
        num_cores=num_cores,
    )

    psd_var = ep.processing.compute_phase_space_density(
        maged_vars["FEDU"], maged_vars["Energy"], particle_species="electron"
    )

    variables_to_save: dict[InternalName, ep.Variable] = {
        "Epoch": binned_time_var,
        "FEDU": maged_vars["FEDU"],
        "Position": maged_vars["xGEO"],
        "Energy_FEDU": maged_vars["Energy"],
        "Alpha": pa_vars["AlphaLocal"],
        "PSD": psd_var,
        "Alpha_Eq": magnetic_field_variables["Alpha_Eq_T89"],
        "MLT": magnetic_field_variables["MLT_T89"],
        "L_star": magnetic_field_variables["L_star_T89"],
        "R_Eq": magnetic_field_variables["R_Eq_T89"],
        "B_Eq": magnetic_field_variables["B_Eq_T89"],
        "B_Calc": magnetic_field_variables["B_Calc_T89"],
        "InvMu": magnetic_field_variables["InvMu_T89"],
        "InvK": magnetic_field_variables["InvK_T89"],
    }

    saving_strategy = ep.saving_strategies.MonthlyRBStrategy(
        Path(processed_data_path),
        mission="GOES",
        satellite=sat_str,
        instrument="MAGED",
        mag_field="T89",
        file_format="nc",
        data_standard=ep.data_standards.GFZStandard(),
    )

    ep.save(variables_to_save, saving_strategy, start_time, end_time, time_var=binned_time_var, append=True)


def _get_eps_maged_variables(
    sat_str: Literal["goes13", "goes15"],
    data_path_stem: str | Path,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, ep.Variable]:
    url = f"https://spdf.gsfc.nasa.gov/pub/data/goes/{sat_str}/eps-maged/19me15_5min/YYYY/"

    ep.setup_logging()

    file_name_stem = f"{sat_str}_eps-maged_5min_YYYYMMDD_.{r'{3}'}.cdf"

    ep.download(
        start_time,
        end_time,
        save_path=data_path_stem,
        file_cadence="monthly",
        download_url=url,
        file_name_stem=file_name_stem,
    )

    extraction_infos_maged = [
        ep.ExtractionInfo(name_or_column="Epoch", unit=ep.units.tt2000, result_key="Epoch"),
        ep.ExtractionInfo(
            name_or_column="dtc_cor_eflux", unit=(u.keV * u.cm**2 * u.s * u.sr) ** (-1), result_key="FEDU"
        ),
        ep.ExtractionInfo(name_or_column="energy", unit=u.keV, result_key="Energy", is_time_dependent=False),
    ]

    return ep.extract_variables_from_files(
        start_time,
        end_time,
        file_cadence="monthly",
        data_path=data_path_stem,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos_maged,
    )


def _get_eps_pitch_angles_variables(
    sat_str: Literal["goes13", "goes15"],
    data_path_stem: str | Path,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, ep.Variable]:
    url = f"https://spdf.gsfc.nasa.gov/pub/data/goes/{sat_str}/eps-pitch-angles_1min/YYYY/"

    file_name_stem = f"{sat_str}_eps-pitch-angles_1min_YYYYMMDD_.{r'{3}'}.cdf"

    ep.download(
        start_time,
        end_time,
        save_path=data_path_stem,
        file_cadence="monthly",
        download_url=url,
        file_name_stem=file_name_stem,
    )

    extraction_infos = [
        ep.ExtractionInfo(name_or_column="Epoch", unit=ep.units.tt2000, result_key="Epoch"),
        ep.ExtractionInfo(name_or_column="pitch_angles", unit=u.degree, result_key="AlphaLocal"),
    ]

    return ep.extract_variables_from_files(
        start_time,
        end_time,
        file_cadence="monthly",
        data_path=data_path_stem,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )


def _geo_cartesian_from_west_lon_incl(
    west_longitude_deg: float, inclination_deg: float, radius_km: float = 42164.0
) -> NDArray[np.float64]:
    lam_w = np.deg2rad(west_longitude_deg)
    i = np.deg2rad(inclination_deg)

    x = radius_km * np.cos(i) * np.cos(lam_w)
    y = -radius_km * np.cos(i) * np.sin(lam_w)
    z = -radius_km * np.sin(i)

    return np.stack([x, y, z], axis=-1)


if __name__ == "__main__":
    start_time = datetime(2017, 10, 25, tzinfo=timezone.utc)
    end_time = datetime(2017, 10, 25, 23, 59, tzinfo=timezone.utc)

    Satellite = Literal["goes13", "goes15"]
    sats: list[Satellite] = ["goes13"]

    for sat in sats:
        process_goes_eps_maged(
            sat_str=sat,
            raw_data_path="goes/raw/",
            processed_data_path="goes/processed/",
            start_time=start_time,
            end_time=end_time,
            num_cores=64,
        )
