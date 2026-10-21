# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""Arase MGF level-2 magnetometer data, shared by the Arase recipes."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u

import el_paso as ep

if TYPE_CHECKING:
    from el_paso.physics import ParticleLiteral
    from el_paso.typing import InternalName

MGF_URL = "https://spdf.gsfc.nasa.gov/pub/data/arase/mgf/l2/8sec/YYYY/"
MGF_FILE_NAME_STEM = "erg_mgf_l2_8sec_YYYYMMDD_.{6}.cdf"

# quality_8sec elements (see its VAR_NOTES): offset error (0-2), spin period error (4), discontinuity (5),
# range switching (6), abnormal raw data (7), MTQ operation (8), and attitude (15) are good if 0;
# the sine-fit RMS (3) is good below 15 (according to cdf)
MGF_QUALITY_ELEMENTS_GOOD_IF_ZERO = [0, 1, 2, 4, 5, 6, 7, 8, 15]
MGF_QUALITY_SINE_FIT_RMS = 3
MGF_MAX_SINE_FIT_RMS = 15


def download_mgf_magnetometer(
    start_time: datetime,
    end_time: datetime,
    save_path: str | Path,
    *,
    skip_existing: bool = True,
) -> None:
    """Download the daily Arase MGF level-2 8-second magnetometer files from SPDF.

    Args:
        start_time (datetime): Start of the time range to download.
        end_time (datetime): End of the time range to download.
        save_path (str | Path): Directory the files are saved to.
        skip_existing (bool): If True, skip files that already exist on disk.
    """
    ep.download(
        start_time,
        end_time,
        save_path=save_path,
        download_url=MGF_URL,
        file_name_stem=MGF_FILE_NAME_STEM,
        file_cadence="daily",
        method="request",
        skip_existing=skip_existing,
    )


def compute_observed_B_variables(
    start_time: datetime,
    end_time: datetime,
    raw_data_path: str | Path,
    binned_time_var: ep.Variable,
    bin_cadence: timedelta,
    energy_var: ep.Variable,
    pa_local_var: ep.Variable,
    particle_species: "ParticleLiteral",
) -> dict["InternalName", ep.Variable]:
    """Compute the observed magnetic field magnitude and the first adiabatic invariant computed from it.

    Reads the MGF files downloaded by `download_mgf_magnetometer`, sets samples with a bad quality_8sec
    flag (offset, sine-fit RMS, spin period, discontinuity, range switching, abnormal raw data, MTQ
    operation, attitude) or a non-positive magnitude to NaN, time-bins the magnitude with a NaN-mean
    onto the recipe's time grid, and computes InvMu from it.

    Args:
        start_time (datetime): Start of the time range; must be the one the recipe binned with.
        end_time (datetime): End of the time range; must be the one the recipe binned with.
        raw_data_path (str | Path): Directory containing the MGF files.
        binned_time_var (ep.Variable): The recipe's binned time variable, binned with `start_time`
            and `end_time`.
        bin_cadence (timedelta): The recipe's time-binning cadence.
        energy_var (ep.Variable): Particle energies, (time, energy), as used for the model InvMu.
        pa_local_var (ep.Variable): Local pitch angles, (time, pitch angle), as used for the model InvMu.
        particle_species (Literal["electron", "proton"]): Particle species.

    Returns:
        dict[InternalName, ep.Variable]: "B_total_obs" (time) and "InvMu_obs" (time, energy, pitch angle).

    Raises:
        ValueError: If the binned magnetometer time grid differs from `binned_time_var`.
    """
    extraction_infos = [
        ep.ExtractionInfo(result_key="Epoch", name_or_column="epoch_8sec", unit=ep.units.tt2000),
        ep.ExtractionInfo(result_key="B_total_obs", name_or_column="magt_8sec", unit=u.nT),
        ep.ExtractionInfo(result_key="quality", name_or_column="quality_8sec", unit=u.dimensionless_unscaled),
    ]

    variables = ep.extract_variables_from_files(
        start_time,
        end_time,
        "daily",
        data_path=Path(raw_data_path),
        file_name_stem=MGF_FILE_NAME_STEM,
        extraction_infos=extraction_infos,
    )

    quality = variables.pop("quality").get_data().astype(np.int64)
    is_bad = (quality[:, MGF_QUALITY_ELEMENTS_GOOD_IF_ZERO] != 0).any(axis=1)
    is_bad |= quality[:, MGF_QUALITY_SINE_FIT_RMS] >= MGF_MAX_SINE_FIT_RMS

    b_total = variables["B_total_obs"].get_data().astype(np.float64)
    b_total[is_bad | ~(b_total > 0)] = np.nan
    variables["B_total_obs"].set_data(b_total, unit="same")
    variables["B_total_obs"].metadata.add_processing_note(
        "Samples with a bad quality_8sec flag (offset, sine-fit RMS, spin period, discontinuity, range "
        "switching, abnormal raw data, MTQ operation, attitude) or non-positive magnitude set to NaN."
    )

    binned_magnetometer_time = ep.processing.bin_by_time(
        variables["Epoch"],
        variables=variables,
        time_bin_method_dict={"B_total_obs": ep.TimeBinMethod.NanMean},
        time_binning_cadence=bin_cadence,
        start_time=start_time,
        end_time=end_time,
    )

    magnetometer_times = binned_magnetometer_time.get_data(ep.units.posixtime)
    recipe_times = binned_time_var.get_data(ep.units.posixtime)
    if magnetometer_times.shape != recipe_times.shape or not np.allclose(
        magnetometer_times, recipe_times, rtol=0, atol=1e-3
    ):
        msg = (
            "The binned magnetometer time grid differs from the recipe's time grid. "
            "The recipe has to bin its variables with the same start_time, end_time, and bin_cadence."
        )
        raise ValueError(msg)

    b_total_obs_var = variables["B_total_obs"]
    inv_mu_obs_var = ep.processing.compute_invariant_mu(energy_var, pa_local_var, b_total_obs_var, particle_species)

    return {"B_total_obs": b_total_obs_var, "InvMu_obs": inv_mu_obs_var}
