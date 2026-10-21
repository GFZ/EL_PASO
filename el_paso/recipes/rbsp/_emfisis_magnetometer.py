# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""EMFISIS level-3 magnetometer data, shared by the RBSP recipes."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u

import el_paso as ep
from el_paso.recipes.rbsp import RBSPSatellite

if TYPE_CHECKING:
    from el_paso.physics import ParticleLiteral
    from el_paso.typing import InternalName


def emfisis_magnetometer_file_name_stem(satellite: RBSPSatellite) -> str:
    """Return the file name stem of the daily EMFISIS level-3 4-second magnetometer files in SM coordinates."""
    return "rbsp-" + satellite + "_magnetometer_4sec-sm_emfisis-l3_YYYYMMDD_.{6}.cdf"


def download_emfisis_magnetometer(
    start_time: datetime,
    end_time: datetime,
    satellite: RBSPSatellite,
    save_path: str | Path,
    *,
    skip_existing: bool = True,
) -> None:
    """Download the daily EMFISIS level-3 4-second magnetometer files in SM coordinates from CDAWeb.

    Args:
        start_time (datetime): Start of the time range to download.
        end_time (datetime): End of the time range to download.
        satellite (Literal["a", "b"]): RBSP satellite identifier.
        save_path (str | Path): Directory the files are saved to; may contain YYYY/MM placeholders.
        skip_existing (bool): If True, skip files that already exist on disk.
    """
    ep.download(
        start_time,
        end_time,
        save_path=save_path,
        download_url=f"https://cdaweb.gsfc.nasa.gov/pub/data/rbsp/rbsp{satellite}/l3/emfisis/magnetometer/4sec/sm/YYYY/",
        file_name_stem=emfisis_magnetometer_file_name_stem(satellite),
        file_cadence="daily",
        method="request",
        skip_existing=skip_existing,
    )


def compute_observed_B_variables(
    start_time: datetime,
    end_time: datetime,
    satellite: RBSPSatellite,
    raw_data_path: str | Path,
    binned_time_var: ep.Variable,
    bin_cadence: timedelta,
    energy_var: ep.Variable,
    pa_local_var: ep.Variable,
    particle_species: "ParticleLiteral",
) -> dict["InternalName", ep.Variable]:
    """Compute the observed magnetic field magnitude and the first adiabatic invariant computed from it.

    Reads the EMFISIS magnetometer files downloaded by `download_emfisis_magnetometer`, sets samples
    flagged by calState, magInvalid, or magFill, or with a non-positive magnitude, to NaN (as the GFZ
    MATLAB reader `emfisis_l3_read_one_file` does), time-bins the magnitude with a NaN-mean onto the
    recipe's time grid, and computes InvMu from it.

    Args:
        start_time (datetime): Start of the time range; must be the one the recipe binned with.
        end_time (datetime): End of the time range; must be the one the recipe binned with.
        satellite (Literal["a", "b"]): RBSP satellite identifier.
        raw_data_path (str | Path): Directory containing the magnetometer files.
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
        ep.ExtractionInfo(result_key="Epoch", name_or_column="Epoch", unit=ep.units.tt2000),
        ep.ExtractionInfo(result_key="B_total_obs", name_or_column="Magnitude", unit=u.nT),
        ep.ExtractionInfo(result_key="magInvalid", name_or_column="magInvalid", unit=u.dimensionless_unscaled),
        ep.ExtractionInfo(result_key="magFill", name_or_column="magFill", unit=u.dimensionless_unscaled),
        ep.ExtractionInfo(result_key="calState", name_or_column="calState", unit=u.dimensionless_unscaled),
    ]

    variables = ep.extract_variables_from_files(
        start_time,
        end_time,
        "daily",
        data_path=Path(raw_data_path),
        file_name_stem=emfisis_magnetometer_file_name_stem(satellite),
        extraction_infos=extraction_infos,
    )

    # all three flags are 0 for valid data; invalid samples carry a negative magnitude (-99999.9)
    b_total = variables["B_total_obs"].get_data().astype(np.float64)
    is_invalid = (
        (variables.pop("magInvalid").get_data() != 0)
        | (variables.pop("magFill").get_data() != 0)
        | (variables.pop("calState").get_data() != 0)
        | ~(b_total > 0)
    )
    b_total[is_invalid] = np.nan
    variables["B_total_obs"].set_data(b_total, unit="same")
    variables["B_total_obs"].metadata.add_processing_note(
        "Samples flagged by magInvalid, magFill, or calState, or with non-positive magnitude, set to NaN."
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
