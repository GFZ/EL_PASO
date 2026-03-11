# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import logging
import typing
from typing import Literal

import astropy.units as u  # type: ignore[reportMissingTypeStubs]
import numpy as np

import el_paso as ep
from el_paso.utils import timed_function

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

DENTON_DENSITY_UPPER_LIMIT = 1500
MAX_ALPHA = 5


@timed_function("Density mapping")
def compute_equatorial_plasmaspheric_density(
    density_var: ep.Variable,
    xgeo_local_var: ep.Variable,
    xgeo_equatorial_var: ep.Variable,
    method: Literal["Denton_average"] = "Denton_average",
) -> ep.Variable:
    r"""Maps the plasma density from a local measurement point to the magnetic equator.

    This function uses a simple empirical model to map plasma density measured
    at a specific local position to the magnetic equator. The mapping is based
    on a power-law density profile, $n(r) = n_{eq} (r / r_{eq})^{-\alpha}$,
    where the exponent $\alpha$ is determined by the chosen method.

    Args:
        density_var (ep.Variable): The input variable containing the local
            plasma density data, expected in units convertible to $cm^{-3}$.
        xgeo_local_var (ep.Variable): The variable containing the GEO coordinates
            corresponding to the density measurements.
        xgeo_equatorial_var (ep.Variable): The variable containing the GEO
            coordinates of the corresponding magnetic field line foot-point
            at the magnetic equator.
        method (Literal["Denton_average"], optional): The method used
            to calculate the power-law exponent $\alpha$.
            - "Denton_average" (default): Uses a fixed average $\\alpha$: $2.5$ inside the
              plasmasphere (density $\ge 10 \cdot (6.6 / r_{eq})^4$) and $0.5$
              outside the plasmasphere. This is an approximation based on Sheeley et al. (2001).

    Returns:
        ep.Variable: A new variable containing the plasma density mapped to the
            magnetic equator, with units of $cm^{-3}$.
    """
    logger.info("Computing equatorial plasmaspheric density...")

    r_local = np.linalg.norm(xgeo_local_var.get_data(ep.units.RE).astype(np.float64), ord=2, axis=1)
    r_eq = np.linalg.norm(xgeo_equatorial_var.get_data(ep.units.RE).astype(np.float64), ord=2, axis=1)
    r_local = typing.cast("NDArray[np.float32]", r_local)
    r_eq = typing.cast("NDArray[np.float32]", r_eq)

    density_data = density_var.get_data(u.cm ** (-3)).astype(np.float64)  # type: ignore[reportUnknownMemberType]

    mapped_density_var = ep.Variable(original_unit=u.cm ** (-3))  # type: ignore[reportUnknownMemberType]
    mapped_density_var.metadata.source_files = density_var.metadata.source_files

    match method:
        case "Denton_average":
            inside_pp = density_data >= 10 * (6.6 / r_eq) ** 4

            alpha = np.full_like(density_data, 0.5)
            alpha[inside_pp] = 2.5

            density_eq_data = density_data / (r_eq / r_local) ** alpha

            mapped_density_var.metadata.add_processing_note(
                "Mapped to the equator using 'compute_equatorial_plasmaspheric_density' assuming the Denton average "
                "approximation with alpha=2.5 inside the plasmasphere (according to the criterion used in "
                "Sheeley et al. 2001) and alpha=1 outside the plasmasphere."
            )

    mapped_density_var.set_data(density_eq_data, u.cm ** (-3))  # type: ignore[reportUnknownMemberType]

    return mapped_density_var
