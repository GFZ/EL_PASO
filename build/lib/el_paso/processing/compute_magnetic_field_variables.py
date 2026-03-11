# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: N806

from __future__ import annotations

import logging
import typing
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
from astropy import units as u  # type: ignore[reportMissingTypeStubs]

import el_paso as ep
import el_paso.processing.magnetic_field_utils as mag_utils
from el_paso import Variable
from el_paso.utils import make_dict_hashable, timed_function

logger = logging.getLogger(__name__)

VariableRequest = Sequence[tuple[mag_utils.MagFieldVarTypes, mag_utils.MagneticFieldLiteral | mag_utils.MagneticField]]


class MagFieldVar(NamedTuple):
    """A named tuple to represent a request for a magnetic field variable.

    Attributes:
        type (mag_utils.MagFieldVarTypes): The type of magnetic field variable to compute (e.g., "B_local", "Lstar").
        mag_field (str | mag_utils.MagneticField): The magnetic field model to use for the computation .
    """

    type: mag_utils.MagFieldVarTypes
    mag_field: str | mag_utils.MagneticField


def compute_magnetic_field_variables(
    time_var: Variable,
    xgeo_var: Variable,
    variables_to_compute: VariableRequest,
    irbem_lib_path: str,
    irbem_options: list[int],
    num_cores: int,
    indices_solar_wind: dict[str, Variable] | None = None,
    pa_local_var: Variable | None = None,
    energy_var: Variable | None = None,
    particle_species: Literal["electron", "proton"] | None = None,
) -> dict[str, Variable]:
    """Computes various magnetic field-related variables using the IRBEM library.

    This function serves as a wrapper to calculate a suite of magnetic field
    and related invariants (like L-star, MLT, B_local, B_eq, invariant Mu,
    and invariant K) based on provided time and geocentric coordinates. It
    leverages the IRBEM library for the underlying computations.

    Args:
        time_var (Variable): A Variable object containing time data. The data should be a 1D array of timestamps.
        xgeo_var (Variable): A Variable object containing geocentric (XGEO)
            coordinates. Expected to be a 2D array (time, 3) where the last
            dimension represents X, Y, Z coordinates.
        variables_to_compute (Sequence[tuple[mag_utils.MagFieldVarTypes, str | mag_utils.MagneticField]]):
            A sequence of tuples, where each tuple specifies a variable to compute. The first element is the
            variable type (e.g., "Lstar"), and the second is the magnetic field model to use (e.g., "IGRF").
        irbem_lib_path (str): The file path to the compiled IRBEM library object
            (a `.so` or `.dll` file).
        irbem_options (list[int]): A list of 5 integer options for the IRBEM library
            calls, controlling aspects like model selection, bounce tracing, etc.
        num_cores (int): The number of CPU cores to use for parallel processing
            within IRBEM calls.
        indices_solar_wind (dict[str, Variable] | None): Optional. A dictionary
            containing solar wind indices (e.g., "Kp", "Dst") as `Variable` objects.
            Defaults to None.
        pa_local_var (Variable | None): Optional. A Variable object containing
            local pitch angle data in degrees. Required if any pitch-angle dependent variables
            (e.g., "PA_eq", "Lstar", "invMu", "invK") are requested. Defaults to None.
        energy_var (Variable | None): Optional. A Variable object containing
            particle energy data in MeV. Required if "invMu" is requested. Defaults to None.
        particle_species (Literal["electron", "proton"] | None): Optional. The
            species of particle (e.g., "electron", "proton"). Required if "invMu"
            is requested. Defaults to None.

    Returns:
        dict[str, Variable]: A dictionary where keys are the computed variable
        names and values are their corresponding `Variable` objects containing
        the calculated data and metadata.

    Raises:
        FileNotFoundError: If no IRBEM library object is found at the provided `irbem_lib_path`.
        ValueError:
            - If `irbem_options` does not contain exactly 5 entries.
            - If a pitch-angle dependent variable is requested but `pa_local_var` is not provided.
            - If an energy-dependent variable ("invMu") is requested but `energy_var` is not provided.
            - If a particle-species dependent variable ("invMu") is requested but `particle_species` is not provided.
        NotImplementedError: If a requested variable name in `variables_to_compute`
            is not supported by this function.

    Notes:
        - The function internally constructs an `IrbemInput` object for each unique
          magnetic field model encountered in `variables_to_compute`.
        - Intermediate computed variables (e.g., `B_eq`, `B_local`) are cached
          within the function to avoid redundant IRBEM calls when multiple
          dependent variables are requested.
    """
    if not Path(irbem_lib_path).is_file():
        msg = f"No library object found under the provided irbem_lib_path: {irbem_lib_path}"
        raise FileNotFoundError(msg)

    if len(irbem_options) != 5:  # noqa: PLR2004
        msg = f"irbem_options must be a list with exactly 5 entries! Got: {irbem_options}"
        raise ValueError(msg)

    mag_variables_to_compute = [
        MagFieldVar(mag_field_var[0], mag_field_var[1]) for mag_field_var in variables_to_compute
    ]

    if _requires_pa_var(mag_variables_to_compute) and pa_local_var is None:
        msg = "Pitch-angle dependent variable is requested but local pitch angles are not provided!"
        raise ValueError(msg)
    pa_local_var = typing.cast("Variable", pa_local_var)

    if _requires_energy_var(mag_variables_to_compute) and energy_var is None:
        msg = "Energy dependent variable is requested but energies are not provided!"
        raise ValueError(msg)
    energy_var = typing.cast("Variable", energy_var)

    if _requires_particle_species(mag_variables_to_compute) and particle_species is None:
        msg = "Particle-species dependent variable is requested but particle_species is not provided!"
        raise ValueError(msg)
    particle_species = typing.cast("Literal['electron', 'proton']", particle_species)

    # collect magnetic_field results in this dictionary
    computed_variables: dict[str, Variable] = {}
    var_names_to_compute: list[str] = []

    for mag_field_var in mag_variables_to_compute:
        var_type = mag_field_var.type
        mag_field = mag_field_var.mag_field

        if isinstance(mag_field, str):
            mag_field = mag_utils.MagneticField(mag_field)

        variable_name = mag_utils.create_var_name(var_type, mag_field)
        var_names_to_compute.append(variable_name)

        # check if the value has been calculated already
        if variable_name in computed_variables:
            continue

        indices_solar_wind_hashable = make_dict_hashable(indices_solar_wind)

        maginput = mag_utils.construct_maginput(time_var, mag_field, indices_solar_wind_hashable)

        irbem_input = mag_utils.IrbemInput(irbem_lib_path, mag_field, maginput, irbem_options, num_cores)  # type: ignore[bad-argument-type]

        computed_variables |= _get_result(
            var_type, xgeo_var, time_var, pa_local_var, energy_var, computed_variables, irbem_input, particle_species
        )

    # only return the requested variables
    computed_variables = {
        var_name: computed_variables[var_name] for var_name in computed_variables if var_name in var_names_to_compute
    }

    return computed_variables


def _get_result(
    var_type: mag_utils.MagFieldVarTypes,
    xgeo_var: Variable,
    time_var: Variable,
    pa_local_var: Variable,
    energy_var: Variable,
    computed_vars: dict[str, Variable],
    irbem_input: mag_utils.IrbemInput,
    particle_species: Literal["electron", "proton"],
) -> dict[str, Variable]:
    """Helper function to get the result for a specific magnetic field variable.

    Args:
        var_type (mag_utils.MagFieldVarTypes): The type of magnetic field variable to compute.
        xgeo_var (Variable): Variable containing geocentric (XGEO) coordinates.
        time_var (Variable): Variable containing time data.
        pa_local_var (Variable): Variable containing local pitch angles.
        energy_var (Variable): Variable containing particle energies.
        computed_vars (dict[str, Variable]): A dictionary of already computed variables to reuse.
        irbem_input (mag_utils.IrbemInput): A named tuple containing all necessary IRBEM inputs.
        particle_species (Literal["electron", "proton"]): The species of the particle.

    Returns:
        dict[str, Variable]: A dictionary containing the newly computed variables.

    Raises:
        NotImplementedError: If the requested variable type is not supported.
    """
    match var_type:
        case "B_local":
            result_dict = mag_utils.get_local_B_field(xgeo_var, time_var, irbem_input)

        case "B_fofl":
            result_dict = mag_utils.get_footpoint_atmosphere(xgeo_var, time_var, irbem_input)

        case "B_mirr":
            result_dict = mag_utils.get_mirror_point(xgeo_var, time_var, pa_local_var, irbem_input)

        case "MLT":
            result_dict = mag_utils.get_MLT(xgeo_var, time_var, irbem_input)

        case "R_eq" | "B_eq" | "xGEO_eq":
            result_dict = mag_utils.get_magequator(xgeo_var, time_var, irbem_input)

        case "Lstar" | "Lm" | "XJ":
            result_dict = mag_utils.get_Lstar(xgeo_var, time_var, pa_local_var, irbem_input)

        case "PA_eq":
            result_dict = _get_pa_eq(
                xgeo_var,
                time_var,
                pa_local_var,
                computed_vars,
                irbem_input,
            )

        case "invMu":
            result_dict = _get_invariant_mu(
                xgeo_var,
                time_var,
                pa_local_var,
                energy_var,
                computed_vars,
                irbem_input,
                particle_species,
            )

        case "invK":
            result_dict = _get_invariant_K(
                xgeo_var,
                time_var,
                pa_local_var,
                computed_vars,
                irbem_input,
            )

        case _:
            msg = f"Variable '{var_type}' is not implemented in compute_magnetic_field_variables."
            raise NotImplementedError(msg)

    return result_dict


def _requires_particle_species(vars_to_compute: list[MagFieldVar]) -> bool:
    """Checks if any of the requested magnetic field variables require particle species information.

    Args:
        vars_to_compute (list[MagFieldVar]): A list of named tuples specifying the variables to compute.

    Returns:
        bool: True if "invMu" is in the list of variables to compute, False otherwise.
    """
    var_types = [mag_field_var.type for mag_field_var in vars_to_compute]

    return any(var_type == "invMu" for var_type in var_types)


def _requires_energy_var(vars_to_compute: list[MagFieldVar]) -> bool:
    """Checks if any of the requested magnetic field variables require energy data.

    Args:
        vars_to_compute (list[MagFieldVar]): A list of named tuples specifying the variables to compute.

    Returns:
        bool: True if "invMu" is in the list of variables to compute, False otherwise.
    """
    var_types = [mag_field_var.type for mag_field_var in vars_to_compute]

    return any(var_type == "invMu" for var_type in var_types)


def _requires_pa_var(vars_to_compute: list[MagFieldVar]) -> bool:
    """Checks if any of the requested magnetic field variables require local pitch angle data.

    Args:
        vars_to_compute (list[MagFieldVar]): A list of named tuples specifying the variables to compute.

    Returns:
        bool: True if any of the specified variables require pitch angle data, False otherwise.
    """
    var_types = [mag_field_var.type for mag_field_var in vars_to_compute]

    return any(var_type in ["Lstar", "PA_eq", "invMu", "invK", "B_mirr", "XJ", "Lm"] for var_type in var_types)


@timed_function("Equatorial pitch angle calculation")
def _get_pa_eq(
    xgeo_var: Variable,
    time_var: Variable,
    pa_local_var: Variable,
    computed_vars: dict[str, Variable],
    irbem_input: mag_utils.IrbemInput,
) -> dict[str, Variable]:
    """Calculates the equatorial pitch angle (PA_eq) and returns the result in a dictionary.

    Args:
        xgeo_var (Variable): Variable containing geocentric (XGEO) coordinates.
        time_var (Variable): Variable containing time data.
        pa_local_var (Variable): Variable containing local pitch angles.
        computed_vars (dict[str, Variable]): A dictionary of already computed variables to reuse.
        irbem_input (mag_utils.IrbemInput): A named tuple containing all necessary IRBEM inputs.

    Returns:
        dict[str, Variable]: A dictionary containing the newly computed PA_eq variable.
    """
    logger.info("\tCalculating equatorial pitch angle ...")

    pa_local = pa_local_var.get_data(u.radian)

    B_eq_name = mag_utils.create_var_name("B_eq", irbem_input.magnetic_field)
    B_local_name = mag_utils.create_var_name("B_local", irbem_input.magnetic_field)

    if B_eq_name not in computed_vars:
        computed_vars |= mag_utils.get_magequator(xgeo_var, time_var, irbem_input)
    if B_local_name not in computed_vars:
        computed_vars |= mag_utils.get_local_B_field(xgeo_var, time_var, irbem_input)

    B_eq = computed_vars[B_eq_name].get_data(u.nT)
    B_local = computed_vars[B_local_name].get_data(u.nT)

    pa_eq_rad = np.asin(np.sin(pa_local) * np.sqrt(B_eq / B_local)[:, np.newaxis])

    pa_var = Variable(data=pa_eq_rad, original_unit=u.radian)
    pa_var.metadata.add_processing_note(
        "Computed equatorial pitch angle from local pitch angle and B_eq/B_local ratio "
        f"using {irbem_input.magnetic_field} and options: {irbem_input.irbem_options}."
    )

    computed_vars[mag_utils.create_var_name("PA_eq", irbem_input.magnetic_field)] = pa_var

    return computed_vars


@timed_function("Invariant mu calculation")
def _get_invariant_mu(
    xgeo_var: Variable,
    time_var: Variable,
    pa_local_var: Variable,
    energy_var: Variable,
    computed_vars: dict[str, Variable],
    irbem_input: mag_utils.IrbemInput,
    particle_species: Literal["electron", "proton"],
) -> dict[str, Variable]:
    """Calculates the first adiabatic invariant (invariant mu) and returns the result in a dictionary.

    Args:
        xgeo_var (Variable): Variable containing geocentric (XGEO) coordinates.
        time_var (Variable): Variable containing time data.
        pa_local_var (Variable): Variable containing local pitch angles.
        energy_var (Variable): Variable containing particle energies.
        computed_vars (dict[str, Variable]): A dictionary of already computed variables to reuse.
        irbem_input (mag_utils.IrbemInput): A named tuple containing all necessary IRBEM inputs.
        particle_species (Literal["electron", "proton"]): The species of the particle.

    Returns:
        dict[str, Variable]: A dictionary containing the newly computed invariant mu variable.
    """
    logger.info("\tCalculating invariant mu ...")

    B_local_name = mag_utils.create_var_name("B_local", irbem_input.magnetic_field)

    if B_local_name not in computed_vars:
        computed_vars |= mag_utils.get_local_B_field(xgeo_var, time_var, irbem_input)

    # load needed data and convert to correct units
    B_local = computed_vars[B_local_name]

    mu_var = ep.processing.compute_invariant_mu(energy_var, pa_local_var, B_local, particle_species)

    computed_vars[mag_utils.create_var_name("invMu", irbem_input.magnetic_field)] = mu_var

    return computed_vars


@timed_function("Invariant K calculation")
def _get_invariant_K(  # noqa: N802
    xgeo_var: Variable,
    time_var: Variable,
    pa_local_var: Variable,
    computed_vars: dict[str, Variable],
    irbem_input: mag_utils.IrbemInput,
) -> dict[str, Variable]:
    """Calculates the second adiabatic invariant (invariant K) and returns the result in a dictionary.

    Args:
        xgeo_var (Variable): Variable containing geocentric (XGEO) coordinates.
        time_var (Variable): Variable containing time data.
        pa_local_var (Variable): Variable containing local pitch angles.
        computed_vars (dict[str, Variable]): A dictionary of already computed variables to reuse.
        irbem_input (mag_utils.IrbemInput): A named tuple containing all necessary IRBEM inputs.

    Returns:
        dict[str, Variable]: A dictionary containing the newly computed invariant K variable.
    """
    logger.info("\tCalculating invariant K ...")

    xj_name = mag_utils.create_var_name("XJ", irbem_input.magnetic_field)
    B_mirr_name = mag_utils.create_var_name("B_mirr", irbem_input.magnetic_field)

    if xj_name not in computed_vars:
        computed_vars |= mag_utils.get_Lstar(xgeo_var, time_var, pa_local_var, irbem_input)
    if B_mirr_name not in computed_vars:
        computed_vars |= mag_utils.get_mirror_point(xgeo_var, time_var, pa_local_var, irbem_input)

    # load needed data and convert to correct units
    B_mirr = computed_vars[B_mirr_name]
    xj = computed_vars[xj_name]

    inv_k_name = mag_utils.create_var_name("invK", irbem_input.magnetic_field)
    computed_vars[inv_k_name] = ep.processing.compute_invariant_K(B_mirr, xj)

    return computed_vars
