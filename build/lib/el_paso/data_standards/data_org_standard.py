# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from astropy import units as u  # type: ignore[reportMissingTypeStubs]

import el_paso as ep
from el_paso.data_standard import ConsistencyCheck, DataStandard
from el_paso.utils import assert_n_dim


class DataOrgStandard(DataStandard):
    """A data standard used historically at the GFZ German Research Centre for Geosciences.

    This standard defines rules for a set of canonical variable names by converting them
    to correct units and checking their array dimensions for consistency. It is tailored
    for compatibility with historical GFZ datasets and internal workflows.
    """

    def __init__(self) -> None:
        """Initializes the DataOrgStandard with a ConsistencyCheck object."""
        self.consistency_check = ConsistencyCheck()

    def standardize_variable(  # noqa: C901, PLR0912, PLR0915
        self,
        standard_name: str,
        variable: ep.Variable,
        *,
        reset_consistency_check: bool,
    ) -> ep.Variable:
        """Standardizes a variable based on its specified standard name.

        Applies unit conversions and dimension checks to a variable, ensuring its structure
        conforms to the expectations for its `standard_name`.

        Parameters:
            standard_name (str): The canonical name of the variable (e.g., 'FEDU', 'xGEO').
            variable (ep.Variable): The variable to be standardized.
            reset_consistency_check (bool): If set to true, the consistency check will be reseted.

        Returns:
            ep.Variable: The standardized variable.

        Raises:
            AssertionError: If the variable's dimensions are incorrect.
            UnitConversionError: If unit conversion fails.
        """
        if reset_consistency_check:
            self.consistency_check = ConsistencyCheck()

        match standard_name:
            case "time":
                variable.convert_to_unit(ep.units.datenum)
                assert_n_dim(variable, 1, standard_name)
            case "Flux":
                variable.convert_to_unit((u.cm**2 * u.s * u.sr * u.keV) ** (-1))  # type: ignore[reportUnknownArgumentType]

                assert_n_dim(variable, 3, standard_name)
                shape = variable.get_data().shape
                self.consistency_check.check_time_size(shape[0], standard_name)
                self.consistency_check.check_energy_size(shape[1], standard_name)
                self.consistency_check.check_pitch_angle_size(shape[2], standard_name)

            case "alpha_local" | "alpha_eq_model":
                variable.convert_to_unit(u.radian)  # type: ignore[reportUnknownArgumentType]

                assert_n_dim(variable, 2, standard_name)
                shape = variable.get_data().shape
                self.consistency_check.check_time_size(shape[0], standard_name)
                self.consistency_check.check_pitch_angle_size(shape[1], standard_name)

            case "energy_channels":
                variable.convert_to_unit(u.MeV)  # type: ignore[reportUnknownArgumentType]

                assert_n_dim(variable, 2, standard_name)
                shape = variable.get_data().shape
                self.consistency_check.check_time_size(shape[0], standard_name)
                self.consistency_check.check_energy_size(shape[1], standard_name)

            case "MLT":
                variable.convert_to_unit(u.hour)  # type: ignore[reportUnknownArgumentType]

                assert_n_dim(variable, 1, standard_name)
                self.consistency_check.check_time_size(variable.get_data().shape[0], standard_name)

            case "Lstar" | "Lm":
                variable.convert_to_unit(u.dimensionless_unscaled)

                assert_n_dim(variable, 2, standard_name)
                shape = variable.get_data().shape
                self.consistency_check.check_time_size(shape[0], standard_name)
                self.consistency_check.check_pitch_angle_size(shape[1], standard_name)

            case "xGEO":
                variable.convert_to_unit(ep.units.RE)
                assert_n_dim(variable, 2, standard_name)
                self.consistency_check.check_time_size(variable.get_data().shape[0], standard_name)

            case "B_eq" | "B_local":
                variable.convert_to_unit(u.nT)  # type: ignore[reportUnknownArgumentType]

                assert_n_dim(variable, 1, standard_name)
                self.consistency_check.check_time_size(variable.get_data().shape[0], standard_name)

            case "R0":
                variable.convert_to_unit(ep.units.RE)

                assert_n_dim(variable, 1, standard_name)
                self.consistency_check.check_time_size(variable.get_data().shape[0], standard_name)

            case "density":
                variable.convert_to_unit(u.cm ** (-3))  # type: ignore[reportUnknownArgumentType]

                assert_n_dim(variable, 1, standard_name)
                self.consistency_check.check_time_size(variable.get_data().shape[0], standard_name)
            case "PSD":
                variable.convert_to_unit((u.m * u.kg * u.m / u.s) ** (-3))  # type: ignore[reportUnknownArgumentType]

                assert_n_dim(variable, 3, standard_name)
                shape = variable.get_data().shape
                self.consistency_check.check_time_size(shape[0], standard_name)
                self.consistency_check.check_energy_size(shape[1], standard_name)
                self.consistency_check.check_pitch_angle_size(shape[2], standard_name)

            case "InvMu":
                variable.convert_to_unit(u.MeV / u.G)  # type: ignore[reportUnknownArgumentType]

                assert_n_dim(variable, 3, standard_name)
                shape = variable.get_data().shape
                self.consistency_check.check_time_size(shape[0], standard_name)
                self.consistency_check.check_energy_size(shape[1], standard_name)
                self.consistency_check.check_pitch_angle_size(shape[2], standard_name)

            case "InvK":
                variable.convert_to_unit(ep.units.RE * u.G**0.5)  # type: ignore[reportUnknownArgumentType]

                assert_n_dim(variable, 2, standard_name)
                shape = variable.get_data().shape
                self.consistency_check.check_time_size(shape[0], standard_name)
                self.consistency_check.check_pitch_angle_size(shape[1], standard_name)

            case _:
                msg = f"Encountered invalid name_in_file: {standard_name}!"
                raise ValueError(msg)

        return variable
