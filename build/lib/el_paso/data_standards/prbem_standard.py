# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from astropy import units as u  # type: ignore[reportMissingTypeStubs]

import el_paso as ep
from el_paso.data_standard import ConsistencyCheck, DataStandard
from el_paso.utils import assert_n_dim


class PRBEMStandard(DataStandard):
    """A data standard of the Panel for Radiation Belt Environment Modeling (PRBEM).

    This class defines and applies a specific set of data standards for variables
    defined by the [PRBEM](https://prbem.github.io/documents/Standard_File_Format.pdf).
    It standardizes variables by converting them to canonical units and performing
    consistency checks on their dimensions and shapes, ensuring they conform to the
    expected format for each standard name.
    """

    def __init__(self) -> None:
        """Initializes the PRBEMStandard with a ConsistencyCheck object."""
        self.consistency_check = ConsistencyCheck()

    def standardize_variable(  # noqa: C901, PLR0912, PLR0915
        self, standard_name: str, variable: ep.Variable, *, reset_consistency_check: bool
    ) -> ep.Variable:
        """Standardizes a variable based on its specified standard name.

        This method first converts the variable to its canonical unit based on the
        `standard_name`. It then performs a series of dimension and shape
        consistency checks to ensure the variable's structure is valid for
        the given data type.

        Args:
            standard_name (str): The name of the data standard to apply (e.g.,
                'FEDU', 'xGEO', 'Lstar').
            variable (ep.Variable): The variable to be standardized.
            reset_consistency_check (bool): If set to true, the consistency check will be reseted.

        Returns:
            ep.Variable: The standardized variable with its unit converted and
                          its consistency validated.
        """
        if reset_consistency_check:
            self.consistency_check = ConsistencyCheck()

        if "FEDU" in standard_name:
            variable.convert_to_unit((u.cm**2 * u.s * u.sr * u.keV) ** (-1))  # type: ignore[reportUnknownArgumentType]

            assert_n_dim(variable, 3, standard_name)
            shape = variable.get_data().shape
            self.consistency_check.check_time_size(shape[0], standard_name)
            self.consistency_check.check_energy_size(shape[1], standard_name)
            self.consistency_check.check_pitch_angle_size(shape[2], standard_name)

            if len(variable.metadata.description) == 0:
                variable.metadata.description = "Processed unidirectional differential electron flux"

        elif "FEDO" in standard_name:
            variable.convert_to_unit((u.cm**2 * u.s * u.sr * u.keV) ** (-1))  # type: ignore[reportUnknownArgumentType]

            assert_n_dim(variable, 2, standard_name)
            shape = variable.get_data().shape
            self.consistency_check.check_time_size(shape[0], standard_name)
            self.consistency_check.check_energy_size(shape[1], standard_name)

            if len(variable.metadata.description) == 0:
                variable.metadata.description = "Processed omnidirectional differential electron flux"

        elif "alpha" in standard_name:
            variable.convert_to_unit(u.deg)  # type: ignore[reportUnknownArgumentType]

            assert_n_dim(variable, 2, standard_name)
            shape = variable.get_data().shape
            self.consistency_check.check_time_size(shape[0], standard_name)
            self.consistency_check.check_pitch_angle_size(shape[1], standard_name)

        elif "energy" in standard_name:
            variable.convert_to_unit(u.MeV)  # type: ignore[reportUnknownArgumentType]

            assert_n_dim(variable, 2, standard_name)
            shape = variable.get_data().shape
            self.consistency_check.check_time_size(shape[0], standard_name)
            self.consistency_check.check_energy_size(shape[1], standard_name)

        elif "xGEO" in standard_name:
            variable.convert_to_unit(ep.units.RE)

            assert_n_dim(variable, 2, standard_name)
            self.consistency_check.check_time_size(variable.get_data().shape[0], standard_name)

        elif "MLT" in standard_name:
            variable.convert_to_unit(u.hour)  # type: ignore[reportUnknownArgumentType]

            assert_n_dim(variable, 1, standard_name)
            self.consistency_check.check_time_size(variable.get_data().shape[0], standard_name)

        elif "R0" in standard_name:
            variable.convert_to_unit(ep.units.RE)

            assert_n_dim(variable, 1, standard_name)
            self.consistency_check.check_time_size(variable.get_data().shape[0], standard_name)

        elif "Lstar" in standard_name or "lm" in standard_name:
            variable.convert_to_unit(u.dimensionless_unscaled)

            assert_n_dim(variable, 2, standard_name)
            shape = variable.get_data().shape
            self.consistency_check.check_time_size(shape[0], standard_name)
            self.consistency_check.check_pitch_angle_size(shape[1], standard_name)

        elif "B_eq" in standard_name or "B_local" in standard_name:
            variable.convert_to_unit(u.nT)  # type: ignore[reportUnknownArgumentType]

            assert_n_dim(variable, 1, standard_name)
            self.consistency_check.check_time_size(variable.get_data().shape[0], standard_name)

        elif "PSD" in standard_name:
            variable.convert_to_unit((u.m * u.kg * u.m / u.s) ** (-3))  # type: ignore[reportUnknownArgumentType]

            assert_n_dim(variable, 3, standard_name)
            shape = variable.get_data().shape
            self.consistency_check.check_time_size(shape[0], standard_name)
            self.consistency_check.check_energy_size(shape[1], standard_name)
            self.consistency_check.check_pitch_angle_size(shape[2], standard_name)

        elif "inv_mu" in standard_name:
            variable.convert_to_unit(u.MeV / u.G)  # type: ignore[reportUnknownArgumentType]

            assert_n_dim(variable, 3, standard_name)
            shape = variable.get_data().shape
            self.consistency_check.check_time_size(shape[0], standard_name)
            self.consistency_check.check_energy_size(shape[1], standard_name)
            self.consistency_check.check_pitch_angle_size(shape[2], standard_name)

        elif "inv_K" in standard_name:
            variable.convert_to_unit(ep.units.RE * u.G**0.5)  # type: ignore[reportUnknownArgumentType]

            assert_n_dim(variable, 2, standard_name)
            shape = variable.get_data().shape
            self.consistency_check.check_time_size(shape[0], standard_name)
            self.consistency_check.check_pitch_angle_size(shape[1], standard_name)

        elif "density" in standard_name:
            variable.convert_to_unit(u.cm ** (-3))  # type: ignore[reportUnknownArgumentType]

            assert_n_dim(variable, 1, standard_name)
            shape = variable.get_data().shape
            self.consistency_check.check_time_size(shape[0], standard_name)

        return variable
