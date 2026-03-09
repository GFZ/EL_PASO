# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from el_paso.processing import magnetic_field_utils
from el_paso.processing.bin_by_time import TimeBinMethod, bin_by_time
from el_paso.processing.compute_equatorial_plasmaspheric_density import compute_equatorial_plasmaspheric_density
from el_paso.processing.compute_invariant_K import compute_invariant_K
from el_paso.processing.compute_invariant_mu import compute_invariant_mu
from el_paso.processing.compute_magnetic_field_variables import VariableRequest, compute_magnetic_field_variables
from el_paso.processing.compute_phase_space_density import compute_phase_space_density
from el_paso.processing.construct_pitch_angle_distribution import construct_pitch_angle_distribution
from el_paso.processing.convert_string_to_datetime import convert_string_to_datetime
from el_paso.processing.fold_pitch_angles_and_flux import fold_pitch_angles_and_flux
from el_paso.processing.get_real_time_tipsod import get_real_time_tipsod
from el_paso.processing.magnetic_field_utils import MagFieldVarTypes
from el_paso.processing.clean_magnetometer_data import clean_magnetometer_variables
from el_paso.processing.calculate_L_mlt_mlat_fce_eq import add_derived_params


__all__ = [
    "MagFieldVarTypes",
    "TimeBinMethod",
    "VariableRequest",
    "bin_by_time",
    "compute_equatorial_plasmaspheric_density",
    "compute_invariant_K",
    "compute_invariant_mu",
    "compute_magnetic_field_variables",
    "compute_phase_space_density",
    "construct_pitch_angle_distribution",
    "convert_string_to_datetime",
    "fold_pitch_angles_and_flux",
    "get_real_time_tipsod",
    "magnetic_field_utils",
    "clean_magnetometer_variables",
    "add_derived_params",
]
