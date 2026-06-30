# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import lazy_loader as lazy

# Each processing step is resolved on first access (SPEC 1), so that e.g.
# `el_paso.processing.bin_by_time` does not also pay for IRBEM, skyfield or sscws.
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["magnetic_field_utils"],
    submod_attrs={
        "bin_by_time": ["TimeBinMethod", "bin_by_time"],
        "calculate_geo_coords_from_file": ["calculate_geo_coords_from_omm", "calculate_geo_coords_from_tle"],
        "compute_electron_gyrofrequency": ["compute_electron_gyrofrequency"],
        "compute_equatorial_plasmaspheric_density": ["compute_equatorial_plasmaspheric_density"],
        "compute_invariant_K": ["compute_invariant_K"],
        "compute_invariant_mu": ["compute_invariant_mu"],
        "compute_LCDS": ["compute_LCDS"],
        "compute_magnetic_field_variables": ["VariableRequest", "compute_magnetic_field_variables"],
        "compute_magnetic_latitude": ["compute_magnetic_latitude"],
        "compute_phase_space_density": ["compute_phase_space_density"],
        "compute_pitch_angles_for_telescopes": ["compute_pitch_angles_for_telescopes"],
        "construct_pitch_angle_distribution": ["construct_pitch_angle_distribution"],
        "convert_string_to_datetime": ["convert_string_to_datetime"],
        "create_quality_flag_from_magnetometer": ["create_quality_flag_from_magnetometer"],
        "fold_pitch_angles_and_flux": ["fold_pitch_angles_and_flux"],
        "get_real_time_tipsod": ["get_real_time_tipsod"],
        "interpolate_in_time": ["interpolate_in_time"],
        "map_to_dipole_equator": ["map_to_dipole_equator"],
    },
)

if TYPE_CHECKING:
    from el_paso.processing import magnetic_field_utils
    from el_paso.processing.bin_by_time import TimeBinMethod, bin_by_time
    from el_paso.processing.calculate_geo_coords_from_file import (
        calculate_geo_coords_from_omm,
        calculate_geo_coords_from_tle,
    )
    from el_paso.processing.compute_electron_gyrofrequency import compute_electron_gyrofrequency
    from el_paso.processing.compute_equatorial_plasmaspheric_density import compute_equatorial_plasmaspheric_density
    from el_paso.processing.compute_invariant_K import compute_invariant_K
    from el_paso.processing.compute_invariant_mu import compute_invariant_mu
    from el_paso.processing.compute_LCDS import compute_LCDS
    from el_paso.processing.compute_magnetic_field_variables import VariableRequest, compute_magnetic_field_variables
    from el_paso.processing.compute_magnetic_latitude import compute_magnetic_latitude
    from el_paso.processing.compute_phase_space_density import compute_phase_space_density
    from el_paso.processing.compute_pitch_angles_for_telescopes import compute_pitch_angles_for_telescopes
    from el_paso.processing.construct_pitch_angle_distribution import construct_pitch_angle_distribution
    from el_paso.processing.convert_string_to_datetime import convert_string_to_datetime
    from el_paso.processing.create_quality_flag_from_magnetometer import create_quality_flag_from_magnetometer
    from el_paso.processing.fold_pitch_angles_and_flux import fold_pitch_angles_and_flux
    from el_paso.processing.get_real_time_tipsod import get_real_time_tipsod
    from el_paso.processing.interpolate_in_time import interpolate_in_time
    from el_paso.processing.map_to_dipole_equator import map_to_dipole_equator

    __all__ = [
        "TimeBinMethod",
        "VariableRequest",
        "bin_by_time",
        "calculate_geo_coords_from_omm",
        "calculate_geo_coords_from_tle",
        "compute_electron_gyrofrequency",
        "compute_equatorial_plasmaspheric_density",
        "compute_invariant_K",
        "compute_invariant_mu",
        "compute_LCDS",
        "compute_magnetic_field_variables",
        "compute_magnetic_latitude",
        "compute_phase_space_density",
        "compute_pitch_angles_for_telescopes",
        "construct_pitch_angle_distribution",
        "convert_string_to_datetime",
        "create_quality_flag_from_magnetometer",
        "fold_pitch_angles_and_flux",
        "get_real_time_tipsod",
        "interpolate_in_time",
        "magnetic_field_utils",
        "map_to_dipole_equator",
    ]
