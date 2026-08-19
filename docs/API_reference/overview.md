<!--
SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
SPDX-FileContributor: Bernhard Haas

SPDX-License-Identifier: Apache-2.0
-->

# Overview

This section provides a detailed reference for all modules, classes, and functions in `el_paso`.

## Core classes

[DataStandard](data_standard.md)

[SavingStrategy](saving_strategy.md)

[Variable](variable.md)

[DataSet](dataset/dataset.md)

[Metadata](metadata.md)

## Core functions

[download](download.md)

[extract_variables_from_files](extract_variables_from_files.md)

[save](save.md)

## Utilities

[Cache](utilities/cache.md)

[General utilities](utilities/general_utilities.md)

[Load geomagnetic indices and solar wind parameters](utilities/load_indices_solar_wind_parameters.md)

[Magnetic field utilities](utilities/magnetic_field_utilities.md)

[Physics](physics.md)

[Scripts](utilities/scripts.md)

[Release mode](utilities/release_mode.md)

[Typing](utilities/typing.md)

[Units](utilities/units.md)

## Processing functions

[bin_by_time](processing/bin_by_time.md)

[calculate_geo_coords_from_omm](processing/calculate_geo_coords_from_omm.md)

[calculate_geo_coords_from_tle](processing/calculate_geo_coords_from_tle.md)

[compute_equatorial_plasmaspheric_density](processing/compute_equatorial_plasmaspheric_density.md)

[compute_invariant_K](processing/compute_invariant_K.md)

[compute_invariant_mu](processing/compute_invariant_mu.md)

[compute_magnetic_field_variables](processing/compute_magnetic_field_variables.md)

[compute_phase_space_density](processing/compute_phase_space_density.md)

[compute_pitch_angles_for_telescopes](processing/compute_pitch_angles_for_telescopes.md)

[construct_pitch_angle_distribution](processing/construct_pitch_angle_distribution.md)

[convert_string_to_datetime](processing/convert_string_to_datetime.md)

[create_quality_flag_from_magnetometer](processing/create_quality_flag_from_magnetometer.md)

[fold_pitch_angles_and_flux](processing/fold_pitch_angles_and_flux.md)

[get_real_time_tipsod](processing/get_real_time_tipsod.md)

[interpolate_in_time](processing/interpolate_in_time.md)

## Saving standards

[DailyLEORBStrategy](saving_strategies/daily_leo_rb.md)

[DailyWaveStrategy](saving_strategies/daily_wave.md)

[DensityNetCDFStrategy](saving_strategies/density_netcdf.md)

[GFZStrategy](saving_strategies/gfz.md)

[MonthlyRBStrategy](saving_strategies/monthly.md)

[SingleFileStrategy](saving_strategies/single_file.md)

## Data Standards

[GFZStandard](data_standards/gfz.md)

[PRBEMStandard](data_standards/prbem.md)

## Datasets

[GFZDataSet](dataset/gfz.md)

[PRBEMDataSet](dataset/prbem.md)

## Recipes

[Arase](recipes/arase.md)

[DMSP](recipes/dmsp.md)

[ESA (NGRM)](recipes/esa.md)

[GOES](recipes/goes.md)

[POES](recipes/poes.md)

[ProbaV](recipes/probav.md)

[RBSP](recipes/rbsp.md)
