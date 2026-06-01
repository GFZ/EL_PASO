# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0


import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from astropy import units as u

import el_paso as ep

ep.setup_logging()

raw_data_path = Path()

url = "https://spdf.gsfc.nasa.gov/pub/data/rbsp/rbspa/l3/ect/hope/pitchangle/rel04/YYYY/"
file_name_stem = "rbspa_rel04_ect-hope-pa-l3_YYYYMMDD_.{6}.cdf"

start_time = datetime(2017, 7, 14, tzinfo=timezone.utc)
end_time = datetime(2017, 7, 14, 23, 59, 59, tzinfo=timezone.utc)

ep.download(
    start_time,
    end_time,
    save_path=raw_data_path,
    download_url=url,
    file_name_stem=file_name_stem,
    file_cadence="daily",
    method="request",
    skip_existing=True,
)

extraction_infos = [
    ep.ExtractionInfo(
        result_key="Epoch",
        name_or_column="Epoch_Ele",
        unit=ep.units.cdf_epoch,
    ),
    ep.ExtractionInfo(
        result_key="Energy",
        name_or_column="HOPE_ENERGY_Ele",
        unit=u.keV,
        is_time_dependent=False,
    ),
    ep.ExtractionInfo(
        result_key="Pitch_angle",
        name_or_column="PITCH_ANGLE",
        unit=u.deg,
        is_time_dependent=False,
    ),
    ep.ExtractionInfo(
        result_key="FEDU",
        name_or_column="FEDU",
        unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
    ),
    ep.ExtractionInfo(
        result_key="xGEO",
        name_or_column="Position_Ele",
        unit=u.km,
    ),
]

variables = ep.extract_variables_from_files(
    start_time=start_time,
    end_time=end_time,
    file_cadence="daily",
    data_path=raw_data_path,
    file_name_stem=file_name_stem,
    extraction_infos=extraction_infos,
)

time_bin_methods = {
    "xGEO": ep.TimeBinMethod.NanMean,
    "Energy": ep.TimeBinMethod.NanMean,
    "FEDU": ep.TimeBinMethod.NanMedian,
    "FEDU_Quality": ep.TimeBinMethod.NanMax,
    "FEDO": ep.TimeBinMethod.NanMedian,
    "Pitch_angle": ep.TimeBinMethod.Repeat,
}

binned_time_variable = ep.processing.bin_by_time(
    time_variable=variables["Epoch"],
    variables=variables,
    time_bin_method_dict=time_bin_methods,
    time_binning_cadence=timedelta(minutes=5),
    start_time=start_time,
    end_time=end_time,
)

variables["FEDU"].transpose_data([0, 2, 1])  # making it having dimensions (time, energy, pitch angle)
variables["FEDU"].apply_thresholds_on_data(lower_threshold=0)  # set negative values to NaN
ep.processing.fold_pitch_angles_and_flux(
    variables["FEDU"],  # fold around 90 degrees
    variables["Pitch_angle"],
)


irbem_options = [1, 1, 4, 4, 0]
mag_field = "T89"  # other options include: "TS04", "T96", "OP77", ...

variables_to_compute: ep.processing.VariableRequest = [
    ("B_Eq", mag_field),
    ("MLT", mag_field),
    ("Alpha_Eq", mag_field),
    ("InvMu", mag_field),
]

magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
    time_var=binned_time_variable,
    xgeo_var=variables["xGEO"],
    variables_to_compute=variables_to_compute,
    irbem_options=irbem_options,
    num_cores=8,
    pa_local_var=variables["Pitch_angle"],
    energy_var=variables["Energy"],
    particle_species="electron",
)

variables_to_save: dict[ep.typing.InternalName, ep.Variable] = {
    "Epoch": binned_time_variable,
    "FEDU": variables["FEDU"],
    "Energy_FEDU": variables["Energy"],
    "Alpha": variables["Pitch_angle"],
    "Alpha_Eq": magnetic_field_variables["Alpha_Eq_" + mag_field],
    "MLT": magnetic_field_variables["MLT_" + mag_field],
    "B_Eq": magnetic_field_variables["B_Eq_" + mag_field],
    "Position": variables["xGEO"],
}


data_standard = ep.data_standards.PRBEMStandard()

strategy_mrb = ep.saving_strategies.MonthlyRBStrategy(
    base_data_path=".",
    mission="RBSP",
    satellite="RBSP_ECT",
    instrument="ECT",
    mag_field="T89",
    file_format=".nc",
    data_standard=ep.data_standards.PRBEMStandard(),
)


strategy_gfz = ep.saving_strategies.GFZStrategy(
    base_data_path=".",
    mission="RBSP",
    satellite="RBSP_ECT",
    instrument="ECT",
    mag_field="T89",
    data_standard=ep.data_standards.PRBEMStandard(),
)


for strategy in (strategy_mrb, strategy_gfz):
    ep.save(variables_to_save, strategy, start_time, end_time, time_var=binned_time_variable, append=True)
