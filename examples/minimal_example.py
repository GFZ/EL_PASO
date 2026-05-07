# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from astropy import units as u

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.getLogger().setLevel(logging.INFO)

import el_paso as ep

raw_data_path = Path()

url = "https://rbsp-ect.newmexicoconsortium.org/data_pub/rbspa/ECT/level3/YYYY/"
file_name_stem = "rbspa_ect-elec-L3_YYYYMMDD_.{6}.cdf"

start_time = datetime(2017, 4, 20, tzinfo=timezone.utc)
end_time = datetime(2017, 4, 21, tzinfo=timezone.utc)

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
    ep.processing.ExtractionInfo(
        result_key="Epoch",
        name_or_column="Epoch",
        unit=ep.units.cdf_epoch,
    ),
    ep.processing.ExtractionInfo(
        result_key="Energy",
        name_or_column="FEDU_Energy",
        unit=u.keV,
        is_time_dependent=False,
    ),
    ep.processing.ExtractionInfo(
        result_key="Pitch_angle",
        name_or_column="FEDU_Alpha",
        unit=u.deg,
        is_time_dependent=False,
    ),
    ep.processing.ExtractionInfo(
        result_key="FEDU",
        name_or_column="FEDU",
        unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
    ),
    ep.processing.ExtractionInfo(
        result_key="xGEO",
        name_or_column="Position",
        unit=u.km,
    ),
]

variables = ep.processing.extract_variables_from_files(
    start_time=start_time,
    end_time=end_time,
    file_cadence="daily",
    data_path=raw_data_path,
    file_name_stem=file_name_stem,
    extraction_infos=extraction_infos,
)

time_bin_methods = {
    "xGEO": ep.TimeBinMethod.NanMean,
    "Energy": ep.TimeBinMethod.Repeat,
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
irbem_lib_path = Path(__file__).parent / ".." / "IRBEM" / "libirbem.so"
mag_field = "T89"  # other options include: "TS04", "T96", "OP77", ...

variables_to_compute: ep.processing.VariableRequest = [
    ("B_eq", mag_field),
    ("MLT", mag_field),
    ("PA_eq", mag_field),
    ("invMu", mag_field),
]

magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
    time_var=binned_time_variable,
    xgeo_var=variables["xGEO"],
    variables_to_compute=variables_to_compute,
    irbem_lib_path=str(irbem_lib_path),
    irbem_options=irbem_options,
    num_cores=8,
    pa_local_var=variables["Pitch_angle"],
    energy_var=variables["Energy"],
    particle_species="electron",
)

variables_to_save = {
    "time": binned_time_variable,
    "flux/FEDU": variables["FEDU"],
    "flux/energy": variables["Energy"],
    "flux/alpha_local": variables["Pitch_angle"],
    "flux/alpha_eq": magnetic_field_variables["PA_eq_" + mag_field],
    f"position/{mag_field}/MLT": magnetic_field_variables["MLT_" + mag_field],
    f"mag_field/{mag_field}/B_eq": magnetic_field_variables["B_eq_" + mag_field],
    "position/xGEO": variables["xGEO"],
}

data_standard = ep.data_standards.PRBEMStandard()

saving_strategy = ep.saving_strategies.MonthlyNetCDFStrategy(
    base_data_path=".", file_name_stem="rbspa_ect_combined", mag_field=mag_field, data_standard=data_standard
)

ep.save(variables_to_save, saving_strategy, start_time, end_time, binned_time_variable)
