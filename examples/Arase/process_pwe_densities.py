# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from astropy import units as u
from get_arase_orbit_variables import (
    get_arase_orbit_level_2_variables,
)

import el_paso as ep
from el_paso.processing.magnetic_field_utils.irbem import Coords


def process_pwe_density(
    start_time: datetime,
    end_time: datetime,
    irbem_lib_path: str | Path,
    mag_field: Literal["T89", "TS04", "OP77Q"],
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    num_cores: int = 4,
    cadence: timedelta = timedelta(minutes=5),
) -> None:
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel(logging.INFO)

    irbem_lib_path = Path(irbem_lib_path)
    raw_data_path = Path(raw_data_path)
    processed_data_path = Path(processed_data_path)

    orb_variables = get_arase_orbit_level_2_variables(start_time, end_time, raw_data_path=raw_data_path)

    file_name_stem = "erg_pwe_hfa_l3_1min_YYYYMMDD_.{6}.cdf"
    url = "https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/pwe/hfa/l3/YYYY/MM/"

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
            name_or_column="Epoch",
            unit=ep.units.tt2000,
        ),
        ep.ExtractionInfo(
            result_key="Density",
            name_or_column="ne_mgf",
            unit=u.cm ** (-3),
        ),
    ]

    pwe_variables = ep.processing.extract_variables_from_files(
        start_time,
        end_time,
        "daily",
        data_path=raw_data_path,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )

    time_bin_methods = {
        "Density": ep.TimeBinMethod.NanMedian,
    }

    binned_time_variable = ep.processing.bin_by_time(
        pwe_variables["Epoch"],
        variables=pwe_variables,
        time_bin_method_dict=time_bin_methods,
        time_binning_cadence=cadence,
        start_time=start_time,
        end_time=end_time,
    )

    pwe_variables["Density"].apply_thresholds_on_data(lower_threshold=1e-21)

    time_bin_methods = {
        "pos_sm": ep.TimeBinMethod.NanMean,
    }

    binned_time_variable = ep.processing.bin_by_time(
        orb_variables["Epoch"],
        variables=orb_variables,
        time_bin_method_dict=time_bin_methods,
        time_binning_cadence=cadence,
        start_time=start_time,
        end_time=end_time,
    )

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in binned_time_variable.get_data(ep.units.posixtime)]

    geo_data = Coords(lib_path=irbem_lib_path).transform(
        time=datetimes,
        pos=orb_variables["pos_sm"].get_data().astype(np.float64),
        sysaxes_in=ep.IRBEM_SYSAXIS_SM,
        sysaxes_out=ep.IRBEM_SYSAXIS_GEO,
    )
    pos_geo_var = ep.Variable(data=geo_data, original_unit=ep.units.RE)

    irbem_options = [1, 1, 4, 4, 0]

    variables_to_compute: ep.processing.VariableRequest = [
        ("MLT", mag_field),
        ("R_eq", mag_field),
        ("xGEO_eq", mag_field),
    ]

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=binned_time_variable,
        xgeo_var=pos_geo_var,
        variables_to_compute=variables_to_compute,
        irbem_lib_path=str(irbem_lib_path),
        irbem_options=irbem_options,
        num_cores=num_cores,
    )

    pwe_variables["Density_mapped_" + mag_field] = ep.processing.compute_equatorial_plasmaspheric_density(
        pwe_variables["Density"], pos_geo_var, magnetic_field_variables["xGEO_eq_" + mag_field], method="Denton_average"
    )

    saving_strategy = ep.saving_strategies.DensityNetCDFStrategy(
        base_data_path=processed_data_path,
        file_name_stem="arase_pwe_density",
        mag_field=mag_field,
    )

    variables_to_save = {
        "time": binned_time_variable,
        "density_local": pwe_variables["Density"],
        "density_eq": pwe_variables["Density_mapped_" + mag_field],
        "MLT": magnetic_field_variables["MLT_" + mag_field],
        "R_eq": magnetic_field_variables["R_eq_" + mag_field],
        "xGEO": pos_geo_var,
        "xGEO_eq": magnetic_field_variables["xGEO_eq_" + mag_field],
    }

    ep.save(variables_to_save, saving_strategy, start_time, end_time, binned_time_variable)


if __name__ == "__main__":
    start_time = datetime(2017, 9, 1, tzinfo=timezone.utc)
    end_time = datetime(2017, 9, 30, 23, 59, tzinfo=timezone.utc)

    with tempfile.TemporaryDirectory() as tmp_dir:
        process_pwe_density(
            start_time,
            end_time,
            "../../libirbem.so",
            "T89",
            raw_data_path=".",
            processed_data_path=".",
            num_cores=32,
        )
