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

import el_paso as ep
from el_paso.processing.magnetic_field_utils.irbem import Coords

sys.path.append(str(Path(__file__).parent / ".." / ".."))
from examples.Arase.get_arase_orbit_variables import (
    get_arase_orbit_level_2_variables,
    get_arase_orbit_level_3_variables,
)


def process_mepe_level_3(  # noqa: PLR0915
    start_time: datetime,
    end_time: datetime,
    irbem_lib_path: str | Path,
    mag_field: Literal["T89", "TS04", "OP77Q"],
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    num_cores: int = 4,
    cadence: timedelta = timedelta(minutes=5),
    save_strategy: Literal["DataOrg", "h5", "netcdf"] = "DataOrg",
    *,
    use_level_3_orbit_data: bool = True,
) -> None:
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel(logging.INFO)

    irbem_lib_path = Path(irbem_lib_path)
    raw_data_path = Path(raw_data_path)
    processed_data_path = Path(processed_data_path)

    if use_level_3_orbit_data:
        orb_variables = get_arase_orbit_level_3_variables(start_time, end_time, mag_field, raw_data_path=raw_data_path)
    else:
        orb_variables = get_arase_orbit_level_2_variables(start_time, end_time, raw_data_path=raw_data_path)

    file_name_stem = "erg_mepe_l3_pa_YYYYMMDD_.{6}.cdf"
    url = "https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/mepe/l3/pa/YYYY/MM/"

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
            name_or_column="epoch",
            unit=ep.units.tt2000,
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
    ]

    mepe_variables = ep.processing.extract_variables_from_files(
        start_time,
        end_time,
        "daily",
        data_path=raw_data_path,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )

    mepe_variables["FEDU"].truncate(mepe_variables["Epoch"], start_time, end_time)
    mepe_variables["Epoch"].truncate(mepe_variables["Epoch"], start_time, end_time)

    # sort energies into ascending order
    idx_sorted = np.argsort(mepe_variables["Energy"].get_data())
    mepe_variables["Energy"].set_data(mepe_variables["Energy"].get_data()[idx_sorted[:-1]], "same")
    mepe_variables["FEDU"].set_data(mepe_variables["FEDU"].get_data()[:, idx_sorted[:-1], :], "same")

    time_bin_methods = {
        "xGEO": ep.TimeBinMethod.NanMean,
        "Energy": ep.TimeBinMethod.Repeat,
        "FEDU": ep.TimeBinMethod.NanMedian,
        "Pitch_angle": ep.TimeBinMethod.Repeat,
    }

    binned_time_variable = ep.processing.bin_by_time(
        mepe_variables["Epoch"],
        variables=mepe_variables,
        time_bin_method_dict=time_bin_methods,
        time_binning_cadence=cadence,
        start_time=start_time,
        end_time=end_time,
    )

    ep.processing.fold_pitch_angles_and_flux(mepe_variables["FEDU"], mepe_variables["Pitch_angle"])

    mepe_variables["FEDU"].apply_thresholds_on_data(lower_threshold=1e-21)

    if use_level_3_orbit_data:
        time_bin_methods = {
            "B_local": ep.TimeBinMethod.NanMedian,
            "B_eq": ep.TimeBinMethod.NanMedian,
            "Lm": ep.TimeBinMethod.NanMean,
            "Lstar": ep.TimeBinMethod.NanMean,
            "MLT": ep.TimeBinMethod.NanMean,
            "R0": ep.TimeBinMethod.NanMean,
        }

        binned_time_variable = ep.processing.bin_by_time(
            orb_variables["Epoch"],
            variables=orb_variables,
            time_bin_method_dict=time_bin_methods,
            time_binning_cadence=cadence,
            start_time=start_time,
            end_time=end_time,
        )

        pa_local = mepe_variables["Pitch_angle"].get_data(u.radian)
        pa_eq = np.asin(
            np.sin(pa_local)
            * np.sqrt(orb_variables["B_eq"].get_data(u.nT) / orb_variables["B_local"].get_data(u.nT))[:, np.newaxis]
        )
        mepe_variables["Pa_eq"] = ep.Variable(data=pa_eq, original_unit=u.radian)

    else:
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

        datetimes = [
            datetime.fromtimestamp(t, tz=timezone.utc) for t in binned_time_variable.get_data(ep.units.posixtime)
        ]

        geo_data = Coords(lib_path=irbem_lib_path).transform(
            time=datetimes,
            pos=orb_variables["pos_sm"].get_data().astype(np.float64),
            sysaxes_in=ep.IRBEM_SYSAXIS_SM,
            sysaxes_out=ep.IRBEM_SYSAXIS_GEO,
        )
        pos_geo_var = ep.Variable(data=geo_data, original_unit=ep.units.RE)

        irbem_options = [1, 1, 4, 4, 0]

        variables_to_compute: ep.processing.VariableRequest = [
            ("B_local", mag_field),
            ("MLT", mag_field),
            ("B_eq", mag_field),
            ("R_eq", mag_field),
            ("PA_eq", mag_field),
            ("Lm", mag_field),
        ]

        magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
            time_var=binned_time_variable,
            xgeo_var=pos_geo_var,
            variables_to_compute=variables_to_compute,
            irbem_lib_path=str(irbem_lib_path),
            irbem_options=irbem_options,
            num_cores=num_cores,
            pa_local_var=mepe_variables["Pitch_angle"],
        )

        orb_variables["R0"] = magnetic_field_variables["R_eq_" + mag_field]
        orb_variables["MLT"] = magnetic_field_variables["MLT_" + mag_field]
        mepe_variables["Pa_eq"] = magnetic_field_variables["PA_eq_" + mag_field]
        orb_variables["Lm"] = magnetic_field_variables["Lm_" + mag_field]

    match mag_field:
        case "T89":
            mag_field_save = "T89"
        case "TS04":
            mag_field_save = "T04s"
        case "OP77Q":
            mag_field_save = "OP77"

    match save_strategy:
        case "DataOrg":
            saving_strategy = ep.saving_strategies.DataOrgStrategy(
                processed_data_path, "ARASE", "arase", "mepe-l3", mag_field_save, ".mat"
            )

            variables_to_save = {
                "time": binned_time_variable,
                "Flux": mepe_variables["FEDU"],
                "energy_channels": mepe_variables["Energy"],
                "alpha_local": mepe_variables["Pitch_angle"],
                "alpha_eq_model": mepe_variables["Pa_eq"],
                "R0": orb_variables["R0"],
                "MLT": orb_variables["MLT"],
                "Lm": orb_variables["Lm"],
            }

        case "h5":
            variables_to_save = {
                "time": binned_time_variable,
                "flux/FEDU": mepe_variables["FEDU"],
                "flux/energy": mepe_variables["Energy"],
                "flux/alpha_local": mepe_variables["Pitch_angle"],
                "flux/alpha_eq": mepe_variables["Pa_eq"],
                f"position/{mag_field}/R0": orb_variables["R0"],
                f"position/{mag_field}/MLT": orb_variables["MLT"],
                f"position/{mag_field}/Lm": orb_variables["Lm"],
            }

            saving_strategy = ep.saving_strategies.MonthlyH5Strategy(
                processed_data_path, "arase_mepe-l3", mag_field=mag_field
            )

        case "netcdf":
            variables_to_save = {
                "time": binned_time_variable,
                "flux/FEDU": mepe_variables["FEDU"],
                "flux/energy": mepe_variables["Energy"],
                "flux/alpha_local": mepe_variables["Pitch_angle"],
                "flux/alpha_eq": mepe_variables["Pa_eq"],
                f"position/{mag_field}/R0": orb_variables["R0"],
                f"position/{mag_field}/MLT": orb_variables["MLT"],
                f"position/{mag_field}/Lm": orb_variables["Lm"],
            }

            saving_strategy = ep.saving_strategies.MonthlyNetCDFStrategy(
                processed_data_path, "arase_mepe-l3", mag_field=mag_field
            )

    ep.save(variables_to_save, saving_strategy, start_time, end_time, binned_time_variable)


if __name__ == "__main__":
    start_time = datetime(2017, 7, 1, tzinfo=timezone.utc)
    end_time = datetime(2017, 9, 30, 23, 59, tzinfo=timezone.utc)

    with tempfile.TemporaryDirectory() as tmp_dir:
        process_mepe_level_3(
            start_time,
            end_time,
            "../../libirbem.so",
            "T89",
            raw_data_path=tmp_dir,
            processed_data_path=".",
            num_cores=32,
            save_strategy="DataOrg",
        )
