from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from astropy import units as u
from get_arase_orbit_variables import get_arase_orbit_level_3_variables

import el_paso as ep


def process_mepe_level_3(start_time:datetime,
                         end_time:datetime,
                         irbem_lib_path:str|Path,
                         mag_field:Literal["T89", "TS04"],
                         raw_data_path:str|Path = ".",
                         processed_data_path:str|Path = ".",
                         num_cores:int=4):

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel(logging.INFO)

    irbem_lib_path = Path(irbem_lib_path)
    raw_data_path = Path(raw_data_path)
    processed_data_path = Path(processed_data_path)

    orb_variables = get_arase_orbit_level_3_variables(start_time, end_time, mag_field, raw_data_path=raw_data_path)

    file_name_stem = "erg_mepe_l3_pa_YYYYMMDD_.{6}.cdf"
    url = "https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/mepe/l3/pa/YYYY/MM/"

    ep.download(start_time, end_time,
                save_path=raw_data_path,
                download_url=url,
                file_name_stem=file_name_stem,
                file_cadence="daily",
                method="request",
                skip_existing=True)

    extraction_infos = [
        ep.ExtractionInfo(
            result_key="Epoch",
            name_or_column="epoch",
            unit=ep.units.tt2000,
        ),
        ep.ExtractionInfo(
            result_key="Energy",
            name_or_column="FEDU_Energy",
            unit=u.keV,
            is_time_dependent=False,
        ),
        ep.ExtractionInfo(
            result_key="Pitch_angle",
            name_or_column="FEDU_Alpha",
            unit=u.deg,
            is_time_dependent=False,
        ),
        ep.ExtractionInfo(
            result_key="FEDU",
            name_or_column="FEDU",
            unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
        ),
    ]

    mepe_variables = ep.extract_variables_from_files(start_time, end_time, "daily",
                                                     data_path=raw_data_path, file_name_stem=file_name_stem,
                                                     extraction_infos=extraction_infos)

    mepe_variables["FEDU"].truncate(mepe_variables["Epoch"], start_time, end_time)
    mepe_variables["Epoch"].truncate(mepe_variables["Epoch"], start_time, end_time)

    # sort energies into ascending order
    idx_sorted = np.argsort(mepe_variables["Energy"].get_data())
    mepe_variables["Energy"].set_data(mepe_variables["Energy"].get_data()[idx_sorted[:-1]], "same")
    mepe_variables["FEDU"].set_data(mepe_variables["FEDU"].get_data()[:,idx_sorted[:-1],:], "same")

    time_bin_methods = {
        "xGEO": ep.TimeBinMethod.NanMean,
        "Energy": ep.TimeBinMethod.Repeat,
        "FEDU": ep.TimeBinMethod.NanMedian,
        "Pitch_angle": ep.TimeBinMethod.Repeat,
        "B_local": ep.TimeBinMethod.NanMedian,
        "B_eq": ep.TimeBinMethod.NanMedian,
        "Lm": ep.TimeBinMethod.NanMean,
        "Lstar": ep.TimeBinMethod.NanMean,
        "MLT": ep.TimeBinMethod.NanMean,
        "R0": ep.TimeBinMethod.NanMean,
    }

    binned_time_variable = ep.processing.bin_by_time(mepe_variables["Epoch"], variables=mepe_variables,
                                                    time_bin_method_dict=time_bin_methods,
                                                    time_binning_cadence=timedelta(minutes=5),
                                                    start_time=start_time,
                                                    end_time=end_time)

    binned_time_variable = ep.processing.bin_by_time(orb_variables["Epoch"], variables=orb_variables,
                                  time_bin_method_dict=time_bin_methods,
                                  time_binning_cadence=timedelta(minutes=5),
                                  start_time=start_time,
                                  end_time=end_time)

    ep.processing.fold_pitch_angles_and_flux(mepe_variables["FEDU"],
                                             mepe_variables["Pitch_angle"])

    mepe_variables["FEDU"].apply_thresholds_on_data(lower_threshold=1e-21)

    pa_local = mepe_variables["Pitch_angle"].get_data(u.radian)
    pa_eq = np.asin(np.sin(pa_local) * np.sqrt(orb_variables["B_eq"].get_data(u.nT) / orb_variables["B_local"].get_data(u.nT))[:, np.newaxis])
    mepe_variables["Pa_eq"] = ep.Variable(data=pa_eq, original_unit=u.radian)

    saving_strategy = ep.saving_strategies.DataOrgStrategy(processed_data_path, "ARASE", "arase", "mepe-l3", "T04s", ".mat")

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

    from matplotlib import pyplot as plt
    plt.plot(binned_time_variable.get_data(), orb_variables["Lm"].get_data()[:,-1])
    plt.savefig("test.png")

    ep.save(variables_to_save, saving_strategy, start_time, end_time, binned_time_variable)

    from matplotlib import pyplot as plt
    plt.plot(binned_time_variable.get_data(), orb_variables["R0"].get_data())
    plt.savefig("test.png")

if __name__ == "__main__":

    start_time = datetime(2017, 9, 6, tzinfo=timezone.utc)
    end_time = datetime(2017, 9, 12, tzinfo=timezone.utc)

    process_mepe_level_3(start_time, end_time, "../../IRBEM/libirbem.so", "TS04",
                         raw_data_path=".", processed_data_path=".", num_cores=16)