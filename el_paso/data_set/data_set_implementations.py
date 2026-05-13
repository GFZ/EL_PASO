import datetime as dt
from typing import Literal, type

import el_paso as ep
from el_paso.data_set.data_set import DataSet
from el_paso.saving_strategy import SavingStrategy

class DataOrgDataSet(DataSet):

    # standard names
    datetime : list[dt.datetime]
    time : NDArray[np.float64]
    energy_channels : NDArray[np.float64]
    alpha_local : NDArray[np.float64]
    alpha_eq_model : NDArray[np.float64]
    alpha_eq_real : NDArray[np.float64]
    InvMu : NDArray[np.float64]
    InvMu_real : NDArray[np.float64]
    InvK : NDArray[np.float64]
    InvV : NDArray[np.float64]
    Lstar : NDArray[np.float64]
    Flux : NDArray[np.float64]
    PSD : NDArray[np.float64]
    MLT : NDArray[np.float64]
    B_SM : NDArray[np.float64]
    B_total : NDArray[np.float64]
    B_sat : NDArray[np.float64]
    xGEO : NDArray[np.float64]
    P : NDArray[np.float64]
    R0 : NDArray[np.float64]
    density : NDArray[np.float64]

    def __init__(
        self,
        mission,
        satellite,
        instrument,
        base_path,
        start_time: dt.datetime | None = None,
        end_time: dt.datetime | None = None,
        preferred_extension: Literal["mat", "pickle", "nc"] = "nc",
        saving_strategy_type: type[SavingStrategy] = ep.saving_strategies.MonthlyFileStrategy,
        *,
        verbose: bool = True,
        enable_dict_loading: bool = False,
    ) -> None:

        saving_strategy = saving_strategy_type(base_path, mission, satellite, instrument, mag_field="OP77", data_standard=ep.data_standards.DataOrgStandard())

        super().__init__(saving_strategy, start_time, end_time, preferred_extension, verbose=verbose, enable_dict_loading=enable_dict_loading)