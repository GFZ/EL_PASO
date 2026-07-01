# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import el_paso as ep
import el_paso.processing.magnetic_field_utils as mag_utils
from el_paso.typing import MagneticFieldLiteral
from el_paso.utils import make_dict_hashable


def compute_LCDS(
    time_var: ep.Variable,
    pa_eq_var: ep.Variable,
    mag_field: MagneticFieldLiteral,
    irbem_options: mag_utils.IrbemOptions,
    indices_solar_wind: dict[str, ep.Variable] | None = None,
    search_params: mag_utils.LCDSSearchParams | None = None,
    *,
    irbem_lib_path: str | Path = Path(ep.__file__).parent / "libirbem.so",
    num_cores: int = 12,
) -> tuple[ep.Variable, ep.Variable]:

    indices_solar_wind_hashable = make_dict_hashable(indices_solar_wind)

    maginput = mag_utils.construct_maginput(time_var, mag_utils.MagneticField(mag_field), indices_solar_wind_hashable)

    irbem_input = mag_utils.IrbemInput(
        magnetic_field=mag_utils.MagneticField(mag_field),
        maginput=maginput,
        irbem_options=irbem_options,
        num_cores=num_cores,
        irbem_lib_path=irbem_lib_path,
    )

    return mag_utils.get_LCDS(time_var, pa_eq_var, irbem_input, search_params)

