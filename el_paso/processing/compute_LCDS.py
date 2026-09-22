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
    """Computes the last closed drift shell (LCDS) and its second adiabatic invariant K.

    Assembles the magnetic field model inputs for the requested model from the given (or
    automatically loaded) geomagnetic indices and solar wind parameters, and then runs the
    LCDS search for every time step and equatorial pitch angle.

    Args:
        time_var (ep.Variable): The variable containing the timestamps.
        pa_eq_var (ep.Variable): The variable containing the equatorial pitch angles, one row
                                 per time step.
        mag_field (MagneticFieldLiteral): The magnetic field model to use (e.g. "T89", "TS04").
        irbem_options (mag_utils.IrbemOptions): The IRBEM-LIB options of the calculation.
        indices_solar_wind (dict[str, ep.Variable] | None): Geomagnetic indices and solar wind
            parameters used to build the model inputs. Defaults to None, in which case they are
            loaded automatically.
        search_params (mag_utils.LCDSSearchParams | None): The settings of the radial search.
            Defaults to None, which uses the settings of Kellerman's LCDS2 routine.
        irbem_lib_path (str | Path): The file path to the compiled IRBEM library.
        num_cores (int): The number of CPU cores used to parallelize over time steps.

    Returns:
        tuple[ep.Variable, ep.Variable]: The LCDS L* and the corresponding invariant K, both of
        shape (number of time steps, number of pitch angles).
    """
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
