# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0
import logging

from astropy import units as u  # type: ignore[reportMissingTypeStubs]

import el_paso as ep
from el_paso.data_standard import ConsistencyCheck, DataStandard, VariableInfo
from el_paso.typing import PRBEMName

logger = logging.getLogger("__name__")


class PRBEMStandard(DataStandard[PRBEMName]):
    """A data standard of the Panel for Radiation Belt Environment Modeling (PRBEM).

    This class defines and applies a specific set of data standards for variables
    defined by the [PRBEM](https://prbem.github.io/documents/Standard_File_Format.pdf).
    It standardizes variables by converting them to canonical units and performing
    consistency checks on their dimensions and shapes, ensuring they conform to the
    expected format for each standard name.
    """

    def __init__(self) -> None:
        """Initializes the PRBEMStandard with a ConsistencyCheck object."""
        self.consistency_check = ConsistencyCheck()

        self.variable_infos: dict[str, VariableInfo[PRBEMName]] = {
            "Epoch": VariableInfo[PRBEMName]("Epoch", "Posix Time", ep.units.posixtime, dependencies=["Epoch"]),
            "FEDU": VariableInfo[PRBEMName](
                "FEDU",
                "Processed unidirectional differential electron flux",
                (u.cm**2 * u.s * u.sr * u.keV) ** (-1),
                dependencies=["Epoch", "Energy_FEDU", "Pitch_angle"],
            ),
            "Alpha": VariableInfo[PRBEMName](
                "Alpha", "Local pitch angle the instrument is looking at", u.deg, dependencies=["Alpha"]
            ),
            "Alpha_Eq": VariableInfo[PRBEMName](
                "Alpha_Eq",
                "Computed equatorial pitch angle the instrument is looking from Alpha, B_Calc and B_Eq",
                u.deg,
                dependencies=["Alpha"],
            ),
            "Energy_FEDU": VariableInfo[PRBEMName](
                "Energy_FEDU",
                "Central energy of unidirectional differential electron flux",
                u.MeV,
                dependencies=["Energy_FEDU"],
            ),
            "Position": VariableInfo[PRBEMName](
                "Position",
                "Spacecraft position in geographic cartesian coordinates",
                u.km,
                dependencies=["Epoch", "Position_components"],
            ),
            "B_Calc": VariableInfo[PRBEMName](
                "B_Calc",
                "Calculated magnetic field strength at the spacecraft position",
                u.nT,
                dependencies=["Epoch"],
            ),
            "B_Eq": VariableInfo[PRBEMName](
                "B_Eq",
                "Calculated magnetic field strength at magnetic equator",
                u.nT,
                dependencies=["Epoch"],
            ),
            "L_m": VariableInfo[PRBEMName](
                "L_m",
                "Calculated L McIlwain's L parameter",
                u.dimensionless_unscaled,
                dependencies=["Epoch", "Alpha"],
            ),
            "L_star": VariableInfo[PRBEMName](
                "L_star",
                "Calculated Roederer's L* parameter",
                u.dimensionless_unscaled,
                dependencies=["Epoch", "Alpha"],
            ),
        }
