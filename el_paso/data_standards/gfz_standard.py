# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from astropy import units as u  # type: ignore[reportMissingTypeStubs]

import el_paso as ep
from el_paso.data_standard import ConsistencyCheck, DataStandard, VariableInfo
from el_paso.typing import GFZVarNames


class GFZStandard(DataStandard[GFZVarNames]):
    """A data standard used historically at the GFZ German Research Centre for Geosciences.

    This standard defines rules for a set of canonical variable names by converting them
    to correct units and checking their array dimensions for consistency. It is tailored
    for compatibility with historical GFZ datasets and internal workflows.
    """

    def __init__(self) -> None:
        """Initializes the GFZStandard with a ConsistencyCheck object."""
        self.consistency_check = ConsistencyCheck()

        self.variable_infos = {
            "Epoch": VariableInfo[GFZVarNames]("time", "Time in MATLAB datenum format.", ep.units.datenum, ["Epoch"]),
            "Position": VariableInfo[GFZVarNames](
                "xGEO", "Position in geographic cartesian coordinates.", ep.units.RE, ["Epoch", "Position_components"]
            ),
            "Energy_FEDU": VariableInfo[GFZVarNames](
                "energy_channels", "Central energy of measured flux.", u.MeV, ["Epoch", "Energy_FEDU"]
            ),
            "FEDU": VariableInfo[GFZVarNames](
                "Flux",
                "Flux of particles. Can be uni/omni-directional and differential/integral.",
                (u.cm**2 * u.s * u.sr * u.keV) ** (-1),
                ["Epoch", "Energy_FEDU", "Alpha"],
            ),
            "Alpha": VariableInfo[GFZVarNames](
                "alpha_local", "Local pitch angles of the particles.", u.radian, ["Epoch", "Alpha"]
            ),
            "Alpha_Eq": VariableInfo[GFZVarNames](
                "alpha_eq_model", "Calculated equatorial pitch angles of the particles.", u.radian, ["Epoch", "Alpha"]
            ),
            "PSD": VariableInfo[GFZVarNames](
                "PSD",
                "Calculated phase space density of particles.",
                (u.m * u.kg * u.m / u.s) ** (-3),
                ["Epoch", "Energy_FEDU", "Alpha"],
            ),
            "MLT": VariableInfo[GFZVarNames](
                "MLT", "Magnetic local time at the satellite location.", u.hour, ["Epoch"]
            ),
            "L_star": VariableInfo[GFZVarNames](
                "Lstar", "Calculated Lstar of the particles.", u.dimensionless_unscaled, ["Epoch", "Alpha"]
            ),
            "L_m": VariableInfo[GFZVarNames](
                "Lm", "Calculated Lm of the particles.", u.dimensionless_unscaled, ["Epoch", "Alpha"]
            ),
            "B_Eq": VariableInfo[GFZVarNames]("B_eq", "Calculated magnetic field at the equator.", u.nT, ["Epoch"]),
            "B_Calc": VariableInfo[GFZVarNames](
                "B_total", "Calculated magnetic field at the satellite location.", u.nT, ["Epoch"]
            ),
            "R_Eq": VariableInfo[GFZVarNames](
                "R0", "Radial distance of the satellite location mapped to the equator.", ep.units.RE, ["Epoch"]
            ),
            "InvMu": VariableInfo[GFZVarNames](
                "InvMu", "Calculated first adiabatic invariant.", u.MeV / u.G, ["Epoch", "Energy_FEDU", "Alpha"]
            ),
            "InvK": VariableInfo[GFZVarNames](
                "InvK", "Calculated modified second adiabatic invariant.", ep.units.RE * u.G**0.5, ["Epoch", "Alpha"]
            ),
        }
