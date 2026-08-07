# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from astropy import units as u

import el_paso as ep
from el_paso.data_standard import ConsistencyCheck, DataStandard, VariableInfo
from el_paso.typing import GFZVarNames, InternalName


class GFZStandard(DataStandard[GFZVarNames]):
    """A data standard used historically at the GFZ German Research Centre for Geosciences.

    This standard defines rules for a set of canonical variable names by converting them
    to correct units and checking their array dimensions for consistency. It is tailored
    for compatibility with historical GFZ datasets and internal workflows.
    """

    def __init__(self) -> None:
        """Initializes the GFZStandard with a ConsistencyCheck object."""
        self.consistency_check = ConsistencyCheck()

        self.variable_infos: dict[InternalName, VariableInfo] = {
            "Epoch": VariableInfo[GFZVarNames]("time", "Time in MATLAB datenum format.", ep.units.datenum, ["Epoch"]),
            "Position": VariableInfo[GFZVarNames](
                "xGEO", "Position in geographic cartesian coordinates.", ep.units.RE, ["Epoch", "Position_components"]
            ),
            "Position_geo_alt": VariableInfo[GFZVarNames](
                "geo_alt", "Altitude in geographic cartesian coordinates.", u.km, ["Epoch"]
            ),
            "Position_geo_lat": VariableInfo[GFZVarNames](
                "geo_lat", "Latitude in geographic cartesian coordinates.", u.deg, ["Epoch"]
            ),
            "Position_geo_lon": VariableInfo[GFZVarNames](
                "geo_lon", "Longitude in geographic cartesian coordinates.", u.deg, ["Epoch"]
            ),
            "Energy_FEDU": VariableInfo[GFZVarNames](
                "energy_channels", "Central energy of measured differential flux.", u.MeV, ["Epoch", "Energy_FEDU"]
            ),
            "Energy_FEIU": VariableInfo[GFZVarNames](
                "energy_FEIU", "Central energy of measured integral flux.", u.MeV, ["Epoch", "Energy_FEIU"]
            ),
            "Energy_FEDO": VariableInfo[GFZVarNames](
                "energy_FEDO", "Central energy of measured omnidirecitonal flux.", u.MeV, ["Epoch", "Energy_FEDO"]
            ),
            "Energy_FPDU": VariableInfo[GFZVarNames](
                "energy_FPDU", "Central energy of measured proton differential flux.", u.MeV, ["Epoch", "Energy_FPDU"]
            ),
            "FEDU": VariableInfo[GFZVarNames](
                "Flux",
                "Electron differential unidirectional flux.",
                (u.cm**2 * u.s * u.sr * u.keV) ** (-1),
                ["Epoch", "Energy_FEDU", "Alpha"],
            ),
            "FEIU": VariableInfo[GFZVarNames](
                "FEIU",
                "Electron integral unidirectional flux.",
                (u.cm**2 * u.s * u.sr) ** (-1),
                ["Epoch", "Energy_FEIU", "Alpha"],
            ),
            "FEDO": VariableInfo[GFZVarNames](
                "FEDO",
                "Electron differential omnidirectional flux.",
                (u.cm**2 * u.s * u.keV) ** (-1),
                ["Epoch", "Energy_FEDO", "Alpha_range"],
            ),
            "FPDU": VariableInfo[GFZVarNames](
                "FPDU",
                "Proton differential unidirectional flux.",
                (u.cm**2 * u.s * u.sr * u.keV) ** (-1),
                ["Epoch", "Energy_FPDU", "Alpha"],
            ),
            "Alpha": VariableInfo[GFZVarNames](
                "alpha_local", "Local pitch angles of the particles.", u.radian, ["Epoch", "Alpha"]
            ),
            "Alpha_range": VariableInfo[GFZVarNames](
                "alpha_local_range",
                "Local pitch angle ranges of the particles.",
                u.radian,
                ["Epoch", "Alpha_range", "min_max"],
            ),
            "Alpha_Eq": VariableInfo[GFZVarNames](
                "alpha_eq_model", "Calculated equatorial pitch angles of the particles.", u.radian, ["Epoch", "Alpha"]
            ),
            "Alpha_Eq_range": VariableInfo[GFZVarNames](
                "alpha_eq_range",
                "Equatorial pitch angle ranges of the particles.",
                u.radian,
                ["Epoch", "Alpha_range", "min_max"],
            ),
            "Alpha_LC": VariableInfo[GFZVarNames](
                "alpha_lc", "Local loss cone size at the satellite location.", u.radian, ["Epoch"]
            ),
            "Alpha_LC_Eq": VariableInfo[GFZVarNames](
                "alpha_lc_eq",
                "Local loss cone size at the satellite location mapped to the equator.",
                u.radian,
                ["Epoch"],
            ),
            "PSD": VariableInfo[GFZVarNames](
                "PSD",
                "Calculated phase space density of particles.",
                (u.m * u.kg * u.m / u.s) ** (-3),
                ["Epoch", ("Energy_FEDU", "Energy_FPDU"), "Alpha"],
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
            "B_total_obs": VariableInfo[GFZVarNames](
                "B_sat", "Observered magnetic field at the satellite location.", u.nT, ["Epoch"]
            ),
            "R_Eq": VariableInfo[GFZVarNames](
                "R0", "Radial distance of the satellite location mapped to the equator.", ep.units.RE, ["Epoch"]
            ),
            "InvMu": VariableInfo[GFZVarNames](
                "InvMu", "Calculated first adiabatic invariant.", u.MeV / u.G, ["Epoch", ("Energy_FEDU", "Energy_FPDU"), "Alpha"]
            ),
            "InvK": VariableInfo[GFZVarNames](
                "InvK", "Calculated modified second adiabatic invariant.", ep.units.RE * u.G**0.5, ["Epoch", "Alpha"]
            ),
            "Wave_frequency": VariableInfo[GFZVarNames](
                "freq", "Frequency of the power spectral density.", u.Hz, ["Wave_frequency"]
            ),
            "Wave_ellipticity": VariableInfo[GFZVarNames](
                "ellipticity",
                "Frequency of the power spectral density.",
                u.dimensionless_unscaled,
                ["Epoch", "Wave_frequency"],
            ),
            "Wave_planarity": VariableInfo[GFZVarNames](
                "planarity",
                "Frequency of the power spectral density.",
                u.dimensionless_unscaled,
                ["Epoch", "Wave_frequency"],
            ),
            "Wave_frequency_bandwidth": VariableInfo[GFZVarNames](
                "freq_bw", "Frequency of the power spectral density.", u.Hz, ["Wave_frequency"]
            ),
            "Wave_normal_angle": VariableInfo[GFZVarNames](
                "wave_wna", "Frequency of the power spectral density.", u.degree, ["Epoch", "Wave_frequency"]
            ),
            "MLat": VariableInfo[GFZVarNames]("MLat", "Frequency of the power spectral density.", u.degree, ["Epoch"]),
            "Magnetic_Power_Spectral_Density": VariableInfo[GFZVarNames](
                "BB", "Frequency of the power spectral density.", u.dimensionless_unscaled, ["Epoch", "Wave_frequency"]
            ),
        }
