# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import el_paso as ep

class MonthlyLEORBStrategy(ep.typing.MonthlyRBStrategy):

    def _get_output_file_entries(self) -> list[ep.typing.InternalName]:
        """Return the standard variable list plus user-defined custom variables."""
        return [
            "FEDU",
            "Epoch",
            "Alpha_Eq",
            "Alpha_Eq_range",
            "Energy_FEDU",
            "Alpha",
            "Alpha_range",
            "B_Calc",
            "B_Eq",
            "InvK",
            "InvMu",
            "Position",
            "PSD",
            "R_Eq",
            "MLT",
            "L_m",
            "L_star",
            "Alpha_LC",
            "Alpha_LC_Eq",
            "Position_geo_alt",
            "Position_geo_lat",
            "Position_geo_lon",
        ]
