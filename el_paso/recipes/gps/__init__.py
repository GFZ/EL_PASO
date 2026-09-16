# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Parvathy Santhini
#
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

from typing import Literal

LANLSatellite = Literal[
    "ns41",
    "ns48",
    "ns53",
    "ns54",
    "ns55",
    "ns56",
    "ns57",
    "ns58",
    "ns59",
    "ns60",
    "ns61",
    "ns62",
    "ns63",
    "ns64",
    "ns65",
    "ns66",
    "ns67",
    "ns68",
    "ns69",
    "ns70",
    "ns71",
    "ns72",
    "ns73",
    "ns74",
    "ns75",
    "ns76",
    "ns77",
    "ns78",
    "ns79",
    "ns80",
    "ns81",
]

from el_paso.recipes.gps.process_gps import gps_cxd_strategy, process_gps_data

__all__ = ["LANLSatellite", "gps_cxd_strategy", "process_gps_data"]
