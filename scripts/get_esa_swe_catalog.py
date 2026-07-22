# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences  # noqa: INP001
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0
import os

import requests
from dotenv import load_dotenv
from tabulate import tabulate

load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.env")))  # noqa: PTH100, PTH118, PTH120

client_id = os.environ.get("CLIENT_ID")
client_secret = os.environ.get("CLIENT_SECRET")

if client_id is None:
    msg = "Client ID not found!"
    raise ValueError(msg)

if client_secret is None:
    msg = "Client secret not found!"
    raise ValueError(msg)

response = requests.post(
    "https://sso.s2p.esa.int/realms/swe/protocol/openid-connect/token",
    data={
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials",
        "scope": "swe_hapiserver",
    },
    timeout=5,
)

token_data = response.json()

access_token = token_data["access_token"]

catalog_response = requests.get(
    "https://swe.ssa.esa.int/hapi/catalog",
    headers={"Authorization": f"Bearer {access_token}"},
    timeout=5,
)

catalog = catalog_response.json()
catalog_to_show = catalog["catalog"]

print(  # noqa: T201
    tabulate(
        catalog_to_show,
        headers={"id": "ID", "title": "Title"},
    )
)
