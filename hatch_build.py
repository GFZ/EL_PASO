# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""Hatchling build hook that compiles the IRBEM Fortran library before packaging."""

import sys
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _irbem_build


class IrbemBuildHook(BuildHookInterface):  # noqa: D101
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:  # noqa: ARG002
        if self.target_name != "wheel":
            return

        source_root = Path(self.root)
        _irbem_build.ensure_libirbem_built(source_root)

        so_path = source_root / "el_paso" / "libirbem.so"
        build_data["force_include"][str(so_path)] = "el_paso/libirbem.so"
