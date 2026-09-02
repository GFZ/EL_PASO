# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""Command line interface for EL-PASO.

:mod:`el_paso.cli.recipe_cli` builds a Typer command for any recipe from that
recipe's own signature; :mod:`el_paso.cli.app` assembles all of them into the
``el-paso`` command tree.
"""

from el_paso.cli.recipe_cli import build_recipe_command, run_recipe_cli

__all__ = [
    "build_recipe_command",
    "run_recipe_cli",
]
