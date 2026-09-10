# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences  # noqa: INP001
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""Guards against a `process_*` recipe entry point constructing a saving strategy inline.

Every `process_*` function under `el_paso/recipes/` should get its `SavingStrategy` from a
named `<...>_strategy(...)` function (e.g. `arase_xep_strategy(path, mag_field)`, defined in
the same file) rather than constructing an `ep.saving_strategies.*Strategy(...)` directly in
the entry point's body. That keeps each entry point readable (the mission/satellite/
instrument/data-standard literals live in one small, named, testable function) without
requiring a central module: new recipes are free to define their strategy function(s)
wherever makes sense in their own file.

Flags any call shaped like `<...>.saving_strategies.<ClassName>(...)`, and any
`from el_paso.saving_strategies import ...`, found directly inside a top-level `process_*`
function (not inside a helper function it calls). Run directly as a script (also wired up as
a pre-commit hook).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPES_DIR = REPO_ROOT / "el_paso" / "recipes"

ALLOWED_VIOLATIONS: dict[Path, set[int]] = {
    RECIPES_DIR / "rbsp" / "process_rbsp_efw_emfisis_density_combined.py": {
        196
    },  # https://github.com/GFZ/EL_PASO/issues/139
}


class Violation(NamedTuple):
    """A disallowed direct `ep.saving_strategies.*` construction inside a `process_*` entry point."""

    path: Path
    lineno: int
    detail: str


def _iter_recipe_files() -> list[Path]:
    return sorted(p for p in RECIPES_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def _is_saving_strategies_construction(func: ast.expr) -> bool:
    """True for `<...>.saving_strategies.<Name>(...)`, however the module was imported."""
    if not isinstance(func, ast.Attribute):
        return False
    value = func.value
    if isinstance(value, ast.Attribute):
        return value.attr == "saving_strategies"
    if isinstance(value, ast.Name):
        return value.id == "saving_strategies"
    return False


def _find_violations_in_function(path: Path, func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[Violation]:
    violations: list[Violation] = []

    for node in ast.walk(func):
        # Don't descend into a nested function/lambda's own body -- a strategy built by a
        # local helper the entry point calls is exactly the pattern this guard wants.
        if node is not func and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Call) and _is_saving_strategies_construction(node.func):
            violations.append(Violation(path, node.lineno, ast.unparse(node.func) + "(...)"))
        elif isinstance(node, ast.ImportFrom) and node.module == "el_paso.saving_strategies":
            names = ", ".join(alias.name for alias in node.names)
            violations.append(Violation(path, node.lineno, f"from el_paso.saving_strategies import {names}"))

    return violations


def _find_violations_in_file(path: Path) -> list[Violation]:
    tree = ast.parse(path.read_text(), filename=str(path))
    violations: list[Violation] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("process_"):
            violations.extend(_find_violations_in_function(path, node))

    return violations


def find_violations() -> list[Violation]:
    """Return every disallowed direct saving-strategy construction inside a `process_*` entry point."""
    violations: list[Violation] = []
    for path in _iter_recipe_files():
        allowed_lines = ALLOWED_VIOLATIONS.get(path, set())
        violations.extend(v for v in _find_violations_in_file(path) if v.lineno not in allowed_lines)
    return violations


def main() -> int:
    """Print every violation found and return a nonzero exit code if there were any."""
    violations = find_violations()

    if not violations:
        return 0

    print(  # noqa: T201
        "A process_* recipe entry point builds a saving strategy inline instead of through a\n"
        "named <...>_strategy(...) function defined in the same file. Violations:\n"
    )
    for violation in violations:
        rel = violation.path.relative_to(REPO_ROOT)
        print(f"  {rel}:{violation.lineno}: {violation.detail}")  # noqa: T201

    return 1


if __name__ == "__main__":
    sys.exit(main())
