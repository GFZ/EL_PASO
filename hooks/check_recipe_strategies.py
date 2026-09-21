# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences  # noqa: INP001
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""Guards recipe files against a few conventions drifting out of sync.

Three independent checks, all against `el_paso/recipes/`:

1. Inline strategy construction. Every `process_*` recipe entry point should get its
   `SavingStrategy` from a named `<...>_strategy(...)` function (e.g.
   `arase_xep_strategy(path, mag_field)`, defined in the same file) rather than constructing
   an `ep.saving_strategies.*Strategy(...)` directly in the entry point's body. That keeps
   each entry point readable (the mission/satellite/instrument/data-standard literals live in
   one small, named, testable function) without requiring a central module: new recipes are
   free to define their strategy function(s) wherever makes sense in their own file.

2. `satellite` typed as `str`. A `satellite` parameter anywhere in a recipe file (the
   `process_*` entry point or a helper it calls, such as its `<...>_strategy` factory) should
   be typed as a `Literal` of the mission's valid satellite names, not a bare `str`. Typing it
   `str` silently accepts any spelling and gives no autocomplete; see
   https://github.com/GFZ/EL_PASO/issues/148.

3. Strategy factories missing from `__init__.py`. Every `<...>_strategy(...)` factory function
   defined in a recipe file should be importable as `el_paso.recipes.<mission>.<name>` (under
   its own name or an alias) and listed in that package's `__all__`, the same as its
   `process_*` entry point already must be. Otherwise the factory is unusable from outside its
   own module; see https://github.com/GFZ/EL_PASO/issues/148.

Run directly as a script (also wired up as a pre-commit hook); always scans the whole
`el_paso/recipes/` tree regardless of which files changed.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPES_DIR = REPO_ROOT / "el_paso" / "recipes"

ALLOWED_INLINE_STRATEGY: dict[Path, set[int]] = {
    RECIPES_DIR / "rbsp" / "process_rbsp_efw_emfisis_density_combined.py": {
        195
    },  # https://github.com/GFZ/EL_PASO/issues/139
}

ALLOWED_STR_SATELLITE: dict[Path, frozenset[str]] = {}


class Violation(NamedTuple):
    """One thing found wrong in a recipe file, plus which check found it."""

    path: Path
    lineno: int
    detail: str


def _iter_recipe_files() -> list[Path]:
    return sorted(p for p in RECIPES_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def _iter_mission_dirs() -> list[Path]:
    return sorted(p for p in RECIPES_DIR.iterdir() if p.is_dir() and "__pycache__" not in p.parts)


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


def _find_inline_strategy_violations_in_function(
    path: Path, func: ast.FunctionDef | ast.AsyncFunctionDef
) -> list[Violation]:
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


def _find_inline_strategy_violations(path: Path, tree: ast.Module) -> list[Violation]:
    allowed_lines = ALLOWED_INLINE_STRATEGY.get(path, set())
    violations: list[Violation] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("process_"):
            violations.extend(_find_inline_strategy_violations_in_function(path, node))

    return [v for v in violations if v.lineno not in allowed_lines]


def _is_str_annotation(annotation: ast.expr | None) -> bool:
    if annotation is None:
        return False
    if isinstance(annotation, ast.Name):
        return annotation.id == "str"
    return isinstance(annotation, ast.Constant) and annotation.value == "str"


def _find_str_satellite_violations(path: Path, tree: ast.Module) -> list[Violation]:
    allowed_functions = ALLOWED_STR_SATELLITE.get(path, frozenset())
    violations: list[Violation] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name in allowed_functions:
            continue

        args = node.args
        violations.extend(
            Violation(path, arg.lineno, f"{node.name}(..., satellite: str, ...)")
            for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)
            if arg.arg == "satellite" and _is_str_annotation(arg.annotation)
        )

    return violations


def _top_level_strategy_function_names(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.endswith("_strategy")
    }


def _import_aliases_from_submodule(init_tree: ast.Module, target_module: str) -> dict[str, list[str]]:
    """Map original name -> bound (possibly aliased) name(s).

    Covers names imported into `__init__.py` from `target_module`.
    """
    aliases: dict[str, list[str]] = {}
    for node in ast.walk(init_tree):
        if isinstance(node, ast.ImportFrom) and node.module == target_module:
            for alias in node.names:
                aliases.setdefault(alias.name, []).append(alias.asname or alias.name)
    return aliases


def _all_list_names(init_tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in init_tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__all__"
            and isinstance(node.value, ast.List)
        ):
            names.update(
                elt.value for elt in node.value.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            )
    return names


def _find_missing_strategy_exports() -> list[Violation]:
    violations: list[Violation] = []

    for mission_dir in _iter_mission_dirs():
        init_path = mission_dir / "__init__.py"
        if not init_path.exists():
            continue

        init_tree = ast.parse(init_path.read_text(), filename=str(init_path))
        exported_names = _all_list_names(init_tree)
        target_module_prefix = f"el_paso.recipes.{mission_dir.name}."

        for module_path in sorted(mission_dir.glob("*.py")):
            if module_path.name == "__init__.py":
                continue

            module_tree = ast.parse(module_path.read_text(), filename=str(module_path))
            strategy_names = _top_level_strategy_function_names(module_tree)
            if not strategy_names:
                continue

            aliases = _import_aliases_from_submodule(init_tree, target_module_prefix + module_path.stem)

            for name in sorted(strategy_names):
                bound_names = aliases.get(name, [])
                rel_module = module_path.relative_to(REPO_ROOT)

                if not bound_names:
                    violations.append(Violation(init_path, 1, f"{name} (defined in {rel_module}) is not imported"))
                elif not any(bound in exported_names for bound in bound_names):
                    imported_as = "/".join(bound_names)
                    violations.append(
                        Violation(
                            init_path,
                            1,
                            f"{name} (defined in {rel_module}, imported as {imported_as}) is missing from __all__",
                        )
                    )

    return violations


def find_violations() -> tuple[list[Violation], list[Violation], list[Violation]]:
    """Return (inline_strategy, str_satellite, missing_exports) violations."""
    inline_strategy: list[Violation] = []
    str_satellite: list[Violation] = []

    for path in _iter_recipe_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        inline_strategy.extend(_find_inline_strategy_violations(path, tree))
        str_satellite.extend(_find_str_satellite_violations(path, tree))

    missing_exports = _find_missing_strategy_exports()

    return inline_strategy, str_satellite, missing_exports


def _print_section(header: str, violations: list[Violation]) -> None:
    if not violations:
        return

    print(header)  # noqa: T201
    for violation in violations:
        rel = violation.path.relative_to(REPO_ROOT)
        print(f"  {rel}:{violation.lineno}: {violation.detail}")  # noqa: T201
    print()  # noqa: T201


def main() -> int:
    """Print every violation found and return a nonzero exit code if there were any."""
    inline_strategy, str_satellite, missing_exports = find_violations()

    _print_section(
        "A process_* recipe entry point builds a saving strategy inline instead of through a\n"
        "named <...>_strategy(...) function defined in the same file. Violations:\n",
        inline_strategy,
    )
    _print_section(
        "A `satellite` parameter is typed as a bare `str` instead of a Literal of the mission's\n"
        "valid satellite names. Violations:\n",
        str_satellite,
    )
    _print_section(
        "A <...>_strategy(...) factory isn't importable as el_paso.recipes.<mission>.<name> (not\n"
        "imported into __init__.py, or missing from its __all__). Violations:\n",
        missing_exports,
    )

    return 1 if (inline_strategy or str_satellite or missing_exports) else 0


if __name__ == "__main__":
    sys.exit(main())
