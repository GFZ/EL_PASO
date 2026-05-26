# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path

import pytest

from el_paso import typing as ep_types


@pytest.mark.parametrize(
    ("attr_name", "import_info"),
    list(ep_types._LAZY_EXPORTS.items()),
)
def test_lazy_exports_resolve(
    attr_name: str,
    import_info: tuple[str, str],
) -> None:
    module_path, attribute_name = import_info

    resolved = getattr(ep_types, attr_name)
    expected = getattr(import_module(module_path), attribute_name)

    assert resolved is expected


def test_dir_includes_lazy_exports_and_is_sorted() -> None:
    exported_names = dir(ep_types)

    assert exported_names == sorted(exported_names)
    assert set(ep_types._LAZY_EXPORTS).issubset(exported_names)


def test_public_typing_definitions_are_in_all() -> None:
    """Test that all public typing definitions are included in __all__."""
    expected_public_names = _get_public_typing_definitions()

    assert expected_public_names.issubset(ep_types.__all__)


def test_public_typing_definitions_missing_from_all_are_detected() -> None:
    source_text = Path(ep_types.__file__).read_text()
    mocked_source = source_text.replace(
        "\n\n_LAZY_EXPORTS: dict[str, tuple[str, str]] = {",
        "\n\nMockedPublicType: TypeAlias = int\n\n_LAZY_EXPORTS: dict[str, tuple[str, str]] = {",
        1,
    )

    discovered_public_names = _get_public_typing_definitions_from_source(mocked_source)

    assert "MockedPublicType" in discovered_public_names
    assert "MockedPublicType" not in ep_types.__all__


@pytest.mark.parametrize("export_name", ep_types.__all__)
def test_all_exports_are_importable(export_name: str) -> None:
    assert getattr(ep_types, export_name) is not None


def test_type_checking_imports_are_lazy_exports_and_in_all() -> None:
    imported_names = _get_type_checking_imported_names()

    missing_from_lazy_exports = imported_names - set(ep_types._LAZY_EXPORTS)
    missing_from_all = imported_names - set(ep_types.__all__)
    assert not missing_from_lazy_exports, (
        f"Type-checking import names missing from _LAZY_EXPORTS: {missing_from_lazy_exports}"
    )
    assert not missing_from_all, f"Type-checking import names missing from __all__: {missing_from_all}"


def _get_public_typing_definitions() -> set[str]:
    source_path = Path(ep_types.__file__).resolve()
    return _get_public_typing_definitions_from_source(source_path.read_text())


def _get_public_typing_definitions_from_source(source_text: str) -> set[str]:
    module_ast = ast.parse(source_text)

    public_names: set[str] = set()

    for node in module_ast.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            public_names.add(node.name)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and not target.id.startswith("_"):
                public_names.add(target.id)

    return public_names


def _get_type_checking_imported_names() -> set[str]:
    source_path = Path(ep_types.__file__).resolve()
    return _get_type_checking_imported_names_from_source(source_path.read_text())


def _get_type_checking_imported_names_from_source(source_text: str) -> set[str]:
    module_ast = ast.parse(source_text)

    imported_names: set[str] = set()

    for node in module_ast.body:
        if not isinstance(node, ast.If) or not _is_type_checking_check(node.test):
            continue

        for child in node.body:
            if isinstance(child, ast.ImportFrom):
                imported_names.update(alias.asname or alias.name for alias in child.names)
            elif isinstance(child, ast.Import):
                imported_names.update(alias.asname or alias.name.split(".", maxsplit=1)[0] for alias in child.names)

    return imported_names


def _is_type_checking_check(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "TYPE_CHECKING") or (
        isinstance(node, ast.Attribute) and node.attr == "TYPE_CHECKING"
    )
