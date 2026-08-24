# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the generated attribute blocks in typing.py, metadata.py, and dataset_implementations.py."""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path
from textwrap import indent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import TYPE_CHECKING

from el_paso.data_standards import GFZStandard, PRBEMStandard

if TYPE_CHECKING:
    from el_paso.data_standard import VariableInfo

REPO_ROOT = Path(__file__).resolve().parent.parent
TYPING_PY = REPO_ROOT / "el_paso" / "typing.py"
METADATA_PY = REPO_ROOT / "el_paso" / "dataset" / "metadata.py"
DATASET_IMPLEMENTATIONS_PY = REPO_ROOT / "el_paso" / "dataset" / "dataset_implementations.py"

# ruff's ambiguous-variable-name rule (E741) flags exactly these single-character names.
_E741_NAMES = {"l", "O", "I"}


def _noqa_for_name(name: str) -> str | None:
    """Return the ruff noqa code a generated attribute name needs, if any."""
    if name in _E741_NAMES:
        return "E741"
    if re.search(r"^[a-z][a-z0-9_]*[A-Z]", name):
        return "N815"

    return None


def _sorted_infos(variable_infos: dict[str, VariableInfo]) -> list[VariableInfo]:
    """Return a data standard's variable infos sorted by standard (attribute) name.

    Args:
        variable_infos (dict[str, VariableInfo]): A data standard's `variable_infos` mapping.

    Returns:
        list[VariableInfo]: The infos sorted by `standard_name`, giving deterministic,
        alphabetically-ordered generated output.
    """
    return sorted(variable_infos.values(), key=lambda info: info.standard_name)


def _replace_between_markers(text: str, marker: str, new_body: str) -> str:
    """Replace the text between a pair of `# BEGIN/END GENERATED <marker>` comments.

    The replacement is re-indented to match the indentation of the `# BEGIN` marker line,
    so callers can pass unindented `new_body` regardless of the marker's nesting depth.

    Args:
        text (str): The full file contents to search and replace within.
        marker (str): The marker name, e.g. `"GFZ_VAR_NAMES"` (without the
            `# BEGIN/END GENERATED` prefix).
        new_body (str): The replacement content, one entry per line, unindented.

    Returns:
        str: `text` with the region between the markers replaced by `new_body`.

    Raises:
        ValueError: If the `# BEGIN GENERATED <marker>` / `# END GENERATED <marker>`
            marker pair is not found in `text`.
    """
    pattern = re.compile(
        rf"( *# BEGIN GENERATED {re.escape(marker)}\n).*?(\n *# END GENERATED {re.escape(marker)})",
        re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        msg = f"Could not find '# BEGIN GENERATED {marker}' / '# END GENERATED {marker}' markers."
        raise ValueError(msg)

    leading_ws = re.match(r" *", match.group(1)).group()  # ty:ignore[possibly-unbound-attribute]
    body = indent(new_body, leading_ws).rstrip("\n") + "\n"

    return pattern.sub(
        lambda _: match.group(1) + body + leading_ws.rstrip() + match.group(2).lstrip("\n"), text, count=1
    )


def _generate_gfz_var_names_literal(infos: list[VariableInfo]) -> str:
    """Generate the quoted, comma-terminated lines for the `GFZVarNames` `Literal`.

    Args:
        infos (list[VariableInfo]): The GFZ data standard's variable infos.

    Returns:
        str: One `"standard_name",` line per info, newline-separated.
    """
    lines = [f'"{info.standard_name}",' for info in infos]
    return "\n".join(lines)


def _generate_class_attrs(infos: list[VariableInfo], attr_type: str) -> str:
    """Generate class-level attribute annotation lines for a metadata/dataset class.

    Args:
        infos (list[VariableInfo]): The data standard's variable infos.
        attr_type (str): The type annotation to give each attribute, e.g.
            `"VariableMetadata"` or `"NDArray[np.float64]"`.

    Returns:
        str: One `name: attr_type` line per info (with a targeted `# noqa` comment
        appended where `_noqa_for_name` requires one), newline-separated.
    """
    lines = []
    for info in infos:
        noqa = _noqa_for_name(info.standard_name)
        suffix = f"  # noqa: {noqa}" if noqa else ""
        lines.append(f"{info.standard_name}: {attr_type}{suffix}")
    return "\n".join(lines)


_MAX_LINE_LENGTH = 120
_DOCSTRING_INDENT = 8
_CONTINUATION_INDENT = 4


def _generate_docstring_attrs(infos: list[VariableInfo], attr_type: str) -> str:
    """Generate Google-style `Attributes:` doc lines, wrapping long descriptions.

    Args:
        infos (list[VariableInfo]): The data standard's variable infos.
        attr_type (str): The type annotation to display for each attribute, e.g.
            `"VariableMetadata"` or `"NDArray[np.float64]"`.

    Returns:
        str: One `name (attr_type): description` line per info, newline-separated;
        descriptions that would exceed ruff's line-length limit are wrapped onto
        indented continuation lines instead of being truncated.
    """
    available = _MAX_LINE_LENGTH - _DOCSTRING_INDENT
    lines = []
    for info in infos:
        head = f"{info.standard_name} ({attr_type}): "
        if len(head) + len(info.description) <= available:
            lines.append(head + info.description)
            continue

        wrapped = textwrap.wrap(
            info.description,
            width=available - _CONTINUATION_INDENT,
            initial_indent=head,
            subsequent_indent=" " * _CONTINUATION_INDENT,
        )
        lines.append("\n".join(wrapped))
    return "\n".join(lines)


def update_typing_py(gfz_infos: list[VariableInfo]) -> None:
    """Regenerate the `GFZVarNames` `Literal` block in `el_paso/typing.py`.

    Args:
        gfz_infos (list[VariableInfo]): The GFZ data standard's variable infos, sorted.
    """
    text = TYPING_PY.read_text()
    text = _replace_between_markers(text, "GFZ_VAR_NAMES", _generate_gfz_var_names_literal(gfz_infos))
    TYPING_PY.write_text(text)


def update_metadata_py(gfz_infos: list[VariableInfo], prbem_infos: list[VariableInfo]) -> None:
    """Regenerate the `GFZMetaData`/`PRBEMMetaData` attribute blocks in `el_paso/dataset/metadata.py`.

    Args:
        gfz_infos (list[VariableInfo]): The GFZ data standard's variable infos, sorted.
        prbem_infos (list[VariableInfo]): The PRBEM data standard's variable infos, sorted.
    """
    text = METADATA_PY.read_text()
    text = _replace_between_markers(
        text, "GFZ_METADATA_ATTRS DOCS", _generate_docstring_attrs(gfz_infos, "VariableMetadata")
    )
    text = _replace_between_markers(text, "GFZ_METADATA_ATTRS", _generate_class_attrs(gfz_infos, "VariableMetadata"))
    text = _replace_between_markers(
        text, "PRBEM_METADATA_ATTRS DOCS", _generate_docstring_attrs(prbem_infos, "VariableMetadata")
    )
    text = _replace_between_markers(
        text, "PRBEM_METADATA_ATTRS", _generate_class_attrs(prbem_infos, "VariableMetadata")
    )
    METADATA_PY.write_text(text)


def update_dataset_implementations_py(gfz_infos: list[VariableInfo], prbem_infos: list[VariableInfo]) -> None:
    """Regenerate the `GFZDataSet`/`PRBEMDataSet` attribute blocks in `el_paso/dataset/dataset_implementations.py`.

    Args:
        gfz_infos (list[VariableInfo]): The GFZ data standard's variable infos, sorted.
        prbem_infos (list[VariableInfo]): The PRBEM data standard's variable infos, sorted.
    """
    text = DATASET_IMPLEMENTATIONS_PY.read_text()
    text = _replace_between_markers(
        text, "GFZ_DATASET_ATTRS DOCS", _generate_docstring_attrs(gfz_infos, "NDArray[np.float64]")
    )
    text = _replace_between_markers(text, "GFZ_DATASET_ATTRS", _generate_class_attrs(gfz_infos, "NDArray[np.float64]"))
    text = _replace_between_markers(
        text, "PRBEM_DATASET_ATTRS DOCS", _generate_docstring_attrs(prbem_infos, "NDArray[np.float64]")
    )
    text = _replace_between_markers(
        text, "PRBEM_DATASET_ATTRS", _generate_class_attrs(prbem_infos, "NDArray[np.float64]")
    )
    DATASET_IMPLEMENTATIONS_PY.write_text(text)


def main() -> None:
    """Regenerate all generated attribute blocks from the current data standards."""
    gfz_infos = _sorted_infos(GFZStandard().variable_infos)
    prbem_infos = _sorted_infos(PRBEMStandard().variable_infos)

    update_typing_py(gfz_infos)
    update_metadata_py(gfz_infos, prbem_infos)
    update_dataset_implementations_py(gfz_infos, prbem_infos)


if __name__ == "__main__":
    main()
