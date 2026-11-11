# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""Entries that any one of several alternative internal names can satisfy.

`OutputFile.names_to_save` and `VariableInfo.dependencies` hold entries that are
either a plain name or a `OneOf`. Read them through `resolve_name`, `alternatives_of`
and `describe` rather than iterating or comparing them directly: a plain name is a
`str`, so iterating it yields characters and `in` on it matches substrings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Container

    from el_paso.typing import InternalName

N = TypeVar("N", bound=str)


@dataclass(frozen=True, init=False)
class OneOf:
    """An entry that any one of several alternative internal names satisfies.

    When resolved against the names that are available, the first alternative
    present wins. Example: `OneOf("FEDU", "FPDU")` accepts electron or proton flux.

    Attributes:
        alternatives (tuple[InternalName, ...]): The alternative names, in order of preference.
    """

    alternatives: tuple[InternalName, ...]

    def __init__(self, *alternatives: InternalName) -> None:
        """Initializes the entry with its alternatives.

        Args:
            *alternatives (InternalName): Two or more distinct internal names, in order of preference.

        Raises:
            ValueError: If fewer than two alternatives are given, or an alternative is repeated.
        """
        if len(alternatives) < 2:
            msg = f"OneOf needs at least two alternatives, got {alternatives}."
            raise ValueError(msg)
        if len(set(alternatives)) != len(alternatives):
            msg = f"OneOf has duplicate alternatives: {alternatives}."
            raise ValueError(msg)
        object.__setattr__(self, "alternatives", alternatives)

    def resolve(self, available: Container[str]) -> InternalName | None:
        """Returns the first alternative contained in `available`, or None if none is."""
        return next((alt for alt in self.alternatives if alt in available), None)

    def __str__(self) -> str:
        return " or ".join(self.alternatives)

    def __repr__(self) -> str:
        return f"OneOf({', '.join(map(repr, self.alternatives))})"


def resolve_name(entry: N | OneOf, available: Container[str]) -> N | InternalName | None:
    """Returns the name `entry` resolves to among `available`, or None if it is not satisfied.

    Args:
        entry (N | OneOf): A plain name or a `OneOf`.
        available (Container[str]): The names that are present.

    Returns:
        N | InternalName | None: The plain name if present, the first present alternative of a
            `OneOf`, or None.
    """
    if isinstance(entry, OneOf):
        return entry.resolve(available)
    _reject_tuple(entry)
    return entry if entry in available else None


def alternatives_of(entry: N | OneOf) -> tuple[N | InternalName, ...]:
    """Returns every name that can satisfy `entry`: its alternatives, or the plain name alone."""
    if isinstance(entry, OneOf):
        return entry.alternatives
    _reject_tuple(entry)
    return (entry,)


def describe(entry: str | OneOf) -> str:
    """Returns a label for messages, e.g. `"FEDU or FPDU"`."""
    _reject_tuple(entry)
    return str(entry)


def _reject_tuple(entry: object) -> None:
    if isinstance(entry, tuple):
        msg = f"Alternatives must be a OneOf, got the tuple {entry!r}. Use OneOf{entry!r} instead."
        raise TypeError(msg)
