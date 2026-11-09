# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import el_paso as ep
import numpy as np
import pytest
from el_paso.data_standard import DataStandard
from el_paso.one_of import OneOf

_GFZ = ep.data_standards.GFZStandard()


@pytest.mark.minimal
@pytest.mark.basic
def test_resolve_dependencies_picks_present_alternative() -> None:
    dependencies = ["Epoch", OneOf("Energy_FEDU", "Energy_FPDU"), "Alpha"]
    assert DataStandard.resolve_dependencies(dependencies, {"Energy_FPDU"}) == ["Epoch", "Energy_FPDU", "Alpha"]


@pytest.mark.minimal
@pytest.mark.basic
def test_resolve_dependencies_passes_plain_names_through_even_if_absent() -> None:
    assert DataStandard.resolve_dependencies(["Epoch", "Position_components"], set()) == [
        "Epoch",
        "Position_components",
    ]


@pytest.mark.minimal
@pytest.mark.basic
def test_resolve_dependencies_returns_none_for_unresolved_entry_only() -> None:
    dependencies = ["Epoch", OneOf("Energy_FEDU", "Energy_FPDU"), "Alpha"]
    assert DataStandard.resolve_dependencies(dependencies, {"Epoch", "Alpha"}) == ["Epoch", None, "Alpha"]


@pytest.mark.minimal
@pytest.mark.basic
def test_get_resolved_dependencies_resolves() -> None:
    assert _GFZ.get_resolved_dependencies("PSD", {"Energy_FEDU"}) == ["Epoch", "Energy_FEDU", "Alpha"]


@pytest.mark.minimal
@pytest.mark.basic
def test_get_resolved_dependencies_raises_naming_variable_and_alternatives() -> None:
    with pytest.raises(ValueError, match=r"dimensions of PSD: none of Energy_FEDU or Energy_FPDU is present"):
        _GFZ.get_resolved_dependencies("PSD", {"Epoch", "Alpha", "PSD"})


@pytest.mark.minimal
@pytest.mark.basic
def test_standardize_variable_raises_if_no_alternative_present() -> None:
    psd = ep.Variable(ep.units.psd_natural_unit, data=np.ones((4, 3, 2)))
    with pytest.raises(ValueError, match="dimensions of PSD"):
        _GFZ.standardize_variable("PSD", psd, reset_consistency_check=True, available_keys={"Epoch", "Alpha", "PSD"})
