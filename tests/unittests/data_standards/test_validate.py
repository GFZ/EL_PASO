# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import el_paso as ep
import numpy as np
import pytest
from astropy import units as u
from el_paso.data_standard import NullDataStandard


def _var() -> ep.Variable:
    return ep.Variable(u.dimensionless_unscaled, data=np.zeros(3))


@pytest.mark.minimal
@pytest.mark.basic
def test_validate_raises_for_unknown_variable() -> None:
    with pytest.raises(ValueError, match=r"not part of PRBEMStandard: \['Wave_frequency'\]"):
        ep.data_standards.PRBEMStandard().validate({"Epoch": _var(), "Wave_frequency": _var()})


@pytest.mark.minimal
@pytest.mark.basic
def test_validate_raises_for_missing_dimension() -> None:
    with pytest.raises(ValueError, match=r"'Energy_FEDU' \(required by: FEDU\)"):
        ep.data_standards.GFZStandard().validate({"Epoch": _var(), "Alpha": _var(), "FEDU": _var()})


@pytest.mark.minimal
@pytest.mark.basic
def test_validate_accepts_any_alternative_of_tuple_dependency() -> None:
    ep.data_standards.GFZStandard().validate({"Epoch": _var(), "Alpha": _var(), "Energy_FPDU": _var(), "PSD": _var()})


@pytest.mark.minimal
@pytest.mark.basic
def test_validate_names_all_alternatives_when_none_present() -> None:
    with pytest.raises(ValueError, match=r"'Energy_FEDU or Energy_FPDU' \(required by: PSD\)"):
        ep.data_standards.GFZStandard().validate({"Epoch": _var(), "Alpha": _var(), "PSD": _var()})


@pytest.mark.minimal
@pytest.mark.basic
def test_null_data_standard_accepts_anything() -> None:
    NullDataStandard().validate({"FEDU": _var(), "Wave_frequency": _var()})


@pytest.mark.minimal
@pytest.mark.basic
def test_null_data_standard_leaves_variable_unchanged() -> None:
    variable = _var()
    result = NullDataStandard().standardize_variable("FEDU", variable, reset_consistency_check=True)
    assert result is variable
