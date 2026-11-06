# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import copy
import dataclasses
import pickle

import el_paso as ep
import pytest
from astropy import units as u
from el_paso.data_standard import VariableInfo
from el_paso.one_of import OneOf, alternatives_of, describe, resolve_name
from el_paso.saving_strategy import OutputFile


@pytest.mark.minimal
@pytest.mark.basic
def test_resolve_picks_first_present_alternative() -> None:
    assert OneOf("FEDU", "FPDU").resolve({"FPDU", "FEDU"}) == "FEDU"
    assert OneOf("FEDU", "FPDU").resolve({"FPDU"}) == "FPDU"
    assert OneOf("FEDU", "FPDU").resolve({"Epoch"}) is None


@pytest.mark.minimal
@pytest.mark.basic
@pytest.mark.parametrize("alternatives", [("FEDU",), ("FEDU", "FEDU")])
def test_one_of_rejects_fewer_than_two_distinct_alternatives(alternatives: tuple[ep.typing.InternalName, ...]) -> None:
    with pytest.raises(ValueError, match="OneOf"):
        OneOf(*alternatives)


@pytest.mark.minimal
@pytest.mark.basic
def test_one_of_value_semantics() -> None:
    one_of = OneOf("FEDU", "FPDU")
    assert one_of == OneOf("FEDU", "FPDU")
    assert one_of != OneOf("FPDU", "FEDU")
    assert one_of != ("FEDU", "FPDU")
    assert one_of != "FEDU"
    assert hash(one_of) == hash(OneOf("FEDU", "FPDU"))
    assert pickle.loads(pickle.dumps(one_of)) == one_of  # noqa: S301
    assert copy.deepcopy(one_of) == one_of
    assert str(one_of) == "FEDU or FPDU"
    assert repr(one_of) == "OneOf('FEDU', 'FPDU')"


@pytest.mark.minimal
@pytest.mark.basic
def test_helpers_treat_plain_name_as_single_alternative() -> None:
    assert resolve_name("Epoch", {"Epoch"}) == "Epoch"
    assert resolve_name("Epoch", {"FEDU"}) is None
    assert alternatives_of("Epoch") == ("Epoch",)
    assert describe("Epoch") == "Epoch"


@pytest.mark.minimal
@pytest.mark.basic
def test_helpers_handle_one_of() -> None:
    one_of = OneOf("FEDU", "FPDU")
    assert resolve_name(one_of, {"FPDU"}) == "FPDU"
    assert alternatives_of(one_of) == ("FEDU", "FPDU")
    assert describe(one_of) == "FEDU or FPDU"


@pytest.mark.minimal
@pytest.mark.basic
def test_helpers_reject_bare_tuple() -> None:
    with pytest.raises(TypeError, match="OneOf"):
        alternatives_of(("FEDU", "FPDU"))  # ty: ignore[invalid-argument-type]


@pytest.mark.minimal
@pytest.mark.basic
def test_output_file_converts_tuple_with_deprecation_warning() -> None:
    with pytest.deprecated_call(match=r"names_to_save as a tuple is deprecated"):
        output_file = OutputFile("flux", ["Epoch", ("FEDU", "FPDU")])
    assert output_file.names_to_save == ["Epoch", OneOf("FEDU", "FPDU")]


@pytest.mark.minimal
@pytest.mark.basic
def test_variable_info_converts_tuple_with_deprecation_warning() -> None:
    with pytest.deprecated_call(match=r"dependencies as a tuple is deprecated"):
        info = VariableInfo("PSD", "", u.dimensionless_unscaled, ["Epoch", ("Energy_FEDU", "Energy_FPDU"), "Alpha"])
    assert info.dependencies == ["Epoch", OneOf("Energy_FEDU", "Energy_FPDU"), "Alpha"]


@pytest.mark.minimal
@pytest.mark.basic
def test_one_of_entries_do_not_warn(recwarn: pytest.WarningsRecorder) -> None:
    OutputFile("flux", ["Epoch", OneOf("FEDU", "FPDU")])
    VariableInfo("PSD", "", u.dimensionless_unscaled, ["Epoch", OneOf("Energy_FEDU", "Energy_FPDU"), "Alpha"])
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


@pytest.mark.minimal
@pytest.mark.basic
def test_dataclasses_replace_also_converts_tuples() -> None:
    output_file = OutputFile("flux", ["Epoch"])
    with pytest.deprecated_call():
        replaced = dataclasses.replace(output_file, names_to_save=[("FEDU", "FPDU")])
    assert replaced.names_to_save == [OneOf("FEDU", "FPDU")]


@pytest.mark.minimal
@pytest.mark.basic
def test_one_of_is_exported_from_typing() -> None:
    assert ep.typing.OneOf is OneOf
