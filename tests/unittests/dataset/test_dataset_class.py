# # SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# # SPDX-FileContributor: Sahil Jhawar
# #
# # SPDX-License-Identifier: Apache-2.0

# from __future__ import annotations

# from datetime import datetime, timezone
# from pathlib import Path
# from typing import TYPE_CHECKING

# import numpy as np
# import pytest

# import el_paso as ep
# from el_paso.dataset import DataSet
# from el_paso.saving_strategy import SavingStrategy

# if TYPE_CHECKING:
#     from el_paso import Variable
#     from el_paso.saving_strategy import OutputFile
#     from el_paso.typing import InternalName, SavedDataDict


# _DATA_STANDARD = ep.data_standards.DataOrgStandard()
# _POSSIBLE_VARIABLES = [
#     "datetime",
#     "time",
#     "energy_channels",
#     "alpha_local",
#     "alpha_eq_model",
#     "alpha_eq_real",
#     "InvMu",
#     "InvMu_real",
#     "InvK",
#     "InvV",
#     "Lstar",
#     "Flux",
#     "PSD",
#     "MLT",
#     "B_SM",
#     "B_total",
#     "B_sat",
#     "xGEO",
#     "P",
#     "R0",
#     "density",
# ]


# class DummySavingStrategy(SavingStrategy):
#     """Minimal saving strategy used to construct `DataSet` instances in dict mode."""

#     def __init__(self) -> None:
#         """Initialize a minimal in-memory saving strategy for tests."""
#         self.base_data_path = Path()
#         self.mission = "TEST"
#         self.satellite = "SAT"
#         self.instrument = "INST"
#         self.mag_field = "T89"
#         self.data_standard = _DATA_STANDARD
#         self.output_files = []

#     def get_time_intervals_to_save(
#         self, start_time: datetime, end_time: datetime
#     ) -> list[tuple[datetime, datetime]]:
#         del start_time, end_time
#         return []

#     def get_file_path(
#         self, interval_start: datetime, interval_end: datetime, output_file: OutputFile
#     ) -> Path:
#         del interval_start, interval_end, output_file
#         return self.base_data_path / "mock"

#     def standardize_variable(
#         self, variable: Variable, internal_name: InternalName, *, first_call_of_interval: bool
#     ) -> Variable:
#         del internal_name, first_call_of_interval
#         return variable

#     def save_single_file(self, file_path: Path, dict_to_save: SavedDataDict, *, append: bool = False) -> None:
#         del file_path, dict_to_save, append

#     def get_file_path_stem(self) -> Path:
#         return self.base_data_path

#     def get_file_name_stem(self) -> str:
#         return "mock"


# def _make_dataset() -> DataSet:
#     dataset = DataSet(
#         saving_strategy=DummySavingStrategy(),
#         start_time=None,
#         end_time=None,
#         preferred_extension="nc",
#         verbose=False,
#     )
#     dataset.__dict__.update(
#         {
#             "possible_variables": list(_POSSIBLE_VARIABLES),
#             "_satellite": "SAT",
#             "_instrument": "INST",
#             "_mfm": "T89",
#         }
#     )
#     return dataset


# @pytest.fixture
# def dict_dataset() -> DataSet:
#     """Return a dict-mode `DataSet` (no file loading)."""
#     return _make_dataset()


# @pytest.fixture
# def matching_dict_dataset() -> DataSet:
#     """Return a second dict-mode dataset with the same metadata for equality tests."""
#     return _make_dataset()


# def _seed_dataset(dataset: DataSet) -> None:
#     dataset.__dict__.update(
#         {
#             "datetime": [datetime(2013, 1, 1, tzinfo=timezone.utc)],
#             "time": np.array([738000.0]),
#             "MLT": np.array([0.0, 6.0, 12.0]),
#             "InvK": np.array([[1.0, 2.0]]),
#             "InvMu": np.array([[[0.1, 0.2], [0.3, 0.4]]]),
#             "Flux": np.array([[1.0, 2.0, 3.0]]),
#         }
#     )


# def test_repr_and_str_include_core_metadata(dict_dataset: DataSet) -> None:
#     """`__repr__` and `__str__` should expose the main dataset metadata."""
#     repr_text = repr(dict_dataset)
#     str_text = str(dict_dataset)

#     for text in (repr_text, str_text):
#         assert "DataSet" in text


# def test_satellite_name_methods(dict_dataset: DataSet) -> None:
#     """The dataset helper methods should reflect the configured saving strategy."""
#     assert dict_dataset.get_satellite_name() == "SAT"
#     assert dict_dataset.get_satellite_and_instrument_name() == "SAT_INST"
#     assert dict_dataset.get_print_name() == "SAT INST"


# def test_get_var_by_internal_name_returns_standardized_variable(dict_dataset: DataSet) -> None:
#     """`get_var_by_internal_name` should map internal names to public dataset attributes."""
#     dict_dataset.__dict__["Flux"] = np.array([[1.0, 2.0, 3.0]])
#     dict_dataset.__dict__["time"] = np.array([738000.0])

#     np.testing.assert_array_equal(dict_dataset.get_var_by_internal_name("FEDU"), dict_dataset.Flux)
#     np.testing.assert_array_equal(dict_dataset.get_var_by_internal_name("Epoch"), dict_dataset.time)

# def test_computed_p_property(dict_dataset: DataSet) -> None:
#     """`P` should be computed from `MLT` when accessed."""
#     dict_dataset.__dict__["MLT"] = np.array([0.0, 6.0, 12.0])

#     expected = ((dict_dataset.MLT + 12) / 12 * np.pi) % (2 * np.pi)
#     np.testing.assert_allclose(dict_dataset.P, expected)


# def test_computed_invv_property(dict_dataset: DataSet) -> None:
#     """`InvV` should be computed from `InvK` and `InvMu` when accessed."""
#     # shapes: time=1, energy=2, alpha=2
#     dict_dataset.__dict__["InvK"] = np.array([[1.0, 2.0]])
#     dict_dataset.__dict__["InvMu"] = np.array([[[0.1, 0.2], [0.3, 0.4]]])

#     inv_K_repeated = np.repeat(dict_dataset.InvK[:, np.newaxis, :], dict_dataset.InvMu.shape[1], axis=1)
#     expected = dict_dataset.InvMu * (inv_K_repeated + 0.5) ** 2

#     # Access computed property
#     np.testing.assert_allclose(dict_dataset.InvV, expected)


# def test_get_loaded_variables_includes_computed(dict_dataset: DataSet) -> None:
#     """Accessing computed properties should make them show up in loaded variables."""
#     dict_dataset.__dict__["MLT"] = np.array([0.0, 6.0, 12.0])
#     dict_dataset.__dict__["InvK"] = np.array([[1.0, 2.0]])
#     dict_dataset.__dict__["InvMu"] = np.array([[[0.1, 0.2]]])

#     _ = dict_dataset.P
#     _ = dict_dataset.InvV

#     loaded = dict_dataset.get_loaded_variables()
#     assert "P" in loaded
#     assert "InvV" in loaded


# def test_getattr_with_valid_variable(dict_dataset: DataSet) -> None:
#     """Setting and reading an existing variable should work through attribute access."""
#     dict_dataset.__dict__["Flux"] = np.array([[1.0, 2.0, 3.0]])

#     result = dict_dataset.Flux
#     assert isinstance(result, np.ndarray)
#     np.testing.assert_array_equal(result, np.array([[1.0, 2.0, 3.0]]))


# def test_getattr_with_unset_known_variable(dict_dataset: DataSet) -> None:
#     """Accessing a known but unset variable should raise the dataset-specific error."""
#     with pytest.raises(AttributeError, match="exists in `VariableLiteral` but has not been set"):
#         _ = dict_dataset.Flux


# def test_getattr_with_unknown_variable_suggests_close_match(dict_dataset: DataSet) -> None:
#     """Unknown variables should still produce the helpful suggestion path when possible."""
#     with pytest.raises(AttributeError):
#         _ = dict_dataset.somethingrandom


# def test_find_similar_variable(dict_dataset: DataSet) -> None:
#     """The similarity helper should return exact matches and close suggestions."""
#     exact, _info_exact = dict_dataset.find_similar_variable("Flux")
#     typo, info_typo = dict_dataset.find_similar_variable("Flx")

#     assert exact == "Flux"
#     assert typo is None
#     assert info_typo["var_name"] == "Flux"


# def test_unknown_attribute_raises(dict_dataset: DataSet) -> None:
#     """Requesting an unknown attribute should raise an AttributeError with helpful text."""
#     with pytest.raises(AttributeError):
#         _ = dict_dataset.somethingrandom


# def test_get_different_variables_reports_changes(
#     dict_dataset: DataSet, matching_dict_dataset: DataSet
# ) -> None:
#     """`get_different_variables` should detect modified variables."""
#     _seed_dataset(dict_dataset)
#     _seed_dataset(matching_dict_dataset)

#     assert dict_dataset == matching_dict_dataset

#     matching_dict_dataset.__dict__["Flux"] = np.array([[9.0, 8.0, 7.0]])

#     different_variables = dict_dataset.get_different_variables(matching_dict_dataset)

#     assert "Flux" in different_variables
#     assert dict_dataset != matching_dict_dataset


# def test_eq_with_nan_arrays(dict_dataset: DataSet, matching_dict_dataset: DataSet) -> None:
#     """Equality should treat NaN values in arrays as equal."""
#     dict_dataset.__dict__["Flux"] = np.array([[1.0, np.nan, 3.0]])
#     matching_dict_dataset.__dict__["Flux"] = np.array([[1.0, np.nan, 3.0]])

#     assert dict_dataset == matching_dict_dataset


# def test_eq_with_list_variables(dict_dataset: DataSet, matching_dict_dataset: DataSet) -> None:
#     """Equality should compare list-backed variables element-wise."""
#     dict_dataset.__dict__["datetime"] = [datetime(2013, 1, 1, tzinfo=timezone.utc)]
#     matching_dict_dataset.__dict__["datetime"] = [datetime(2013, 1, 1, tzinfo=timezone.utc)]

#     assert dict_dataset == matching_dict_dataset


# def test_eq_with_different_values(dict_dataset: DataSet, matching_dict_dataset: DataSet) -> None:
#     """Equality should fail when values differ."""
#     dict_dataset.__dict__["Flux"] = np.array([[1.0, 2.0, 3.0]])
#     matching_dict_dataset.__dict__["Flux"] = np.array([[4.0, 5.0, 6.0]])

#     assert dict_dataset != matching_dict_dataset
