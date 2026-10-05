# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import el_paso as ep
import numpy as np
import pytest
from astropy import units as u
from el_paso.saving_strategy import OutputFile, SavingStrategy

if TYPE_CHECKING:
    from datetime import datetime

    from el_paso.data_standard import DataStandard
    from el_paso.typing import InternalName

# Concrete stub so the ABC can be instantiated in tests.
_GFZ = ep.data_standards.GFZStandard()


class _StubStrategy(SavingStrategy):
    """Minimal concrete subclass that satisfies all abstract methods."""

    def __init__(self, data_standard: DataStandard = _GFZ, output_files: list[OutputFile] | None = None) -> None:
        self.data_standard = data_standard
        self.output_files = output_files or []
        self.satellite = "TEST"
        self.mission = "TEST"
        self.instrument = "TEST"
        self.mag_field = "T89"
        self.base_data_path = Path("/tmp")  # noqa: S108

    def get_time_intervals_to_save(self, start_time: datetime, end_time: datetime) -> list:
        return [(start_time, end_time)]

    def get_file_path(self, interval_start: datetime, interval_end: datetime, output_file: OutputFile) -> Path:  # noqa: ARG002
        return Path("/tmp/test.nc")  # noqa: S108

    def get_file_path_stem(self) -> Path:
        return Path("/tmp/test")  # noqa: S108

    def get_file_name_stem(self) -> str:
        return "test"


@pytest.mark.basic
def test_output_file_defaults() -> None:
    of = OutputFile(name="flux", names_to_save=["FEDU", "Epoch"])
    assert of.name == "flux"
    assert of.names_to_save == ["FEDU", "Epoch"]
    assert of.save_incomplete is False


@pytest.mark.basic
def test_output_file_save_incomplete_true() -> None:
    of = OutputFile(name="wave", names_to_save=["FEDU"], save_incomplete=True)
    assert of.save_incomplete is True


@pytest.mark.basic
def test_get_target_variables_skips_optional_output_with_no_content() -> None:
    """An optional output file whose only resolved name is the shared time axis must be skipped.

    This is what lets a saving strategy always register e.g. a "solar_wind_indices" output file
    without every caller having to opt in/out depending on whether it actually computed and saved
    any solar wind index variables alongside the rest of its data.
    """
    strategy = _StubStrategy()
    output_file = OutputFile(name="solar_wind_indices", names_to_save=["Epoch", "Kp", "Dst"], save_incomplete=True)
    variables_dict: dict[InternalName, ep.Variable] = {
        "Epoch": ep.Variable(original_unit=ep.units.posixtime, data=np.array([1.0, 2.0]))
    }

    result = strategy.get_target_variables(output_file, variables_dict, None, None, None)

    assert result is None


@pytest.mark.basic
def test_get_target_variables_keeps_optional_output_with_partial_content() -> None:
    strategy = _StubStrategy()
    output_file = OutputFile(name="solar_wind_indices", names_to_save=["Epoch", "Kp", "Dst"], save_incomplete=True)
    variables_dict: dict[InternalName, ep.Variable] = {
        "Epoch": ep.Variable(original_unit=ep.units.posixtime, data=np.array([1.0, 2.0])),
        "Kp": ep.Variable(original_unit=u.dimensionless_unscaled, data=np.array([2.0, 3.0])),
    }

    result = strategy.get_target_variables(output_file, variables_dict, None, None, None)

    assert result is not None
    assert set(result.keys()) == {"Epoch", "Kp", "Dst"}
    assert result["Dst"].get_data().size == 0


@pytest.mark.basic
def test_get_all_standard_names() -> None:
    strategy = _StubStrategy(
        output_files=[
            OutputFile(name="rb", names_to_save=["Epoch", "FEDU"]),
            OutputFile(name="pos", names_to_save=["Position"]),
        ]
    )
    names = strategy.get_all_standard_names()
    # GFZStandard standard names for those internal names
    assert "time" in names
    assert "xGEO" in names


@pytest.mark.basic
def test_merge_non_overlapping_appends() -> None:
    strategy = _StubStrategy()
    merged = strategy._merge_and_sort_data(
        {"time": np.array([1.0, 2.0, 3.0])},
        {"Epoch": np.array([4.0, 5.0])},
    )
    np.testing.assert_array_equal(merged["Epoch"], [1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.mark.basic
def test_merge_duplicate_epoch_replaced_by_new() -> None:
    strategy = _StubStrategy()
    # 2.0 and 3.0 overlap → new values replace old ones
    merged = strategy._merge_and_sort_data(
        {"time": np.array([1.0, 2.0, 3.0])},
        {"Epoch": np.array([2.0, 3.0, 4.0])},
    )
    np.testing.assert_array_equal(merged["Epoch"], [1.0, 2.0, 3.0, 4.0])


@pytest.mark.basic
def test_merge_result_sorted_ascending() -> None:
    strategy = _StubStrategy()
    # New data lies before existing → result must be sorted
    merged = strategy._merge_and_sort_data(
        {"time": np.array([3.0, 4.0, 5.0])},
        {"Epoch": np.array([1.0, 2.0])},
    )
    np.testing.assert_array_equal(merged["Epoch"], [1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.mark.basic
def test_merge_key_only_in_existing() -> None:
    strategy = _StubStrategy()
    merged = strategy._merge_and_sort_data(
        {"time": np.array([1.0, 2.0]), "geo_alt": np.array([100.0, 200.0])},
        {"Epoch": np.array([3.0])},
    )
    assert "Position_geo_alt" in merged


@pytest.mark.basic
def test_merge_key_only_in_new() -> None:
    strategy = _StubStrategy()
    merged = strategy._merge_and_sort_data(
        {"time": np.array([1.0, 2.0])},
        {"Epoch": np.array([3.0]), "Position_geo_alt": np.array([300.0])},
    )
    assert "Position_geo_alt" in merged
    np.testing.assert_array_equal(merged["Position_geo_alt"], [300.0])


@pytest.mark.basic
def test_merge_metadata_combined() -> None:
    strategy = _StubStrategy()
    merged = strategy._merge_and_sort_data(
        {"metadata": {"source": "old"}, "time": np.array([1.0])},
        {"metadata": {"instrument": "EPT"}, "Epoch": np.array([2.0])},
    )
    assert merged["metadata"]["source"] == "old"
    assert merged["metadata"]["instrument"] == "EPT"


@pytest.mark.basic
def test_merge_column_vector_normalized() -> None:
    """A (N, 1) Epoch column vector is treated as 1-D after normalization."""
    strategy = _StubStrategy()
    merged = strategy._merge_and_sort_data(
        {"time": np.array([[1.0], [2.0]])},  # shape (2, 1)
        {"Epoch": np.array([3.0])},
    )
    assert merged["Epoch"].ndim == 1
    np.testing.assert_array_equal(merged["Epoch"], [1.0, 2.0, 3.0])


@pytest.mark.basic
def test_repr_returns_string() -> None:
    strategy = _StubStrategy()
    result = repr(strategy)
    assert isinstance(result, str)
    assert "_StubStrategy" in result
