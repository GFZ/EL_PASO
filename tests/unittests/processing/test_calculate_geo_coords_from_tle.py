# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

import logging
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pytest
from el_paso.processing import calculate_geo_coords_from_tle


@pytest.fixture
def sample_tle_file():
    """Create a temporary TLE file for testing."""
    tle_content = """1 25544C 98067A   26070.00000000  .00112600  00000+0  20674-2 0    75
2 25544  51.6320  65.8754 0008329 179.3321 194.2182 15.48545888    16
1 25544C 98067A   26070.25000000  .00035963  00000+0  66012-3 0    86
2 25544  51.6323  64.6423 0008037 180.0733 148.1113 15.48574346    14
"""
    with NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write(tle_content)
        temp_path = f.name

    yield temp_path

    Path(temp_path).unlink()


@pytest.mark.basic
class TestCalculateGeoCoordsReturn:
    """Test the return values of calculate_geo_coords."""

    def test_returns_tuple(self, sample_tle_file: str):
        """Test that the function returns a tuple of three elements."""
        result = calculate_geo_coords_from_tle(sample_tle_file)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_satellite_name(self, sample_tle_file: str):
        """Test that the satellite name is parsed correctly."""
        satellite_name, _, _ = calculate_geo_coords_from_tle(sample_tle_file)
        assert satellite_name == "25544C"

    def test_tle_times_length_and_type(self, sample_tle_file: str):
        """Test that TLE times are a list of UTC datetimes."""
        _, tle_times, _ = calculate_geo_coords_from_tle(sample_tle_file)
        assert len(tle_times) == 2
        assert all(isinstance(t, datetime) for t in tle_times)
        assert all(t.tzinfo == timezone.utc for t in tle_times)

    def test_tle_time_values(self, sample_tle_file: str):
        """Test that extracted times are in the expected year."""
        _, tle_times, _ = calculate_geo_coords_from_tle(sample_tle_file)
        assert all(t.year == 2026 for t in tle_times)
        assert all(t.month >= 1 for t in tle_times)

    def test_geo_coords_shape(self, sample_tle_file: str):
        """Test that geo coordinates have correct shape and dtype."""
        _, _, coords_var = calculate_geo_coords_from_tle(sample_tle_file)
        coords = coords_var.get_data()
        assert coords.shape == (2, 3)
        assert coords.dtype == np.float64

    def test_geo_coords_values(self, sample_tle_file: str):
        """Test that geo coordinates are reasonable (ISS orbits ~7000 km from Earth's center)."""
        _, _, coords_var = calculate_geo_coords_from_tle(sample_tle_file)
        distances = np.linalg.norm(coords_var.get_data(), axis=1)
        assert all(6000 < d < 8000 for d in distances)

    def test_geo_coords_no_inf(self, sample_tle_file: str):
        """Test that coordinates don't contain infinite values."""
        _, _, coords_var = calculate_geo_coords_from_tle(sample_tle_file)
        assert not np.isinf(coords_var.get_data()).any()

    def test_nan_warning_logging(self, caplog, monkeypatch, sample_tle_file: str):  # noqa: ANN001
        """Test that NaN values in coordinates trigger a warning log."""
        original_isnan = np.isnan

        def fake_isnan(values: np.ndarray) -> np.ndarray:
            isnan_values = original_isnan(values)
            isnan_values[0, 0] = True
            return isnan_values

        monkeypatch.setattr("numpy.isnan", fake_isnan)

        with caplog.at_level(logging.WARNING):
            calculate_geo_coords_from_tle(sample_tle_file)

        assert any("NaN values found in GEO coordinates" in record.message for record in caplog.records)
