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

from el_paso.processing import TLE


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
class TestTLEInitialization:
    """Test TLE class initialization and parsing."""

    def test_tle_initialization(self, sample_tle_file: str):
        """Test that TLE object initializes correctly."""
        tle = TLE(sample_tle_file)
        assert tle.tle_filename == sample_tle_file
        assert isinstance(tle.tle_data, list)
        assert len(tle.tle_data) == 2  # noqa: PLR2004
        assert tle.satellite_name == "25544C"

    def test_tle_data_structure(self, sample_tle_file: str):
        """Test that tle_data contains proper tuple pairs."""
        tle = TLE(sample_tle_file)
        for tle_pair in tle.tle_data:
            assert isinstance(tle_pair, tuple)
            assert len(tle_pair) == 2  # noqa: PLR2004
            assert isinstance(tle_pair[0], str)
            assert isinstance(tle_pair[1], str)

    def test_tle_frozen_dataclass(self, sample_tle_file: str):
        """Test that TLE is a frozen dataclass."""
        tle = TLE(sample_tle_file)
        with pytest.raises(Exception):  # noqa: B017, PT011
            tle.satellite_name = "CHANGED"


class TestReadTLEFile:
    """Test the _read_tle_file method."""

    def test_read_tle_file(self, sample_tle_file: str):
        """Test reading TLE file."""
        tle = TLE(sample_tle_file)
        tle_data, satellite_name = tle._read_tle_file(Path(sample_tle_file))  # noqa: SLF001

        assert len(tle_data) == 2  # noqa: PLR2004
        assert satellite_name == "25544C"
        assert tle_data[0][0].startswith("1")
        assert tle_data[0][1].startswith("2")

    def test_read_tle_file_with_path_object(self, sample_tle_file: str):
        """Test _read_tle_file with Path object."""
        tle = TLE(sample_tle_file)
        tle_data, _ = tle._read_tle_file(Path(sample_tle_file))  # noqa: SLF001
        assert isinstance(tle_data, list)


class TestGetTLETime:
    """Test the get_tle_time method."""

    def test_get_tle_time(self, sample_tle_file: str):
        """Test TLE time extraction."""
        tle = TLE(sample_tle_file)
        tle_times = tle.tle_time_list

        assert len(tle_times) == 2  # noqa: PLR2004
        assert all(isinstance(t, datetime) for t in tle_times)
        assert all(t.tzinfo == timezone.utc for t in tle_times)

    def test_tle_time_values(self, sample_tle_file: str):
        """Test that extracted times are reasonable."""
        tle = TLE(sample_tle_file)
        tle_times = tle.tle_time_list

        # Times should be in 2021
        assert all(2026 <= t.year <= 2026 for t in tle_times)  # noqa: PLR2004
        assert all(t.month >= 1 for t in tle_times)


class TestCalculateGeoCoords:
    """Test the calculate_geo_coords method."""

    def test_calculate_geo_coords_shape(self, sample_tle_file: str):
        """Test that geo coordinates have correct shape."""
        tle = TLE(sample_tle_file)
        coords = tle.calculate_geo_coords()

        assert isinstance(coords, np.ndarray)
        assert coords.shape == (2, 3)
        assert coords.dtype == np.float64

    def test_calculate_geo_coords_values(self, sample_tle_file: str):
        """Test that geo coordinates are reasonable."""
        tle = TLE(sample_tle_file)
        coords = tle.calculate_geo_coords()

        # 25544C orbits at ~7000 km from Earth's center
        distances = np.linalg.norm(coords, axis=1)
        assert all(6000 < d < 8000 for d in distances)  # noqa: PLR2004

    def test_calculate_geo_coords_no_inf(self, sample_tle_file: str):
        """Test that coordinates don't contain infinite values."""
        tle = TLE(sample_tle_file)
        coords = tle.calculate_geo_coords()

        assert not np.isinf(coords).any()

    def test_calculate_geo_coords_with_nan_logging(self, caplog, monkeypatch, sample_tle_file: str):  # noqa: ANN001
        """Test NaN warning logging in geo coordinates."""
        tle = TLE(sample_tle_file)

        original_isnan = np.isnan

        def fake_isnan(values: np.ndarray) -> np.ndarray:
            isnan_values = original_isnan(values)
            isnan_values[0, 0] = True
            return isnan_values

        monkeypatch.setattr("el_paso.processing.tle.np.isnan", fake_isnan)

        with caplog.at_level(logging.WARNING):
            coords = tle.calculate_geo_coords()

        assert isinstance(coords, np.ndarray)
        assert any("NaN values found in GEO coordinates" in record.message for record in caplog.records)
