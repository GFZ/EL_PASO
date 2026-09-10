# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from el_paso.processing import calculate_geo_coords_from_omm

_ISS_OMM_LINE = {
    "OBJECT_NAME": "ISS (ZARYA)",
    "OBJECT_ID": "1998-067A",
    "EPOCH": "2024-01-02T10:04:56.890560",
    "MEAN_MOTION": "15.50006585",
    "ECCENTRICITY": "0.00033780",
    "INCLINATION": "51.6403",
    "RA_OF_ASC_NODE": "61.6579",
    "ARG_OF_PERICENTER": "343.5198",
    "MEAN_ANOMALY": "16.5681",
    "EPHEMERIS_TYPE": "0",
    "CLASSIFICATION_TYPE": "U",
    "NORAD_CAT_ID": "25544",
    "ELEMENT_SET_NO": "999",
    "REV_AT_EPOCH": "43269",
    "BSTAR": "0.00029671000000",
    "MEAN_MOTION_DOT": "0.00016518",
    "MEAN_MOTION_DDOT": "0.0000000000000",
}

_CSV_HEADER = ",".join(_ISS_OMM_LINE.keys())
_TARGET_TIME = datetime(2024, 1, 2, 12, tzinfo=timezone.utc)


def _write_omm_csv(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [_CSV_HEADER, *(",".join(row.values()) for row in rows)]
    path.write_text("\n".join(lines) + "\n")


def test_calculate_geo_coords_from_omm_dict_matches_file_single_record(tmp_path: Path) -> None:
    """The dict and single-record-file overloads should produce identical results."""
    csv_path = tmp_path / "single.csv"
    _write_omm_csv(csv_path, [_ISS_OMM_LINE])

    target_times = [_TARGET_TIME]

    from_dict = calculate_geo_coords_from_omm(_ISS_OMM_LINE, target_times)
    from_file = calculate_geo_coords_from_omm(csv_path, target_times)

    np.testing.assert_array_equal(from_dict.get_data(), from_file.get_data())
    assert csv_path.name in from_file.metadata.source_files
    assert from_dict.metadata.source_files == []


def test_calculate_geo_coords_from_omm_file_accepts_str_path(tmp_path: Path) -> None:
    csv_path = tmp_path / "single.csv"
    _write_omm_csv(csv_path, [_ISS_OMM_LINE])

    result = calculate_geo_coords_from_omm(str(csv_path), [_TARGET_TIME])
    assert result.get_data().shape == (1, 3)


def test_calculate_geo_coords_from_omm_file_uses_first_record(tmp_path: Path) -> None:
    """`download_omm` writes one record per file; if a file ever holds more, the first is used."""
    first = dict(_ISS_OMM_LINE, EPOCH="2024-01-01T00:00:00.000000")
    second = dict(_ISS_OMM_LINE, EPOCH="2024-01-05T00:00:00.000000")

    csv_path = tmp_path / "multi.csv"
    _write_omm_csv(csv_path, [first, second])

    from_file = calculate_geo_coords_from_omm(csv_path, [_TARGET_TIME])
    from_dict = calculate_geo_coords_from_omm(first, [_TARGET_TIME])

    np.testing.assert_array_equal(from_file.get_data(), from_dict.get_data())


def test_calculate_geo_coords_from_omm_empty_file_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text(_CSV_HEADER + "\n")

    with pytest.raises(ValueError, match="is empty"):
        calculate_geo_coords_from_omm(csv_path, [_TARGET_TIME])


@pytest.mark.basic
def test_calculate_geo_coords_from_omm_values_are_reasonable() -> None:
    """ISS orbits ~6800 km from Earth's center."""
    result = calculate_geo_coords_from_omm(_ISS_OMM_LINE, [_TARGET_TIME])
    distance = np.linalg.norm(result.get_data()[0])
    assert 6000 < distance < 8000
