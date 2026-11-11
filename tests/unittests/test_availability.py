# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for satellite availability tables (`el_paso.cli.availability`) and their generator script."""

from __future__ import annotations

import types
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import el_paso as ep
import pytest
from el_paso.cli.availability import (
    AvailabilityTable,
    check_availability_table,
    load_availability_table,
)
from scripts.generate_satellite_availability import (
    ListingCache,
    Product,
    build_table,
    find_days,
    merge_days,
    write_table,
)

TABLE: AvailabilityTable = {
    "a": [(date(2020, 1, 1), date(2020, 1, 31)), (date(2020, 6, 1), date(2020, 6, 30))],
    "b": [(date(2020, 3, 1), date(2020, 3, 31))],
}


def table_recipe(
    start_time: datetime,
    end_time: datetime,
    satellite: Literal["a", "b"] = "a",
) -> None:
    """Stub recipe whose module declares `TABLE`.

    Args:
        start_time (datetime): Start of the range.
        end_time (datetime): End of the range.
        satellite (Literal["a", "b"]): Which satellite.
    """


def recipe_module(**attributes: object) -> types.ModuleType:
    """A stand-in recipe module with the given module-level attributes."""
    module = types.ModuleType("stub_recipe_module")
    module.__dict__.update(attributes)
    return module


# --- satellite_availability -------------------------------------------------------------------------------------


@pytest.mark.minimal
@pytest.mark.basic
def test_query_returns_only_satellites_with_data_and_their_overlapping_intervals() -> None:
    coverage = ep.satellite_availability(TABLE, date(2020, 1, 15), date(2020, 3, 5))

    assert coverage == {
        "a": [(date(2020, 1, 1), date(2020, 1, 31))],
        "b": [(date(2020, 3, 1), date(2020, 3, 31))],
    }


@pytest.mark.minimal
@pytest.mark.basic
@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (date(2019, 12, 1), date(2020, 1, 1), ["a"]),  # ends on the first day with data
        (date(2020, 1, 31), date(2020, 2, 15), ["a"]),  # starts on the last day with data
        (date(2019, 12, 1), date(2019, 12, 31), []),  # ends the day before
        (date(2020, 2, 1), date(2020, 2, 29), []),  # in a gap between intervals
    ],
)
def test_query_overlap_at_the_edges(start: date, end: date, expected: list[str]) -> None:
    coverage = ep.satellite_availability(TABLE, start, end)

    assert coverage is not None
    assert list(coverage) == expected


@pytest.mark.minimal
@pytest.mark.basic
def test_query_compares_datetimes_by_utc_day() -> None:
    # 2020-02-01T00:30 at UTC+01:00 is still 2020-01-31 in UTC
    plus_one = timezone(timedelta(hours=1))
    coverage = ep.satellite_availability(
        TABLE, datetime(2020, 2, 1, 0, 30, tzinfo=plus_one), datetime(2020, 2, 2, tzinfo=plus_one)
    )

    assert coverage is not None
    assert list(coverage) == ["a"]


@pytest.mark.minimal
@pytest.mark.basic
def test_no_table_is_distinct_from_no_data() -> None:
    without_table = recipe_module()
    with_table = recipe_module(SATELLITE_AVAILABILITY=TABLE)

    assert ep.satellite_availability(without_table, date(2020, 2, 1), date(2020, 2, 2)) is None
    assert ep.satellite_availability(with_table, date(2020, 2, 1), date(2020, 2, 2)) == {}


@pytest.mark.minimal
@pytest.mark.basic
def test_query_accepts_a_recipe_function() -> None:
    coverage = ep.satellite_availability(
        ep.recipes.lanl_geogrid.process_lanl_geogrid_mpa_electrons, date(2017, 4, 1), date(2017, 4, 30)
    )

    assert coverage is not None
    assert sorted(coverage) == ["010", "070", "130", "190", "310"]


@pytest.mark.minimal
@pytest.mark.basic
def test_recipes_sharing_a_product_share_a_table() -> None:
    lanl = ep.recipes.lanl_geogrid
    april = (date(2017, 4, 1), date(2017, 4, 30))

    assert ep.satellite_availability(lanl.process_lanl_geogrid_mpa_electrons, *april) == ep.satellite_availability(
        lanl.process_lanl_geogrid_mpa_ions, *april
    )
    assert ep.satellite_availability(lanl.process_lanl_geogrid_sopa_esp_electrons, *april) == ep.satellite_availability(
        lanl.process_lanl_geogrid_sopa_protons, *april
    )


# --- load_availability_table ------------------------------------------------------------------------------------


@pytest.mark.minimal
@pytest.mark.basic
def test_load_toml_table(tmp_path: Path) -> None:
    table_file = tmp_path / "availability.toml"
    table_file.write_text(
        """max_gap_days = 5

[satellites]
"a" = [
  [2020-01-01, 2020-01-31],
  [2020-06-01, 2020-06-30],
]
"b" = [
  [2020-03-01, 2020-03-31],
]
""",
        encoding="utf-8",
    )

    assert load_availability_table(table_file) == TABLE


@pytest.mark.minimal
@pytest.mark.basic
def test_load_rejects_a_file_without_satellites(tmp_path: Path) -> None:
    table_file = tmp_path / "availability.toml"
    table_file.write_text("max_gap_days = 5\n", encoding="utf-8")

    with pytest.raises(TypeError, match=r"\[satellites\]"):
        load_availability_table(table_file)


@pytest.mark.minimal
@pytest.mark.basic
@pytest.mark.parametrize(
    ("table", "error"),
    [
        ({"a": "2020-01-01"}, TypeError),
        ({"a": [(date(2020, 1, 1),)]}, TypeError),
        ({"a": [(datetime(2020, 1, 1, tzinfo=timezone.utc), date(2020, 1, 2))]}, TypeError),
        ({"a": [("2020-01-01", "not a date")]}, ValueError),
    ],
)
def test_load_rejects_malformed_intervals(table: dict, error: type[Exception]) -> None:
    with pytest.raises(error, match="satellite 'a'"):
        load_availability_table(table)


# --- check_availability_table -----------------------------------------------------------------------------------


@pytest.mark.minimal
@pytest.mark.basic
def test_check_accepts_a_valid_table() -> None:
    check_availability_table(table_recipe, TABLE)


@pytest.mark.minimal
@pytest.mark.basic
def test_check_reports_every_problem() -> None:
    table = {
        "c": [(date(2020, 1, 1), date(2020, 1, 2))],  # not a satellite of the recipe
        "a": [
            (date(2020, 1, 10), date(2020, 1, 5)),  # ends before it starts
            (date(2020, 3, 1), date(2020, 3, 31)),
            (date(2020, 3, 15), date(2020, 4, 15)),  # overlaps the one before
        ],
        "b": [(date(2020, 6, 1), date(2020, 6, 30)), (date(2020, 1, 1), date(2020, 1, 31))],  # unsorted
    }

    with pytest.raises(ValueError, match="Invalid availability table of table_recipe") as error:
        check_availability_table(table_recipe, table)

    message = str(error.value)
    assert "'c': not a value of the satellite parameter" in message
    assert "'a': interval 2020-01-10 to 2020-01-05 ends before it starts" in message
    assert "'a': interval 2020-03-15 to 2020-04-15 is unsorted or overlaps" in message
    assert "'b': interval 2020-01-01 to 2020-01-31 is unsorted or overlaps" in message


@pytest.mark.minimal
@pytest.mark.basic
def test_check_requires_a_table() -> None:
    with pytest.raises(ValueError, match="declares no SATELLITE_AVAILABILITY"):
        check_availability_table(table_recipe)


@pytest.mark.minimal
@pytest.mark.basic
def test_check_requires_a_satellite_literal() -> None:
    def no_satellite(start_time: datetime, end_time: datetime) -> None:
        """Recipe without a satellite parameter."""

    with pytest.raises(TypeError, match="no 'satellite' parameter"):
        check_availability_table(no_satellite, TABLE)


# --- generator script -------------------------------------------------------------------------------------------


@pytest.mark.minimal
@pytest.mark.basic
def test_merge_days_bridges_short_gaps_only() -> None:
    days = {date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 5), date(2020, 1, 20)}

    # two missing days (3rd, 4th) are bridged, fourteen (6th to 19th) are not
    assert merge_days(days, max_gap_days=2) == [
        (date(2020, 1, 1), date(2020, 1, 5)),
        (date(2020, 1, 20), date(2020, 1, 20)),
    ]
    assert merge_days(days, max_gap_days=1) == [
        (date(2020, 1, 1), date(2020, 1, 2)),
        (date(2020, 1, 5), date(2020, 1, 5)),
        (date(2020, 1, 20), date(2020, 1, 20)),
    ]
    assert merge_days(set(), max_gap_days=5) == []


class FakeListings(ListingCache):
    """Serves directory listings from a dict and records which ones were requested."""

    def __init__(self, listings: dict[str, str]) -> None:
        super().__init__()
        self._fake = listings
        self.fetched: list[str] = []

    def _request(self, url: str) -> str | None:
        self.fetched.append(url)
        return self._fake.get(url)


def _listing(*names: str) -> str:
    return "".join(f'<a href="{name}">{name}</a>\n' for name in names)


@pytest.mark.minimal
@pytest.mark.basic
def test_find_days_matches_the_recipe_pattern_and_fetches_each_listing_once() -> None:
    listings = FakeListings(
        {
            "https://archive/": _listing("2017/"),
            "https://archive/2017": _listing(
                "20170401_X_010_MPA_v1.0.0.txt",
                "20170401_X_070_MPA_v1.0.0.txt",
                "20170401_X_070_SOPA_v1.0.0.txt",  # another product
                "20170402_X_010_MPA_v1.0.1.txt",
            ),
        }
    )

    days = find_days(
        "https://archive/YYYY",
        r"YYYYMMDD_X_{satellite}_MPA_v\d+\.\d+\.\d+\.txt",
        r"\d{3}",
        date(2016, 12, 31),
        date(2017, 4, 3),
        listings,
    )

    assert days == {"010": {date(2017, 4, 1), date(2017, 4, 2)}, "070": {date(2017, 4, 1)}}
    # the parent is listed once; 2016 is not linked from it, so it is never requested
    assert listings.fetched.count("https://archive/") == 1
    assert listings.fetched.count("https://archive/2017") == 1
    assert "https://archive/2016" not in listings.fetched


@pytest.mark.minimal
@pytest.mark.basic
def test_build_table_fills_the_satellite_into_per_satellite_urls(tmp_path: Path) -> None:
    listings = FakeListings(
        {
            "https://archive/sat1/": _listing("2020/"),
            "https://archive/sat1/2020/": _listing("sat1_20200101_v01.cdf", "sat1_20200102_v01.cdf"),
            "https://archive/sat2/": _listing("2019/"),
        }
    )
    # `.{3}` is a regex quantifier, not a format field
    product = Product(
        download_url="https://archive/{satellite}/YYYY/",
        file_name_stem="{satellite}_YYYYMMDD_.{3}.cdf",
        satellites=("sat1", "sat2"),
        first_day=date(2020, 1, 1),
        output=tmp_path / "availability.toml",
    )

    table = build_table(product, max_gap_days=5, last_day=date(2020, 1, 5), listings=listings)

    assert table == {"sat1": [(date(2020, 1, 1), date(2020, 1, 2))]}


@pytest.mark.minimal
@pytest.mark.basic
def test_written_table_round_trips_and_carries_a_reuse_header(tmp_path: Path) -> None:
    product = Product(
        download_url="https://archive/YYYY",
        file_name_stem="unused",
        satellite_pattern=r"\d{3}",
        first_day=date(2020, 1, 1),
        output=tmp_path / "availability.toml",
    )
    table = {"010": [(date(2020, 1, 1), date(2020, 1, 31))], "070": [(date(2020, 2, 1), date(2020, 2, 2))]}

    path = write_table(product, table, max_gap_days=5)

    text = path.read_text(encoding="utf-8")
    # REUSE-IgnoreStart
    assert text.startswith("# SPDX-FileCopyrightText: ")
    assert "# SPDX-License-Identifier: Apache-2.0" in text
    # REUSE-IgnoreEnd
    assert load_availability_table(path) == table
