# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: INP001, T201

"""Regenerate the satellite availability tables of recipes from their remote archives.

A satellite availability table records, per satellite, the day intervals in which the remote
archive has files of a recipe's product (see :mod:`el_paso.cli.availability`). This script builds such
a table from the archive's directory listings and writes it as a TOML file, with its REUSE header,
into the recipe package::

    python scripts/generate_satellite_availability.py lanl-geogrid-mpa lanl-geogrid-sopa-esp

Each product repeats the download URL and file name stem of its recipes (the stem with a
``{satellite}`` field); keep them in sync when a recipe's remote location changes. For every day they
are filled with :func:`el_paso.utils.fill_str_template_with_time` and matched against the directory
listing, as :func:`el_paso.download` does.

Each directory listing is fetched once (e.g. once per year directory), not once per day.
Consecutive days are merged into intervals, and gaps of up to ``--max-gap-days`` missing days do not
split an interval, so the table records whether a satellite was populated in a period rather than
every outage.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import tomllib
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import get_args

import requests

from el_paso.recipes.poes import POESSatellite
from el_paso.utils import fill_str_template_with_time

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MAX_GAP_DAYS = 5
"""Missing days bridged without splitting an interval, unless ``--max-gap-days`` says otherwise."""

_HTTP_NOT_FOUND = 404


@dataclasses.dataclass(frozen=True)
class Product:
    """A remote product whose availability is tabulated. Recipes sharing a product share its table.

    Attributes:
        download_url (str): The recipes' download URL, with time placeholders (e.g. ``.../YYYY``).
            It may contain a ``{satellite}`` field if every satellite has its own directory, which
            requires `satellites`.
        file_name_stem (str): The recipes' file name regex, with time placeholders and a
            ``{satellite}`` field.
        first_day (date): First day to look for files. The last day defaults to today.
        output (Path): The TOML file to write, which the recipe modules declare as
            ``SATELLITE_AVAILABILITY``.
        satellite_pattern (str): Regex for the ``{satellite}`` field of `file_name_stem`, matching
            any satellite in one listing. Not used with `satellites`.
        satellites (tuple[str, ...]): The satellites to look for one by one, filling ``{satellite}``
            in `download_url` and `file_name_stem` with each. Empty to match `satellite_pattern`
            instead.
    """

    download_url: str
    file_name_stem: str
    first_day: date
    output: Path
    satellite_pattern: str = ""
    satellites: tuple[str, ...] = ()


_LANL_GEOGRID_URL = (
    "https://www.ngdc.noaa.gov/stp/space-weather/satellite-data/satellite-systems/lanl_geogrid/data/LowRes/YYYY"
)
_LANL_GEOGRID_DIR = REPO_ROOT / "el_paso" / "recipes" / "lanl_geogrid"
_POES_SATELLITES = get_args(POESSatellite)

PRODUCTS: dict[str, Product] = {
    "lanl-geogrid-mpa": Product(
        download_url=_LANL_GEOGRID_URL,
        file_name_stem=r"YYYYMMDD_LANL_GeoGrid_{satellite}_LowRes_MPA_v\d+\.\d+\.\d+\.txt",
        satellite_pattern=r"\d{3}",
        first_day=date(2008, 1, 1),
        output=_LANL_GEOGRID_DIR / "mpa_availability.toml",
    ),
    "lanl-geogrid-sopa-esp": Product(
        download_url=_LANL_GEOGRID_URL,
        file_name_stem=r"YYYYMMDD_LANL_GeoGrid_{satellite}_LowRes_SOPA_ESP_v\d+\.\d+\.\d+\.txt",
        satellite_pattern=r"\d{3}",
        first_day=date(2008, 1, 1),
        output=_LANL_GEOGRID_DIR / "sopa_esp_availability.toml",
    ),
    "poes-sem2-fluxes-2sec": Product(
        download_url="https://spdf.gsfc.nasa.gov/pub/data/noaa/{satellite}/sem2_fluxes-2sec/YYYY/",
        file_name_stem="{satellite}_poes-sem2_fluxes-2sec_YYYYMMDD_.{3}.cdf",
        satellites=_POES_SATELLITES,
        first_day=date(1998, 1, 1),
        output=REPO_ROOT / "el_paso" / "recipes" / "poes" / "sem2_fluxes_2sec_availability.toml",
    ),
}
"""Products this script can tabulate, by the name given on the command line."""

_REUSE_HEADER = """\
# SPDX-FileCopyrightText: {year} GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0
"""


class ListingCache:
    """Fetches each directory listing once, however many days share it.

    Before a listing is fetched, its parent directory is listed (once): a directory the parent
    does not link to is taken as empty without a request, so that looking up a satellite's years
    before its launch or after its end costs one request, not one per year.
    """

    def __init__(self, session: requests.Session | None = None) -> None:
        """Create an empty cache, reusing `session` for every request."""
        self._session = session or requests.Session()
        self._listings: dict[str, str | None] = {}

    def _fetch(self, url: str) -> str | None:
        """Return the listing at `url`, or ``None`` if the server has none (404).

        Raises:
            requests.HTTPError: If the server answers with an error other than 404.
        """
        if url not in self._listings:
            print(f"  listing {url}", file=sys.stderr)
            response = self._session.get(url, timeout=120)
            if response.status_code == _HTTP_NOT_FOUND:
                self._listings[url] = None
            else:
                response.raise_for_status()
                self._listings[url] = response.text
        return self._listings[url]

    def get(self, url: str) -> str:
        """Return the listing at `url`, or an empty string if the server has none.

        Raises:
            requests.HTTPError: If the server answers with an error other than 404.
        """
        parent, _, name = url.rstrip("/").rpartition("/")
        parent_listing = self._fetch(f"{parent}/")
        if parent_listing is None or (
            f'href="{name}/"' not in parent_listing and f'href="{name}"' not in parent_listing
        ):
            return ""
        return self._fetch(url) or ""


def find_days(
    download_url: str,
    file_name_stem: str,
    satellite_pattern: str,
    first_day: date,
    last_day: date,
    listings: ListingCache,
) -> dict[str, set[date]]:
    """Collect the days with a file of the product, per satellite.

    Args:
        download_url (str): See `Product.download_url`, with ``{satellite}`` filled in if it has one.
        file_name_stem (str): See `Product.file_name_stem`.
        satellite_pattern (str): Regex for the ``{satellite}`` field of `file_name_stem`.
        first_day (date): First day to look at.
        last_day (date): Last day to look at, inclusive.
        listings (ListingCache): Where the directory listings come from.

    Returns:
        dict[str, set[date]]: The days with at least one matching file, per satellite.
    """
    satellite_group = f"(?P<satellite>{satellite_pattern})"
    days: dict[str, set[date]] = defaultdict(set)

    day = first_day
    while day <= last_day:
        time = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        listing = listings.get(fill_str_template_with_time(download_url, time))
        if listing:
            stem = fill_str_template_with_time(file_name_stem, time).replace("{satellite}", satellite_group)
            # the same match as `ep.download`, with the satellite as a group
            for match in re.finditer(stem, listing, flags=re.IGNORECASE):
                days[match["satellite"]].add(day)
        day += timedelta(days=1)

    return days


def merge_days(days: set[date], max_gap_days: int) -> list[tuple[date, date]]:
    """Merge days into closed intervals, bridging up to `max_gap_days` missing days.

    Args:
        days (set[date]): Days with data.
        max_gap_days (int): Longest run of missing days that does not split an interval.

    Returns:
        list[tuple[date, date]]: Sorted, non-overlapping ``(first, last)`` day intervals.
    """
    intervals: list[tuple[date, date]] = []
    for day in sorted(days):
        if intervals and (day - intervals[-1][1]).days - 1 <= max_gap_days:
            intervals[-1] = (intervals[-1][0], day)
        else:
            intervals.append((day, day))
    return intervals


def build_table(
    product: Product,
    *,
    max_gap_days: int,
    first_day: date | None = None,
    last_day: date | None = None,
    listings: ListingCache | None = None,
) -> dict[str, list[tuple[date, date]]]:
    """Build a product's availability table from the remote archive.

    Args:
        product (Product): The product to tabulate.
        max_gap_days (int): See `merge_days`.
        first_day (date | None): First day to look at. Defaults to the product's first day.
        last_day (date | None): Last day to look at. Defaults to today.
        listings (ListingCache | None): Listings to reuse between products.

    Returns:
        dict[str, list[tuple[date, date]]]: Intervals per satellite, satellites sorted; satellites
        without any file are left out.
    """
    if product.satellites:
        targets = [
            (product.download_url.replace("{satellite}", satellite), re.escape(satellite))
            for satellite in product.satellites
        ]
    elif "{satellite}" in product.download_url:
        msg = "A download URL with a {satellite} field needs the satellites to look for."
        raise ValueError(msg)
    else:
        targets = [(product.download_url, product.satellite_pattern)]

    listings = listings or ListingCache()
    days: dict[str, set[date]] = defaultdict(set)
    for download_url, satellite_pattern in targets:
        found = find_days(
            download_url,
            product.file_name_stem,
            satellite_pattern,
            first_day or product.first_day,
            last_day or datetime.now(timezone.utc).date(),
            listings,
        )
        for satellite, satellite_days in found.items():
            days[satellite] |= satellite_days
    return {satellite: merge_days(days[satellite], max_gap_days) for satellite in sorted(days)}


def write_table(product: Product, table: dict[str, list[tuple[date, date]]], *, max_gap_days: int) -> Path:
    """Write a table as TOML, with a REUSE header.

    Returns:
        Path: The file written.
    """
    today = datetime.now(timezone.utc).date()
    # JSON strings are valid TOML basic strings; one interval per line, so that a regenerated
    # table reviews as a readable diff
    lines = [
        _REUSE_HEADER.format(year=today.year),
        f"download_url = {json.dumps(product.download_url)}",
        f"max_gap_days = {max_gap_days}",
        f"generated = {today.isoformat()}",
        "",
        "[satellites]",
    ]
    for satellite, intervals in table.items():
        lines.append(f"{json.dumps(satellite)} = [")
        lines.extend(f"  [{first.isoformat()}, {last.isoformat()}]," for first, last in intervals)
        lines.append("]")
    text = "\n".join(lines) + "\n"
    tomllib.loads(text)  # guard against a formatting slip

    product.output.write_text(text, encoding="utf-8")
    return product.output


def main(argv: list[str] | None = None) -> None:
    """Regenerate the availability tables of the products named on the command line."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", maxsplit=1)[0])
    parser.add_argument("products", nargs="+", choices=sorted(PRODUCTS), help="Products to regenerate.")
    parser.add_argument(
        "--max-gap-days",
        type=int,
        default=DEFAULT_MAX_GAP_DAYS,
        help=f"Missing days bridged without splitting an interval (default {DEFAULT_MAX_GAP_DAYS}).",
    )
    parser.add_argument("--first-day", type=date.fromisoformat, help="First day (default: the product's first day).")
    parser.add_argument("--last-day", type=date.fromisoformat, help="Last day (default: today).")
    args = parser.parse_args(argv)

    listings = ListingCache()
    for name in args.products:
        product = PRODUCTS[name]
        print(f"{name}:", file=sys.stderr)
        table = build_table(
            product,
            max_gap_days=args.max_gap_days,
            first_day=args.first_day,
            last_day=args.last_day,
            listings=listings,
        )
        path = write_table(product, table, max_gap_days=args.max_gap_days)
        total = sum(len(intervals) for intervals in table.values())
        print(f"  wrote {len(table)} satellites, {total} intervals to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
