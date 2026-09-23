# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""MkDocs build hook that generates docs/getting_started/supported_versions.md.

The page documents which versions of Python and el_paso's core scientific
dependencies are currently required under SPEC-0 (scientific-python.org's
"Minimum Supported Dependencies" policy: drop Python 3 years after release,
drop core packages 2 years after release), rendered as a Mermaid gantt chart,
plus a table comparing those floors against what pyproject.toml actually pins.

Release dates are fetched live from PyPI and endoflife.date on every docs
build, so the page stays current without anyone having to hand-edit dates.
Every successful fetch is written to a local cache (.cache/spec0_releases.json,
gitignored); if a later build has no network access, it reuses that cache
instead of failing outright. There is no hand-maintained date table in this
file -- if a package has neither a live result nor a prior cache, generation
fails loudly rather than silently rendering stale numbers.
"""

from __future__ import annotations

import json
import re
import tomllib
from datetime import date, datetime
from itertools import zip_longest
from pathlib import Path

import requests
from dateutil.relativedelta import relativedelta

DOCS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = DOCS_DIR.parent
OUTPUT_FILE = DOCS_DIR / "getting_started" / "supported_versions.md"
PYPROJECT_FILE = REPO_ROOT / "pyproject.toml"

# Last-known-good release dates, refreshed automatically on every successful
# fetch. Gitignored ("!/.cache" -- see .gitignore) -- this is a build cache,
# not a source of truth anyone should hand-edit.
CACHE_FILE = REPO_ROOT / ".cache" / "spec0_releases.json"

PYTHON_SUPPORT_MONTHS = 36
CORE_PACKAGE_SUPPORT_MONTHS = 24
RELEASE_LOOKBACK_YEARS = 3


CORE_PACKAGES = ["numpy", "scipy", "matplotlib", "pandas", "xarray"]
LABELS = {
    "python": "Python",
    "numpy": "NumPy",
    "scipy": "SciPy",
    "matplotlib": "Matplotlib",
    "pandas": "pandas",
    "xarray": "xarray",
}

Segment = tuple[str, date, date]  # (version, floor_start, floor_end)


def read_cache() -> dict[str, list[tuple[str, str]]]:
    if not CACHE_FILE.exists():
        return {}
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def write_cache(cache: dict[str, list[tuple[str, str]]]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def fetch_pypi_feature_releases(package: str) -> list[tuple[str, date]]:
    """Feature releases (X.Y.0) of `package`, oldest first, with the date its
    first file was uploaded to PyPI (yanked releases still count -- a release
    that gets pulled after the fact was still the SPEC-0 clock start)."""
    resp = requests.get(f"https://pypi.org/pypi/{package}/json", timeout=15)
    resp.raise_for_status()
    releases = resp.json()["releases"]

    out: list[tuple[str, date]] = []
    for version, files in releases.items():
        if not re.fullmatch(r"\d+\.\d+\.0", version):
            continue
        upload_times = [f["upload_time_iso_8601"] for f in files if f.get("upload_time_iso_8601")]
        if not upload_times:
            continue
        released = datetime.fromisoformat(min(upload_times).replace("Z", "+00:00")).date()
        out.append((version, released))

    if not out:
        raise ValueError(f"no feature releases found for {package}")
    return _recent(out)


def fetch_python_feature_releases() -> list[tuple[str, date]]:
    """CPython feature releases, oldest first, from endoflife.date."""
    resp = requests.get("https://endoflife.date/api/python.json", timeout=15)
    resp.raise_for_status()
    out: list[tuple[str, date]] = []
    for row in resp.json():
        cycle, released = row.get("cycle"), row.get("releaseDate")
        if not cycle or not released:
            continue
        out.append((cycle, date.fromisoformat(released)))
    if not out:
        raise ValueError("no python feature releases found")
    return _recent(out)


def _recent(releases: list[tuple[str, date]]) -> list[tuple[str, date]]:
    releases = sorted(releases, key=lambda t: t[1])
    cutoff = date.today() - relativedelta(years=RELEASE_LOOKBACK_YEARS)
    trimmed = [r for r in releases if r[1] >= cutoff]
    return trimmed or releases[-1:]  # always keep at least the newest known release


def load_releases() -> dict[str, list[tuple[str, date]]]:
    """Fetch each package's feature-release dates live. On failure, fall back
    to whatever was cached from the most recent *successful* fetch -- never
    to a hand-written date. If neither is available, raise: a docs build
    that would otherwise silently render made-up numbers should fail instead.
    """
    cache = read_cache()
    updated_cache = dict(cache)
    releases: dict[str, list[tuple[str, date]]] = {}

    fetchers = {"python": fetch_python_feature_releases}
    fetchers.update({pkg: (lambda p=pkg: fetch_pypi_feature_releases(p)) for pkg in CORE_PACKAGES})

    for key, fetch in fetchers.items():
        try:
            fetched = fetch()
            updated_cache[key] = [[version, released.isoformat()] for version, released in fetched]
            releases[key] = fetched
        except Exception as exc:  # noqa: BLE001 - a fetch failure for one package must not abort the rest
            print(f"[supported_versions] live fetch failed for {key}: {exc}")
            if key not in cache:
                raise RuntimeError(
                    f"no live data and no cached data for {key} -- cannot generate supported_versions.md"
                ) from exc
            print(f"[supported_versions] using cached data for {key} from {CACHE_FILE}")
            releases[key] = [(version, date.fromisoformat(released)) for version, released in cache[key]]

    write_cache(updated_cache)
    return releases


def compute_segments(releases: list[tuple[str, date]], support_months: int) -> list[Segment]:
    """Non-overlapping "required minimum version" windows: a version is the
    floor from the moment the previous floor expires until its own SPEC-0
    drop date (release + support_months)."""
    segments: list[Segment] = []
    for i, (version, released) in enumerate(releases):
        drop = released + relativedelta(months=support_months)
        start = released if i == 0 else max(released, segments[i - 1][2])
        segments.append((version, start, drop))
    return segments


def floor_at(segments: list[Segment], as_of: date) -> Segment | None:
    for seg in segments:
        if seg[1] <= as_of < seg[2]:
            return seg
    if as_of >= segments[-1][2]:
        return None  # beyond the last known release's window: floor unknown
    return segments[0]


def parse_version_tuple(version: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", version)]


def compare_versions(a: str, b: str) -> int:
    for x, y in zip_longest(parse_version_tuple(a), parse_version_tuple(b), fillvalue=0):
        if x != y:
            return -1 if x < y else 1
    return 0


def load_project_pins() -> dict[str, str]:
    data = tomllib.loads(PYPROJECT_FILE.read_text(encoding="utf-8"))
    project = data["project"]

    requires_python = project["requires-python"]
    pins = {"python": re.sub(r"^[^\d]*", "", requires_python)}

    dep_pattern = re.compile(r"^([A-Za-z0-9_.-]+)\s*(?:>=|==)\s*([0-9][0-9A-Za-z.\-]*)")
    for dep in project["dependencies"]:
        match = dep_pattern.match(dep)
        if not match:
            continue
        name, version = match.group(1).lower(), match.group(2)
        if name in CORE_PACKAGES:
            pins[name] = version
    return pins


def render_mermaid(all_segments: dict[str, list[Segment]], today: date) -> str:
    # Rendered as a custom "mermaid-pz" fence rather than mkdocs-material's
    # built-in "mermaid" one: Material renders mermaid diagrams into a closed
    # shadow root, which is unreachable from outside JS -- including the
    # svg-pan-zoom script this page needs for interactivity. The "mermaid-pz"
    # fence (see mkdocs.yml's custom_fences and docs/js/mermaid-panzoom.js)
    # is rendered by our own script directly into the light DOM instead.
    history_horizon = today - relativedelta(months=15)
    lines = [
        "```mermaid-pz",
        "gantt",
        "    dateFormat YYYY-MM-DD",
        "    axisFormat %m/%Y",
        "    title SPEC-0 minimum supported versions",
    ]
    for pkg in ["python", *CORE_PACKAGES]:
        lines.append(f"    section {LABELS[pkg]}")
        for version, start, end in all_segments[pkg]:
            if end < history_horizon:
                continue
            if end <= today:
                status = "done, "
            elif start <= today < end:
                status = "active, "
            else:
                status = ""
            lines.append(f"    {version} :{status}{start.isoformat()}, {end.isoformat()}")
    lines.append("```")
    return "\n".join(lines)


def render_compliance_table(all_segments: dict[str, list[Segment]], pins: dict[str, str], today: date) -> str:
    rows = [
        "| Dependency | `el_paso` requires | SPEC-0 floor (today) | Status | Next required bump |",
        "|---|---|---|---|---|",
    ]
    for pkg in ["python", *CORE_PACKAGES]:
        floor = floor_at(all_segments[pkg], today)
        required = pins.get(pkg, "?")
        if floor is None:
            rows.append(f"| {LABELS[pkg]} | ≥{required} | unknown | ⚠️ check upstream | — |")
            continue
        floor_version, _, floor_end = floor
        compliant = compare_versions(required, floor_version) >= 0
        status = "✅ compliant" if compliant else "❌ action needed"
        rows.append(f"| {LABELS[pkg]} | ≥{required} | ≥{floor_version} | {status} | by {floor_end.isoformat()} |")
    return "\n".join(rows)


def generate() -> None:
    today = date.today()
    releases = load_releases()

    all_segments = {
        "python": compute_segments(releases["python"], PYTHON_SUPPORT_MONTHS),
    }
    for pkg in CORE_PACKAGES:
        all_segments[pkg] = compute_segments(releases[pkg], CORE_PACKAGE_SUPPORT_MONTHS)

    pins = load_project_pins()

    content = f"""

<!-- Generated by docs/scripts/generate_spec0_gantt.py during the docs build. Do not edit by hand. -->

# Supported versions

`el_paso` follows [SPEC-0](https://scientific-python.org/specs/spec-0000/), the scientific Python
ecosystem's shared policy for minimum dependency support: Python versions are dropped 3 years
after release, and core packages (NumPy, SciPy, Matplotlib, pandas, xarray) are dropped 2 years
after release. The chart below shows which version of each is currently required, computed from
live release data as of the last docs build ({today.isoformat()}); the vertical line marks *today*
whenever you're viewing this page.

{render_mermaid(all_segments, today)}

## Current status

{render_compliance_table(all_segments, pins, today)}

"`el-paso` requires" reflects the floor pinned in `pyproject.toml`. When it is stricter than the
SPEC-0 floor, that's a deliberate choice (a feature the project relies on), not a compliance gap.
"""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(content, encoding="utf-8")


def on_pre_build(config):  # noqa: ANN001, ARG001 - MkDocs hook signature
    generate()


if __name__ == "__main__":
    generate()
    print(f"wrote {OUTPUT_FILE}")
