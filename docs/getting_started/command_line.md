<!--
SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
SPDX-FileContributor: Bernhard Haas

SPDX-License-Identifier: Apache-2.0
-->

# Command line interface

Every recipe shipped with EL-PASO is runnable from the command line, and they all
accept the same options. Installing the package provides the `el-paso` command:

```bash
el-paso list-recipes                      # show every available recipe
el-paso process poes meped --help         # options for one recipe
el-paso process poes meped 2013-03-16 2013-03-16T23:59:59
el-paso plot poes meped 2013-03-16 2013-03-16T23:59:59
el-paso list-satellites poes meped
el-paso list-variables                    # internal variable names for plotting
```

```asciinema-player
{
    "file": "../../figures/ep.cast",
    "auto_play": true
}
```

A recipe can equally be run as a module, which is useful inside job scripts:

```bash
python -m el_paso.recipes.poes.process_poes_meped \
    2013-03-16 2013-03-16T23:59:59
```

Both forms expose exactly the same options, because both are generated from the
recipe function's own signature.


## Shared options

| Option | Meaning |
| --- | --- |
| `START END` | Processing time range, the first two arguments. Anything `dateutil` understands, e.g. `2013-03-16` or `2013-03-16T23:59:59`. Naive values are read as UTC. |
| `--satellite`, `-s` | Which spacecraft to process. Comma-separated to process several in one run, e.g. `a,b` (repeating the option also works), or pass `all` for every spacecraft the recipe supports (only those with data in the time range, if the recipe has an [availability table](#which-satellites-have-data)). |
| `--mag-field` | Magnetic field model used for the derived quantities, e.g. `T89`, `T96`, `TS04`. |
| `--bin-cadence` | Time binning cadence: `10s`, `5min`, `1h`, `2d`. A bare number means seconds. |
| `--raw-data-path`, `-r` | Where raw files are downloaded to and read from. Defaults to the `EL_PASO_RAW_DATA_PATH` environment variable if set, otherwise `.`. |
| `--processed-data-path`, `-o` | Where processed output is written. Defaults to the `EL_PASO_PROCESSED_DATA_PATH` environment variable if set, otherwise `.`. |
| `--num-cores` | Cores used for the IRBEM magnetic field computations. |
| `--save-strategy` | Which saving strategy to use, where the recipe supports a choice. |
| `--dry-run` | Print the resolved arguments and exit without processing anything. |
| `-v` / `--quiet` | Raise or lower the log level. |
| `--skip-download` / `--exit-after-download` | Reuse already downloaded files, or stop after downloading. |
| `--version` | Print the EL-PASO version. |

The start and end time (`START END`) are required, unless the recipe declares defaults for them; everything else has a default.

Not every recipe has every option: the command line mirrors the recipe's
signature, so a recipe that does no time binning has no `--bin-cadence`. Run
`--help` on a recipe to see what it actually accepts.

Processing several satellites means giving a comma-separated list (repeating the option works too).
The same goes for every other option that takes several values, e.g. `--alpha-eq 10,90`:

```bash
el-paso process rbsp mageis-electrons \
    2017-10-15 2017-10-15T23:59:59 \
    --satellite a,b --mag-field TS04
```

`--satellite all` runs the recipe for every spacecraft it supports, e.g. `goes16` to
`goes19` for `el-paso process goes mps-high`:

```bash
el-paso process goes mps-high \
    2023-10-01 2023-10-31T23:59:59 \
    --satellite all
```

When several satellites are processed, a satellite that fails (for example because it
has no data in the time range) does not stop the others. The run continues, logs which
satellites failed, and exits with code 1 at the end.

### Which satellites have data

Most missions' satellites cover different eras: POES and MetOp satellites were launched and
retired at different times, and the LANL GeoGrid "satellites" are geosynchronous longitude
slots, only a handful of which are populated at any time. Recipes for such missions ship a
*satellite availability table*: for each satellite, the date intervals in which the remote
archive has data. It is part of the package, so using it needs no network access.

Ask which satellites have data with `el-paso list-satellites <mission> <recipe>`. It takes no
time range: it prints the intervals each satellite has data for, without processing anything:

```bash
$ el-paso list-satellites poes meped
poes meped: 7 of its 15 satellites have data, listed below
Satellite  Intervals with data
metop1     2012-10-03 to 2015-06-08
           2015-06-17 to 2019-01-03
           2019-01-29 to 2026-11-10
noaa15     2013-01-01 to 2015-06-08
...
No data in the table: noaa05, noaa06, noaa07, noaa08, noaa10, noaa12, noaa14, noaa17
```

A recipe without a table only lists its satellite names.

With a table, the recipe commands use it as follows:

- `--satellite all` runs only the satellites with data somewhere in the time range, and logs
  which ones it skipped. If none has data, the command stops with an error naming the range.
- A satellite you name explicitly is always run, even if the table says it has no data, since
  the table is a hint and not a guarantee: you get a warning before the run starts instead.
- The run report (and `--dry-run`) gets an `availability` line per satellite.

A time range counts as covered if it overlaps an interval at all, also partially at either end.
Short gaps of a few days are merged into the surrounding interval, so a covered day can still
miss its file; downloading reports missing files as before. Recipes without a table behave as
if every satellite had data, and `el-paso list-satellites` lists only their satellite names.

The same query is available from Python, returning `None` for a recipe without a table:

```python
from datetime import date

import el_paso as ep

coverage = ep.satellite_availability(
    ep.recipes.lanl_geogrid.process_lanl_geogrid_mpa_electrons, date(2017, 4, 1), date(2017, 4, 30)
)
sorted(coverage)  # ['010', '070', '130', '190', '310']
```

Recipes declare their table in a `SATELLITE_AVAILABILITY` attribute of the recipe module, next to
`CLI_DEFAULTS`. Maintainers regenerate the tables from the remote archives with
`python scripts/generate_satellite_availability.py <product>`.

## What a run reports

Every run begins by printing the settings it resolved, so a long job is never a
black box and the log says exactly what produced it:

```
         Running process_poes_meped_electron
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃Parameter            ┃Value                          ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│satellite            │noaa15                         │
│mag field            │T89                            │
│raw data path        │.                              │
│processed data path  │.                              │
│bin cadence          │10s                            │
│num cores            │16                             │
│save strategy        │netcdf                         │
│skip existing        │True                           │
│calculate Lstar      │True                           │
│save indices sw      │False                          │
│append               │False                          │
│start time           │2013-03-16T00:00:00+00:00      │
│end time             │2013-03-16T23:59:59+00:00      │
│availability         │data 2013-01-01 to 2015-06-08  │
└─────────────────────┴───────────────────────────────┘
```

When several satellites are processed in one invocation, each is announced in
turn as `(1/2)`, `(2/2)`, and so on. Anything that looks like a credential
(`--client-secret`, `--erg-password`) is shown as `<hidden>` rather than echoed.
`--quiet` suppresses the summary along with the rest of the informational output.

Use `--dry-run` to check what a command resolves to before committing to a long
run. It prints the same summary, headed `Would run`, and stops:

```bash
$ el-paso process poes meped 2013-03-16 2013-03-16T23:59:59 \
    -s noaa15 --bin-cadence 1min --dry-run
         Would run process_poes_meped_electron
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃Parameter            ┃Value                          ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│satellite            │noaa15                         │
│mag field            │T89                            │
│raw data path        │.                              │
│processed data path  │.                              │
│bin cadence          │1min                           │
│num cores            │16                             │
│save strategy        │netcdf                         │
│skip existing        │True                           │
│calculate Lstar      │True                           │
│save indices sw      │False                          │
│append               │False                          │
│start time           │2013-03-16T00:00:00+00:00      │
│end time             │2013-03-16T23:59:59+00:00      │
│availability         │data 2013-01-01 to 2015-06-08  │
└─────────────────────┴───────────────────────────────┘
```

## Defaults

Each recipe defines its own defaults. Most come from the recipe function's
signature; the ones that have no sensible library default are declared in a `CLI_DEFAULTS` dictionary at the bottom of the recipe
module, next to its entry point:

```python
CLI_DEFAULTS = {
    "start_time": datetime(2013, 3, 16, tzinfo=timezone.utc),
    "end_time": datetime(2013, 3, 16, 23, 59, 59, tzinfo=timezone.utc),
}

if __name__ == "__main__":
    ep.run_recipe_cli(process_poes_meped_electron, defaults=CLI_DEFAULTS)
```

## Plotting

`el-paso plot <mission> <recipe>` draws a quick-look figure of data you have already
processed. It takes the same arguments as the processing command, so the settings that
located the files when you processed them locate them again:

```bash
# 1. process
el-paso process rbsp mageis-electrons \
    2017-10-15 2017-10-16T23:59:59 \
    --satellite a --mag-field T89 \
    --raw-data-path ~/ep/raw --processed-data-path ~/ep/processed

# 2. plot (same time range, satellite, mag field and processed data path)
el-paso plot rbsp mageis-electrons \
    2017-10-15 2017-10-16T23:59:59 \
    --satellite a --mag-field T89 \
    --processed-data-path ~/ep/processed
```

Plotting never processes anything. The only thing it downloads are space-weather
indices and solar wind parameters (`Kp`, `Dst`, `Vsw`, ...), and only if they are
missing from the local cache. Raw-data and processing options such as
`--raw-data-path` or `--num-cores` are therefore not accepted. Use
`el-paso plot <mission> <recipe> --help` to see the options of one recipe.

Run `--dry-run` first to see which files are expected and which of them exist. If
files are missing, the command stops and prints the processing command that would
create them. `--satellite all` only plots the satellites with data in the time range, as
in [processing](#which-satellites-have-data), and the dry run says per satellite whether
the recipe's availability table has data for it.

### Choosing what to plot

Variables are named by their *internal* names (`FEDU`, `L_star`, `Energy_FEDU`, `Kp`),
so the same command works for the GFZ and the PRBEM data standard. Each variable
gets its own panel, and the plot type follows from the variable's dimensions: a line for
time series, a spectrogram for variables with an energy dimension.

```bash
el-paso plot rbsp mageis-electrons \
    2017-10-15 2017-10-16T23:59:59 \
    --processed-data-path ~/ep/processed \
    --variable Kp,FEDU,L_star
```

A recipe can define default panels (`PLOT_DEFAULTS`, as RBSP MagEIS does), in which case
`--variable` is optional. Settings are layered: defaults per plot type, then the recipe's
`PLOT_DEFAULTS`, then the command-line flags, which win.

| Option | Meaning |
| --- | --- |
| `--variable` | Comma-separated internal names of the variables to plot, one panel each, e.g. `FEDU,L_star`. `el-paso list-variables` lists all internal names. Any space-weather index or solar wind parameter can be requested too; these are drawn once at the top of the figure, shared by all satellites. A variable missing from the files is an error. |
| `--energy` | Energy to show for variables with an energy dimension, comma-separated, e.g. `470,1575`. Bare numbers are in keV; a unit may be added, e.g. `1MeV`. Each is drawn as one line, labelled with the matched energy. |
| `--pitch-angle` | Pitch angle in degrees at which to cut (default 90). |
| `--vmin`, `--vmax` | Colour limits of spectrograms and scatter panels, or y-limits of `--energy` line panels. Default: the 5th and 95th percentile of the data. |
| `--colormap` | A [colorcet](https://colorcet.holoviz.org) or matplotlib colormap name. Default `rainbow4`. |
| `--along` | One of `L_star`, `L_m`, `R_Eq` (case-insensitive). Draw each variable as a scatter plot over time with that coordinate on the y axis and the value as colour. |
| `--output-dir` | Directory to write the figure to, created if missing. Default: the current directory. |
| `--filename` | Name of the file; the format follows the suffix (`.png`, `.pdf`, ...). Default: `<recipe>_<satellite>_<start>_<end>.png`. |
| `--show` | Open the figure in an interactive window instead of only writing a file. |
| `--export-script` | Write a standalone Python script that makes this figure to the given file, instead of plotting. The script holds every setting as a variable at the top and draws the figure with matplotlib, so it can be edited into a custom figure.|

For example, electron flux at three energies against `L_star`, written as a PDF:

```bash
el-paso plot rbsp mageis-electrons \
    2017-10-15 2017-10-16T23:59:59 \
    --processed-data-path ~/ep/processed \
    --variable FEDU --energy 108,470,1575 \
    --along L_star --vmin 1 --vmax 1e6 \
    --output-dir ~/ep/figures --filename fedu_lstar.pdf
```

With several satellites (`--satellite a,b`) the figure holds one block of panels per satellite
on a shared time axis. The plot runs headless by default (it writes a file, so it works
on a cluster).

### From Python

The same plots are available on a loaded dataset. Build the saving strategy the
recipe uses, load it with `ep.DataSet`, and call `plot`:

```python
from datetime import datetime, timezone
from pathlib import Path

import astropy.units as u
import el_paso as ep
from el_paso.recipes.rbsp.process_rbsp_mageis_electrons import rbsp_mageis_electron_strategy

start = datetime(2017, 10, 15, tzinfo=timezone.utc)
end = datetime(2017, 10, 16, 23, 59, 59, tzinfo=timezone.utc)

strategy = rbsp_mageis_electron_strategy(
    processed_data_path=Path("~/ep/processed").expanduser(), satellite="a", mag_field="T89", save_strategy="netcdf"
)
ds = ep.DataSet(strategy, start, end)

fig = ds.plot(["FEDU", "L_star"], energies=[470, 1575] * u.keV, pitch_angle=90 * u.deg)
fig.savefig("quicklook.png")
```

Importing `el_paso` does not import matplotlib; it is loaded the first time you plot.

## Shell completion

```bash
el-paso --install-completion
```

## Giving your own recipe a command line

[`run_recipe_cli`][el_paso.cli.recipe_cli.run_recipe_cli] gives any function the
same command line, deriving the options from its type annotations and its
Google-style docstring. `datetime`, `timedelta`, `Literal[...]`, `Path`,
`str | Path`, `bool`, `int`, `float` and `str | None` parameters are all
supported; anything it cannot represent stays a Python-only parameter.
