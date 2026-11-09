<!--
SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
SPDX-FileContributor: Bernhard Haas

SPDX-License-Identifier: Apache-2.0
-->

# Command line interface

Every recipe shipped with EL-PASO is runnable from the command line, and they all
accept the same options. Installing the package provides the `el-paso` command:

```bash
el-paso list                      # show every available recipe
el-paso poes meped --help         # options for one recipe
el-paso poes meped --start-time 2013-03-16 --end-time 2013-03-16T23:59:59
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
    --start-time 2013-03-16 --end-time 2013-03-16T23:59:59
```

Both forms expose exactly the same options, because both are generated from the
recipe function's own signature.


## Shared options

| Option | Meaning |
| --- | --- |
| `--start-time`, `--end-time` | Processing time range. Anything `dateutil` understands, e.g. `2013-03-16` or `2013-03-16T23:59:59`. Naive values are read as UTC. |
| `--satellite` | Which spacecraft to process. Repeat the option to process several in one run, or pass `all` for every spacecraft the recipe supports. |
| `--mag-field` | Magnetic field model used for the derived quantities, e.g. `T89`, `T96`, `TS04`. |
| `--bin-cadence` | Time binning cadence: `10s`, `5min`, `1h`, `2d`. A bare number means seconds. |
| `--raw-data-path` | Where raw files are downloaded to and read from. |
| `--processed-data-path` | Where processed output is written. |
| `--num-cores` | Cores used for the IRBEM magnetic field computations. |
| `--save-strategy` | Which saving strategy to use, where the recipe supports a choice. |
| `--dry-run` | Print the resolved arguments and exit without processing anything. |
| `-v` / `--quiet` | Raise or lower the log level. |
| `--skip-download` / `--exit-after-download` | Reuse already downloaded files, or stop after downloading. |
| `--version` | Print the EL-PASO version. |

`--start-time` and `--end-time` are required; everything else has a default.

Not every recipe has every option: the command line mirrors the recipe's
signature, so a recipe that does no time binning has no `--bin-cadence`. Run
`--help` on a recipe to see what it actually accepts.

Processing several satellites means repeating the option:

```bash
el-paso rbsp mageis-electrons \
    --start-time 2017-10-15 --end-time 2017-10-15T23:59:59 \
    --satellite a --satellite b --mag-field TS04
```

`--satellite all` runs the recipe for every spacecraft it supports, e.g. `goes16` to
`goes19` for `el-paso goes mps-high`:

```bash
el-paso goes mps-high \
    --start-time 2023-10-01 --end-time 2023-10-31T23:59:59 \
    --satellite all
```

When several satellites are processed, a satellite that fails (for example because it
has no data in the time range) does not stop the others. The run continues, logs which
satellites failed, and exits with code 1 at the end.

## What a run reports

Every run begins by printing the settings it resolved, so a long job is never a
black box and the log says exactly what produced it:

```
Running process_poes_meped_electron
start time           2013-03-16T00:00:00+00:00
end time             2013-03-16T23:59:59+00:00
satellite            noaa15
mag field            T89
raw data path        .
processed data path  .
bin cadence          10s
num cores            32
```

When several satellites are processed in one invocation, each is announced in
turn as `(1/2)`, `(2/2)`, and so on. Anything that looks like a credential
(`--client-secret`, `--erg-password`) is shown as `<hidden>` rather than echoed.
`--quiet` suppresses the summary along with the rest of the informational output.

Use `--dry-run` to check what a command resolves to before committing to a long
run — it prints the same summary, headed `Would run`, and stops:

```bash
$ el-paso poes meped --dry-run \
    --start-time 2013-03-16 --end-time 2013-03-16T23:59:59 \
    --satellite noaa15 --bin-cadence 1min
el_paso.recipes.poes.process_poes_meped.process_poes_meped_electron(
    start_time=datetime.datetime(2013, 3, 16, 0, 0, tzinfo=datetime.timezone.utc),
    ...
    bin_cadence=datetime.timedelta(seconds=60),
)
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
el-paso rbsp mageis-electrons \
    --start-time 2017-10-15 --end-time 2017-10-16T23:59:59 \
    --satellite a --mag-field T89 \
    --raw-data-path ~/ep/raw --processed-data-path ~/ep/processed

# 2. plot (same time range, satellite, mag field and processed data path)
el-paso plot rbsp mageis-electrons \
    --start-time 2017-10-15 --end-time 2017-10-16T23:59:59 \
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
create them.

### Choosing what to plot

Variables are named by their *internal* names (`FEDU`, `L_star`, `Energy_FEDU`, `Kp`),
so the same command works for the GFZ and the PRBEM data standard. Each `--variable`
gets its own panel, and the plot type follows from the variable's dimensions: a line for
time series, a spectrogram for variables with an energy dimension.

```bash
el-paso plot rbsp mageis-electrons \
    --start-time 2017-10-15 --end-time 2017-10-16T23:59:59 \
    --processed-data-path ~/ep/processed \
    --variable Kp --variable FEDU --variable L_star
```

A recipe can define default panels (`PLOT_DEFAULTS`, as RBSP MagEIS does), in which case
`--variable` is optional. Settings are layered: defaults per plot type, then the recipe's
`PLOT_DEFAULTS`, then the command-line flags, which win.

| Option | Meaning |
| --- | --- |
| `--variable` | Internal name of a variable to plot. Repeat for several panels. Any space-weather index or solar wind parameter can be requested too; these are drawn once at the top of the figure, shared by all satellites. A variable missing from the files is an error. |
| `--energy` | Energy to show for variables with an energy dimension, e.g. `470` (keV) or `1MeV`. Repeat for several; each is drawn as one line, labelled with the matched energy. |
| `--pitch-angle` | Pitch angle in degrees at which to cut (default 90). |
| `--vmin`, `--vmax` | Colour limits of spectrograms and scatter panels, or y-limits of `--energy` line panels. Default: the 5th and 95th percentile of the data. |
| `--colormap` | A [colorcet](https://colorcet.holoviz.org) or matplotlib colormap name. Default `rainbow4`. |
| `--along-L-star`, `--along-L-m`, `--along-R-eq` | Draw each variable as a scatter plot over time with that coordinate on the y axis and the value as colour. At most one of them per call. |
| `--output-dir` | Directory to write the figure to, created if missing. Default: the current directory. |
| `--filename` | Name of the file; the format follows the suffix (`.png`, `.pdf`, ...). Default: `<recipe>_<satellite>_<start>_<end>.png`. |
| `--show` | Open the figure in an interactive window instead of only writing a file. |

For example, electron flux at three energies against `L_star`, written as a PDF:

```bash
el-paso plot rbsp mageis-electrons \
    --start-time 2017-10-15 --end-time 2017-10-16T23:59:59 \
    --processed-data-path ~/ep/processed \
    --variable FEDU --energy 108 --energy 470 --energy 1575 \
    --along-L-star --vmin 1 --vmax 1e6 \
    --output-dir ~/ep/figures --filename fedu_lstar.pdf
```

With several `--satellite` options the figure holds one block of panels per satellite
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
