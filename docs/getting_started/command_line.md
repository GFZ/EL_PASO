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

A recipe can equally be run as a module, which is useful inside job scripts:

```bash
python -m el_paso.recipes.poes.process_poes_meped \
    --start-time 2013-03-16 --end-time 2013-03-16T23:59:59
```

Both forms expose exactly the same options, because both are generated from the
recipe function's own signature.

## Try it

This is a genuine run of the RBSP HOPE electrons recipe for a single day, with
the (comparatively slow) L\* calculation switched off. It is not a canned
transcript: the block below is generated when these docs are built by
actually running the command, downloading the real CDF from NASA SPDF and
running the real IRBEM magnetic-field computations.

<!-- LIVE_CLI_EXAMPLE -->
## Shared options

| Option | Meaning |
| --- | --- |
| `--start-time`, `--end-time` | Processing time range. Anything `dateutil` understands, e.g. `2013-03-16` or `2013-03-16T23:59:59`. Naive values are read as UTC. |
| `--satellite` | Which spacecraft to process. Repeat the option to process several in one run. |
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
