<!--
SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
SPDX-FileContributor: Bernhard Haas

SPDX-License-Identifier: Apache-2.0
-->

# el-paso plot: design and implementation plan

2026-11-03 · Bernhard Haas

## Summary

`el-paso plot <mission> <recipe>` plots data that a recipe has already processed, using the same arguments as the processing run. It loads data with the existing DataSet class, built from the strategy that the recipe's factory returns, so file names are worked out in one place.

Decisions so far:

- **Subcommand, not a `--plot` flag.** Plotting reads processed output and never triggers processing.
- **The recipe identifies the data.** The user gives the recipe and every argument its strategy factory needs. Nothing is discovered from a path alone: no provenance metadata, no `el-paso ls`, no plotting of a bare file.
- **The strategy factory defines the plot command's data options.** Its parameters, plus `--start-time`/`--end-time`, are what `plot` accepts besides the plot-only options. Processing-only options (`--num-cores`, `--raw-data-path`, download flags) drop out automatically.
- **Defaults come in layers:** per variable type (shared), then the recipe's `PLOT_DEFAULTS`, then CLI flags.
- **Loading uses `ep.DataSet`** as it is: `DataSet(strategy, start_time, end_time)`. The only addition is a public way to report file coverage.

## Command surface

The plot command mirrors the processing command, with `plot` in front:

```bash
el-paso rbsp mageis-electrons --start-time 2017-10-15 --end-time 2017-10-15T23:59:59 \
    --satellite a --mag-field TS04 --processed-data-path /data/processed

el-paso plot rbsp mageis-electrons --start-time 2017-10-15 --end-time 2017-10-15T23:59:59 \
    --satellite a --mag-field TS04 --processed-data-path /data/processed
```

Options come from three sources:

| Source | Options | Notes |
| --- | --- | --- |
| Shared | `--start-time`, `--end-time`, `-v`/`--quiet`, `--dry-run` | Same parsing as processing (dateutil, naive = UTC). |
| Strategy factory signature | e.g. `--satellite`, `--mag-field`, `--processed-data-path`, `--save-strategy` | Built by the same machinery as `run_recipe_cli`. `--satellite` can be repeated or set to `all`. |
| Plot-only | `--variable`, `--energy`, `--pitch-angle`, `--vmin`, `--vmax`, `--colormap`, `--output`, `--show` | `--variable` and `--energy` take one value and are repeated for several (`--variable FEDU --variable L_star`). `--variable` is required unless the recipe defines `PLOT_DEFAULTS`. |

Behaviour:

- With several satellites, panels are stacked vertically in one block per spacecraft (e.g. rbspa: FEDU, L_star; then rbspb: FEDU, L_star), with a single shared time axis at the bottom. Every variable of every satellite gets its own panel; nothing is overlaid.
- `--output plot.png` writes a file (format from its suffix). Without `--output` and without `--show`, the default is `<recipe>_<satellite>_<start>_<end>.png` in the current directory.
- `--show` opens an interactive window. On headless systems the Agg backend is used, and `--show` fails with a clear message.
- `--dry-run` prints the strategy (its `repr`), the expected files and which of them exist, then stops.
- Like processing runs, a plot run prints its resolved settings first.

## Recipe registration

Each recipe registers its existing strategy factory and an optional `PLOT_DEFAULTS` dict next to `CLI_DEFAULTS`. The same registration has to reach both `python -m` entry points and the `el-paso` command registry that backs `el-paso list`.

```python
def rbsp_mageis_electron_strategy(
    processed_data_path: str | Path,          # renamed from base_data_path
    mag_field: ep.typing.MagneticFieldLiteral,
    satellite: RBSPSatellite,
    save_strategy: Literal["netcdf"] = "netcdf",  # replaces file_format
) -> ep.SavingStrategy: ...

PLOT_DEFAULTS = {
    "panels": ["FEDU", "L_star", "MLT"],
    "FEDU": {
        "pitch_angle": 90,
        "energies": STANDARD_ENERGIES_KEV[[3, 9, 15]],  # ~108 keV, 470 keV, 1.58 MeV
        "vmin": 1e0,
        "vmax": 1e6,
    },
}

if __name__ == "__main__":
    ep.run_recipe_cli(
        process_rbsp_mageis_electrons,
        strategy=rbsp_mageis_electron_strategy,
        plot_defaults=PLOT_DEFAULTS,
    )
```

The factory needs some changes so that it can be exposed to users:

1. **Same parameter names as the recipe.** `base_data_path` becomes `processed_data_path`; otherwise users would see `--base-data-path` in `plot` but `--processed-data-path` in processing.
2. **`save_strategy` is passed through.** Today the recipe deletes `save_strategy` and the factory has its own `file_format="nc"`. Both commands must use the same switch, so the recipe calls `rbsp_mageis_electron_strategy(processed_data_path, mag_field, satellite, save_strategy)` and the factory maps it to a format.
3. **Only CLI-representable parameters.** Every factory parameter must be a type `run_recipe_cli` supports; nothing discovered during processing.

The value of `--satellite` is passed to the factory once per spacecraft, so a factory takes a single satellite. The recipe signature and the factory must agree on each shared parameter's type and allowed values; a registration-time check enforces that (see Implementation plan).

## Loading

`plot` loads data with the existing `DataSet`, so no new loading code is needed:

```python
strategy = rbsp_mageis_electron_strategy(processed_data_path, mag_field, satellite, save_strategy)
ds = ep.DataSet(strategy, start_time, end_time, verbose=verbose)

time = ds.datetime
fedu = ds.get_by_internal_name("FEDU")
energy = ds.get_by_internal_name("Energy_FEDU")
```

What `DataSet` already provides that the plot command relies on:

- **Same files as saving.** It lists files with the strategy's `get_time_intervals_to_save` and `get_file_path`.
- **Lazy loading.** Each output file is read on first access to one of its variables, cut to `[start, end]` and concatenated along time.
- **Missing data handled.** Missing files are skipped with a warning; a variable found in no file becomes an empty array. The plotter treats a zero-length array as missing.
- **Standard-agnostic access.** `get_by_internal_name` and `datetime` let plotting code work the same for GFZ and PRBEM output. `--variable` and `PLOT_DEFAULTS` therefore use internal names (`FEDU`, `L_star`, `Energy_FEDU`).
- **Typo suggestions.** `find_similar_variable` already produces "Maybe you meant …" for misspelled names.

Units and dimensions come from the data standard (`variable_infos[name].unit`, `get_dependencies(name)`), which layer-1 defaults use to choose a plot type.

Limits of `DataSet` today, and how `plot` handles them:

| Behaviour | Effect on plotting | Handling |
| --- | --- | --- |
| Output files with empty `names_to_save` ("save everything") give an empty `possible_variables`, and loading raises "not part of the chosen saving strategy" | Such recipes cannot be plotted | Recipes with plotting list their names, or `DataSet` falls back to the file's own variables |
| Interval list is private (`_date_list`); coverage is only logged per file | No clean summary of missing months | Add a public coverage method (Phase 1) |
| Time-independent variables come from the last file loaded, without a consistency check | Usually harmless | Accept for now |
| Chunks whose non-time shapes differ raise `ValueError` on access | One bad panel would stop the plot | Catch per panel, report, continue |
| `verbose=True` logs every file loaded | Noisy for long ranges | Pass `verbose` only with `-v` |
| `preferred_extension` is stored but not used when loading | Format comes from the strategy | Pass `save_strategy` through to the factory (Recipe registration) |

`DataSet` also has `interp_flux`, but `--energy` and `--pitch-angle` do not use it: it handles only FEDU, interpolates in equatorial pitch angle (`Alpha_Eq`, which needs field-model output) and runs a process pool, which is too slow for a quick-look. Nearest-channel matching is used instead; see Plot defaults.

## Code layout

Plotting is a `DataSet` method, `ds.plot(...)`, attached with the same `_LazyMethod` pattern as `interp_flux` and `identify_orbits`. matplotlib is imported only when `plot` is first called. Notebooks and scripts get the same quick-look as the CLI (`ep.DataSet(strategy, start, end).plot(["FEDU", "L_star"])`), with no new top-level export.

```python
def plot(self, variables: Sequence[str], *, axes=None) -> Figure:
    # variables: internal names, one panel each, in this order
    # axes=None: create a figure with one panel per variable
    # axes given: draw into them (one satellite's block of a multi-satellite figure)
```

`variables` is required. A `DataSet` knows its saving strategy but not the recipe that produced it, so it cannot look up `PLOT_DEFAULTS`, and "every saved variable" is not a usable default (it includes coordinate variables such as `Energy_FEDU` and is empty for recipes that save everything). The CLI resolves the panel list and passes it in.

There is no `mag_field` argument. The model used for labels is read from the strategy (`getattr(self.saving_strategy, "mag_field", None)`); strategies without a field model (density, solar wind, waves) do not set it, and those labels then carry no model. GFZ strategies store `TS04` as `T04s`, so labels show the name used in the file.

A `DataSet` covers one satellite, so the multi-satellite layout belongs to the CLI. The CLI creates a single column of panels with a shared time axis (`sharex=True`) and calls `ds.plot(variables, axes=block)` once per satellite, where `block` is that satellite's slice of rows.

| File | Contents |
| --- | --- |
| `el_paso/dataset/plot.py` | `DataSet.plot`: layer-1 plot-type selection from dimensions, one panel per given variable. Energy/pitch-angle selection (Phase 4) goes here too. |
| `el_paso/cli/plot_cli.py` | The `plot` command: options, panel list (`--variable` or `PLOT_DEFAULTS`), settings summary, `--dry-run`, coverage errors, multi-satellite figure, `--output`/`--show`. |
| `el_paso/cli/recipe_cli.py` | Signature-to-option machinery split out so `plot_cli.py` can reuse it. |
| `el_paso/cli/app.py` | Registers the `plot` group (mission → recipe, only recipes with a registered `strategy`). |

## Plot defaults

Settings are resolved in three layers, each overriding the one before: per variable type, then the recipe's `PLOT_DEFAULTS`, then CLI flags. This is the same layering the CLI already uses for recipe arguments.

The panel list is the exception: layer 1 has none. It comes from `--variable`, otherwise from `PLOT_DEFAULTS`; if the recipe has no `PLOT_DEFAULTS`, `--variable` is required and the CLI exits with an error naming the option.

**Layer 1, per variable type (shared code).** The plot type is chosen from the variable's dimensions in the data standard, so every recipe gets a reasonable plot without configuration:

| Dimensions | Plot | Default selection |
| --- | --- | --- |
| (time) | Line | none |
| (time, 3) e.g. position | Line, one per component | all components |
| (time, energy) | Spectrogram, energy on y (log) | all channels |
| (time, energy, pitch angle) e.g. FEDU, PSD | Spectrogram, energy on y (log) | pitch angle nearest 90° |
| (time, pitch angle) | Spectrogram, pitch angle on y | all |

Flux-like variables get a log colour scale. Variables tied to a magnetic field model (L\*, MLT, InvMu…) are labelled with the strategy's `mag_field`.

**Layer 2, recipe `PLOT_DEFAULTS`.** These hold only what is specific to the instrument: which panels make a good quick-look, which energies or pitch angles to show as line panels, colour limits for the instrument's dynamic range.

**Layer 3, CLI flags.** `--variable` replaces the panel list and switches layer 2 off entirely: the named variables get the type defaults plus any `--energy`/`--pitch-angle`/`--vmin`/`--vmax`, not the recipe's per-variable settings. `--energy`, `--pitch-angle`, `--vmin` and `--vmax` override the matching settings for every panel they apply to.

**Selected energies are drawn as lines.** When energies are selected for a variable (from `--energy` or `PLOT_DEFAULTS`), that variable gets one line panel (log y) with one line per selected energy, and a legend gives the energy actually matched. In such a panel `--vmin`/`--vmax` set the y-limits; in a spectrogram they set the colour limits.

**Space-weather indices and solar wind parameters** (every `SW_Index` of `el_paso/load_indices_solar_wind_parameters.py`: `Kp`, `Dst`, `Vsw`, `Nsw`, `Pdyn`, `BzIMF`, `ByIMF`, `G1`, `G2`, `G3`, `W_params`) are internal names in both data standards, so they are requested like any other variable (`--variable Kp`, or in the `PLOT_DEFAULTS` panel list). Kp and Dst are drawn as steps, `W_params` as one line per component (W1–W6), the rest as lines; the y label carries the unit. They are loaded with `ep.load_indices_solar_wind_parameters` (swvo) for the plotted time range and **are downloaded if they are missing from the local cache**; this is the only thing plotting downloads. If loading fails (e.g. offline), a warning is logged and the panel shows "no data".

Two rules apply in every layer:

- **Energies and pitch angles are matched per time step.** MagEIS energies vary between files (`Energy_FEDU` has a time dimension), so `--energy 470` selects the nearest channel at each time step, not a fixed index. The selected value is shown in the panel label; if the nearest channel is more than 10% away, those time steps are left blank.
- **Missing variables are an error.** Any requested variable that is not in the files, whether from `--variable` or `PLOT_DEFAULTS` (e.g. `B_total_obs` without `--add-B-obs`), makes the command exit with an error naming it. Recipes therefore list in `PLOT_DEFAULTS` only variables their processing always saves.

## Errors and diagnostics

The usual failure is arguments that almost match what was processed, so errors say what does exist.

- **Partial coverage:** plotting continues, gaps are shown as blank, and a warning lists missing intervals: `No processed files for rbspa (TS04) in 2017-11; plotting 2017-10 only.`
- **No files at all:** exit code 1, the expected path of the first missing file, and the processing command that would create the data.
- **Near-miss hint:** when nothing is found, the factory is called again for every other allowed value of each `Literal` parameter (`satellite`, `mag_field`, `save_strategy`), varying one parameter at a time, and `DataSet.get_file_coverage()` is checked for each. Matches are reported: `No files for rbspa T89 in 2017-10; found files for: --mag-field TS04.` This reuses the forward logic only, with no directory scanning or file-name parsing. Free-form parameters such as paths are not varied.
- **Several satellites:** as in processing, a satellite with no data does not stop the others; its panels are omitted, and the run exits with code 1 at the end.

Example:

```
$ el-paso plot rbsp mageis-electrons --start-time 2017-10-15 --end-time 2017-10-16 --satellite a
No processed files for rbspa (T89) between 2017-10-15 and 2017-10-16.
Expected: ./RBSP/rbspa/mageis/...
Found files for: --mag-field TS04
To create them: el-paso rbsp mageis-electrons --start-time 2017-10-15 --end-time 2017-10-16 --satellite a --mag-field T89
```

## Non-goals and open questions

Out of scope for the first version:

- Discovering data from a path alone (`el-paso ls`, plotting a bare file, provenance metadata).
- Processing on demand from `plot`. A `--process-if-missing` flag can be added later if needed.
- Publication-quality styling. The CLI produces quick-looks; detailed figures stay in the Python API, which `plot` uses too.

Open questions:

- [x] Is matplotlib an optional extra (`pip install el-paso[plot]`) or a core dependency? -> core dependency
- [x] Should solar wind indices (Kp, Dst) be available as extra panels? That needs a second `DataSet` from the solar wind recipe's strategy. -> use swvo to load them on the fly, downloaded if missing (revised 2026-11-04), requested by their internal names; every `SW_Index` is available, not only Kp and Dst (revised 2026-11-04); planned in Phase 4
- [x] Should the plotting function be public (`ep.plot(dataset, ...)`) so notebooks and scripts get the same quick-look? -> yes, as the method `DataSet.plot(...)` (see Code layout)
- [x] Should `--energy` / `--pitch-angle` use `DataSet.interp_flux` instead of nearest-channel matching? -> no, nearest-channel matching (revised 2026-11-03: `interp_flux` handles only FEDU, needs `Alpha_Eq` and is slow; see Loading)
- [ ] Should `DataSet` support output files with empty `names_to_save`, or do recipes with plotting simply list their names? -> I don't understand
- [x] Which recipes get `PLOT_DEFAULTS` in the first release beyond RBSP MagEIS? -> None

## Implementation plan

Five phases, all developed on the single branch `add-plotting-cli` and merged as one PR (commit per phase). Phases 1 and 2 are independent; 3 needs both; 4 and 5 build on 3. RBSP MagEIS electrons is the pilot recipe throughout. File paths other than `el_paso/cli/recipe_cli.py` and the MagEIS recipe are placeholders until checked against the repo.

### Phase 1: Coverage reporting in `DataSet`

Loading itself exists; this phase only adds what the plot command needs on top.

- [x] Add a public `DataSet.get_file_coverage(names=None)` returning, per interval and output file, the expected path and whether it exists. It reuses the interval list `DataSet` already builds and replaces direct use of `_date_list`.
- [x] Confirm a save → `DataSet` round trip for MagEIS-shaped data across a month boundary, including time-dependent `Energy_FEDU`; add the test if it is not covered yet.
- [x] Tests for coverage: one month missing, no files at all, a range inside one file.

Done when: the plot command can report missing months and drive `--dry-run` without touching private attributes.

### Phase 2: Strategy factories ready for the CLI

- [x] Rename `base_data_path` to `processed_data_path` in every strategy factory. (Done: RBSP MagEIS electrons. Remaining recipes: follow-up PRs, merging gfz/netcdf factories into one.)
- [x] Replace `file_format` with `save_strategy` in factories and pass it through from each recipe instead of `del save_strategy`.
- [x] Add `strategy=` and `plot_defaults=` to `run_recipe_cli` and to the `el-paso` recipe registry.
- [x] Registration-time check: every factory parameter exists in the recipe signature with the same type and allowed values.
- [x] Tests: the check runs over all registered recipes in CI, so a mismatch fails the build.

Done when: every recipe registers its factory and the check passes for all of them.

### Phase 3: `el-paso plot` command, minimal

- [x] Split the signature-to-option machinery out of `_make_command` in `recipe_cli.py` so the plot command can reuse it.
- [x] Add the `plot` subcommand (`el_paso/cli/plot_cli.py`, registered in `app.py`); build its options from the shared options + factory signature + plot-only options. Plot-only options in this phase: `--variable`, `--output`, `--show`.
- [x] Settings summary and `--dry-run` (strategy `repr`, expected files, which exist).
- [x] `DataSet.plot(variables, *, axes=None)` in `el_paso/dataset/plot.py` via `_LazyMethod`, with layer 1 defaults only: plot type from dimensions, one panel per given variable, field-model labels from the strategy's `mag_field`. `--variable` is required in this phase (no `PLOT_DEFAULTS` yet).
- [x] Multi-satellite figure in the CLI: panels stacked vertically in one block of rows per satellite, one panel per variable per satellite, single shared time axis at the bottom; each block drawn by `ds.plot(axes=...)`.
- [x] Headless-safe output: Agg by default, `--output`, default file name, `--show`.
- [x] Errors for partial and no coverage, with the processing command to run.
- [x] Tests: CLI help snapshot for MagEIS; `--dry-run` output; an image is produced from a small synthetic file; processing-only options are rejected; `DataSet.plot` on a synthetic dataset, and `import el_paso` does not import matplotlib.

Done when: `el-paso plot rbsp mageis-electrons ... --variable FEDU --variable L_star` produces a quick-look from data written by `el-paso rbsp mageis-electrons ...` with no recipe-specific code.

### Phase 4: Recipe defaults and selection

- [x] Plot-only options `--energy`, `--pitch-angle`, `--vmin`, `--vmax`. (`--pitch-angle` picks the cut and `--vmin`/`--vmax` set spectrogram colour limits; `--energy` is parsed and passed to `DataSet.plot`, but only takes effect with the selection tasks below.)
- [x] `PLOT_DEFAULTS` support and layering with CLI flags; `--variable` becomes optional for recipes that define `PLOT_DEFAULTS`. (Layering lives in `DataSet.plot(panel_settings=...)`; `PLOT_DEFAULTS` is validated when the command is built, and settings carry astropy units, e.g. `"pitch_angle": 90 * u.deg`.)
- [x] `--energy` / `--pitch-angle` nearest-value matching per time step, with tolerance and labels (not `interp_flux`).
- [x] Selected energies drawn as one line panel per variable (log y), one line per energy, legend with the matched energy; `--vmin`/`--vmax` set y-limits there and colour limits in spectrograms.
- [x] Error on any requested variable that is missing from the files (`--variable` or `PLOT_DEFAULTS`); no skipping.
- [x] `PLOT_DEFAULTS` for RBSP MagEIS.
- [x] Solar wind indices and parameters (every `SW_Index`: Kp, Dst, Vsw, Nsw, Pdyn, BzIMF, ByIMF, G1–G3, W_params) as extra panels, loaded with swvo for the plotted time range (as in `el_paso/load_indices_solar_wind_parameters.py`, no `DataSet`). Drawn once at the top of the figure and shared by all satellites, not repeated per satellite block. Requested by their internal names (`--variable Kp`, `--variable Vsw`, ..., or in `PLOT_DEFAULTS` panels). Downloaded if missing from the local cache; if loading fails, a warning is logged and the panel is left empty.
- [x] Plot-only option `--colormap` (colorcet name, else matplotlib name; default colorcet `rainbow4`) for spectrograms and `--along-*` scatter panels. It is a figure-wide setting, not part of `PLOT_DEFAULTS`; `colorcet` is a core dependency.
- [x] Plot-only options `--along-L-star`, `--along-L-m` and `--along-R-eq`, mutually exclusive (at most one per call). Each variable still gets one panel with time on the x axis, but the panel is a scatter plot with the chosen coordinate (`L_star`, `L_m` or `R_Eq`, i.e. the satellite's radial distance mapped to the magnetic equator) on the y axis and the variable's value as colour, with a colour bar. Colour scale and `--vmin`/`--vmax` work as in spectrograms. Energy and pitch-angle selection apply as usual. The labels of `L_star` and `L_m` name the strategy's `mag_field`. (Revised 2026-11-04: a scatter panel colours one value per time step, so a variable with an energy dimension gets one panel per selected energy, or one per channel without a selection; `DataSet.plot(along=...)`, with `count_panels` to lay out the figure. Pitch angles of the variable and of `L_star`/`L_m` are cut at `--pitch-angle`, default 90°. Variables with other dimensions, e.g. vectors, are reported and left empty. A missing coordinate is an error, like a missing variable.)
- [ ] Tests: layering order; time-dependent energies with shifted channels; `B_total_obs` absent without `--add-B-obs` is an error; with several satellites an index panel appears once, with swvo mocked (no network).

### Phase 5: Diagnostics, docs and rollout

- [ ] Near-miss hint by varying `Literal` factory parameters.
- [ ] Exit code 1 when any satellite has no data, matching processing.
- [x] Add a "Plotting" section to `command_line.md` and a short tutorial example.
- [ ] `PLOT_DEFAULTS` for further recipes, starting with the ones decided under Open questions.
- [x] Decide and implement the matplotlib dependency (optional extra vs core). -> core (already in `dependencies`, with `colorcet`); it is still imported lazily so `import el_paso` stays cheap.
