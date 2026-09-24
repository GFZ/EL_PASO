# EL-PASO architecture

Deep reference for the data pipeline and recipe plugin architecture. Linked from the `add-new-recipe` skill rather than duplicated there.

## Core data model
- **`ep.Variable`** (`el_paso/variable.py`) is the unit of data passed through the whole pipeline: a numpy array plus `VariableMetadata` (astropy unit, original cadence, source files, processing notes). Methods such as `convert_to_unit`, `apply_thresholds_on_data`, `truncate`, and `merge` keep data and metadata consistent, so prefer them over editing the array directly.
- **`el_paso/units.py`** defines the time units used in raw files (`ep.units.cdf_epoch`, `ep.units.posixtime`, TT2000, datenum) and conversions between them. `el_paso/physics.py` holds physical helpers (rest energy, energy to momentum).
- Times are UTC. Entry points call `enforce_utc_timezone` on incoming datetimes.

## Data pipeline
Every recipe follows the same shape:

1. **`ep.download(...)`** (`el_paso/download.py`) fetches raw files (CDF, etc.) for a time range via HTTP/FTP, with caching (`skip_existing`) and parallel chunked downloads (`richpool.JoblibPool`). `download_omm.py` is a separate, orbit-metadata-specific downloader exposed as its own `el-paso omm` command.
2. **`ep.ExtractionInfo` + `el_paso/extract_variables_from_files.py`** declaratively describe which columns/variables to pull out of the raw files and their units (via `astropy.units`), producing `ep.Variable` objects.
3. **`el_paso/processing/`** is a library of transform functions applied to `Variable`s: time-binning (`bin_by_time.py`), magnetic-field-derived quantities via IRBEM (`magnetic_field_utils/`, `compute_magnetic_field_variables.py`), invariants (`compute_invariant_K.py`, `compute_invariant_mu.py`), phase-space density (`compute_phase_space_density.py`), pitch-angle folding (`fold_pitch_angles_and_flux.py`), and more. Each recipe calls the specific subset of these it needs, in sequence.
4. **`ep.save(...)`** (`el_paso/save.py`) writes the processed variables using a `SavingStrategy` (`el_paso/saving_strategy.py`). A strategy pairs an output layout (e.g. `GFZStrategy`, `MonthlyRBStrategy`; see `el_paso/saving_strategies/`) with a `DataStandard` (`el_paso/data_standards/`: GFZ or PRBEM naming/metadata conventions) that validates and names the output variables.

## Recipe plugin architecture
- Each mission subpackage (`el_paso/recipes/<mission>/`) uses `lazy_loader.attach` in its `__init__.py` to lazily expose its `process_*` functions and `<...>_strategy` factories. Nothing is imported until it is actually used, so `import el_paso` and CLI tab-completion stay fast even with 9 missions and 20+ recipes.
- The CLI (`el_paso/cli/app.py`) declares a flat `RECIPES: tuple[RecipeEntry, ...]` registry mapping `(mission, command) → (module, function)`. Each mission becomes a `LazyRecipeGroup` sub-command group. A recipe's `typer` command is built on demand from its function signature (`build_recipe_command`, `el_paso/cli/recipe_cli.py`), only when that specific sub-command is invoked or its help is requested, so `el-paso --help` never imports any recipe's (heavy) scientific dependencies. Each recipe module's `if __name__ == "__main__": ep.run_recipe_cli(...)` goes through the same builder, so `python -m el_paso.recipes.<mission>.<module>` exposes exactly the same options as `el-paso <mission> <command>`. An optional module-level `CLI_DEFAULTS` dict overrides defaults for both.
- `hooks/check_recipe_strategies.py` (a pre-commit hook, see the `add-new-recipe` skill) statically enforces that this registry, the `__init__.py` exports, and the recipe files themselves all stay in sync.

## Working with saved data: `el_paso/dataset/`
This subpackage works on the pipeline's *output*, not on raw files:
- `DataSet` (`dataset.py`) loads saved `.mat` / `.nc` files; `GFZDataSet` and `PRBEMDataSet` (`dataset_implementations.py`) are typed per data standard, with generated attribute blocks (see below).
- Analysis helpers: `identify_orbits` (splits a time series into orbit segments between radial extrema), `linearize_trajectories`, and `bin_and_interpolate_to_model_grid` (maps observations onto a model's grid and time axis).
- `el_paso/dataset/scripts/` holds one-off data-preparation scripts (`create_RBSP_line_data.py`, which is excluded from `ty`).

## Runtime configuration and caches
Environment variables read by the library:

| Variable | Effect |
|---|---|
| `EL_PASO_SKIP_DOWNLOAD` | Skip every `ep.download` call. Same as setting `ep.skip_download = True`. |
| `EL_PASO_EXIT_AFTER_DOWNLOAD` | Exit the process after `ep.download` finishes (useful for download-only runs on HPC). Same as setting `ep.exit_after_download = True`. |
| `EL_PASO_MODEL_DATA_PATH` | Where model coefficient data is stored. Default `~/.elpaso`. |
| `EL_PASO_INDICES_SW_PARAM_DATA_PATH` | Where solar-wind indices/parameters are stored. Default `~/.elpaso`. The test suite's `conftest.py` points it at a temporary directory unless `--indices_sw_param_data_path` is given. |
| `EL_PASO_USE_MAG_FIELD_CACHE` | Set to `0`/`false`/`no`/`off` to disable the magnetic-field joblib cache. |

The two download flags are read with `el_paso.utils.env_flag_enabled`: unset, empty, and `0`/`false`/`no`/`off` (any case) mean off, and any other value means on. Use that helper for any new boolean environment variable.

Credentials come from the environment when not passed as arguments: `CLIENT_ID`/`CLIENT_SECRET` (ESA NGRM and PROBA-V EPT recipes; CI stores them as the `ESA_CLIENT_ID`/`ESA_CLIENT_SECRET` secrets), `ERG_USER`/`ERG_PASSWORD` (Arase XEP real-time), and `SPACETRACK_USER`/`SPACETRACK_PASS` (`download_omm`). Never print or log their values.

`compute_magnetic_field_variables` caches IRBEM results with joblib in `~/.elpaso/joblib_cache` (`el_paso/cache.py`). The cache is removed at exit after a successful run and kept after a crash, and `import el_paso` cleans up stale caches.

## Release mode
`ep.activate_release_mode(...)` (`el_paso/release_mode.py`) stamps processed output with the user, package version, and git commit hash, and refuses to run on a dirty repository unless `dirty_ok=True`. Production data runs use it, so don't weaken the clean-repo check.

## Data standards
`el_paso/data_standards/gfz_standard.py` and `prbem_standard.py` each declare a `variable_infos` mapping of standard variable name → `VariableInfo` (units, description). `hooks/generate_metadata_stubs.py` uses these to regenerate matching typed attribute blocks in `el_paso/typing.py` (the `GFZVarNames` Literal), `el_paso/dataset/metadata.py`, and `el_paso/dataset/dataset_implementations.py`. Never hand-edit those generated blocks; change the standard's `variable_infos` and let the hook regenerate them instead. Class-body blocks sit between `# BEGIN/END GENERATED <NAME>` comments. The matching `Attributes:` docstring blocks sit between `<!-- BEGIN/END GENERATED <NAME> DOCS -->` lines, indented one level deeper than the attribute entries. That form is deliberate: griffe reads each marker as a continuation of the previous entry's description, the rendered docs hide it, and `mkdocs build --strict` stays warning-free. A `#`-style marker there would break the strict build.

### Adding a standardized variable
1. Add an entry to `self.variable_infos` in `gfz_standard.py` and/or `prbem_standard.py`, keyed by the internal name recipes use: `"<InternalName>": VariableInfo[GFZVarNames]("<standard_name>", "<description>", <unit>, [<dimension names>], <optional ("<dim>", "ascending"|"descending")>)`.
2. If the internal name (the dict key) is new, add it **by hand** to the `InternalName` Literal in `el_paso/typing.py`. It is not generated.
3. Regenerate with `.venv/bin/python hooks/generate_metadata_stubs.py` (pre-commit also runs it when a standard file changes). It updates `GFZVarNames` in `el_paso/typing.py` and the attribute blocks in `metadata.py` and `dataset_implementations.py`. Commit those regenerated files together with the standard. Until both steps are done, `ty` reports `invalid-assignment` (unknown internal name) and `invalid-argument-type` (unknown `standard_name`).
4. Run `tests/unittests/data_standards/`, `tests/unittests/test_typing.py`, and `tests/unittests/dataset/test_data_set_implementations.py`, following the `build-test-verify` skill.

#### Worked example: extending a standard with a family of related variables
PR #160 ("Add themis waves") reused `GFZStandard`'s existing EMFISIS/RBSP wave entries (`Wave_normal_angle`, `Magnetic_Power_Spectral_Density`, ...) for a second mission, THEMIS, and added the new variables THEMIS needed alongside them:

```python
# el_paso/typing.py — step 2, by hand
MagFieldVarTypes: TypeAlias = Literal[
    ...,
    "f_ce",
    "f_ce_Eq",
]

InternalName: TypeAlias = (
    Literal[
        ...,
        "Number_density",
        "Number_density_Eq",              # new: equatorial-mapped variant
        "Number_density_emfisis",         # new: per-instrument variant
        "Number_density_emfisis_Eq",
        "Number_density_efw",
        "Number_density_efw_Eq",
        "Number_density_hiss_derived",
        "Number_density_hiss_derived_Eq",
        "Wave_normal_angle",              # unchanged, already existed
        ...,
    ]
    | MagFieldVarTypes
)
```

```python
# el_paso/data_standards/gfz_standard.py — step 1
"f_ce": VariableInfo[GFZVarNames]("fce", "Electron gyrofrequency at the satellite location.", u.Hz, ["Epoch"]),
"f_ce_Eq": VariableInfo[GFZVarNames](
    "fce_eq", "Electron gyrofrequency mapped to the magnetic equator.", u.Hz, ["Epoch"]
),
"Number_density": VariableInfo[GFZVarNames](
    "density", "Electron number density at the satellite location.", u.cm ** (-3), ["Epoch"]
),
"Number_density_Eq": VariableInfo[GFZVarNames](
    "density_eq", "Electron number density at the magnetic equator.", u.cm ** (-3), ["Epoch"]
),
```

Two things worth noticing, both non-obvious from the 4-step procedure alone:
- **A new internal name doesn't have to go into every standard.** This PR added `f_ce`/`f_ce_Eq` and `Number_density`/`Number_density_Eq` to `GFZStandard`, and (separately) `Number_density`/`Number_density_Eq`/`xGEO_Eq` to `PRBEMStandard` — each standard only got the entries its own recipes actually save. Add to `gfz_standard.py` and/or `prbem_standard.py`, whichever a recipe needs.
- **A naming convention, not something the code enforces:** a quantity computed both at the satellite and mapped to the magnetic equator gets two internal names, `<Base>` and `<Base>_Eq`; the same physical quantity derived from different instruments gets one internal name per instrument (`Number_density_emfisis`, `Number_density_efw`, `Number_density_hiss_derived`). Match this pattern when the new variable is a variant of one that already exists, so `DataSet` users and other recipes can find it the same way.

A `variable_infos` entry only declares the variable's shape and unit; something still has to compute it. This PR also added `el_paso/processing/compute_electron_gyrofrequency.py` and `map_to_dipole_equator.py` as new public processing functions (see `export-public-api` for exporting one) and called them from `process_themis_fft_waves.py` and `process_rbsp_emfisis_waves.py` (see `add-new-recipe`).
