---
name: add-new-recipe
description: Add support for a new satellite mission or instrument recipe under el_paso/recipes/. Use when asked to add a new data source, processing recipe, or satellite/instrument support.
---

# Adding a new recipe

A "recipe" is one `process_*` function (e.g. `process_rbsp_hope_electrons`) that downloads raw files for a mission/instrument, extracts and processes variables, and saves them via a saving strategy. Recipes live under `el_paso/recipes/<mission>/`, one subpackage per mission (arase, dmsp, esa, goes, gps, poes, probav, rbsp, themis).

## Shape of a recipe file
Copy the structure of an existing one, e.g. `el_paso/recipes/rbsp/process_rbsp_hope_electrons.py`:
1. One or more `<...>_strategy(...)` factory functions, each returning an `ep.SavingStrategy` (e.g. `ep.saving_strategies.GFZStrategy(...)`), defined in the same file.
2. The `process_*` entry point: `ep.download(...)`, then `ep.ExtractionInfo`-based extraction, then `el_paso/processing/` steps, then `ep.save(...)` with a strategy from step 1. Its Google-style docstring becomes the CLI `--help` text, so keep the `Args:` section accurate.
3. A `satellite` parameter, if any, typed as the mission's `Literal[...]` alias (e.g. `RBSPSatellite` from the mission `__init__.py`), never bare `str`.
4. At the bottom, `if __name__ == "__main__": ep.run_recipe_cli(process_<name>)`, or `ep.run_recipe_cli(process_<name>, defaults=CLI_DEFAULTS)` if the module defines a `CLI_DEFAULTS` dict. This keeps `python -m` and the `el-paso` CLI exposing identical options.

## Register it in four places
1. **Mission `__init__.py`:** add the `process_*` function and every `<...>_strategy` factory to both the runtime `lazy.attach(submod_attrs=...)` and the `TYPE_CHECKING` imports and `__all__`. The `export-public-api` skill explains why both are needed.
2. **`RECIPES` in `el_paso/cli/app.py`:** add a `RecipeEntry(mission, command, module, function)`. `command` is the kebab-case sub-command name.
3. **`docs/API_reference/recipes/<mission>.md`:** add a `::: el_paso.recipes.<mission>.<module>.<function>` line.
4. **`README.md`:** add it to the "Available processing scripts" list.

**New mission** (not just a new recipe in an existing one): also add the mission to `el_paso/recipes/__init__.py` (in `submodules=[...]` and in the `TYPE_CHECKING` block), create `docs/API_reference/recipes/<mission>.md`, and add that page to the `Recipes:` section of `mkdocs.yml`'s nav.

## What `hooks/check_recipe_strategies.py` enforces (a pre-commit hook; violations fail the commit)
1. `process_*` gets its `SavingStrategy` from a named `<...>_strategy(...)` function. No inline `ep.saving_strategies.*Strategy(...)`.
2. `satellite` parameters are `Literal`, not `str`.
3. Every `<...>_strategy` factory is in the mission `__init__.py`'s `__all__`.
4. Every `process_*` function has a `RecipeEntry`, and every `RecipeEntry` points at a real function.

The hook does not check the runtime `lazy.attach` entry, the docs page, or the README. The runtime entry is covered by the export tests listed under Tests below; the docs page and README are up to you.

`hooks/generate_metadata_stubs.py` fires when `el_paso/data_standards/gfz_standard.py`, `prbem_standard.py`, or the generator itself changes. For a recipe it matters only if the recipe also needs a new standardized variable. In that case, follow "Adding a standardized variable" in `.claude/agent-guides/architecture.md`.

## Tests
- Add a case for each new strategy factory to the `CASES` list in `tests/unittests/test_recipe_saving_strategies.py`, and mark any new test `@pytest.mark.basic`.
- These existing tests then cover the new recipe automatically. Run them:
  ```bash
  .venv/bin/python -m pytest tests/unittests/test_recipe_saving_strategies.py tests/unittests/test_recipe_cli.py -m "not visual" -o log_cli=false
  ```
  `test_recipe_cli.py` builds every `RECIPES` entry's command, checks the shared option surface, and checks that each `process_*` resolves at runtime from its mission package. `test_every_recipe_strategy_is_exported_from_its_mission_package` does the same for strategy factories.
- A full end-to-end test belongs in `tests/system/test_<mission>.py`. It needs reference data in the Zenodo archive, which a maintainer has to upload, so flag that in the PR. Say too if downloading the data needs credentials.

## Try it without downloading
`--dry-run` resolves and prints the recipe's arguments, then exits without downloading or processing:
```bash
.venv/bin/el-paso <mission> <command> --start-time 2013-03-16 --end-time 2013-03-17 \
    --raw-data-path /tmp/el_paso-raw --processed-data-path /tmp/el_paso-processed --dry-run
```
For a real run, keep both paths outside the repo. They default to `.`, and `.mat`/`.h5` output is not gitignored.

For how download, extraction, processing, saving strategies, and data standards fit together, and how the CLI builds commands lazily, see `.claude/agent-guides/architecture.md`.
