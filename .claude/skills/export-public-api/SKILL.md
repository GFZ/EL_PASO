---
name: export-public-api
description: Add, rename, or remove a public function, class, or module in EL-PASO (e.g. a new processing step in el_paso/processing/, a dataset helper, a top-level ep.* function). Covers the lazy_loader export pattern, the API docs pages, updating tutorials/examples on renames, and building the docs locally. Use whenever something must be importable as ep.<name> or el_paso.<subpackage>.<name>, or when building or checking the mkdocs site.
---

# Exporting public API

`el_paso/__init__.py`, `el_paso/processing/__init__.py`, `el_paso/processing/magnetic_field_utils/__init__.py`, `el_paso/dataset/__init__.py`, and every `el_paso/recipes/**/__init__.py` use `lazy_loader.attach`. Nothing is imported until someone accesses it, which keeps `import el_paso` and the CLI fast. As a result, every public symbol is declared twice in its package's `__init__.py`, and the two copies must agree:

1. **Runtime:** add it to the `lazy.attach(...)` call, under `submod_attrs={"<module>": [..., "<name>"]}` (or `submodules=[...]` for a whole subpackage). If you miss this, `ep.<name>` raises `AttributeError` at runtime, and ty and ruff will not catch it.
2. **Type checking:** add the matching import inside the `if TYPE_CHECKING:` block, and add the name to that block's `__all__` list. If you miss this, type checkers and IDEs don't see the symbol, even though it works at runtime.

Look at the neighboring entries in the same `__init__.py` and copy their form exactly. For recipe packages, the runtime side is covered by tests: `test_every_recipe_is_exported_from_its_mission_package` (`tests/unittests/test_recipe_cli.py`) and `test_every_recipe_strategy_is_exported_from_its_mission_package` (`tests/unittests/test_recipe_saving_strategies.py`). `hooks/check_recipe_strategies.py` checks only the `TYPE_CHECKING`-side `__all__`. For the other packages, nothing checks either side, so the `Check` step below is the only guard.

**`el_paso/typing.py` is different.** It doesn't use `lazy_loader`. Its own `__getattr__` resolves names from a `_LAZY_EXPORTS = {"<name>": ("<module>", "<attribute>")}` dict. A lazily exported type needs an entry there plus the matching `TYPE_CHECKING` import and `__all__` entry. `tests/unittests/test_typing.py` checks that all three agree, so run it after touching `typing.py`.

The module you add the symbol to should follow the same rule: import heavy third-party packages inside the function that uses them (with `# noqa: PLC0415`), or under `if TYPE_CHECKING:` for annotation-only use, not at module level.

**`el_paso/saving_strategies/__init__.py` and `el_paso/data_standards/__init__.py` are different too.** Neither uses `lazy_loader` — each is just a flat `from ... import ...` per class plus a matching `__all__`. Adding a new `SavingStrategy` or `DataStandard`:

1. **Class:** `el_paso/saving_strategies/<name>_strategy.py`, subclassing `ep.SavingStrategy` (`el_paso/saving_strategy.py`) and implementing its abstract methods (`get_time_intervals_to_save`, `get_file_path`, `get_file_path_stem`, `get_file_name_stem`; override `standardize_variable` too only if the strategy needs different standardization, as `SingleFileStrategy` does). For a new `DataStandard`, subclass `el_paso.data_standard.DataStandard` and populate `self.variable_infos` — see "Adding a standardized variable" in `.claude/agent-guides/architecture.md` for the `VariableInfo` shape.
2. **Register:** add the class to both the import list and `__all__` in `saving_strategies/__init__.py` (or `data_standards/__init__.py`). Nothing checks this automatically — a missed entry only surfaces as an `AttributeError`/`ImportError` the first time someone uses `ep.saving_strategies.<Name>` / `ep.data_standards.<Name>`, so verify it by hand (see Check below).
3. **Docs:** add `docs/API_reference/saving_strategies/<name>.md` (or `data_standards/<name>.md`) with a `::: el_paso.saving_strategies.<module>.<Class>` mkdocstrings directive — copy an existing page's `options: members:` list — and list it under the matching `Saving Strategies:` / `Data Standards:` section of `mkdocs.yml`'s nav.
4. **Tests:** a strategy factory used by a recipe needs a case in `tests/unittests/test_recipe_saving_strategies.py`'s `CASES` list (see the `add-new-recipe` skill). Either way, add or extend a test under `tests/unittests/saving_strategies/test_<name>_strategy.py` or `tests/unittests/data_standards/`, mirroring an existing one.

## Docs
The API reference is hand-listed, not auto-discovered:
- `docs/API_reference/<area>/...md` files hold `::: el_paso.<dotted.path>` mkdocstrings directives. Add one for the new symbol, next to its siblings.
- A new page must also be added to the `nav:` section of `mkdocs.yml`, or it won't appear on the site.
- Public functions need Google-style docstrings (`ruff.toml` sets `convention = "google"`). mkdocstrings renders them, and for recipes the CLI builds its `--help` text from them.

To build the docs locally, do what ReadTheDocs does, in a separate environment so `.venv` is left untouched (`uv sync` removes any package that isn't in the lockfile).
```bash
UV_PROJECT_ENVIRONMENT=/tmp/el_paso-docs-venv uv sync --locked --no-default-groups --group docs
/tmp/el_paso-docs-venv/bin/mkdocs build --strict        # output goes to site/, which is gitignored
```
The strict build is warning-free, so any warning or error is from your change. A common cause is a docstring `Attributes:`/`Args:` line that isn't in `name (type): description` form. Two machine-local failures are not your change:
- A notebook page fails with an import from a personal Jupyter config (e.g. `jupyter_contrib_nbextensions`). Rerun with `JUPYTER_CONFIG_DIR` pointed at an empty directory.
- `Couldn't load inventory ... CERTIFICATE_VERIFY_FAILED`. Rerun with `SSL_CERT_FILE="$(/tmp/el_paso-docs-venv/bin/python -c 'import certifi; print(certifi.where())')"`.

## Renaming or removing public API
Tutorials, examples, and the executable paper call the public API directly, and nothing tests them in CI. Before renaming or removing a symbol, find every use:
```bash
grep -rn "<old_name>" el_paso tests docs tutorials examples paper scripts README.md
```
Notebooks (`.ipynb`) are JSON, so edit their code cells carefully and keep their outputs intact. Removing or renaming a public name is a breaking change: say so in the PR (the template's Feature section asks about backwards compatibility).

## Check
Confirm the export resolves both ways without running the test suite:
```bash
.venv/bin/python -c "import el_paso as ep; print(ep.processing.<name>)"   # adjust the path to the package
.venv/bin/ty check
```
Then run the tests that cover the changed module, following the `build-test-verify` skill.
