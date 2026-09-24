---
name: build-test-verify
description: Install EL-PASO, run its tests (only those covering the changed files, never the whole suite), lint, and type-check. Use before committing any change to el_paso/ or tests/, or whenever asked to verify a change works.
---

# Build, test, verify

## Running tools
Each shell command an agent runs starts in a fresh shell, so `source .venv/bin/activate` does not carry over to the next command. Call tools through the venv explicitly: `.venv/bin/python -m pytest`, `.venv/bin/ruff`, `.venv/bin/ty`, `.venv/bin/pre-commit`, `.venv/bin/python`.

## Install
```bash
uv sync --locked                                  # same as CI
.venv/bin/python examples/minimal_example.py      # verifies the install
.venv/bin/pre-commit install                      # once per checkout, before making changes
```
`uv sync --locked` creates `.venv` with the package (editable), the locked dependencies, and the `dev` group: pytest, ruff, ty, pre-commit, reuseify, coverage. It also compiles `libirbem.so` via the custom `hatchling` build hook. If this step fails, the problem is the Fortran build, not your Python change. (A plain `uv pip install -e .` installs the package but none of the dev tools.)

## Tests: run only the tests for the files you changed, never the full suite
**Never run bare `pytest`** (it runs every unit test listed in `pytest.ini`'s `testpaths`) **or `pytest tests`** (that adds the system and comparison tests too). A PreToolUse hook (`.claude/hooks/guard_bash.py`) blocks both.

Find the tests that cover each changed file, and run only those:
```bash
grep -rl "<changed_module_name>" tests/unittests    # e.g. grep -rl "bin_by_time" tests/unittests
.venv/bin/python -m pytest <those files> -m "not visual" -o log_cli=false
```
`pytest.ini` turns on live INFO logging (`log_cli = 1`), which floods the output. `-o log_cli=false` turns it off; failures still show their captured logs.

`tests/unittests/` roughly mirrors `el_paso/` (`dataset/`, `processing/`, `data_standards/`, `saving_strategies/`, and top-level modules at `tests/unittests/test_<module>.py`), but the grep is what counts. If nothing covers the changed file, say so instead of running a broader set.

Recipe changes (`el_paso/recipes/**`, `el_paso/cli/**`) are covered by `tests/unittests/test_recipe_saving_strategies.py` and `tests/unittests/test_recipe_cli.py`. Changes to `el_paso/typing.py` are covered by `tests/unittests/test_typing.py`.

**Use `-m "not visual"`, not `-m basic`, for targeted runs.** `-m basic` silently deselects any test that lacks the marker and still reports a pass, so an unmarked test would look green without running. `-m visual` tests only produce plots for someone to check by eye. Run them only when asked.

**Mark every new test `@pytest.mark.basic`** (the repo's convention is a per-test decorator). CI runs only `-m basic`, so an unmarked test never runs in CI.

System tests (`tests/system/test_<mission>.py`) download real mission data, need the Zenodo reference data (`./download_data_for_tests.sh`), and some need credentials (`CLIENT_ID`/`CLIENT_SECRET`, `SPACETRACK_USER`/`SPACETRACK_PASS`). Run one only when asked. Files prefixed `_test_` are deliberately disabled.

Regenerate reference solutions only when asked: `.venv/bin/python -m pytest <file> --renew_solution=true`.

## Lint & type-check
```bash
.venv/bin/ruff check el_paso tests      # the scope pre-commit uses; CI lints el_paso/
.venv/bin/ruff format el_paso tests
.venv/bin/ty check                      # scoped to el_paso/, see pyproject.toml [tool.ty.src]
```
Run all pre-commit hooks (including the recipe-consistency and metadata-stub hooks and REUSE/SPDX linting) on just the files you changed:
```bash
.venv/bin/pre-commit run --files <changed files>
```
Use `--all-files` only before opening a PR. Its whitespace and end-of-file fixers can rewrite files you didn't touch.
