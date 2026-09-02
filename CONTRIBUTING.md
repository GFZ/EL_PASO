<!--
SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
SPDX-FileContributor: Bernhard Haas
SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
SPDX-FileContributor: Sahil Jhawar

SPDX-License-Identifier: Apache-2.0
-->

# Contributing to EL-PASO

Thanks for your interest in contributing. This document covers how to report issues, set up a dev environment, and get a pull request merged.

## Ways to contribute

- **Bug reports and feature requests:** use [GitHub Issues](https://github.com/GFZ/EL_PASO/issues).
- **Code contributions:** fork the repo, make your changes, open a pull request.
- **Documentation:** fixes and additions to `docs/` and `tutorials/` are welcome.

## Reporting bugs

Open a new issue and include:

- A short summary of the problem.
- Steps to reproduce, with a minimal code sample if possible.
- Expected behavior vs. what actually happened.
- Python version, OS, and `EL-PASO` version (`pip show el_paso` or the git commit).

## Development setup

`EL-PASO` requires Python 3.12+ and uses [uv](https://docs.astral.sh/uv/) for dependency management. The `IRBEM` FORTRAN library is compiled automatically during install via the custom `setup.py` build hook.

```bash
git clone https://github.com/GFZ/EL_PASO.git
cd EL_PASO
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -e .
```

Verify the install:

```bash
python examples/minimal_example.py
```

### Pre-commit hooks

**Always** install and enable pre-commit before making changes:

```bash
uv pip install pre-commit
pre-commit install
```

This runs on every commit:

- `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml`, `check-added-large-files`
- `reuse`: SPDX license header compliance
- `ruff-check` and `ruff-format` : linting and formatting (scoped to `el_paso/`)
- `ty` : static type checking (scoped to `el_paso/`)

You can run all hooks manually against the full codebase with:

```bash
pre-commit run --all-files
```

## Code style

- Formatting and linting are enforced by `ruff`, configured in `ruff.toml`. Run `ruff check .` and `ruff format .` before committing if you're not using pre-commit.
- Type hints are required on new and modified code in `el_paso/`. Type checking is done with `ty`; run `ty check` locally to catch issues before pushing.
- Follow existing naming and module structure. If you're unsure where new functionality belongs, open an issue first to discuss.

## License headers (REUSE)

This project is dual-licensed under Apache-2.0 and LGPL-3.0-only, and follows the [REUSE](https://reuse.software/) specification. Every source file needs an SPDX header, for example:

```python
# SPDX-FileCopyrightText: {Year} GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: {Your Name}
#
# SPDX-License-Identifier: Apache-2.0
```

Add yourself as a `SPDX-FileContributor` on files you meaningfully change. Run `reuse lint` to check compliance before opening a PR.

## Tests

Tests live in `tests/` and use `pytest`. Some tests require external test data, fetch it first:

```bash
./download_data_for_tests.sh
```

Then run the suite:

```bash
pytest
```

If you add a new feature or fix a bug, add or update a test that covers it. Coverage is tracked via `coverage` and reported on Coveralls.

## Submitting a pull request

1. Fork the repository and clone your fork.
2. Create a branch: `git checkout -b feature-branch` or `git checkout -b fix-branch`.
3. Make your changes, keeping the diff focused on one issue or feature.
4. Make sure pre-commit hooks pass and `pytest` is green.
5. Push to your fork: `git push origin feature-branch`.
6. Open a PR against `main` in `GFZ/EL_PASO`, describing what changed and why. Reference any related issue. Adhere to the PR template provided.

A maintainer will review your PR, may request changes, and will merge once it's ready.

## Adding a new processing script or satellite mission

If you're contributing support for a new satellite/instrument, look at an existing entry under `el_paso/` for structure (input format handling, metadata, output to PRBEM-style standards) and add a corresponding example under `examples/` if practical. Update the "Available processing scripts" list in `README.md`.

## Questions

If anything here is unclear, open an issue or start a discussion on an existing PR. That's easier for others to find later than a private email.
