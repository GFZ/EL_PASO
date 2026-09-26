---
name: create-pull-request
description: Prepare a change for a pull request against GFZ/EL_PASO, covering SPDX/REUSE headers, the PR template's per-type checklist, and commit conventions. Use before opening or updating a PR.
---

# Creating a pull request

## License headers (REUSE/SPDX)
Every new or changed source file needs an SPDX header. The project is dual-licensed Apache-2.0 / LGPL-3.0-only:
```python
# SPDX-FileCopyrightText: {Year} GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: {Your Name}
#
# SPDX-License-Identifier: Apache-2.0
```
Add yourself as `SPDX-FileContributor` on any file you meaningfully change, and don't remove existing contributors. Check with `.venv/bin/reuse lint` (it also runs in pre-commit and in CI's `reuse_compliance.yml`).

Files that can't hold a comment header, or whose header would get in the way (JSON, images, the PR template, the `.claude/` Markdown files that agents load), get a sidecar `<file>.license` with the same SPDX lines instead. Copy an existing one, e.g. `.claude/CLAUDE.md.license`.

## Before opening the PR
- Never push, open a PR, or comment on a PR/issue without the user's explicit go-ahead for that specific action. Draft the title and body, show them, and wait.
- PRs target `main` on `GFZ/EL_PASO`, from a topic branch (`feature-...` / `fix-...`).
- `.venv/bin/pre-commit run --all-files` is clean.
- Tests pass, scoped per the `build-test-verify` skill (not the full suite). New tests carry `@pytest.mark.basic`, since CI runs only `-m basic`.
- `README.md` / `CONTRIBUTING.md` updated if this changes install steps, usage, or the "Available processing scripts" list.

## PR template
`.github/pull_request_template.md` has one collapsible `<details>` section per PR type: **New processing recipe / satellite mission**, **Bug fix**, **Feature / enhancement**, **Documentation**, **Dependency bump / chore**, **Miscellaneous**. Fill in the top-level summary, then fill the one section matching this PR, and delete the rest. Don't leave every section in the description. When opening the PR, strip the `<details>`/`<summary>` wrapper tags from the section you keep, leaving its heading as a plain `###`/bold line and its content visible by default rather than collapsed.

Notable per-type items:
- **New recipe**: recipe follows an existing mission's structure, mission `__init__.py` updated, README recipe list updated, a test covers it, and any credentials/secrets needed for download are called out. See the `add-new-recipe` skill for the full pattern.
- **Dependency bump / chore**: confirm the IRBEM build still succeeds if the change touches build hooks or the Fortran toolchain; test CI workflow file changes via a draft PR or `workflow_dispatch` rather than assuming. The Apptainer image (`el_paso.def`) copies only `el_paso/`, `scripts/`, `pyproject.toml`, `uv.lock`, the two build-hook files, and `README.md`. If install starts needing another root-level file, add it to the `%files` section too.

The repo has no CHANGELOG file. Ignore the template's "Changelog updated" items rather than creating one, and describe user-visible changes in the PR body instead.

## Commit style
Short, imperative, loosely Conventional-Commits-flavored (`feat:`, `fix(...)`, `chore(...)`, or plain "add X" / "update Y"). This is an observed convention, not enforced by tooling. Keep each PR's diff focused on one issue or feature, per `CONTRIBUTING.md`.
