<!--
Expand the section below that matches this PR, fill it in, and delete the other sections.
-->

## Summary

<!-- What does this PR change, and why? -->

## Checklist
- [ ] SPDX header present on new/changed files, with yourself as `SPDX-FileContributor`.
- [ ] `ruff check .` / `ruff format .` and `ty check` pass (or pre-commit ran clean).
- [ ] Tests added/updated as needed; `pytest -m basic` (and/or `-m visual`) passes locally.
- [ ] `README.md` / `CONTRIBUTING.md` updated if this changes install steps, usage, or the processing-script list.
- [ ] `pre-commit` executed without any failures

---

<details>
<summary><strong>New processing recipe / satellite mission</strong></summary>

- Mission/instrument:
- Data product(s):
- Related issue:

**Checklist**

- [ ] Recipe lives under `el_paso/recipes/<mission>/` and follows the structure of an existing recipe (input handling, metadata, output to PRBEM-style standard names).
- [ ] `el_paso/recipes/<mission>/__init__.py` updated if a new module was added.
- [ ] `README.md` "Available processing scripts" list updated to include this recipe.
- [ ] Type hints added on new/changed code.
- [ ] A test covers the new recipe.
- [ ] If the recipe needs new reference/system test data, it's been added to the Zenodo dataset (or flagged in this PR for a maintainer to do so).
- [ ] If the recipe needs credentials/secrets to download data (e.g. `ESA_CLIENT_ID`), that's called out below.

**Notes for reviewers**

<!-- Anything unusual about the data source, quirks in the raw files, units, or assumptions worth flagging. -->

</details>

<details>
<summary><strong>Bug fix</strong></summary>

- Bug: [Brief description of the bug]
- Root cause: [Explanation of what caused the bug]
- Related issue: [Link to the related issue]

**Reproduction**

<!-- Minimal steps or code sample that showed the bug before this fix. -->

**Checklist**

- [ ] Changelog/docs updated if the fix changes documented behavior.

**Notes for reviewers**

</details>

<details>
<summary><strong>Feature / enhancement</strong></summary>

- Feature:
- Related issue/discussion:

**Design notes**

<!-- Any non-obvious API or architecture decisions, alternatives considered, and why this approach. -->

**Checklist**

- [ ] Public API additions have type hints and docstrings (Google convention, per `ruff.toml`).
- [ ] `examples/minimal_example.py` (or another example) updated if this changes the primary usage pattern.
- [ ] `README.md` updated if this changes install steps, usage, or the feature set described there.
- [ ] Backwards compatibility considered: existing recipes and tests still pass unmodified, or breaking changes are called out.

**Notes for reviewers**

<!-- Anything that needs special attention: perf implications, new dependencies, follow-up work planned. -->

</details>

<details>
<summary><strong>Documentation</strong></summary>

<!-- What docs changed (README, CONTRIBUTING, docs/, tutorials/, docstrings) and why? -->

**Checklist**

- [ ] Content is accurate against current code behavior (commands, flags, file paths verified to still exist).
- [ ] Links checked (internal anchors and external URLs resolve).
- [ ] Code samples in docs actually run against the current API.
- [ ] No unrelated formatting churn mixed into the diff.

**Notes for reviewers**

<!-- Anything that needs a second pair of eyes, e.g. rendering in the docs site (readthedocs) if applicable. -->

</details>

<details>
<summary><strong>Dependency bump / chore</strong></summary>

- What changed:
- Why (security fix, new feature needed elsewhere, upstream deprecation, etc.):

**Checklist**

- [ ] `pyproject.toml` version constraint updated consistently.

- [ ] IRBEM build still succeeds if the change touches build hooks or Fortran toolchain.
- [ ] CI workflows (`.github/workflows/*.yml`) still pass; any workflow file changes tested via `workflow_dispatch` or a draft PR run.
- [ ] Changelog/release notes updated if this affects the published package.

**Notes for reviewers**

<!-- Breaking changes in the dependency, migration steps, or anything that needs a version bump (`bumpver`). -->

</details>


<details>
<summary><strong>Miscellaneous</strong></summary>

<!-- Any other changes that don't fit into the above categories. -->

</details>
