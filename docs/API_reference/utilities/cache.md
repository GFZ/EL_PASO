<!--
SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
SPDX-FileContributor: Bernhard Haas

SPDX-License-Identifier: Apache-2.0
-->

# Cache

Disk caching utilities used by `compute_magnetic_field_variables` to avoid
re-running expensive IRBEM computations when downstream code crashes.

The default cache location is `$HOME/.elpaso/joblib_cache`.  Stale entries
older than 7 days are purged automatically at `import el_paso` time, and
the entire cache is cleared on graceful interpreter exit via `atexit`.

::: el_paso.cache
