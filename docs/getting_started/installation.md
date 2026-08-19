<!--
SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
SPDX-FileContributor: Bernhard Haas

SPDX-License-Identifier: Apache-2.0
-->

# Installation

## Installing EL-PASO

After cloning the repository, the main package can be installed using a virtual environment and pip. Make sure your current directory is set to the EL-PASO repository:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install .
```

## Validation of installation

You can validate your installation by running the minimal example located in _examples_:

```bash
python3 examples/minimal_example.py
```
