<!--
SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
SPDX-FileContributor: Bernhard Haas

SPDX-License-Identifier: Apache-2.0
-->

# Installation

## Step 0: Install prerequisites

EL-PASO uses [uv](https://docs.astral.sh/uv/) for environment and dependency management. Install it if you do not have it yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Installing EL-PASO also compiles the IRBEM Fortran library, so a Fortran compiler (`gfortran`) and `make` must be available on your `PATH`. For example:

```bash
# Debian/Ubuntu
sudo apt install gfortran make

# macOS (Homebrew)
brew install gcc make
```

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
