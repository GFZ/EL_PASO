<!--
SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
SPDX-FileContributor: Bernhard Haas

SPDX-License-Identifier: Apache-2.0
-->

|            |                                                                                                                                                                                                                                                                                                                                                            |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Versions   | [![PyPi](https://badge.fury.io/py/el-paso.svg)](https://badge.fury.io/py/el-paso) [![Python version](https://img.shields.io/pypi/pyversions/el-paso.svg)](https://badge.fury.io/py/el-paso) [![Zenodo](https://img.shields.io/badge/Zenodo-10.5281/zenodo.20760790-blue)](https://doi.org/10.5281/zenodo.20760790)                                     |
| Status     | [![Tests](https://github.com/GFZ/EL_PASO/actions/workflows/test.yml/badge.svg)](https://github.com/GFZ/EL_PASO/actions/workflows/test.yml) [![Coverage Status](https://coveralls.io/repos/github/GFZ/EL_PASO/badge.svg?branch=main)](https://coveralls.io/github/GFZ/EL_PASO?branch=main) [![Docs](https://app.readthedocs.org/projects/el-paso/badge/?version=latest)](https://el-paso.readthedocs.io/en/latest/) [![REUSE status](https://api.reuse.software/badge/github.com/GFZ/EL_PASO)](https://api.reuse.software/info/github.com/GFZ/EL_PASO) |
| Tools      | [![Pre-Commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty) |
| License    | [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)                                                                                                                    |
| Paper      | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/GFZ/EL_PASO/executable_paper?urlpath=%2Fdoc%2Ftree%2Fpaper%2Fel_paso_executable_paper.ipynb) [![Paper](https://img.shields.io/badge/Open_Access-10.22541/essoar.15002644/v1-blue)](https://essopenarchive.org/doi/abs/10.22541/essoar.15002644/v1)                          |
| SPEC       | [![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/) [![SPEC 1 — Lazy Loading of Submodules and Functions](https://img.shields.io/badge/SPEC-1-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0001/) |

# ELaborative Particle Analysis from Satellite Observations (EL-PASO)

`EL-PASO` is a Python framework designed to streamline the download, processing, and saving of satellite particle observation data.

Its primary purpose is to prepare and standardize particle data for use in radiation belt modeling.

<img width="1493" height="963" alt="el_paso_recipe" src="https://github.com/user-attachments/assets/876072eb-00a9-44c6-9b83-a404bc49e8e1" />

## Features

- **Format Flexibility:** Capable of handling different input formats including `cdf`, `netcdf`, `h5`, `ascii`, and `json`
- **Integrated Processing:** Provides a comprehensive set of functions for common particle data analysis tasks
- **Supports Metadata:** Stores all processing and metadata alongside the data, ensuring full traceability and reproducibility.
- **Standardized output files:** Saving processed data in different standards (e.g. PRBEM) to enable easy loading and sharing of processed data

Full documentation can be viewed [here](https://el-paso.readthedocs.io/en/latest/).

## Available processing scripts

Color code: <span style="color:green">publication ready</span>, <span style="color:E6D753">needs verification</span>, <span style="color:E67D53">draft</span>

- **Arase**
    - <span style="color:E6D753"> MEPe
    - <span style="color:green"> XEP (archived and real-time)
    - <span style="color:green"> PWE density
- **DMSP**
    - <span style="color:E6D753"> SSJ (electrons)
- **GOES (13 and 15)**
    - <span style="color:green"> EPS MAGED and EPEAD combined
- **GOES-R**
    - <span style="color:E6D753"> MPS-High (archived and real-time)
- **GPS**
    - <span style="color:E6D753"> GPS processing
- **ESA**
    - <span style="color:E67D53"> NGRM satellites
    - <span style="color:E67D53"> GSSC EMU (electrons and protons)
- **POES (no proton correction)**
    - <span style="color:green"> MEPED (electrons)
    - <span style="color:green"> TED (electrons)
- **PROBA-V**
    - <span style="color:E6D753"> EPT (electrons and protons)
- **THEMIS**
    - <span style="color:E6D753"> FFT waves
    - <span style="color:E6D753"> Spacecraft-potential density
- **Van Allen Probes**
    - <span style="color:green"> HOPE (electrons)
    - <span style="color:E67D53"> HOPE (protons)
    - <span style="color:green"> MagEIS (electrons)
    - <span style="color:E67D53"> MagEIS (protons)
    - <span style="color:E67D53"> RBSPICE (protons)
    - <span style="color:green"> ECT-combined
    - <span style="color:green"> EMFISIS waves
    - <span style="color:green"> EMFISIS and EFW density
- **Models**
    - <span style="color:green"> Last closed drift shell (LCDS)
    - <span style="color:green"> Solar wind and geomagnetic indices

Every recipe shares one command line interface; see
[Running a recipe](#running-a-recipe) below.

## Installation

### Step 0: Install Prerequisites

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

### Step 1: Clone the Repository

Begin by cloning the EL-PASO repository and navigating into its directory.

```bash
git clone https://github.com/GFZ/EL_PASO.git
cd EL_PASO
```

### Step 2: Set up a Python Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install .
```

The custom `setup.py` script will automatically download and compile the IRBEM Fortran library during this step.

### Verifying the Installation

You can validate your installation by running the minimal example located in *examples*:

```bash
python examples/minimal_example.py
```
> [!TIP]
> #### Using the Apptainer Image
>
> Instead of setting up a Python environment yourself, you can pull a prebuilt [Apptainer](https://apptainer.org/) [el_paso](https://github.com/GFZ/EL_PASO/pkgs/container/el_paso) image:
> ```bash
> apptainer pull -F elpaso.sif oras://ghcr.io/gfz/el_paso:latest
> ```
> Run a command inside the image with `apptainer exec` or `apptainer run`, e.g.:
> ```bash
> apptainer exec elpaso.sif python examples/minimal_example.py
> ```
> Available tags mirror the CI build: `latest` (most recent build on `main`), a specific commit SHA, or a released package version (e.g. `oras://ghcr.io/gfz/el_paso:2.1.2`).

## Running a recipe

Installing EL-PASO provides the `el-paso` command, which exposes every recipe
under a common set of options:

```bash
el-paso list                 # show every available recipe
el-paso poes meped --help    # options for one recipe

el-paso poes meped \
    --start-time 2013-03-16 --end-time 2013-03-16T23:59:59 \
    --satellite noaa15 --mag-field T89 --bin-cadence 10s \
    --raw-data-path ./raw --processed-data-path ./processed
```

Recipes can equally be run as modules, which is convenient inside job scripts:

```bash
python -m el_paso.recipes.poes.process_poes_meped \
    --start-time 2013-03-16 --end-time 2013-03-16T23:59:59
```

Both forms accept the same options, because both are generated from the recipe
function's own signature. `--start-time` and `--end-time` are required; the rest
have per-recipe defaults. Common options are
`--satellite` (repeat it to process several, or pass `all`), `--mag-field`, `--bin-cadence`
(`10s`, `5min`, `1h`), `--raw-data-path`, `--processed-data-path` and
`--num-cores`. Add `--dry-run` to print the resolved arguments without
processing anything, and `--install-completion` to set up shell completion.
Which options a given recipe accepts depends on what it actually supports, so
check `--help`.

See the [command line documentation](docs/getting_started/command_line.md) for
the full list.

## Testing

### Step 1: Download the Test Data

Most tests rely on reference/system test data hosted on [Hugging Face](https://huggingface.co/datasets/spsw/EL-PASO-Test-Data). Download it by running the following script from the repository root:

```bash
bash download_data_for_tests.sh
```

This fetches the pinned revision of the dataset into `tests/system/data/` using the Hugging Face CLI (run through `uvx`, so [uv](https://docs.astral.sh/uv/) must be installed). You only need to do this once (rerun it if the pinned revision changes).

`pytest` is installed as part of the regular dependencies (see [Installation](#installation)), so no separate test install step is needed.

### Step 2: Run the Tests

Run the full test suite with `pytest`:

```bash
pytest tests
```

Tests are grouped using pytest markers, defined in `pytest.ini`:

- `basic`: quick tests suitable for fast, everyday verification of the code. This is what CI runs on every push/PR:

  ```bash
  pytest tests -m basic
  ```

- `visual`: tests that produce plots or other visual output which must be checked manually rather than being asserted automatically:

  ```bash
  pytest tests -m visual
  ```

You can combine or exclude markers using standard pytest marker expressions, e.g. to run everything except visual tests:

```bash
pytest tests -m "not visual"
```

Some system tests compare against previously stored reference solutions. Pass `--renew_solution` to regenerate and overwrite those reference solutions instead of comparing against them:

```bash
pytest tests --renew_solution=true
```

Use this only when you intend to intentionally update the stored reference outputs.

## Contributing

Please read our comprehensive [contribution guide](CONTRIBUTING.md) before submitting a pull request.

## Citation

If you use `EL-PASO` in your research, please cite the associated preprint:

> Haas, B., Drozdov, A. Y., and Jhawar, S. 	EL-PASO: An Open-Source Python Library for Processing and Standardizing Particle Measurements Taken in Space. ESS Open Archive. https://essopenarchive.org/doi/full/10.22541/essoar.15002644/v1

```bibtex
@article{
doi:10.22541/essoar.15002644/v1,
author = {Bernhard Haas  and Alexander Y. Drozdov  and Sahil Jhawar },
title = {EL-PASO: An Open-Source Python Library for Processing and Standardizing Particle Measurements Taken in Space},
journal = {ESS Open Archive},
volume = {2026},
number = {0502},
pages = {},
year = {2026},
doi = {10.22541/essoar.15002644/v1},
URL = {https://essopenarchive.org/doi/abs/10.22541/essoar.15002644/v1},
eprint = {https://essopenarchive.org/doi/pdf/10.22541/essoar.15002644/v1}}
```

To cite this repository you can use [CITATION.cff](CITATION.cff).

## Acknowledgements

This work has been funded by the German Research Foundation (NFDI4Earth, DFG project no. 460036893, https://www.nfdi4earth.de/).
The authors acknowledge the work of Mátyás Szabó-Roberts who led the foundation for the EL-PASO framework.

The thank the authors of the [IRBEM library](https://github.com/PRBEM/IRBEM) for providing their code.
