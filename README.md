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
| Paper      | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/GFZ/EL_PASO/executable_paper?urlpath=%2Fdoc%2Ftree%2Fpaper%2Fel_paso_executable_paper.ipynb) [![Paper](https://img.shields.io/badge/DOI-10.1029/2026EA005044-blue)](https://doi.org/10.1029/2026EA005044)                          |
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

Color code: ${\color{#4A8A4F}\textsf{publication ready}}$, ${\color{#E6D753}\textsf{needs verification}}$, ${\color{#E67D53}\textsf{draft}}$

- **Arase**
    - ${\color{#E6D753}\textsf{MEPe}}$
    - ${\color{#E6D753}\textsf{HEP (L and H)}}$
    - ${\color{#4A8A4F}\textsf{XEP (archived and real-time)}}$
    - ${\color{#4A8A4F}\textsf{PWE density}}$
- **DMSP (F16, F17, F18)**
    - ${\color{#E6D753}\textsf{SSJ (electrons)}}$
- **GOES (13 and 15)**
    - ${\color{#4A8A4F}\textsf{EPS MAGED and EPEAD combined}}$
- **GOES-R**
    - ${\color{#E6D753}\textsf{MPS-High (archived and real-time)}}$
- **GPS**
    - ${\color{#E6D753}\textsf{GPS processing}}$
- **LANL GeoGrid**
    - ${\color{#E67D53}\textsf{SOPA and ESP (electrons)}}$
    - ${\color{#E67D53}\textsf{SOPA (protons)}}$
    - ${\color{#E67D53}\textsf{MPA (electrons)}}$
    - ${\color{#E67D53}\textsf{MPA (ions)}}$
- **ESA**
    - ${\color{#E67D53}\textsf{NGRM satellites}}$
    - ${\color{#E67D53}\textsf{GSSC EMU (electrons and protons)}}$
- **POES (no proton correction)**
    - ${\color{#4A8A4F}\textsf{MEPED (electrons)}}$
    - ${\color{#4A8A4F}\textsf{TED (electrons)}}$
- **PROBA-V**
    - ${\color{#E6D753}\textsf{EPT (electrons and protons)}}$
- **THEMIS**
    - ${\color{#E6D753}\textsf{FFT waves}}$
    - ${\color{#E6D753}\textsf{Spacecraft-potential density}}$
- **Van Allen Probes**
    - ${\color{#4A8A4F}\textsf{HOPE (electrons)}}$
    - ${\color{#E67D53}\textsf{HOPE (protons)}}$
    - ${\color{#4A8A4F}\textsf{MagEIS (electrons)}}$
    - ${\color{#E67D53}\textsf{MagEIS (protons)}}$
    - ${\color{#E67D53}\textsf{RBSPICE (protons)}}$
    - ${\color{#4A8A4F}\textsf{REPT (electrons)}}$
    - ${\color{#4A8A4F}\textsf{ECT-combined}}$
    - ${\color{#4A8A4F}\textsf{EMFISIS waves}}$
    - ${\color{#4A8A4F}\textsf{EMFISIS and EFW density}}$
- **Models**
    - ${\color{#4A8A4F}\textsf{Last closed drift shell (LCDS)}}$
    - ${\color{#4A8A4F}\textsf{Solar wind and geomagnetic indices}}$

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

### Plotting processed data

`el-paso plot <mission> <recipe>` draws a quick-look figure of data you have
already processed. Pass the same arguments you used for processing, so the same
files are located again:

```bash
el-paso plot rbsp mageis-electrons \
    --start-time 2017-10-15 --end-time 2017-10-16T23:59:59 \
    --satellite a --mag-field T89 \
    --processed-data-path ./processed
```
See the [plotting section](docs/getting_started/command_line.md#plotting) of the command line documentation for details.

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

If you use `EL-PASO` in your research, please cite the associated paper:

> Haas, B., Drozdov, A. Y., Szabó-Roberts, M., & Jhawar, S. (2026). EL-PASO: An Open-Source Python Library for Processing and Standardizing Particle Measurements Taken in Space. *Earth and Space Science, 13*, e2026EA005044. https://doi.org/10.1029/2026EA005044

```bibtex
@article{haas2026elpaso,
  author  = {Haas, Bernhard and Drozdov, Alexander Y. and Szab{\'o}-Roberts, M{\'a}ty{\'a}s and Jhawar, Sahil},
  title   = {{EL-PASO}: An Open-Source Python Library for Processing and Standardizing Particle Measurements Taken in Space},
  journal = {Earth and Space Science},
  volume  = {13},
  number  = {11},
  pages   = {e2026EA005044},
  year    = {2026},
  doi     = {10.1029/2026EA005044},
  url     = {https://doi.org/10.1029/2026EA005044}
}
```

To cite this repository you can use [CITATION.cff](CITATION.cff).

## Acknowledgements

This work has been funded by the German Research Foundation (NFDI4Earth, DFG project no. 460036893, https://www.nfdi4earth.de/).
The authors acknowledge the work of Mátyás Szabó-Roberts who led the foundation for the EL-PASO framework.

The thank the authors of the [IRBEM library](https://github.com/PRBEM/IRBEM) for providing their code.
