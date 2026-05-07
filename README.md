<!--
SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
SPDX-FileContributor: Bernhard Haas

SPDX-License-Identifier: Apache-2.0
-->

[![PyPi](https://badge.fury.io/py/el-paso.svg)](https://badge.fury.io/py/el-paso)
[![Python version](https://img.shields.io/pypi/pyversions/el-paso.svg)](https://badge.fury.io/py/el-paso)
[![Tests](https://github.com/GFZ/EL_PASO/actions/workflows/test.yml/badge.svg)](https://github.com/GFZ/EL_PASO/actions/workflows/test.yml)
[![Coverage Status](https://coveralls.io/repos/github/GFZ/EL_PASO/badge.svg?branch=main)](https://coveralls.io/github/GFZ/EL_PASO?branch=main)
[![Docs](https://app.readthedocs.org/projects/el-paso/badge/?version=latest)](https://el-paso.readthedocs.io/en/latest/)
[![REUSE status](https://api.reuse.software/badge/github.com/GFZ/EL_PASO)](https://api.reuse.software/info/github.com/GFZ/EL_PASO)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-blue.svg)](http://creativecommons.org/publicdomain/zero/1.0/)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

# ELaborative Particle Analysis from Satellite Observations (EL-PASO)

`EL-PASO` is a Python framework designed to streamline the download, processing, and saving of satellite particle observation data.

Its primary purpose is to prepare and standardize particle data for use in radiation belt modeling.

## Features

- **Format Flexibility:** Capable of handling different input formats including `cdf`, `netcdf`, `h5`, `ascii`, and `json`
- **Integrated Processing:** Provides a comprehensive set of functions for common particle data analysis tasks
- **Supports Metadata:** Stores all processing and metadata alongside the data, ensuring full traceability and reproducibility.
- **Standardized output files:** Saving processed data in different standards (e.g. PRBEM) to enable easy loading and sharing of processed data

Full documentation can be viewed [here](https://el-paso.readthedocs.io/en/latest/).

## Available processing scripts

- **Arase**
    - MEPe
    - XEP real-time
    - PWE density
- **GOES-R**
    - MPS-High real-time
    - MPS-High
- **ESA**
    - NGRM satellites
- **POES**
    - MEPED (electrons)
    - TED (electrons)
- **PROBA-V**
    - EPT
- **Van Allen Probes**
    - HOPE (electrons) and ECT-combined
    - EMFISIS and EFW density

## Installation Guide

### Step 1: Clone the Repository

Begin by cloning the EL-PASO repository and navigating into its directory.

```bash
git clone https://github.com/GFZ/EL_PASO.git
cd EL_PASO
```

### Step 2: Set up a Python Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install the EL PASO Package

Install the core EL-PASO package using pip.

```bash
pip install .
```

The custom `setup.py` script will automatically download and compile the IRBEM Fortran library during this step.

### Verifying the Installation

You can validate your installation by running the minimal example located in *examples*:

```bash
python3 examples/minimal_example.py
```

## Acknowledgements

This work has been funded by the German Research Foundation (NFDI4Earth, DFG project no. 460036893, https://www.nfdi4earth.de/).
The authors acknowledge the work of Mátyás Szabó-Roberts who led the foundation for the EL-PASO framework.

The thank the authors of the [IRBEM library](https://github.com/PRBEM/IRBEM) for providing their code.
