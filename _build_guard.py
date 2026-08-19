# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""PEP 517 build backend wrapper that fails fast if setup.py is missing.

el_paso's setup.py compiles the IRBEM Fortran library via a custom
build_py command. Without it, setuptools silently falls back to a
pyproject.toml-only build and produces a wheel missing libirbem.so.
This wrapper aborts the build immediately instead.
"""

import pathlib

from setuptools import build_meta as _orig

if not pathlib.Path("setup.py").exists():
    msg = "setup.py is required to build el_paso (it builds the IRBEM library) but is missing."
    raise RuntimeError(msg)

build_wheel = _orig.build_wheel
build_sdist = _orig.build_sdist
build_editable = _orig.build_editable
get_requires_for_build_wheel = _orig.get_requires_for_build_wheel
get_requires_for_build_sdist = _orig.get_requires_for_build_sdist
get_requires_for_build_editable = _orig.get_requires_for_build_editable
prepare_metadata_for_build_wheel = _orig.prepare_metadata_for_build_wheel
prepare_metadata_for_build_editable = _orig.prepare_metadata_for_build_editable
