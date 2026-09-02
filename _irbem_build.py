# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""Builds the IRBEM Fortran shared library into el_paso/libirbem.so.

Invoked from the Hatchling build hook in hatch_build.py, which runs before
both standard and editable wheel builds and, unlike a setuptools build_py
override, does not have its exceptions silently swallowed for editable
installs (see https://hatch.pypa.io/latest/plugins/build-hook/reference/).
"""

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

IRBEM_REPO_URL = "https://github.com/radiation-belts/IRBEM.git"
IRBEM_REPO_BRANCH = "GFZ"


class IrbemBuildError(Exception):
    """Raised when the IRBEM library cannot be built."""


def ensure_libirbem_built(source_root: Path) -> None:
    """Build el_paso/libirbem.so in source_root, rebuilding on every call."""
    dest_so = source_root / "el_paso" / "libirbem.so"

    tmp_dir = tempfile.mkdtemp(prefix="irbem_build_")
    try:
        _clone_irbem_repo(tmp_dir)
        _compile_irbem(tmp_dir)

        so_path = _find_so_file(tmp_dir)
        if so_path is None:
            msg = (
                "libirbem.so was not produced by the IRBEM build. "
                "Check that gfortran and make are installed and on PATH."
            )
            raise IrbemBuildError(msg)

        # make can exit 0 and still leave a broken/incomplete library
        try:
            ctypes.CDLL(so_path)
        except OSError as exc:
            msg = f"libirbem.so was built but is not loadable: {exc}"
            raise IrbemBuildError(msg) from exc

        shutil.copy2(so_path, dest_so)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _find_so_file(search_root: str) -> str | None:
    for dirpath, _, filenames in os.walk(search_root):
        for fname in filenames:
            if fname == "libirbem.so":
                return os.path.join(dirpath, fname)  # noqa: PTH118
    return None


def _clone_irbem_repo(tmp_dir: str) -> None:
    cmd = ["git", "clone", "--depth=1", "--branch", IRBEM_REPO_BRANCH, IRBEM_REPO_URL, tmp_dir]
    try:
        subprocess.run(cmd, check=True, text=True, capture_output=True)  # noqa: S603
    except FileNotFoundError as exc:
        msg = f"{cmd[0]!r} not found; it is required to build IRBEM."
        raise IrbemBuildError(msg) from exc
    except subprocess.CalledProcessError as exc:
        msg = (
            f"IRBEM build step failed: {' '.join(cmd)}\n"
            f"exit code: {exc.returncode}\n"
            f"--- stdout ---\n{exc.stdout}\n"
            f"--- stderr ---\n{exc.stderr}"
        )
        raise IrbemBuildError(msg) from exc


def _compile_irbem(irbem_dir: str) -> None:
    if sys.platform == "darwin":
        base_cmd = ["make", "OS=osx64"]
        subprocess.check_call([*base_cmd, "all"], cwd=irbem_dir)  # noqa: S603
        subprocess.check_call([*base_cmd, "install"], cwd=irbem_dir)  # noqa: S603
    else:
        subprocess.check_call(["make", "all"], cwd=irbem_dir)  # noqa: S607
        subprocess.check_call(["make", "install", "."], cwd=irbem_dir)  # noqa: S607
