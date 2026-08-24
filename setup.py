# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0


"""Custom setup.py to install IRBEM Fortran library before building the Python package."""

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.errors import BaseError, ExecError, FileError

IRBEM_REPO_URL = "https://github.com/PRBEM/IRBEM.git"


class IrbemBuildError(BaseError):
    """Raised when the IRBEM library cannot be built."""


class CustomBuild(build_py):
    """Custom build command that builds IRBEM before building Python package."""

    def run(self):
        self.build_irbem()
        super().run()

    def build_irbem(self):
        self._clone_and_build()

    def _get_fortran_compiler_darwin(self):
        try:
            fc = subprocess.check_output(["bash", "-c", "compgen -c gfortran | sort -V | tail -n1"], text=True).strip()
            fc = fc or "gfortran"
        except subprocess.CalledProcessError:
            fc = "gfortran"
        return fc

    def _find_so_file(self, search_root):
        """Recursively search for libirbem.so under search_root."""
        for dirpath, _, filenames in os.walk(search_root):
            for fname in filenames:
                if fname == "libirbem.so":
                    return os.path.join(dirpath, fname)
        return None

    def _clone_and_build(self):
        dest_dir = os.path.join(self.build_lib, "el_paso")
        os.makedirs(dest_dir, exist_ok=True)
        dest_so = os.path.join(dest_dir, "libirbem.so")

        cwd = os.getcwd()
        dest_so_el_paso_dir = os.path.join(cwd, "el_paso")

        tmp_dir = tempfile.mkdtemp(prefix="irbem_build_")

        try:
            self._clone_irbem_repo(tmp_dir)
            self._compile_and_install_irbem(tmp_dir)

            so_path = self._find_so_file(tmp_dir)
            if so_path is None:
                raise FileError(
                    "libirbem.so was not produced by the IRBEM build. "
                    "Check that gfortran and make are installed and on PATH."
                )

            # make can exit 0 and still leave a broken/incomplete library
            try:
                ctypes.CDLL(so_path)
            except OSError as exc:
                raise IrbemBuildError(f"libirbem.so was built but is not loadable: {exc}") from exc

            shutil.copy2(so_path, dest_so)
            shutil.copy2(so_path, dest_so_el_paso_dir)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _clone_irbem_repo(self, tmp_dir, cwd=None):
        """Run a command, aborting the build with captured output on failure."""
        try:
            subprocess.run(
                ["git", "clone", "--depth=1", IRBEM_REPO_URL, tmp_dir],
                cwd=cwd,
                check=True,
                text=True,
                capture_output=True,
            )
        except FileNotFoundError as exc:
            raise ExecError(f"{cmd[0]!r} not found; it is required to build IRBEM.") from exc
        except subprocess.CalledProcessError as exc:
            raise IrbemBuildError(
                f"IRBEM build step failed: {' '.join(cmd)}\n"
                f"exit code: {exc.returncode}\n"
                f"--- stdout ---\n{exc.stdout}\n"
                f"--- stderr ---\n{exc.stderr}"
            ) from exc

    def _compile_and_install_irbem(self, irbem_dir):
        fflags = "-fpic -fno-second-underscore -std=legacy -ffixed-line-length-none -O2 -march=native"
        cflags = "-fpic -O2 -march=native"
        if sys.platform == "darwin":
            fc = self._get_fortran_compiler_darwin()
            base_cmd = [
                "make",
                "OS=osx64",
                f"FC={fc}",
                f"LD={fc}",
                f"FFLAGS={fflags}",
                f"CFLAGS={cflags}",
            ]
            subprocess.check_call(base_cmd + ["all"], cwd=irbem_dir)
            subprocess.check_call(base_cmd + ["install"], cwd=irbem_dir)
        else:
            base_cmd = [
                "make",
                f"FFLAGS={fflags}",
                f"CFLAGS={cflags}",
            ]
            subprocess.check_call(base_cmd + ["all"], cwd=irbem_dir)
            subprocess.check_call(base_cmd + ["install", "."], cwd=irbem_dir)


setup(
    name="el_paso",
    packages=find_packages(),
    package_data={"el_paso": ["libirbem.so"]},
    include_package_data=True,
    python_requires=">=3.12",
    cmdclass={"build_py": CustomBuild},
)
