# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0


"""Custom setup.py to install IRBEM Fortran library before building the Python package."""

import os
import shutil
import subprocess
import sys
import tempfile

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py

IRBEM_REPO_URL = "https://github.com/PRBEM/IRBEM.git"


class CustomBuild(build_py):
    """Custom build command that builds IRBEM before building Python package."""

    def run(self):
        self.build_irbem()
        super().run()

    def build_irbem(self):
        try:
            self._clone_and_build()
        except subprocess.CalledProcessError:
            sys.exit(1)
        except Exception:
            sys.exit(1)

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
            subprocess.check_call(["git", "clone", "--depth=1", IRBEM_REPO_URL, tmp_dir])

            self._compile_and_install_irbem(tmp_dir)

            so_path = self._find_so_file(tmp_dir)

            if so_path is None:
                raise FileNotFoundError(
                    "libirbem.so was not found after build. Check that gfortran and make are installed correctly."
                )

            shutil.copy2(so_path, dest_so)
            shutil.copy2(so_path, dest_so_el_paso_dir)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _compile_and_install_irbem(self, irbem_dir):
        if sys.platform == "darwin":
            fc = self._get_fortran_compiler_darwin()
            base_cmd = ["make", "OS=osx64", f"FC={fc}", f"LD={fc}"]
            subprocess.check_call(base_cmd + ["all"], cwd=irbem_dir)
            subprocess.check_call(base_cmd + ["install"], cwd=irbem_dir)
        else:
            subprocess.check_call(["make"], cwd=irbem_dir)
            subprocess.check_call(["make", "install", "."], cwd=irbem_dir)


setup(
    name="el_paso",
    packages=find_packages(),
    package_data={"el_paso": ["libirbem.so"]},
    include_package_data=True,
    python_requires=">=3.12",
    cmdclass={"build_py": CustomBuild},
)
