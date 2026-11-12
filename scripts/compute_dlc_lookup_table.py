# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: INP001, T201

import argparse
import os
from pathlib import Path

from el_paso import setup_logging
from el_paso.processing.magnetic_field_utils import compute_dlc_lookup_table, get_dlc_table_path


def compute_dlc_lookup_tables(
    years: list[int],
    *,
    output_dir: Path | None = None,
    num_cores: int = 4,
    lat_step: float = 0.5,
    lon_step: float = 5.0,
    dlc_tol: float = 0.1,
    drift_shell_resolution: int = 3,
    overwrite: bool = False,
) -> None:
    """Computes and saves the IGRF drift loss cone lookup table of every year.

    `compute_magnetic_field_variables` takes the drift loss cone from these tables instead of tracing it at every
    time step whenever IGRF is the internal field model, also with external field models such as T89, whose effect
    on the drift shell is then neglected (small deviations at L > 5). By default they are written to where it
    looks for them, the `dlc_lookup_tables` subdirectory of `EL_PASO_MODEL_DATA_PATH` (default `~/.elpaso`).
    Each table takes about two hours on 24 cores with the default grid.

    Args:
        years (list[int]): The years to compute the tables for.
        output_dir (Path | None): Directory to save the tables to. Defaults to the directory
            `compute_magnetic_field_variables` reads them from.
        num_cores (int): Number of processes for the IRBEM calculations. Defaults to 4.
        lat_step (float): Latitude spacing of the foot point grid in degrees. Defaults to 0.5.
        lon_step (float): Longitude spacing of the foot point grid in degrees. Defaults to 5.
        dlc_tol (float): Bisection tolerance on the equatorial drift loss cone in degrees. Defaults to 0.1.
        drift_shell_resolution (int): IRBEM drift shell resolution, 1 to 3 recommended. Defaults to 3.
        overwrite (bool): If True, recompute tables that already exist. Defaults to False.
    """
    for year in years:
        path = get_dlc_table_path(year)
        if output_dir is not None:
            path = output_dir / path.name

        if path.is_file() and not overwrite:
            print(f"Skipping {year}: {path} already exists.")
            continue

        table = compute_dlc_lookup_table(
            year,
            lat_step=lat_step,
            lon_step=lon_step,
            dlc_tol=dlc_tol,
            drift_shell_resolution=drift_shell_resolution,
            num_cores=num_cores,
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_netcdf(path)
        print(f"Saved drift loss cone lookup table for {year} to {path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the IGRF drift loss cone lookup tables of el_paso.")
    parser.add_argument("--years", type=int, nargs="+", required=True, help="Years to compute the tables for.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save the tables to. Defaults to where compute_magnetic_field_variables reads them from.",
    )
    parser.add_argument("--num-cores", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--lat-step", type=float, default=0.5, help="Latitude spacing in degrees.")
    parser.add_argument("--lon-step", type=float, default=5.0, help="Longitude spacing in degrees.")
    parser.add_argument("--dlc-tol", type=float, default=0.1, help="Bisection tolerance in degrees.")
    parser.add_argument("--drift-shell-resolution", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true", help="Recompute tables that already exist.")
    args = parser.parse_args()

    setup_logging()

    compute_dlc_lookup_tables(
        args.years,
        output_dir=args.output_dir,
        num_cores=args.num_cores,
        lat_step=args.lat_step,
        lon_step=args.lon_step,
        dlc_tol=args.dlc_tol,
        drift_shell_resolution=args.drift_shell_resolution,
        overwrite=args.overwrite,
    )
