# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import shutil
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import pytest

import el_paso as ep
from el_paso.dataset import GFZDataSet
from el_paso.recipes.rbsp import process_rbsp_ect_combined


@pytest.mark.parametrize(
    ("mag_field", "save_strategy"),
    [
        pytest.param("T89", "gfz", marks=pytest.mark.basic),
        pytest.param("T89", "netcdf", marks=pytest.mark.basic),
    ],
)
def test_rbsp_ect_combined_snapshot(
    mag_field: Literal["T89", "TS04", "OP77", "T96"],
    save_strategy: Literal["gfz", "netcdf"],
    tmpdir: Path,
    skip_if_unreachable: Callable[..., None],
    *,
    renew_solution: bool,
) -> None:
    skip_if_unreachable("https://spdf.gsfc.nasa.gov")

    start_time = datetime(2017, 9, 8, tzinfo=timezone.utc)
    end_time = start_time + timedelta(days=0.4, seconds=-1)

    processed_data_path = tmpdir

    process_rbsp_ect_combined(
        start_time=start_time,
        end_time=end_time,
        satellite="a",
        mag_field=mag_field,
        raw_data_path=Path(__file__).parent / "data" / "raw",
        processed_data_path=processed_data_path,
        num_cores=32,
        bin_cadence=timedelta(hours=1),
        save_strategy=save_strategy,
    )

    start_date = start_time.replace(day=1)
    end_date = end_time.replace(day=30)

    match save_strategy:
        case "gfz":
            out_path = (
                processed_data_path
                / "RBSP"
                / "rbspa"
                / "Processed_Mat_Files"
                / f"rbspa_ect_combined_{start_date:%Y%m%d}to{end_date:%Y%m%d}_flux_ver4.mat"
            )
            assert out_path.exists()

            if renew_solution:
                shutil.copytree(processed_data_path, Path(__file__).parent / "data" / "processed", dirs_exist_ok=True)

            rbsp_proc = GFZDataSet(
                saving_strategy=ep.saving_strategies.GFZStrategy(
                    str(tmpdir), "RBSP", "rbspa", "ect_combined", mag_field
                ),
                start_time=start_time,
                end_time=end_time,
            )

            rbsp_true = GFZDataSet(
                saving_strategy=ep.saving_strategies.GFZStrategy(
                    Path(__file__).parent / "data" / "processed", "RBSP", "rbspa", "ect_combined", mag_field
                ),
                start_time=start_time,
                end_time=end_time,
            )

        case "netcdf":
            out_path = (
                processed_data_path
                / "RBSP"
                / "rbspa"
                / f"rbspa_ect_combined_{start_date:%Y%m%d}to{end_date:%Y%m%d}_{mag_field}.nc"
            )
            assert out_path.exists()

            if renew_solution:
                shutil.copy(out_path, Path(__file__).parent / "data" / "processed" / "RBSP" / "rbspa")

            rbsp_proc = GFZDataSet(
                start_time=start_time,
                end_time=end_time,
                saving_strategy=ep.saving_strategies.MonthlyRBStrategy(
                    tmpdir,
                    "RBSP",
                    "rbspa",
                    "ect_combined",
                    mag_field,
                    data_standard=ep.data_standards.GFZStandard(),
                    file_format="nc",
                ),
            )

            rbsp_true = GFZDataSet(
                start_time=start_time,
                end_time=end_time,
                saving_strategy=ep.saving_strategies.MonthlyRBStrategy(
                    Path(__file__).parent / "data" / "processed",
                    "RBSP",
                    "rbspa",
                    "ect_combined",
                    mag_field,
                    data_standard=ep.data_standards.GFZStandard(),
                    file_format="nc",
                ),
            )

    rbsp_proc.assert_equal(rbsp_true)
