# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from swvo.io.RBMDataSet import RBMDataSet

from el_paso.dataset.data_set_implementations import DataOrgDataSet
from examples.VanAllenProbes.process_ect_combined import process_ect_combined


@pytest.mark.parametrize(
    ("mag_field", "save_strategy"),
    [
        pytest.param("T89", "dataorg", marks=pytest.mark.basic),
        ("OP77", "dataorg"),
        ("T96", "dataorg"),
        ("TS04", "dataorg"),
        pytest.param("T89", "netcdf", marks=pytest.mark.basic),
    ],
)
def test_rbsp_ect_combined_snapshot(
    mag_field: Literal["T89", "TS04", "OP77", "T96"],
    save_strategy: Literal["dataorg", "h5", "netcdf"],
    tmpdir: Path,
    *,
    renew_solution: bool,
) -> None:
    start_time = datetime(2017, 9, 8, tzinfo=timezone.utc)
    end_time = start_time + timedelta(days=0.4, seconds=-1)

    irbem_lib_path = Path(__file__).parent / "../../libirbem.so"

    processed_data_path = tmpdir

    process_ect_combined(
        start_time=start_time,
        end_time=end_time,
        sat_str="a",
        irbem_lib_path=irbem_lib_path,
        mag_field=mag_field,
        raw_data_path=Path(__file__).parent / "data" / "raw",
        processed_data_path=processed_data_path,
        num_cores=32,
        cadence=timedelta(hours=1),
        save_strategy=save_strategy,
    )

    start_date = start_time.replace(day=1)
    end_date = end_time.replace(day=30)

    match save_strategy:
        case "dataorg":
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

        case "h5":
            out_path = processed_data_path / "RBSP" / "rbspa" / f"rbspa_ect_combined_{start_date:%Y%m%d}to{end_date:%Y%m%d}_{mag_field}.h5"
            assert out_path.exists()

            if renew_solution:
                shutil.copy(out_path, Path(__file__).parent / "data" / "processed" / "RBSP" / "rbspa")

        case "netcdf":
            out_path = processed_data_path / "RBSP" / "rbspa" / f"rbspa_ect_combined_{start_date:%Y%m%d}to{end_date:%Y%m%d}_{mag_field}.nc"
            assert out_path.exists()

            if renew_solution:
                shutil.copy(out_path, Path(__file__).parent / "data" / "processed" / "RBSP" / "rbspa")

    if save_strategy == "dataorg":
        rbsp_proc = RBMDataSet(
            start_time=start_time,
            end_time=end_time,
            folder_path=tmpdir,
            satellite="RBSPA",
            instrument="ect_combined",
            mfm=mag_field,
        )

        rbsp_true = RBMDataSet(
            start_time=start_time,
            end_time=end_time,
            folder_path=Path(__file__).parent / "data" / "processed",
            satellite="RBSPA",
            instrument="ect_combined",
            mfm=mag_field,
            preferred_extension="mat",
        )
    elif save_strategy == "netcdf":
        rbsp_proc = DataOrgDataSet(
            start_time=start_time,
            end_time=end_time,
            base_path=str(tmpdir),
            mission="RBSP",
            satellite="RBSPA",
            instrument="ect_combined",
            mag_field=mag_field,
        )

        rbsp_true = RBMDataSet(
            start_time=start_time,
            end_time=end_time,
            folder_path=Path(__file__).parent / "data" / "processed",
            satellite="RBSPA",
            instrument="ect_combined",
            mfm=mag_field,
        )
    else:
        msg = "Test not implemented for this save strategy."
        raise NotImplementedError(msg)

    # np.testing.assert_array_almost_equal(rbsp_proc.InvMu, rbsp_true.InvMu)
    # np.testing.assert_array_almost_equal(rbsp_proc.Lstar, rbsp_true.Lstar)
    # np.testing.assert_allclose(rbsp_proc.time, rbsp_true.time)

    assert rbsp_proc == rbsp_true