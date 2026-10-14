# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import shutil
import typing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from el_paso.dataset import DataSet
from el_paso.recipes.model import compute_lcds, lcds_strategy
from el_paso.typing import MagneticFieldLiteral

# Aliases share an IRBEM kext with a tested model: TS05 and T04s with TS04, OP77Q with OP77
MAG_FIELDS = [
    mag_field for mag_field in typing.get_args(MagneticFieldLiteral) if mag_field not in ("TS05", "T04s", "OP77Q")
]


@pytest.mark.basic
@pytest.mark.parametrize("mag_field", MAG_FIELDS)
def test_lcds(
    tmpdir: Path,
    mag_field: MagneticFieldLiteral,
    *,
    renew_solution: bool,
) -> None:

    # 12-hour cadence over [May 10 00:00, May 12 12:00) gives 5 time steps
    start_time = datetime(2024, 5, 10, tzinfo=timezone.utc)
    end_time = datetime(2024, 5, 12, 12, tzinfo=timezone.utc)

    processed_data_path = Path(tmpdir)

    compute_lcds(
        start_time=start_time,
        end_time=end_time,
        mag_field=mag_field,
        raw_data_path=processed_data_path,
        processed_data_path=processed_data_path,
        bin_cadence=timedelta(hours=12),
        num_cores=5,
    )

    proc_strategy = lcds_strategy(processed_data_path, mag_field)
    interval_start, interval_end = proc_strategy.get_time_intervals_to_save(start_time, end_time)[0]
    out_path = proc_strategy.get_file_path(interval_start, interval_end, proc_strategy.output_files[0])
    assert out_path.exists(), "Output path does not exist!"

    true_data_path = Path(__file__).parent / "data" / "processed" / "LCDS"

    if renew_solution:
        true_data_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(out_path, true_data_path)

    lcds_proc = DataSet(start_time=start_time, end_time=end_time, saving_strategy=proc_strategy)
    lcds_true = DataSet(
        start_time=start_time, end_time=end_time, saving_strategy=lcds_strategy(true_data_path, mag_field)
    )

    lcds_proc.assert_equal(lcds_true)
