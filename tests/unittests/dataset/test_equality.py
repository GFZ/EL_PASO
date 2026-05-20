# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import pytest
from astropy import units as u  # type: ignore[reportMissingTypeStubs]

import el_paso as ep
from el_paso.dataset import DataSet
from el_paso.dataset.utils import python2matlab

if TYPE_CHECKING:
    from pathlib import Path

    from el_paso.typing import DataStandard, InternalName, MFSFormats


def _mock_monthly_variables() -> dict[InternalName, ep.Variable]:
    """Create mocked monthly product variables without running processing code."""
    time_size = 144
    energy_size = 3
    alpha_size = 4

    start_time = datetime(2013, 1, 1, tzinfo=timezone.utc)
    datetimes = [start_time + i * np.timedelta64(6000, "s") for i in range(time_size)]
    epoch = np.array([python2matlab(i) for i in datetimes])

    variables: dict[InternalName, ep.Variable] = {
        "Epoch": ep.Variable(original_unit=ep.units.datenum, data=epoch),
        "FEDU": ep.Variable(
            original_unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
            data=np.arange(time_size * energy_size * alpha_size, dtype=float).reshape(
                time_size,
                energy_size,
                alpha_size,
            ),
        ),
        "Alpha_Eq": ep.Variable(original_unit=u.deg, data=np.full((time_size, alpha_size), 45.0)),
        "Energy_FEDU": ep.Variable(
            original_unit=u.MeV,
            data=np.tile(np.asarray([0.5, 1.0, 2.0]), (time_size, 1)),
        ),
        "Alpha": ep.Variable(
            original_unit=u.deg,
            data=np.tile(np.asarray([10.0, 30.0, 60.0, 90.0]), (time_size, 1)),
        ),
        "B_Calc": ep.Variable(original_unit=u.nT, data=np.full(time_size, 75.0)),
        "B_Eq": ep.Variable(original_unit=u.nT, data=np.full(time_size, 50.0)),
        "InvK": ep.Variable(
            original_unit=ep.units.RE * u.G**0.5,
            data=np.full((time_size, alpha_size), 1.5),
        ),
        "InvMu": ep.Variable(
            original_unit=u.MeV / u.G,
            data=np.full((time_size, energy_size, alpha_size), 2.5),
        ),
        "Position": ep.Variable(
            original_unit=ep.units.RE,
            data=np.arange(time_size * 3, dtype=float).reshape(time_size, 3),
        ),
        "PSD": ep.Variable(
            original_unit=(u.m * u.kg * u.m / u.s) ** (-3),
            data=np.full((time_size, energy_size, alpha_size), 3.5),
        ),
        "R_Eq": ep.Variable(original_unit=ep.units.RE, data=np.full(time_size, 6.0)),
        "MLT": ep.Variable(original_unit=u.hour, data=np.full(time_size, 12.0)),
        "L_m": ep.Variable(
            original_unit=u.dimensionless_unscaled,
            data=np.full((time_size, alpha_size), 4.5),
        ),
        "L_star": ep.Variable(
            original_unit=u.dimensionless_unscaled,
            data=np.full((time_size, alpha_size), 5.5),
        ),
    }

    for variable in variables.values():
        variable.metadata.source_files = ["mocked_input.cdf"]

    return variables


@pytest.mark.basic
@pytest.mark.parametrize("file_format", ["nc", "h5", "cdf", "mat"])
def test_dataset_equality_rejects_data_saved_with_different_standards(tmp_path: Path, file_format: MFSFormats) -> None:
    variables = _mock_monthly_variables()
    start_time = datetime(2013, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2013, 1, 2, tzinfo=timezone.utc)

    gfz_strategy = ep.saving_strategies.MonthlyFileStrategy(
        base_data_path=tmp_path / "gfz",
        mission="GOES",
        satellite="primary",
        instrument="MAGED",
        mag_field="T89",
        file_format=file_format,
        data_standard=ep.data_standards.GFZStandard(),
    )
    prbem_strategy = ep.saving_strategies.MonthlyFileStrategy(
        base_data_path=tmp_path / "prbem",
        mission="GOES",
        satellite="primary",
        instrument="MAGED",
        mag_field="T89",
        file_format=file_format,
        data_standard=ep.data_standards.PRBEMStandard(),
    )

    for strategy in (gfz_strategy, prbem_strategy):
        ep.save(
            variables,
            strategy,
            start_time=start_time,
            end_time=end_time,
            time_var=variables["Epoch"],
        )
        interval_start, interval_end = strategy.get_time_intervals_to_save(start_time, end_time)[0]
        assert strategy.get_file_path(interval_start, interval_end, strategy.output_files[0]).exists()

    gfz_dataset = DataSet(
        saving_strategy=gfz_strategy,
        start_time=start_time,
        end_time=end_time,
        preferred_extension=file_format,
        verbose=False,
    )
    prbem_dataset = DataSet(
        saving_strategy=prbem_strategy,
        start_time=start_time,
        end_time=end_time,
        preferred_extension=file_format,
        verbose=False,
    )

    gfz_dataset.load(gfz_strategy.data_standard.get_standard_name("FEDU"))
    prbem_dataset.load(prbem_strategy.data_standard.get_standard_name("FEDU"))

    assert "Flux" in gfz_dataset.get_loaded_variables()
    assert "FEDU" in prbem_dataset.get_loaded_variables()
    assert gfz_dataset != prbem_dataset
    with pytest.raises(AssertionError, match="Data standards are different"):
        gfz_dataset.assert_equal(prbem_dataset)


@pytest.mark.basic
@pytest.mark.parametrize("file_format", ["nc", "h5", "cdf", "mat"])
@pytest.mark.parametrize("data_standard", [ep.data_standards.PRBEMStandard, ep.data_standards.GFZStandard])
def test_dataset_equality_accepts_data_saved_with_different_strategies_but_same_standards(
    tmp_path: Path, file_format: MFSFormats, data_standard: type[DataStandard]
) -> None:
    variables = _mock_monthly_variables()
    start_time = datetime(2013, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2013, 1, 2, tzinfo=timezone.utc)

    gfz_strategy = ep.saving_strategies.GFZStrategy(
        base_data_path=tmp_path / "gfz",
        mission="GOES",
        satellite="primary",
        instrument="MAGED",
        mag_field="T89",
        data_standard=data_standard(),
    )
    mfs_strategy = ep.saving_strategies.MonthlyFileStrategy(
        base_data_path=tmp_path / "prbem",
        mission="GOES",
        satellite="primary",
        instrument="MAGED",
        mag_field="T89",
        file_format=file_format,
        data_standard=data_standard(),
    )

    for strategy in (gfz_strategy, mfs_strategy):
        ep.save(
            variables,
            strategy,
            start_time=start_time,
            end_time=end_time,
            time_var=variables["Epoch"],
        )
        interval_start, interval_end = strategy.get_time_intervals_to_save(start_time, end_time)[0]
        assert strategy.get_file_path(interval_start, interval_end, strategy.output_files[0]).exists()

    gfz_strategy_dataset = DataSet(
        saving_strategy=gfz_strategy,
        start_time=start_time,
        end_time=end_time,
        preferred_extension=file_format,
        verbose=False,
    )
    mfs_strategy_dataset = DataSet(
        saving_strategy=mfs_strategy,
        start_time=start_time,
        end_time=end_time,
        preferred_extension=file_format,
        verbose=False,
    )

    assert gfz_strategy_dataset.datetime == mfs_strategy_dataset.datetime

    gfz_strategy_dataset.load(gfz_strategy.data_standard.get_standard_name("FEDU"))
    mfs_strategy_dataset.load(mfs_strategy.data_standard.get_standard_name("FEDU"))

    assert gfz_strategy.data_standard.get_standard_name("FEDU") in gfz_strategy_dataset.get_loaded_variables()
    assert mfs_strategy.data_standard.get_standard_name("FEDU") in mfs_strategy_dataset.get_loaded_variables()
    gfz_strategy_dataset.assert_equal(mfs_strategy_dataset)
    assert gfz_strategy_dataset == mfs_strategy_dataset
