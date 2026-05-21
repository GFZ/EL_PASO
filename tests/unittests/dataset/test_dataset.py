# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from unittest import mock

import numpy as np
import pytest
from astropy import units as u

import el_paso as ep
from el_paso.dataset import DataSet
from el_paso.dataset.utils import python2matlab

if TYPE_CHECKING:
    from pathlib import Path

    from el_paso.typing import InternalName, MFSFormats


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


@pytest.fixture(params=["nc", "h5", "mat", "cdf"])
def mock_dataset(request, tmp_path: Path) -> DataSet:  # noqa: ANN001
    formats: MFSFormats = request.param

    variables = _mock_monthly_variables()
    start_time = datetime(2013, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2013, 1, 2, tzinfo=timezone.utc)

    strategy = ep.saving_strategies.MonthlyRBStrategy(
        base_data_path=tmp_path,
        mission="GOES",
        satellite="primary",
        instrument="MAGED",
        mag_field="T89",
        file_format=formats,
        data_standard=ep.data_standards.GFZStandard(),
    )

    ep.save(
        variables,
        strategy,
        start_time=start_time,
        end_time=end_time,
        time_var=variables["Epoch"],
    )

    return DataSet(
        saving_strategy=strategy,
        start_time=start_time,
        end_time=end_time,
        preferred_extension=formats,
        verbose=True,
    )


@pytest.mark.basic
class TestDataSet:  # noqa: D101
    def test_init_datetime_timezone(self, mock_dataset: DataSet):
        """Test timezone handling for input datetimes."""
        assert mock_dataset._start_time.tzinfo == timezone.utc
        assert mock_dataset._end_time.tzinfo == timezone.utc

    def test_get_satellite_name(self, mock_dataset: DataSet):
        """Test get_satellite_name method."""
        assert mock_dataset.get_satellite_name() == "primary"

    def test_get_satellite_and_instrument_name(self, mock_dataset: DataSet):
        """Test get_satellite_and_instrument_name method."""
        assert mock_dataset.get_satellite_and_instrument_name() == "primary_MAGED"

    def test_get_print_name(self, mock_dataset: DataSet):
        """Test get_print_name method."""
        assert mock_dataset.get_print_name() == "primary MAGED"

    def test_getattr_with_valid_variable(self, mock_dataset: DataSet):
        """Test __getattr__ with a valid variable."""
        with mock.patch.object(mock_dataset, "_load_variable") as _:
            mock_dataset.Flux = np.array([1.0, 2.0, 3.0])
            result = mock_dataset.Flux
            assert isinstance(result, np.ndarray)
            assert (result == np.array([1.0, 2.0, 3.0])).all()

    def test_getattr_with_invalid_variable(self, mock_dataset: DataSet):
        """Test __getattr__ with an invalid variable."""
        with pytest.raises(AttributeError):
            _ = mock_dataset.NonExistentAttribute

    def test_getattr_with_similar_variable(self, mock_dataset: DataSet):
        """Test __getattr__ suggests similar variable name."""
        with pytest.raises(AttributeError) as e:
            _ = mock_dataset.Flx

        assert "Maybe you meant Flux?" in str(e.value)

    def test_computed_invv_variable(self, mock_dataset: DataSet):
        """Test computed InvV variable."""
        mock_dataset.InvK = np.array([[1.0, 2.0]])
        mock_dataset.InvMu = np.array([[0.1, 0.2], [0.3, 0.4]])

        mock_dataset._load_variable("InvV")

        expected = (
            mock_dataset.InvMu
            * (
                np.repeat(
                    mock_dataset.InvK[:, np.newaxis, :],
                    mock_dataset.InvMu.shape[1],
                    axis=1,
                )
                + 0.5
            )
            ** 2
        )
        np.testing.assert_array_equal(mock_dataset.InvV, expected)

    def test_computed_p_variable(self, mock_dataset: DataSet):
        """Test computed P variable."""
        mock_dataset.MLT = np.array([0.0, 6.0, 12.0, 18.0])

        mock_dataset._load_variable("P")

        expected = ((mock_dataset.MLT + 12) / 12 * np.pi) % (2 * np.pi)
        np.testing.assert_array_equal(mock_dataset.P, expected)

    def test_load_variable_real_file(self, mock_dataset: DataSet):
        mock_dataset._load_variable("alpha_local")

        assert hasattr(
            mock_dataset,
            "alpha_local",
        ), "Dataset should have 'alpha_local' attribute after loading."
        assert isinstance(
            mock_dataset.alpha_local,
            np.ndarray,
        ), "'alpha_local' should be a NumPy array."

    def test_setattr_sets_standard_variable_without_file_loading(
        self,
        tmp_path: Path,
    ) -> None:
        strategy = ep.saving_strategies.MonthlyRBStrategy(
            base_data_path=tmp_path,
            mission="GOES",
            satellite="primary",
            instrument="MAGED",
            mag_field="T89",
            file_format="nc",
            data_standard=ep.data_standards.GFZStandard(),
        )
        start = datetime(2013, 1, 1, tzinfo=timezone.utc)
        end = datetime(2013, 1, 2, tzinfo=timezone.utc)

        dataset = DataSet(strategy, start, end, verbose=False)

        flux = np.array([[1.0, 2.0, 3.0]])
        dataset.Flux = flux

        np.testing.assert_array_equal(dataset.Flux, flux)

        with pytest.raises(
            AttributeError,
            match="Cannot set attribute 'NonExistentVariable'. It is not part of .",  # noqa: RUF043
        ):
            dataset.NonExistentVariable = 1

        with pytest.raises(
            AttributeError,
            match="Cannot set attribute 'flux'. Maybe you meant 'Flux'?",  # noqa: RUF043
        ):
            dataset.flux = 1

    def test_setattr_overrides_file_loaded_standard_variable(
        self,
        mock_dataset: DataSet,
    ) -> None:
        mock_dataset.load("Flux")

        loaded_flux = mock_dataset.Flux.copy()
        replacement_flux = np.full_like(loaded_flux, 42.0)

        mock_dataset.Flux = replacement_flux

        np.testing.assert_array_equal(mock_dataset.Flux, replacement_flux)

    def test_all_variables_in_dir(self, mock_dataset: DataSet):
        mock_dataset._load_variable("time")

        for var in ep.data_standards.GFZStandard().variable_infos.values():
            assert var.standard_name in mock_dataset.__dir__()

    def test_accessing_second_variable_does_not_reload_file(
        self,
        mock_dataset: DataSet,
        caplog: pytest.LogCaptureFixture,
    ):
        with caplog.at_level(logging.INFO):
            _ = mock_dataset.energy_channels

        assert "/GOES/primary/primary_maged_20130101to20130131_T89" in caplog.text

        log_count_after_first_load = caplog.text.count("/GOES/primary/primary_maged_20130101to20130131_T89")

        with (
            caplog.at_level(logging.INFO),
            mock.patch.object(
                mock_dataset,
                "_load_variable",
                wraps=mock_dataset._load_variable,
            ) as mock_load,
        ):
            _ = [getattr(mock_dataset, var) for var in mock_dataset.get_loaded_variables()]

            assert mock_load.call_count == 0, (
                "_load_variable should not be called again, variables was already loaded on first access"
            )

            assert (
                caplog.text.count("/GOES/primary/primary_maged_20130101to20130131_T89") == log_count_after_first_load
            ), "File should not have been loaded again"
