# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import pathlib
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from unittest import mock

import numpy as np
import pytest
import xarray as xr
from astropy import units as u

import el_paso as ep
from el_paso.dataset import DataSet, GFZDataSet
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


@pytest.fixture
def gfz_dataset(tmp_path: Path) -> DataSet:
    variables = _mock_monthly_variables()
    start_time = datetime(2013, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2013, 1, 2, tzinfo=timezone.utc)
    strategy = ep.saving_strategies.MonthlyRBStrategy(
        tmp_path, "TEST", "sat", "inst", "T89",
        data_standard=ep.data_standards.GFZStandard(),
        file_format="nc",
    )
    ep.save(variables, strategy, start_time=start_time, end_time=end_time, time_var=variables["Epoch"])
    return DataSet(saving_strategy=strategy, start_time=start_time, end_time=end_time, preferred_extension="nc")


@pytest.fixture
def prbem_dataset(tmp_path: Path) -> DataSet:
    variables = _mock_monthly_variables()
    start_time = datetime(2013, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2013, 1, 2, tzinfo=timezone.utc)
    strategy = ep.saving_strategies.MonthlyRBStrategy(
        tmp_path, "TEST", "sat", "inst", "T89",
        data_standard=ep.data_standards.PRBEMStandard(),
        file_format="nc",
    )
    ep.save(variables, strategy, start_time=start_time, end_time=end_time, time_var=variables["Epoch"])
    return DataSet(saving_strategy=strategy, start_time=start_time, end_time=end_time, preferred_extension="nc")


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

    def test_datetime_access_with_missing_files_returns_empty_value(self, tmp_path: Path) -> None:
        strategy = ep.saving_strategies.MonthlyRBStrategy(
            base_data_path=tmp_path,
            mission="GOES",
            satellite="primary",
            instrument="MAGED",
            mag_field="T89",
            file_format="nc",
            data_standard=ep.data_standards.GFZStandard(),
        )
        start_time = datetime(2013, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2013, 1, 2, tzinfo=timezone.utc)

        dataset = DataSet(strategy, start_time, end_time, verbose=False)

        assert dataset.datetime == []
        assert "datetime" in dataset.get_loaded_variables()

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

        for standard_name in mock_dataset.saving_strategy.get_all_standard_names():
            assert standard_name in mock_dataset.__dir__()

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

    def test_usable_without_context_manager(self, mock_dataset: DataSet):
        """Direct usage (no `with`, no explicit close()) must keep working unchanged."""
        mock_dataset.load("Flux")

        assert isinstance(mock_dataset.Flux, np.ndarray)
        assert mock_dataset.get_satellite_name() == "primary"
        # Resources, if any were opened, are still open -- nothing forces a close
        # unless the caller opts into `with` or calls close() explicitly.
        mock_dataset.close()  # should still be safe to call directly, though optional

    def test_context_manager_closes_open_resources(self, mock_dataset: DataSet):
        """Entering/exiting a `with` block should close any open lazy file handles."""
        with mock_dataset as ds:
            assert ds is mock_dataset
            ds.load("Flux")

        assert mock_dataset._open_resources == []

    def test_context_manager_keeps_loaded_data_accessible_after_exit(self, mock_dataset: DataSet):
        """Already-loaded variables must remain readable after the `with` block exits."""
        with mock_dataset as ds:
            ds.load("Flux")
            loaded_flux = ds.Flux.copy()

        np.testing.assert_array_equal(mock_dataset.Flux, loaded_flux)

    def test_close_is_idempotent(self, mock_dataset: DataSet):
        """Calling close() multiple times should not raise."""
        mock_dataset.load("Flux")
        mock_dataset.close()
        mock_dataset.close()

        assert mock_dataset._open_resources == []

    def test_close_materializes_still_lazy_variables(self, mock_dataset: DataSet):
        """After close(), no loaded variable should still be a lazy `xr.Variable`.

        Regression test: loading "Flux" also caches every other variable found in the
        same file (e.g. "alpha_local"), but only as a lazy, unmaterialized `xr.Variable`
        if it was never itself accessed. Accessing such a variable for the first time,
        even after close(), used to silently reopen the underlying file via xarray's
        caching file manager, bypassing `_open_resources` entirely.
        """
        mock_dataset.load("Flux")
        mock_dataset.close()

        raw_dict = object.__getattribute__(mock_dataset, "__dict__")
        for var_name in mock_dataset.get_loaded_variables():
            if var_name == "metadata":
                continue
            assert not isinstance(raw_dict[var_name], xr.Variable), (
                f"'{var_name}' is still a lazy xr.Variable after close()"
            )

    def test_overwrite_raises_permission_error_while_dataset_still_open(self, mock_dataset: DataSet):
        """Rewriting the backing `.nc` file while it is still open must fail.

        This documents the actual failure mode `close()` addresses: as long as a
        `DataSet` has lazily loaded data from a `.nc` file and has not been closed,
        reopening that same file for writing raises `PermissionError`. Closing it
        first releases the file so the write succeeds.
        """
        if mock_dataset._preferred_ext != "nc":
            pytest.skip("File-locking behavior is specific to the .nc backend.")

        mock_dataset.load("Flux")

        overwrite_variables = _mock_monthly_variables()

        with pytest.raises(PermissionError):
            ep.save(
                overwrite_variables,
                mock_dataset.saving_strategy,
                start_time=mock_dataset._start_time,
                end_time=mock_dataset._end_time,
                time_var=overwrite_variables["Epoch"],
            )

        mock_dataset.close()

        ep.save(
            overwrite_variables,
            mock_dataset.saving_strategy,
            start_time=mock_dataset._start_time,
            end_time=mock_dataset._end_time,
            time_var=overwrite_variables["Epoch"],
        )

    def test_overwrite_succeeds_after_close_even_after_accessing_new_variable(
        self,
        mock_dataset: DataSet,
    ):
        """Regression test for closing a file, accessing a new var, then overwriting it.

        Reproduces a real failure: close() the dataset, then access a variable that was
        cached but never materialized ("alpha_local"). Before the fix, this silently
        reopened the file outside of close()'s tracking, so a subsequent overwrite of
        the same file failed with PermissionError even though close() had been called.
        """
        if mock_dataset._preferred_ext != "nc":
            pytest.skip("Lazy-loading reopen behavior is specific to the .nc backend.")

        mock_dataset.load("Flux")
        mock_dataset.close()

        _ = mock_dataset.alpha_local  # first-ever access, after close()

        mock_dataset.close()

        overwrite_variables = _mock_monthly_variables()
        ep.save(
            overwrite_variables,
            mock_dataset.saving_strategy,
            start_time=mock_dataset._start_time,
            end_time=mock_dataset._end_time,
            time_var=overwrite_variables["Epoch"],
        )

    def test_context_manager_propagates_exceptions(self, mock_dataset: DataSet):
        """Exceptions raised inside a `with` block must still propagate, after closing."""

        def _load_then_raise() -> None:
            with mock_dataset as ds:
                ds.load("Flux")
                raise ValueError("boom")  # noqa: EM101

        with pytest.raises(ValueError, match="boom"):
            _load_then_raise()

        assert mock_dataset._open_resources == []


@pytest.mark.basic
@pytest.mark.parametrize("file_format", ["nc", "h5", "cdf", "mat"])
def test_dataset_equality_rejects_data_saved_with_different_standards(tmp_path: Path, file_format: MFSFormats) -> None:
    variables = _mock_monthly_variables()
    start_time = datetime(2013, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2013, 1, 2, tzinfo=timezone.utc)

    gfz_strategy = ep.saving_strategies.MonthlyRBStrategy(
        base_data_path=tmp_path / "gfz",
        mission="GOES",
        satellite="primary",
        instrument="MAGED",
        mag_field="T89",
        file_format=file_format,
        data_standard=ep.data_standards.GFZStandard(),
    )
    prbem_strategy = ep.saving_strategies.MonthlyRBStrategy(
        base_data_path=tmp_path / "prbem",
        mission="GOES",
        satellite="primary",
        instrument="MAGED",
        mag_field="T89",
        file_format=file_format,
        data_standard=ep.data_standards.PRBEMStandard(),
    )

    for strategy in (gfz_strategy, prbem_strategy):
        ep.save(variables, strategy, start_time=start_time, end_time=end_time, time_var=variables["Epoch"])

    gfz_dataset = DataSet(
        saving_strategy=gfz_strategy, start_time=start_time, end_time=end_time,
        preferred_extension=file_format, verbose=False,
    )
    prbem_dataset = DataSet(
        saving_strategy=prbem_strategy, start_time=start_time, end_time=end_time,
        preferred_extension=file_format, verbose=False,
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
def test_comparison_loads_all_strategy_variables(tmp_path: Path, file_format: MFSFormats) -> None:
    """After == comparison every standard-name variable must be in get_loaded_variables()."""
    variables = _mock_monthly_variables()
    start_time = datetime(2013, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2013, 1, 2, tzinfo=timezone.utc)

    strategy = ep.saving_strategies.MonthlyRBStrategy(
        base_data_path=tmp_path,
        mission="GOES",
        satellite="primary",
        instrument="MAGED",
        mag_field="T89",
        file_format=file_format,
        data_standard=ep.data_standards.GFZStandard(),
    )
    ep.save(variables, strategy, start_time=start_time, end_time=end_time, time_var=variables["Epoch"])

    ds1 = DataSet(saving_strategy=strategy, start_time=start_time, end_time=end_time,
                  preferred_extension=file_format, verbose=False)
    ds2 = DataSet(saving_strategy=strategy, start_time=start_time, end_time=end_time,
                  preferred_extension=file_format, verbose=False)

    assert [v for v in ds1.get_loaded_variables() if v != "metadata"] == [], (
        "No data variables should be loaded before comparison"
    )

    _ = ds1 == ds2

    loaded = set(ds1.get_loaded_variables())
    expected = set(strategy.get_all_standard_names())
    assert expected.issubset(loaded), f"Variables not loaded after comparison: {expected - loaded}"


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
    mfs_strategy = ep.saving_strategies.MonthlyRBStrategy(
        base_data_path=tmp_path / "prbem",
        mission="GOES",
        satellite="primary",
        instrument="MAGED",
        mag_field="T89",
        file_format=file_format,
        data_standard=data_standard(),
    )

    for strategy in (gfz_strategy, mfs_strategy):
        ep.save(variables, strategy, start_time=start_time, end_time=end_time, time_var=variables["Epoch"])

    gfz_strategy_dataset = DataSet(
        saving_strategy=gfz_strategy, start_time=start_time, end_time=end_time,
        preferred_extension=file_format, verbose=False,
    )
    mfs_strategy_dataset = DataSet(
        saving_strategy=mfs_strategy, start_time=start_time, end_time=end_time,
        preferred_extension=file_format, verbose=False,
    )

    assert gfz_strategy_dataset.datetime == mfs_strategy_dataset.datetime

    gfz_strategy_dataset.load(gfz_strategy.data_standard.get_standard_name("FEDU"))
    mfs_strategy_dataset.load(mfs_strategy.data_standard.get_standard_name("FEDU"))

    assert gfz_strategy.data_standard.get_standard_name("FEDU") in gfz_strategy_dataset.get_loaded_variables()
    assert mfs_strategy.data_standard.get_standard_name("FEDU") in mfs_strategy_dataset.get_loaded_variables()
    gfz_strategy_dataset.assert_equal(mfs_strategy_dataset)
    assert gfz_strategy_dataset == mfs_strategy_dataset


_PROCESSED = pathlib.Path(__file__).parent.parent.parent / "system" / "data" / "processed"
_ECT_START = datetime(2017, 9, 8, tzinfo=timezone.utc)
_ECT_END = _ECT_START + timedelta(days=0.4, seconds=-1)


@pytest.mark.basic
def test_dataset_equals_gfz_dataset_on_ect_combined() -> None:
    """DataSet and GFZDataSet loaded from the same file must compare equal."""
    strategy = ep.saving_strategies.MonthlyRBStrategy(
        _PROCESSED,
        "RBSP",
        "rbspa",
        "ect_combined",
        "T89",
        data_standard=ep.data_standards.GFZStandard(),
        file_format="nc",
    )
    base_ds = DataSet(saving_strategy=strategy, start_time=_ECT_START, end_time=_ECT_END, verbose=False)
    gfz_ds = GFZDataSet(saving_strategy=strategy, start_time=_ECT_START, end_time=_ECT_END, verbose=False)

    assert base_ds == gfz_ds

@pytest.mark.basic
def test_gfz_epoch_resolves_to_time(gfz_dataset: DataSet) -> None:
    result = gfz_dataset.get_by_internal_name("Epoch")
    np.testing.assert_array_equal(result, gfz_dataset.time)


@pytest.mark.basic
def test_gfz_fedu_resolves_to_flux(gfz_dataset: DataSet) -> None:
    result = gfz_dataset.get_by_internal_name("FEDU")
    np.testing.assert_array_equal(result, gfz_dataset.Flux)


@pytest.mark.basic
def test_gfz_r_eq_resolves_to_r0(gfz_dataset: DataSet) -> None:
    result = gfz_dataset.get_by_internal_name("R_Eq")
    np.testing.assert_array_equal(result, gfz_dataset.R0)


@pytest.mark.basic
def test_gfz_l_star_resolves_to_lstar(gfz_dataset: DataSet) -> None:
    result = gfz_dataset.get_by_internal_name("L_star")
    np.testing.assert_array_equal(result, gfz_dataset.Lstar)


@pytest.mark.basic
def test_gfz_energy_fedu_resolves_to_energy_channels(gfz_dataset: DataSet) -> None:
    result = gfz_dataset.get_by_internal_name("Energy_FEDU")
    np.testing.assert_array_equal(result, gfz_dataset.energy_channels)


@pytest.mark.basic
def test_gfz_alpha_eq_resolves_to_alpha_eq_model(gfz_dataset: DataSet) -> None:
    result = gfz_dataset.get_by_internal_name("Alpha_Eq")
    np.testing.assert_array_equal(result, gfz_dataset.alpha_eq_model)


@pytest.mark.basic
def test_prbem_epoch_resolves_to_epoch(prbem_dataset: DataSet) -> None:
    result = prbem_dataset.get_by_internal_name("Epoch")
    np.testing.assert_array_equal(result, prbem_dataset.Epoch)


@pytest.mark.basic
def test_prbem_r_eq_resolves_to_r_eq(prbem_dataset: DataSet) -> None:
    result = prbem_dataset.get_by_internal_name("R_Eq")
    np.testing.assert_array_equal(result, prbem_dataset.R_Eq)


@pytest.mark.basic
def test_prbem_l_star_resolves_to_l_star(prbem_dataset: DataSet) -> None:
    result = prbem_dataset.get_by_internal_name("L_star")
    np.testing.assert_array_equal(result, prbem_dataset.L_star)


@pytest.mark.basic
def test_unknown_internal_name_raises(gfz_dataset: DataSet) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        gfz_dataset.get_by_internal_name("NotAnInternalName")  # ty:ignore[invalid-argument-type]
