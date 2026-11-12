# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""Tests of the IGRF drift loss cone lookup table, on synthetic tables and without any IRBEM tracing."""

from __future__ import annotations

import importlib
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import el_paso as ep
import numpy as np
import pytest
import xarray as xr
from astropy import units as u
from el_paso.processing.magnetic_field_utils import (
    InternalFieldModel,
    IrbemInput,
    IrbemOptions,
    MagneticField,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

# synthetic tables and monkeypatched IRBEM calls only, so every test here is fast
pytestmark = [pytest.mark.minimal, pytest.mark.basic]

# the module names are shadowed by the functions the lazy loader exports under them
cmfv = importlib.import_module("el_paso.processing.compute_magnetic_field_variables")
dlc_lookup_table = importlib.import_module("el_paso.processing.magnetic_field_utils.dlc_lookup_table")

LAT = np.array([0.0, 10.0, 20.0])
LON = np.array([0.0, 90.0, 180.0, 270.0])


def _write_table(year: int, b_m_crit: NDArray[np.float64]) -> Path:
    """Writes a synthetic table where `compute_magnetic_field_variables` looks for it."""
    table = xr.Dataset(
        data_vars={"B_m_crit": (("lat", "lon"), b_m_crit, {"units": "nT"})},
        coords={"lat": LAT, "lon": LON},
        attrs={
            "epoch": datetime(year, 7, 2, tzinfo=timezone.utc).isoformat(),
            "stop_alt_km": 100.0,
            "lat_step_deg": 10.0,
            "lon_step_deg": 90.0,
            "dlc_tol_deg": 0.1,
            "drift_shell_resolution": 3,
        },
    )
    path = dlc_lookup_table.get_dlc_table_path(year)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_netcdf(path)
    return path


def _irbem_input(
    n_points: int,
    magnetic_field: MagneticField = MagneticField.Dip,
    internal_field_model: InternalFieldModel = InternalFieldModel.IGRF,
) -> IrbemInput:
    return IrbemInput(
        magnetic_field=magnetic_field,
        maginput={"Kp": np.zeros(n_points)},
        irbem_options=IrbemOptions(internal_field_model=internal_field_model),
    )


def _time_var(*dates: datetime) -> ep.Variable:
    return ep.Variable(data=np.array([d.timestamp() for d in dates]), original_unit=ep.units.posixtime)


def _computed_vars(
    b_calc: list[float], b_eq: list[float], b_fofl: list[float], model: str = "Dip"
) -> dict[str, ep.Variable]:
    return {
        f"B_Calc_{model}": ep.Variable(data=np.asarray(b_calc), original_unit=u.nT),
        f"B_Eq_{model}": ep.Variable(data=np.asarray(b_eq), original_unit=u.nT),
        # the weaker foot point is the given one, the other hemisphere's is stronger
        f"B_fofl_{model}": ep.Variable(
            data=np.column_stack([np.asarray(b_fofl), 3 * np.asarray(b_fofl)]), original_unit=u.nT
        ),
    }


@pytest.fixture
def model_data_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("EL_PASO_MODEL_DATA_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture
def foot_points(monkeypatch: pytest.MonkeyPatch) -> list[tuple[float, float]]:
    """Northern foot points (latitude, longitude) returned for the positions, in order."""
    points: list[tuple[float, float]] = []

    def fake_foot_point(
        xgeo_var: ep.Variable, _time_var: ep.Variable, irbem_input: IrbemInput, **_kwargs: object
    ) -> NDArray[np.float64]:
        # the table labels field lines of the IGRF field, whatever model the drift loss cone is converted with
        assert irbem_input.magnetic_field == MagneticField.Dip
        n_points = len(xgeo_var.get_data(ep.units.RE))
        taken = [points.pop(0) for _ in range(n_points)]
        return np.array([[100.0, lat, lon] for lat, lon in taken])

    monkeypatch.setattr(dlc_lookup_table, "_get_northern_foot_point", fake_foot_point)
    return points


def test_interpolation_wraps_longitude_and_is_linear_in_log() -> None:
    b_m_crit = np.tile([100.0, 200.0, 400.0, 800.0], (len(LAT), 1))
    table = xr.Dataset({"B_m_crit": (("lat", "lon"), b_m_crit)}, coords={"lat": LAT, "lon": LON})

    lat = np.array([5.0, 5.0, 5.0, 5.0])
    lon = np.array([45.0, 315.0, -45.0, 360.0])
    result = dlc_lookup_table._interpolate_b_m_crit(table, lat, lon)

    # geometric means between neighbouring nodes, across 360 degrees as well
    np.testing.assert_allclose(result, [np.sqrt(100 * 200), np.sqrt(800 * 100), np.sqrt(800 * 100), 100.0])


def test_interpolation_does_not_extrapolate_or_bridge_missing_nodes() -> None:
    b_m_crit = np.full((len(LAT), len(LON)), 1000.0)
    b_m_crit[2, 1] = np.nan
    table = xr.Dataset({"B_m_crit": (("lat", "lon"), b_m_crit)}, coords={"lat": LAT, "lon": LON})

    lat = np.array([25.0, 15.0, 5.0, np.nan])
    lon = np.array([0.0, 45.0, 45.0, 0.0])
    result = dlc_lookup_table._interpolate_b_m_crit(table, lat, lon)

    assert np.isnan(result[0])  # beyond the grid
    assert np.isnan(result[1])  # next to a node without a drift loss cone
    np.testing.assert_allclose(result[2], 1000.0)
    assert np.isnan(result[3])  # no foot point


def test_suspicious_nodes_flags_spikes_and_holes_across_the_date_line() -> None:
    lat = np.linspace(0, 1, 6)[:, np.newaxis]
    lon = np.linspace(0, 1, 8)[np.newaxis, :]
    b_m_crit = 20000 * (1 + 0.02 * lat + 0.01 * lon)  # smooth, a few percent across the grid
    eligible = np.ones(b_m_crit.shape, dtype=bool)

    assert not dlc_lookup_table._suspicious_nodes(b_m_crit, eligible).any()

    b_m_crit[2, 0] = 4000.0  # spurious ALL_LOST, next to the longitude wrap
    b_m_crit[4, 5] = np.nan  # failed node among valid ones
    b_m_crit[0, :] = np.nan  # a whole edge without drift loss cone is not a hole
    eligible[0, :] = False

    flagged = dlc_lookup_table._suspicious_nodes(b_m_crit, eligible).reshape(b_m_crit.shape)
    assert set(zip(*np.nonzero(flagged), strict=True)) == {(2, 0), (4, 5)}


@pytest.mark.usefixtures("model_data_path")
def test_pitch_angles_from_table_are_bounded_by_bounce_loss_cone(foot_points: list[tuple[float, float]]) -> None:
    path = _write_table(2017, np.full((len(LAT), len(LON)), 1000.0))
    foot_points.extend([(5.0, 45.0), (5.0, 45.0)])

    time_var = _time_var(datetime(2017, 3, 1, tzinfo=timezone.utc), datetime(2017, 3, 2, tzinfo=timezone.utc))
    xgeo_var = ep.Variable(data=np.ones((2, 3)), original_unit=ep.units.RE)
    # The second field line has a weak foot point, so its bounce loss cone exceeds the tabulated drift loss cone.
    computed_vars = _computed_vars(b_calc=[500.0, 500.0], b_eq=[100.0, 100.0], b_fofl=[2000.0, 600.0])

    out = dlc_lookup_table.get_drift_loss_cone_from_lookup_table(xgeo_var, time_var, _irbem_input(2), computed_vars)

    dlc = out["Alpha_DLC_Dip"].get_data(u.deg)
    dlc_eq = out["Alpha_DLC_Eq_Dip"].get_data(u.deg)
    np.testing.assert_allclose(dlc, [45.0, np.degrees(np.arcsin(np.sqrt(500 / 600)))])
    np.testing.assert_allclose(dlc_eq, [np.degrees(np.arcsin(np.sqrt(0.1))), np.degrees(np.arcsin(np.sqrt(1 / 6)))])

    for name in ("Alpha_DLC_Dip", "Alpha_DLC_Eq_Dip"):
        notes = out[name].metadata.processing_notes
        assert "NOT traced per time step" in notes
        assert str(path) in notes
        assert "No lookup table" not in notes
        assert "L > 5" not in notes


@pytest.mark.usefixtures("model_data_path")
def test_external_field_model_converts_igrf_table_with_its_own_fields(foot_points: list[tuple[float, float]]) -> None:
    _write_table(2017, np.full((len(LAT), len(LON)), 1000.0))
    foot_points.append((5.0, 45.0))

    time_var = _time_var(datetime(2017, 3, 1, tzinfo=timezone.utc))
    xgeo_var = ep.Variable(data=np.ones((1, 3)), original_unit=ep.units.RE)
    # T89 stretches the field: weaker at the equator than IGRF would be
    computed_vars = _computed_vars(b_calc=[480.0], b_eq=[40.0], b_fofl=[5000.0], model="T89")

    out = dlc_lookup_table.get_drift_loss_cone_from_lookup_table(
        xgeo_var, time_var, _irbem_input(1, MagneticField.T89), computed_vars
    )

    dlc = out["Alpha_DLC_T89"].get_data(u.rad)[0]
    dlc_eq = out["Alpha_DLC_Eq_T89"].get_data(u.rad)[0]
    np.testing.assert_allclose(np.sin(dlc) ** 2, 480 / 1000)
    np.testing.assert_allclose(np.sin(dlc_eq) ** 2, 40 / 1000)

    # Pitch angles mapped to the equator with T89 stay on the same side of the drift loss cone.
    for local_pa in (dlc - 1e-3, dlc + 1e-3):
        mapped = np.asin(np.sqrt(40 / 480) * np.sin(local_pa))
        assert (mapped < dlc_eq) == (local_pa < dlc)

    notes = out["Alpha_DLC_T89"].metadata.processing_notes
    assert "B_Calc and B_Eq of model MagneticField.T89" in notes
    assert "small deviations at L > 5" in notes


@pytest.mark.usefixtures("model_data_path")
def test_years_without_table_are_traced(
    foot_points: list[tuple[float, float]], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _write_table(2017, np.full((len(LAT), len(LON)), 1000.0))
    foot_points.append((5.0, 45.0))

    traced_calls: list[int] = []

    def fake_get_drift_loss_cone(
        xgeo_var: ep.Variable, _time_var: ep.Variable, irbem_input: IrbemInput
    ) -> dict[str, ep.Variable]:
        n_points = len(xgeo_var.get_data(ep.units.RE))
        assert len(irbem_input.maginput["Kp"]) == n_points
        assert irbem_input.magnetic_field == MagneticField.Dip
        traced_calls.append(n_points)
        return {
            "Alpha_DLC_Dip": ep.Variable(data=np.full(n_points, np.radians(80.0)), original_unit=u.rad),
            "Alpha_DLC_Eq_Dip": ep.Variable(data=np.full(n_points, np.radians(10.0)), original_unit=u.rad),
        }

    def fake_get_magequator(
        xgeo_var: ep.Variable, _time_var: ep.Variable, irbem_input: IrbemInput
    ) -> dict[str, ep.Variable]:
        assert irbem_input.magnetic_field == MagneticField.Dip
        n_points = len(xgeo_var.get_data(ep.units.RE))
        return {"B_Eq_Dip": ep.Variable(data=np.full(n_points, 100.0), original_unit=u.nT)}

    monkeypatch.setattr(dlc_lookup_table, "get_drift_loss_cone", fake_get_drift_loss_cone)
    monkeypatch.setattr(dlc_lookup_table, "get_magequator", fake_get_magequator)

    time_var = _time_var(datetime(2017, 12, 31, tzinfo=timezone.utc), datetime(2018, 1, 1, tzinfo=timezone.utc))
    xgeo_var = ep.Variable(data=np.ones((2, 3)), original_unit=ep.units.RE)
    computed_vars = _computed_vars(b_calc=[500.0, 500.0], b_eq=[100.0, 100.0], b_fofl=[2000.0, 1e5])

    with caplog.at_level(logging.WARNING):
        out = dlc_lookup_table.get_drift_loss_cone_from_lookup_table(xgeo_var, time_var, _irbem_input(2), computed_vars)

    assert traced_calls == [1]
    assert "No drift loss cone lookup table for 2018" in caplog.text
    # The traced year is converted through its critical mirror field, B_min / sin^2(10 deg), like the table.
    b_m_crit = 100.0 / np.sin(np.radians(10.0)) ** 2
    np.testing.assert_allclose(
        out["Alpha_DLC_Dip"].get_data(u.deg), [45.0, np.degrees(np.arcsin(np.sqrt(500 / b_m_crit)))]
    )
    np.testing.assert_allclose(out["Alpha_DLC_Eq_Dip"].get_data(u.deg)[1], 10.0)

    notes = out["Alpha_DLC_Dip"].metadata.processing_notes
    assert "NOT traced per time step" in notes
    assert "No lookup table for 2018" in notes


@pytest.mark.parametrize("magnetic_field", [MagneticField.Dip, MagneticField.T89])
def test_lookup_table_requires_igrf_internal_field(magnetic_field: MagneticField) -> None:
    assert dlc_lookup_table.uses_dlc_lookup_table(_irbem_input(1, magnetic_field))

    irbem_input = _irbem_input(1, magnetic_field, InternalFieldModel.CENTERED_DIPOLE)
    assert not dlc_lookup_table.uses_dlc_lookup_table(irbem_input)

    with pytest.raises(ValueError, match="requires IGRF as the internal field model"):
        dlc_lookup_table.get_drift_loss_cone_from_lookup_table(
            ep.Variable(data=np.ones((1, 3)), original_unit=ep.units.RE),
            _time_var(datetime(2017, 3, 1, tzinfo=timezone.utc)),
            irbem_input,
        )


@pytest.mark.parametrize(
    ("magnetic_field", "internal_field_model", "expected"),
    [
        (MagneticField.Dip, InternalFieldModel.IGRF, "table"),
        (MagneticField.Dip, InternalFieldModel.CENTERED_DIPOLE, "traced"),
        (MagneticField.T89, InternalFieldModel.IGRF, "table"),
        (MagneticField.T89, InternalFieldModel.CENTERED_DIPOLE, "traced"),
    ],
)
def test_compute_magnetic_field_variables_dispatches_to_table(
    monkeypatch: pytest.MonkeyPatch,
    magnetic_field: MagneticField,
    internal_field_model: InternalFieldModel,
    expected: str,
) -> None:
    calls: list[str] = []

    def fake(method: str):  # noqa: ANN202
        def inner(*_args: object, **_kwargs: object) -> dict[str, ep.Variable]:
            calls.append(method)
            return {}

        return inner

    monkeypatch.setattr(cmfv.mag_utils, "get_drift_loss_cone_from_lookup_table", fake("table"), raising=False)
    monkeypatch.setattr(cmfv.mag_utils, "get_drift_loss_cone", fake("traced"), raising=False)

    irbem_input = _irbem_input(1, magnetic_field, internal_field_model)
    for var_type in ("Alpha_DLC", "Alpha_DLC_Eq"):
        cmfv._get_result(var_type, None, None, None, None, {}, irbem_input, "electron")  # ty: ignore[invalid-argument-type]

    assert calls == [expected, expected]
