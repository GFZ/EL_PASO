# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the loss cone partition of Tu et al. 2010 (https://doi.org/10.1029/2009JA014949), figure 1.

That figure plots equatorial pitch angle against dipole longitude at L = 4.5, split into trapped,
quasi-trapped and untrapped electrons by the bounce loss cones of the two foot points and the drift loss
cone. A second test maps the same boundaries over the globe at low Earth orbit altitude, as local pitch
angles, which is what a spacecraft such as ELFIN or POES measures against.

Both use the internal field only ("Dip", i.e. no external model), so they run offline.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import pytest
from astropy import units as u
from matplotlib import pyplot as plt

import el_paso as ep
from el_paso.processing.magnetic_field_utils import (
    Coords,
    IrbemInput,
    IrbemOptions,
    LstarQuantity,
    MagneticField,
    construct_maginput,
    get_drift_loss_cone,
)
from el_paso.processing.magnetic_field_utils.irbem import MagFields

if TYPE_CHECKING:
    from numpy.typing import NDArray

NUM_CORES = min(os.cpu_count() or 1, 32)

# Colours of the published figure.
COLOR_TRAPPED = "#57c84d"
COLOR_QUASI_TRAPPED = "#4a52c8"
COLOR_UNTRAPPED = "#f2554e"

# Tu et al. 2010, figure 1
TU_EPOCH = datetime(2013, 8, 21, tzinfo=timezone.utc)
TU_L = 4.5
TU_DLON = 3.0
STOP_ALT_KM = 100.0

# Global map at low Earth orbit
MAP_EPOCH = datetime(2021, 1, 1, tzinfo=timezone.utc)
MAP_ALTITUDE_KM = 500.0
MAP_DLON, MAP_DLAT = 5.0, 4.0
MAP_L_RANGE = (1.5, 8.0)
L_CONTOURS = (2.0, 3.0, 6.0)


def _time_var(epoch: datetime, n_points: int) -> ep.Variable:
    return ep.Variable(data=np.full(n_points, epoch.timestamp()), original_unit=ep.units.posixtime)


def _l_m(x_geo: NDArray[np.float64], epoch: datetime, irbem_options: IrbemOptions) -> NDArray[np.float64]:
    """McIlwain L of the field line through each position, from el-paso's L_m for locally mirroring particles."""
    n_points = len(x_geo)
    out = ep.processing.compute_magnetic_field_variables(
        _time_var(epoch, n_points),
        ep.Variable(data=x_geo, original_unit=ep.units.RE),
        [("L_m", "Dip")],
        irbem_options,
        num_cores=NUM_CORES,
        pa_local_var=ep.Variable(data=np.full((n_points, 1), 90.0), original_unit=u.deg),
        cache_dir=None,
    )
    return out["L_m_Dip"].get_data()[:, 0]


def _mag_equator_to_geo(lon_deg: NDArray[np.float64], radius: NDArray[np.float64], epoch: datetime) -> NDArray:
    """Points in the magnetic equatorial plane (MAG z = 0), as GEO cartesian positions in Earth radii."""
    angle = np.radians(lon_deg)
    x_mag = np.column_stack([radius * np.cos(angle), radius * np.sin(angle), np.zeros(lon_deg.size)])
    return Coords().transform([epoch] * lon_deg.size, x_mag, "MAG", "GEO")


def _radius_for_l(mag_lon: NDArray[np.float64], l_value: float, irbem_options: IrbemOptions) -> NDArray:
    """Radius in the magnetic equatorial plane at which L_m equals `l_value`, for each MAG longitude."""
    radii = np.arange(l_value - 0.8, l_value + 0.8, 0.02)
    lon_grid, radius_grid = np.meshgrid(mag_lon, radii, indexing="ij")

    x_geo = _mag_equator_to_geo(lon_grid.ravel(), radius_grid.ravel(), TU_EPOCH)
    lm = _l_m(x_geo, TU_EPOCH, irbem_options).reshape(lon_grid.shape)

    # L_m increases with radius, so plain interpolation is well posed
    return np.array([np.interp(l_value, row[np.isfinite(row)], radii[np.isfinite(row)]) for row in lm])


@pytest.mark.visual
def test_replot_tu2010_figure() -> None:
    """Regenerate the left panel of Tu et al. 2010, figure 1, and check its key features."""
    irbem_options = IrbemOptions(
        lstar_quantity=LstarQuantity.NONE,
        igrf_update_interval_days=0,
        field_line_resolution=3,
        drift_shell_resolution=3,
    )

    # --- positions: one field line per MAG longitude, all on the L = 4.5 shell ---------------------------
    mag_lon = np.arange(0.0, 360.0, TU_DLON)
    x_geo = _mag_equator_to_geo(mag_lon, _radius_for_l(mag_lon, TU_L, irbem_options), TU_EPOCH)
    time_var = _time_var(TU_EPOCH, len(x_geo))
    xgeo_var = ep.Variable(data=x_geo, original_unit=ep.units.RE)

    # --- bounce loss cones of the two foot points ---------------------------------------------------------
    b_eq = ep.processing.compute_magnetic_field_variables(
        time_var, xgeo_var, [("B_Eq", "Dip")], irbem_options, num_cores=NUM_CORES, cache_dir=None
    )["B_Eq_Dip"].get_data(u.nT)

    model = MagFields(kext=0, sysaxes=ep.IRBEM_SYSAXIS_GEO, options=irbem_options)
    b_foot = {
        hemi: np.array(
            [
                model.find_foot_point(TU_EPOCH, {"x1": x[0], "x2": x[1], "x3": x[2]}, {}, STOP_ALT_KM, hemi).b_foot_mag
                for x in x_geo
            ]
        )
        for hemi in (1, -1)
    }
    blc_north = np.degrees(np.arcsin(np.sqrt(b_eq / b_foot[1])))
    blc_south = np.degrees(np.arcsin(np.sqrt(b_eq / b_foot[-1])))
    blc_local = np.fmax(blc_north, blc_south)  # the weaker foot point wins

    # --- drift loss cone, traced independently at every longitude ------------------------------------------
    irbem_input = IrbemInput(
        magnetic_field=MagneticField.Dip,
        maginput=construct_maginput(time_var, MagneticField.Dip),
        irbem_options=irbem_options,
        num_cores=NUM_CORES,
    )
    dlc = get_drift_loss_cone(xgeo_var, time_var, irbem_input, stop_alt=STOP_ALT_KM, dlc_tol=0.01)
    dlc = dlc["Alpha_DLC_Eq_Dip"].get_data(u.deg)

    # --- dipole longitude from the SAA, as in the published figure ------------------------------------------
    # Zero where the southern foot point field is weakest, i.e. where the southern bounce loss cone peaks.
    i_saa = int(np.argmin(b_foot[-1]))
    dipole_lon = (mag_lon - mag_lon[i_saa]) % 360
    order = np.argsort(dipole_lon)
    order = np.append(order, order[0])  # close the circle so the curves reach 360
    lon = np.append(dipole_lon[order[:-1]], 360.0)
    blc_north, blc_south, blc_local, dlc = (arr[order] for arr in (blc_north, blc_south, blc_local, dlc))

    # --- figure ----------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    top, bottom = 6.5, 3.5
    ax.fill_between(lon, dlc, top, color=COLOR_TRAPPED)
    ax.fill_between(lon, blc_local, dlc, color=COLOR_QUASI_TRAPPED)
    ax.fill_between(lon, bottom, blc_local, color=COLOR_UNTRAPPED)
    ax.plot(lon, blc_south, "k-", lw=2, label="southern bounce loss cone")
    ax.plot(lon, blc_north, "k--", lw=2, label="northern bounce loss cone")
    ax.plot(lon, dlc, color="k", lw=1, label="drift loss cone (traced)")
    ax.axhline(np.nanmax(blc_local), color="w", lw=1, ls=(0, (4, 4)), label="largest bounce loss cone on the shell")
    for x, y, text in ((15, 6.0, "(a) Trapped"), (180, 5.0, "(b) Quasi-Trapped"), (15, 3.9, "(c) Untrapped")):
        ax.text(x, y, text, ha="left" if x < 100 else "center", fontsize=13, fontweight="bold", color="w")
    ax.set_xlim(0, 360)
    ax.set_ylim(bottom, top)
    ax.set_xticks(np.arange(0, 361, 50))
    ax.set_xlabel("Dipole longitude from the SAA [°]")
    ax.set_ylabel("Equatorial pitch angle [°]")
    ax.set_title(f"L={TU_L:g}", loc="left", fontsize=15)
    ax.set_title(f"IGRF {TU_EPOCH:%Y-%m-%d}, lost below {STOP_ALT_KM:.0f} km", loc="right", fontsize=9)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.85)
    fig.savefig("tu2010_loss_cones_vs_longitude.png", dpi=150)
    plt.close(fig)

    # --- checks ----------------------------------------------------------------------------------------
    # A particle survives its drift only if it clears the loss cone at every longitude, so the drift loss cone
    # is the largest bounce loss cone anywhere on the shell - and, being a property of the shell, flat.
    assert np.isfinite(dlc).all()
    assert np.ptp(dlc) < 0.06, f"drift loss cone should be flat, spans {np.ptp(dlc):.3f} deg"
    np.testing.assert_allclose(dlc, np.nanmax(blc_local), atol=0.05)
    assert (dlc >= blc_local - 0.01).all(), "the drift loss cone can never be narrower than the bounce loss cone"

    # Values read off the published figure.
    assert abs(np.nanmax(blc_south) - 5.5) < 0.15
    assert abs(np.nanmin(blc_south) - 4.2) < 0.15
    assert 4.2 < np.nanmin(blc_north) < np.nanmax(blc_north) < 4.8
    assert abs(np.nanmean(dlc) - 5.5) < 0.15

    # The two bounce loss cones cross twice, once on either side of the southern minimum.
    crossings = np.count_nonzero(np.diff(np.sign(blc_south - blc_north)))
    assert crossings == 2, f"expected the two bounce loss cones to cross twice, got {crossings}"


@pytest.mark.visual
def test_replot_drift_loss_cone_map() -> None:
    """Map the precipitating and drift loss cones at 500 km, as seen by a low Earth orbit spacecraft."""
    irbem_options = IrbemOptions(
        lstar_quantity=LstarQuantity.NONE,
        igrf_update_interval_days=0,
        field_line_resolution=0,
        drift_shell_resolution=1,
    )

    lon_edges = np.arange(-180.0, 180.0 + MAP_DLON / 2, MAP_DLON)
    lat_edges = np.arange(-90.0, 90.0 + MAP_DLAT / 2, MAP_DLAT)
    lon_c = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_c = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_grid, lat_grid = np.meshgrid(lon_c, lat_c)

    gdz = np.column_stack([np.full(lon_grid.size, MAP_ALTITUDE_KM), lat_grid.ravel(), lon_grid.ravel()])
    x_geo = Coords().transform([MAP_EPOCH] * len(gdz), gdz, "GDZ", "GEO")

    # Only where the local field line traps particles at all: this carves out the gap around the magnetic
    # equator, where nothing is trapped at this altitude, and the polar caps.
    lm = _l_m(x_geo, MAP_EPOCH, irbem_options)
    in_band = (lm >= MAP_L_RANGE[0]) & (lm <= MAP_L_RANGE[1])

    out = ep.processing.compute_magnetic_field_variables(
        _time_var(MAP_EPOCH, int(in_band.sum())),
        ep.Variable(data=x_geo[in_band], original_unit=ep.units.RE),
        [("Alpha_LC", "Dip"), ("Alpha_DLC", "Dip")],
        irbem_options,
        num_cores=NUM_CORES,
        cache_dir=None,
    )

    def on_grid(values: NDArray[np.float64]) -> NDArray[np.float64]:
        grid = np.full(lon_grid.size, np.nan)
        grid[in_band] = values
        return grid.reshape(lon_grid.shape)

    lc = on_grid(out["Alpha_LC_Dip"].get_data(u.deg))
    dlc = on_grid(out["Alpha_DLC_Dip"].get_data(u.deg))
    lm_grid = np.where(in_band, lm, np.nan).reshape(lon_grid.shape)

    # --- figure ----------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), constrained_layout=True)
    for ax, values, title in ((axes[0], lc, "Bounce Loss Cone"), (axes[1], dlc, "Drift Loss Cone")):
        mesh = ax.pcolormesh(lon_edges, lat_edges, values, cmap="jet", vmin=60, vmax=90)
        contours = ax.contour(lon_c, lat_c, lm_grid, levels=list(L_CONTOURS), colors="w", linestyles="dashed")
        ax.clabel(contours, fmt=lambda level: f"L={level:g}", fontsize=8)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xticks(np.arange(-180, 181, 60))
        ax.set_yticks(np.arange(-90, 91, 30))
        ax.set_xlabel("GEO longitude [°]")
        ax.set_ylabel("GEO latitude [°]")
        ax.set_title(title, color="tab:blue")
        fig.colorbar(mesh, ax=ax, extend="min", label="Local pitch angle [°]")
    fig.suptitle(f"{MAP_ALTITUDE_KM:.0f} km, IGRF {MAP_EPOCH:%Y-%m-%d}, lost below {STOP_ALT_KM:.0f} km", fontsize=10)
    fig.savefig("drift_loss_cone_map.png", dpi=150)
    plt.close(fig)

    # --- checks ----------------------------------------------------------------------------------------
    finite = np.isfinite(dlc) & np.isfinite(lc)
    assert finite.sum() / in_band.sum() > 0.98, "too many points without a drift loss cone"
    # Where the local field line is already the weakest on its drift shell the two are equal by definition,
    # but come from different code paths, so allow for round-off.
    assert (dlc[finite] >= lc[finite] - 1e-6).all(), "the drift loss cone can never be narrower than the bounce one"

    # Away from the South Atlantic Anomaly, everything seen at 500 km drifts into it and is lost...
    north = finite & (lat_grid > 0)
    assert np.mean(dlc[north] > 89.9) > 0.95
    far_south = finite & (lat_grid < 0) & (lon_grid > 90)
    assert np.mean(dlc[far_south] > 89.9) > 0.9

    # ...while under it, the local field line is already the weakest on its drift shell, so part of the
    # distribution seen there is stably trapped.
    saa = finite & (lat_grid < -25) & (lat_grid > -60) & (lon_grid > -60) & (lon_grid < 30)
    assert np.nanmin(dlc[saa]) < 75
