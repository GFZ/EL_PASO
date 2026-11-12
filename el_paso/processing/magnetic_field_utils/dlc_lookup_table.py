# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""Precomputed drift loss cone of the IGRF field, looked up instead of traced at every time step.

Without an external field model ("Dip", i.e. kext 0) the IGRF field is static in GEO apart from its secular
variation, so the drift loss cone only depends on the field line a particle is on. It is tabulated once per year
on a grid of field lines and interpolated for any satellite position.

The tabulated quantity is the critical mirror field `B_m_crit` of the drift shell, i.e. the mirror field of a
marginally trapped particle. A particle mirroring at a field stronger than it reaches below the stopping altitude
somewhere on its drift orbit. Unlike the pitch angle, `B_m_crit` is conserved along the drift and therefore varies
smoothly from field line to field line, even across the South Atlantic Anomaly. The pitch angles follow exactly as
`sin^2(Alpha_DLC) = B_Calc / B_m_crit` and `sin^2(Alpha_DLC_Eq) = B_Eq / B_m_crit`.

The table is also used with external field models such as T89: `B_m_crit` is a field strength near the stopping
altitude, set by the internal field, and converting it with `B_Calc` and `B_Eq` of the external field model keeps
the drift loss cone consistent with pitch angles mapped by that model. The effect of the external field on the drift
shell itself is neglected, which can lead to small deviations at L > 5.

Field lines are labelled by their foot point in the northern magnetic hemisphere at the stopping altitude, which
names every closed field line exactly once and gives a regular latitude/longitude grid.
"""

from __future__ import annotations

import dataclasses
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u

import el_paso as ep
from el_paso.processing.magnetic_field_utils.irbem import (
    DEFAULT_LIBIRBEM_PATH,
    Coords,
    InternalFieldModel,
    IrbemOptions,
    LstarQuantity,
    MagFields,
)
from el_paso.processing.magnetic_field_utils.mag_field_enum import MagneticField
from el_paso.processing.magnetic_field_utils.magnetic_field_functions import (
    IrbemInput,
    _get_northern_foot_point,
    _get_worker_context,
    _run_irbem_parallel,
    create_var_name,
    get_drift_loss_cone,
    get_footpoint_atmosphere,
    get_local_B_field,
    get_magequator,
)
from el_paso.utils import get_el_paso_model_data_path, timed_function

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

DLC_TABLE_SUBDIR = "dlc_lookup_tables"
"""Subdirectory of the el_paso model data path (`EL_PASO_MODEL_DATA_PATH`) holding the lookup tables."""

_SCRIPT = "scripts/compute_dlc_lookup_table.py"


def get_dlc_table_path(year: int) -> Path:
    """Returns the path of the IGRF drift loss cone lookup table of a year.

    The tables are stored in the `dlc_lookup_tables` subdirectory of the el_paso model data path, which is
    `EL_PASO_MODEL_DATA_PATH` if set and `~/.elpaso` otherwise.

    Args:
        year (int): The year of the table.

    Returns:
        Path: The path of the table, whether it exists or not.
    """
    return get_el_paso_model_data_path() / DLC_TABLE_SUBDIR / f"dlc_lookup_table_IGRF_{year}.nc"


def uses_dlc_lookup_table(irbem_input: IrbemInput) -> bool:
    """Checks whether the drift loss cone of this IRBEM configuration is taken from the IGRF lookup table.

    This is the case whenever IGRF is the internal field model, with or without an external field model.

    Args:
        irbem_input (IrbemInput): The IRBEM input of the calculation.

    Returns:
        bool: True if the lookup table applies.
    """
    return IrbemOptions(*irbem_input.irbem_options).internal_field_model == InternalFieldModel.IGRF


def _table_irbem_input(
    irbem_options: IrbemOptions, num_cores: int, irbem_lib_path: str | Path, n_points: int
) -> IrbemInput:
    return IrbemInput(
        magnetic_field=MagneticField.Dip,
        maginput={"Kp": np.zeros(n_points)},
        irbem_options=irbem_options,
        num_cores=num_cores,
        irbem_lib_path=irbem_lib_path,
    )


_SPIKE_LOG_THRESHOLD = np.log(1.2)
"""Largest deviation of log(B_m_crit) of a node from the median of its neighbours before it counts as suspicious.

B_m_crit typically changes by a few percent between neighbouring nodes, a spurious drift loss cone by a factor of
several.
"""


def _suspicious_nodes(b_m_crit: NDArray[np.float64], eligible: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Flags nodes whose B_m_crit stands out of their 8 neighbours, or is missing in the middle of valid ones.

    The grid has latitude along the first and longitude along the second axis, which wraps around.
    """
    import warnings  # noqa: PLC0415

    log_b = np.log(b_m_crit)
    n_lat = log_b.shape[0]
    padded = np.pad(log_b, ((1, 1), (0, 0)), constant_values=np.nan)
    neighbours = np.stack(
        [
            np.roll(padded, -dlon, axis=1)[1 + dlat : n_lat + 1 + dlat]
            for dlat in (-1, 0, 1)
            for dlon in (-1, 0, 1)
            if (dlat, dlon) != (0, 0)
        ]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN neighbourhoods
        median = np.nanmedian(neighbours, axis=0)
    n_valid = np.isfinite(neighbours).sum(axis=0)

    spikes = np.isfinite(log_b) & (n_valid >= 3) & (np.abs(log_b - median) > _SPIKE_LOG_THRESHOLD)
    holes = eligible & ~np.isfinite(log_b) & (n_valid >= 6)

    return (spikes | holes).ravel()


def _trace_field_line_parallel(it: int) -> tuple[float, float, NDArray[np.float64]]:
    """Traces the field line of a node and returns its Bmin, its largest geocentric distance and a start point.

    The drift loss cone is started a quarter of the way along the field line: IRBEM does not close the drift shell
    when starting exactly at the magnetic equator, and the equatorial drift loss cone is the same from any other
    point of the field line.
    """
    context = _get_worker_context()

    trace = context.model.trace_field_line(
        context.datetimes[it], context.position_at(it), context.maginput_at(it), r0=1.0
    )
    posit = trace.posit[: trace.n_posit]
    if trace.n_posit < 3 or not np.isfinite(posit).all():
        return np.nan, np.nan, np.full(3, np.nan)

    return trace.bmin, float(np.linalg.norm(posit, axis=1).max()), posit[(trace.n_posit - 1) // 4]


@timed_function()
def compute_dlc_lookup_table(
    year: int,
    *,
    lat_step: float = 0.5,
    lon_step: float = 5.0,
    lat_range: tuple[float, float] = (-20.0, 90.0),
    max_eq_radius: float = 10.0,
    stop_alt: float = 100.0,
    dlc_tol: float = 0.1,
    drift_shell_resolution: int = 3,
    num_cores: int = 4,
    irbem_lib_path: str | Path = DEFAULT_LIBIRBEM_PATH,
) -> xr.Dataset:
    """Computes the IGRF drift loss cone lookup table of a year.

    For every node of a latitude/longitude grid at `stop_alt`, the field line through it is traced, and the
    drift loss cone is calculated with `get_drift_loss_cone` from a point a quarter of the way along it. Nodes in the
    southern magnetic hemisphere are skipped, since the lookup labels field lines by their northern foot point,
    as are field lines reaching beyond `max_eq_radius`. IGRF is evaluated at the middle of the year (2 July).

    IRBEM occasionally fails for single field lines, e.g. reporting a spurious ALL_LOST at one drift shell
    resolution. Since `B_m_crit` varies smoothly between field lines, nodes deviating by more than 20 % from the
    median of their neighbours, or missing among valid ones, are recomputed with successively coarser drift shell
    resolutions, and set to NaN if they still deviate.

    This costs about a dozen drift shell traces per node: with the default grid of about 16,000 nodes, roughly two
    hours on 24 cores. `scripts/compute_dlc_lookup_table.py` computes and saves the tables.

    Args:
        year (int): The year of the IGRF field.
        lat_step (float, optional): Latitude spacing of the grid in degrees. Defaults to 0.5.
        lon_step (float, optional): Longitude spacing of the grid in degrees. Must divide 360. Defaults to 5.
        lat_range (tuple[float, float], optional): Geodetic latitude range of the grid in degrees. It must include
            every northern foot point of interest, which reach south of the geographic equator over the
            Atlantic. Defaults to (-20, 90).
        max_eq_radius (float, optional): Field lines crossing the magnetic equator beyond this geocentric
            distance in Earth radii are not traced. Defaults to 10.
        stop_alt (float, optional): Geodetic altitude in km below which a particle counts as lost; also the
            altitude of the grid. Defaults to 100.
        dlc_tol (float, optional): Bisection tolerance on the equatorial drift loss cone in degrees.
            Defaults to 0.1.
        drift_shell_resolution (int, optional): IRBEM `options(4)`, setting how finely the drift orbit samples
            longitude. 0 can step over the bottom of the South Atlantic Anomaly. Defaults to 3.
        num_cores (int, optional): Number of processes for the IRBEM calculations. Defaults to 4.
        irbem_lib_path (str | Path, optional): Path to the IRBEM shared library.

    Returns:
        xr.Dataset: The table with coordinates `lat` and `lon` (degrees) and the variables `B_m_crit` (nT),
        `Alpha_DLC_Eq` (rad), `B_Eq` (nT), `R_Eq_GEO` (RE) and `drift_shell_resolution_used`, NaN where no drift
        loss cone was calculated. The settings are stored as attributes.

    Raises:
        ValueError: If `lon_step` does not divide 360.
    """
    import xarray as xr  # noqa: PLC0415

    if not np.isclose(360.0 % lon_step, 0.0):
        msg = f"lon_step must divide 360 degrees, got {lon_step}"
        raise ValueError(msg)

    epoch = datetime(year, 7, 2, tzinfo=timezone.utc)

    lat = np.arange(lat_range[0], lat_range[1] + lat_step / 2, lat_step)
    lon = np.arange(0.0, 360.0 - lon_step / 2, lon_step)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    n_nodes = lat_grid.size

    logger.info(f"Computing IGRF drift loss cone lookup table for {epoch:%Y-%m-%d} on {n_nodes} nodes ...")

    gdz = np.column_stack([np.full(n_nodes, stop_alt), lat_grid.ravel(), lon_grid.ravel()])

    irbem_options = IrbemOptions(
        lstar_quantity=LstarQuantity.NONE,
        igrf_update_interval_days=0,
        field_line_resolution=0,
        drift_shell_resolution=drift_shell_resolution,
        internal_field_model=InternalFieldModel.IGRF,
    )

    # transform and evaluate the field in chunks, as IRBEM's vectorized routines are limited to NTIME_MAX
    model = MagFields(
        lib_path=irbem_lib_path, kext=MagneticField.Dip.get_kext(), sysaxes=ep.IRBEM_SYSAXIS_GEO, options=irbem_options
    )
    chunk = model.ntime_max.value
    coords = Coords(lib_path=irbem_lib_path)

    x_geo = np.empty((n_nodes, 3))
    b_radial = np.empty(n_nodes)
    for start in range(0, n_nodes, chunk):
        sl = slice(start, min(start + chunk, n_nodes))
        n_chunk = sl.stop - sl.start
        x_geo[sl] = coords.transform([epoch] * n_chunk, gdz[sl], ep.IRBEM_SYSAXIS_GDZ, ep.IRBEM_SYSAXIS_GEO)
        b_geo = model.get_field_multi(
            [epoch] * n_chunk,
            {"x1": x_geo[sl, 0], "x2": x_geo[sl, 1], "x3": x_geo[sl, 2]},
            {"Kp": np.zeros(n_chunk)},
        ).bgeo
        b_radial[sl] = np.einsum("ij,ij->i", b_geo, x_geo[sl])

    # the field points into the Earth in the northern magnetic hemisphere
    northern = b_radial < 0

    b_eq = np.full(n_nodes, np.nan)
    r_eq = np.full(n_nodes, np.nan)
    x_start = np.full((n_nodes, 3), np.nan)
    alpha_dlc_eq = np.full(n_nodes, np.nan)
    resolution_used = np.zeros(n_nodes, dtype=np.int32)

    def equatorial_dlc(nodes: NDArray[np.intp], resolution: int) -> NDArray[np.float64]:
        dlc = get_drift_loss_cone(
            ep.Variable(data=x_start[nodes], original_unit=ep.units.RE),
            ep.Variable(data=np.full(nodes.size, epoch.timestamp()), original_unit=ep.units.posixtime),
            _table_irbem_input(
                irbem_options._replace(drift_shell_resolution=resolution), num_cores, irbem_lib_path, nodes.size
            ),
            stop_alt=stop_alt,
            dlc_tol=dlc_tol,
        )
        return dlc[create_var_name("Alpha_DLC_Eq", MagneticField.Dip)].get_data(u.rad).astype(np.float64)

    idx = np.flatnonzero(northern)
    if idx.size > 0:
        traces = _run_irbem_parallel(
            _trace_field_line_parallel,
            _table_irbem_input(irbem_options, num_cores, irbem_lib_path, idx.size),
            [epoch] * idx.size,
            sysaxes=ep.IRBEM_SYSAXIS_GEO,
            desc="Tracing field lines",
            x_geo=x_geo[idx],
        )
        b_eq[idx] = [trace[0] for trace in traces]
        r_eq[idx] = [trace[1] for trace in traces]
        x_start[idx] = np.array([trace[2] for trace in traces]).reshape(-1, 3)

    eligible = np.isfinite(b_eq) & (r_eq <= max_eq_radius) & np.isfinite(x_start).all(axis=1)
    idx_eligible = np.flatnonzero(eligible)
    if idx_eligible.size > 0:
        alpha_dlc_eq[idx_eligible] = equatorial_dlc(idx_eligible, drift_shell_resolution)
        resolution_used[idx_eligible] = drift_shell_resolution

    # IRBEM occasionally fails for single field lines at one drift shell resolution, e.g. reporting a spurious
    # ALL_LOST. Such nodes stand out of the smooth B_m_crit of their neighbours, so they are recomputed with a
    # coarser drift shell resolution, and discarded if they still stand out.
    shape = lat_grid.shape
    for resolution in range(drift_shell_resolution - 1, 0, -1):
        suspicious = np.flatnonzero(
            _suspicious_nodes((b_eq / np.sin(alpha_dlc_eq) ** 2).reshape(shape), eligible.reshape(shape))
        )
        if suspicious.size == 0:
            break
        logger.warning(
            f"Recomputing {suspicious.size} nodes that disagree with their neighbours with drift shell "
            f"resolution {resolution} ..."
        )
        alpha_dlc_eq[suspicious] = equatorial_dlc(suspicious, resolution)
        resolution_used[suspicious] = resolution

    rejected = _suspicious_nodes((b_eq / np.sin(alpha_dlc_eq) ** 2).reshape(shape), eligible.reshape(shape))
    rejected &= np.isfinite(alpha_dlc_eq)
    if rejected.any():
        logger.warning(f"Discarding {rejected.sum()} nodes that still disagree with their neighbours.")
    alpha_dlc_eq[rejected] = np.nan
    resolution_used[rejected] = 0

    b_m_crit = b_eq / np.sin(alpha_dlc_eq) ** 2

    return xr.Dataset(
        data_vars={
            "B_m_crit": (("lat", "lon"), b_m_crit.reshape(shape), {"units": "nT"}),
            "Alpha_DLC_Eq": (("lat", "lon"), alpha_dlc_eq.reshape(shape), {"units": "rad"}),
            "B_Eq": (("lat", "lon"), b_eq.reshape(shape), {"units": "nT"}),
            "R_Eq_GEO": (
                ("lat", "lon"),
                r_eq.reshape(shape),
                {"units": "RE", "long_name": "largest geocentric distance of the field line"},
            ),
            "drift_shell_resolution_used": (
                ("lat", "lon"),
                resolution_used.reshape(shape),
                {"long_name": "IRBEM drift shell resolution of the node, 0 where no drift loss cone was calculated"},
            ),
        },
        coords={
            "lat": ("lat", lat, {"units": "deg", "long_name": "geodetic latitude of the northern foot point"}),
            "lon": ("lon", lon, {"units": "deg", "long_name": "geodetic longitude of the northern foot point"}),
        },
        attrs={
            "description": "Critical mirror field of the drift loss cone of the IGRF field (no external field), "
            "on a grid of northern magnetic hemisphere foot points at stop_alt_km.",
            "year": year,
            "epoch": epoch.isoformat(),
            "internal_field_model": "IGRF",
            "lat_step_deg": lat_step,
            "lon_step_deg": lon_step,
            "max_eq_radius_RE": max_eq_radius,
            "stop_alt_km": stop_alt,
            "dlc_tol_deg": dlc_tol,
            "drift_shell_resolution": drift_shell_resolution,
            "irbem_options": str(irbem_options),
            "el_paso_version": ep.__version__,
            "created": datetime.now(timezone.utc).isoformat(),
        },
    )


def _interpolate_b_m_crit(table: xr.Dataset, lat: NDArray[np.float64], lon: NDArray[np.float64]) -> NDArray[np.float64]:
    """Interpolates the critical mirror field of the table at foot points, linearly in its logarithm.

    Longitude wraps around. Positions outside the table, or next to a node without a drift loss cone, are NaN.
    """
    from scipy.interpolate import RegularGridInterpolator  # noqa: PLC0415

    lat_nodes = table["lat"].to_numpy()
    lon_nodes = table["lon"].to_numpy()
    log_b = np.log(table["B_m_crit"].to_numpy())

    # close the longitude circle
    lon_ext = np.append(lon_nodes, lon_nodes[0] + 360.0)
    log_b_ext = np.concatenate([log_b, log_b[:, :1]], axis=1)

    interpolator = RegularGridInterpolator((lat_nodes, lon_ext), log_b_ext, bounds_error=False, fill_value=np.nan)

    b_m_crit = np.full(len(lat), np.nan)
    finite = np.isfinite(lat) & np.isfinite(lon)
    b_m_crit[finite] = np.exp(interpolator(np.column_stack([lat[finite], np.mod(lon[finite], 360.0)])))

    return b_m_crit


def _subset_irbem_input(irbem_input: IrbemInput, idx: NDArray[np.intp]) -> IrbemInput:
    return dataclasses.replace(
        irbem_input, maginput={key: np.asarray(value)[idx] for key, value in irbem_input.maginput.items()}
    )


def _table_note(path: Path, table: xr.Dataset) -> str:
    attrs = table.attrs
    return (
        f"precomputed IGRF lookup table {path} (epoch {attrs['epoch'][:10]}, northern foot point grid at "
        f"{attrs['stop_alt_km']:g} km with {attrs['lat_step_deg']:g} deg latitude x {attrs['lon_step_deg']:g} deg "
        f"longitude, dlc_tol {attrs['dlc_tol_deg']:g} deg, drift_shell_resolution {attrs['drift_shell_resolution']})"
    )


@timed_function()
def get_drift_loss_cone_from_lookup_table(
    xgeo_var: ep.Variable,
    time_var: ep.Variable,
    irbem_input: IrbemInput,
    computed_vars: dict[str, ep.Variable] | None = None,
) -> dict[str, ep.Variable]:
    """Calculates the drift loss cone from the precomputed IGRF lookup table of each year.

    Each position is traced, in the IGRF field without external field, to the foot point of its field line in the
    northern magnetic hemisphere, where the critical mirror field `B_m_crit` of the table is interpolated, linearly
    in its logarithm. It is converted to the local and equatorial drift loss cone with `B_Calc` and `B_Eq` of the
    requested magnetic field model, `sin^2(Alpha_DLC) = B_Calc / B_m_crit` and
    `sin^2(Alpha_DLC_Eq) = B_Eq / B_m_crit`, bounded from below by the bounce loss cone of that model. Both share
    the ratio `B_Eq / B_Calc` with which local pitch angles are mapped to the equator, so a measurement inside the
    local drift loss cone is also inside the equatorial one.

    The drift shell is always the one of the IGRF field, held at the table epoch (2 July) for the whole year. For an
    external field model, its effect on the drift shell is neglected, which can lead to small deviations at L > 5.

    Years without a table are traced per time step in the IGRF field with `get_drift_loss_cone` instead, after a
    warning, and converted the same way; `scripts/compute_dlc_lookup_table.py` computes the missing tables. The
    processing notes state which method was used.

    Args:
        xgeo_var (ep.Variable): The variable containing satellite position data in GEO coordinates.
        time_var (ep.Variable): The variable containing the timestamps.
        irbem_input (IrbemInput): A data class with all required IRBEM input parameters. Its options apply to
            the foot point tracing and the magnetic fields, not to the tabulated drift loss cone.
        computed_vars (dict[str, ep.Variable] | None, optional): Already computed `B_Calc`, `B_Eq` and `B_fofl`
            variables of the requested model to reuse. Newly computed ones are added to it. Defaults to None.

    Returns:
        dict[str, ep.Variable]: A dictionary containing the `Alpha_DLC` and `Alpha_DLC_Eq` variables, in radians.

    Raises:
        ValueError: If the internal field model is not IGRF.
    """
    import xarray as xr  # noqa: PLC0415

    if not uses_dlc_lookup_table(irbem_input):
        msg = (
            "The drift loss cone lookup table holds the IGRF field, so it requires IGRF as the internal field "
            f"model, got {InternalFieldModel(IrbemOptions(*irbem_input.irbem_options).internal_field_model).name}. "
            "Use get_drift_loss_cone instead."
        )
        raise ValueError(msg)

    logger.info("\tCalculating drift loss cone from lookup table ...")

    if computed_vars is None:
        computed_vars = {}

    mag_field = irbem_input.magnetic_field
    # the drift shell, and with it the table, belongs to the IGRF field without external field
    igrf_input = dataclasses.replace(irbem_input, magnetic_field=MagneticField.Dip)

    timestamps = time_var.get_data(ep.units.posixtime).astype(np.float64)
    x_geo = xgeo_var.get_data(ep.units.RE).astype(np.float64)
    n_points = len(timestamps)

    years = timestamps.astype("datetime64[s]").astype("datetime64[Y]").astype(np.int64) + 1970

    b_m_crit = np.full(n_points, np.nan)
    table_notes: list[str] = []
    traced_years: list[int] = []

    for year in np.unique(years):
        idx = np.flatnonzero(years == year)
        sub_xgeo = ep.Variable(data=x_geo[idx], original_unit=ep.units.RE)
        sub_time = ep.Variable(data=timestamps[idx], original_unit=ep.units.posixtime)
        sub_input = _subset_irbem_input(igrf_input, idx)

        path = get_dlc_table_path(int(year))
        if not path.is_file():
            logger.warning(
                f"No drift loss cone lookup table for {year} at {path}; tracing the drift loss cone at every time "
                f"step instead, which is slow. Compute the table with '{_SCRIPT} --years {year}'."
            )
            # B_m_crit from the traced equatorial drift loss cone, which is not capped where all locally
            # observable pitch angles are lost
            traced = get_drift_loss_cone(sub_xgeo, sub_time, sub_input)
            alpha_eq = traced[create_var_name("Alpha_DLC_Eq", MagneticField.Dip)].get_data(u.rad)
            b_min = get_magequator(sub_xgeo, sub_time, sub_input)[create_var_name("B_Eq", MagneticField.Dip)]
            b_m_crit[idx] = b_min.get_data(u.nT) / np.sin(alpha_eq) ** 2
            traced_years.append(int(year))
            continue

        table = xr.load_dataset(path)
        foot = _get_northern_foot_point(sub_xgeo, sub_time, sub_input, stop_alt=float(table.attrs["stop_alt_km"]))
        b_m_crit[idx] = _interpolate_b_m_crit(table, foot[:, 1], foot[:, 2])
        table_notes.append(_table_note(path, table))

    names = {name: create_var_name(name, mag_field) for name in ("B_Calc", "B_Eq", "B_fofl")}
    if names["B_Calc"] not in computed_vars:
        computed_vars |= get_local_B_field(xgeo_var, time_var, irbem_input)
    if names["B_Eq"] not in computed_vars:
        computed_vars |= get_magequator(xgeo_var, time_var, irbem_input)
    if names["B_fofl"] not in computed_vars:
        computed_vars |= get_footpoint_atmosphere(xgeo_var, time_var, irbem_input)

    b_local = computed_vars[names["B_Calc"]].get_data(u.nT).astype(np.float64)
    b_eq = computed_vars[names["B_Eq"]].get_data(u.nT).astype(np.float64)
    # the bounce loss cone is set by the weaker of the two foot points; fmin skips one that was not found
    b_fofl = np.fmin.reduce(computed_vars[names["B_fofl"]].get_data(u.nT).astype(np.float64), axis=1)

    def pitch_angle(b_ratio: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asin(np.sqrt(np.clip(b_ratio, 0, 1)))

    # The drift loss cone is never narrower than the bounce loss cone; interpolation, or a different field model
    # for the bounce loss cone, could undercut it where the local field line is the weakest on its drift shell.
    dlc = pitch_angle(b_local / b_m_crit)
    dlc_eq = pitch_angle(b_eq / b_m_crit)
    alpha_dlc = np.where(np.isfinite(dlc), np.fmax(dlc, pitch_angle(b_local / b_fofl)), np.nan)
    alpha_dlc_eq = np.where(np.isfinite(dlc_eq), np.fmax(dlc_eq, pitch_angle(b_eq / b_fofl)), np.nan)

    conversion = (
        f"converted to local and equatorial pitch angles with B_Calc and B_Eq of model {mag_field} "
        f"(sin^2(Alpha_DLC) = B_Calc / B_m_crit, sin^2(Alpha_DLC_Eq) = B_Eq / B_m_crit), bounded from below by the "
        "bounce loss cone of that model."
    )
    if mag_field != MagneticField.Dip:
        conversion += (
            f" The drift shell is the one of the IGRF field without external field: the effect of the external field "
            f"of {mag_field} on the drift shell is neglected, which can lead to small deviations at L > 5."
        )

    notes: list[str] = []
    if table_notes:
        notes.append(
            "Drift loss cone NOT traced per time step but taken from the "
            + "; ".join(table_notes)
            + ": the critical mirror field B_m_crit of the drift shell was interpolated (linear in log) at the "
            "northern foot point of each field line, traced in the IGRF field without external field, and "
            + conversion
            + " IGRF is held at the table epoch for the whole year; the options "
            f"{irbem_input.irbem_options} only apply to the foot point tracing and the fields."
        )
    if traced_years:
        notes.append(
            f"No lookup table for {', '.join(map(str, traced_years))}: there, the drift loss cone was traced per "
            f"time step in the IGRF field without external field with options {irbem_input.irbem_options}, and "
            "its critical mirror field B_m_crit " + conversion
        )

    alpha_dlc_var = ep.Variable(data=alpha_dlc, original_unit=u.radian)
    alpha_dlc_eq_var = ep.Variable(data=alpha_dlc_eq, original_unit=u.radian)
    for var in (alpha_dlc_var, alpha_dlc_eq_var):
        for note in notes:
            var.metadata.add_processing_note(note)

    return {
        create_var_name("Alpha_DLC", mag_field): alpha_dlc_var,
        create_var_name("Alpha_DLC_Eq", mag_field): alpha_dlc_eq_var,
    }
