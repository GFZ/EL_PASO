<!--
SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
SPDX-FileContributor: Bernhard Haas

SPDX-License-Identifier: Apache-2.0
-->

# Compute magnetic field variables

This function serves as a wrapper to calculate a suite of magnetic field and related invariants (like L-star, MLT, B_local, B_eq, invariant Mu,
and invariant K) based on provided time and geocentric coordinates. It leverages the IRBEM library for the underlying computations.

*variables_to_compute* must be a list of tuples of a variable name and a magnetic field model. The returned dictionary is keyed by
the variable name followed by the magnetic field identifier string.

**Supported variable names**:

- B_Calc
- B_fofl
- B_mirr
- B_Eq
- R_Eq
- MLT
- MLT_Eq
- xGEO_Eq
- L_star
- L_m
- I
- Alpha_Eq
- InvMu
- InvK
- Alpha_LC
- Alpha_LC_Eq
- Alpha_DLC
- Alpha_DLC_Eq

**Supported magnetic field strings**:

- Dip
- OP77|OP77Q
- T89
- T96
- T01
- T01s
- TS04|TS05|T04s

Examples of valid entries: *("B_Calc", "T89")*, returned as *B_Calc_T89*, and *("L_star", "TS04")*, returned as *L_star_TS04*.

## Loss cones

Two loss cones are available, each both as a local pitch angle at the satellite and mapped to the magnetic equator (*_Eq*):

- **Alpha_LC**: the bounce loss cone, given separately for the foot points of the local field line in the northern and
  southern magnetic hemisphere, with shape (time, 2) along the *north_south* dimension (index 0 north, 1 south). Particles
  below a hemisphere's value mirror below 100 km in that hemisphere. Each value is set by that hemisphere's foot point
  field at 100 km (*B_fofl*, likewise of shape (time, 2)). Particles are lost within a bounce if they are inside either,
  so the overall bounce loss cone is the larger of the two values. Where a foot point field is below the local one,
  every particle seen locally is lost at that end and the hemisphere's *Alpha_LC* is 90 degrees.
- **Alpha_DLC**: the drift loss cone. Particles between the two cones survive the local bounce but mirror below 100 km
  somewhere else on their drift orbit, essentially always over the South Atlantic Anomaly, and are lost within a drift
  period. Above it they are stably trapped. *Alpha_DLC* is never smaller than either value of *Alpha_LC*, and 90 degrees is
  a genuine result:
  over most of the globe at low Earth orbit, every pitch angle seen locally is inside the drift loss cone.

Without a lookup table (see below), the drift loss cone is traced by bisection over full drift shells, which costs about a
dozen drift shell traces, roughly one second, per time step, so it is by far the most expensive quantity here. *Alpha_DLC* and *Alpha_DLC_Eq* come out of the same
calculation, so requesting both costs no more than requesting one. The IRBEM option *drift_shell_resolution* sets how finely
the drift orbit samples longitude, and with it how well the South Atlantic Anomaly is resolved; use 1 to 3. The stopping
altitude and bisection tolerance can be set by calling
`el_paso.processing.magnetic_field_utils.get_drift_loss_cone` directly.

### Drift loss cone lookup table for the IGRF field

Whenever IGRF is the internal field model, the drift loss cone is **not** traced per time step but taken from a
precomputed lookup table of the year, for *Dip* as well as for external field models such as *T89*.

Without an external field, the IGRF field is static apart from its secular variation, so the drift loss cone only depends
on the field line. The table holds the critical mirror field *B_m_crit* of the drift shell (the mirror field of a
marginally trapped particle) on a grid of field lines, labelled by their foot point at 100 km in the northern magnetic
hemisphere. Each position is traced to that foot point in the IGRF field, *B_m_crit* is interpolated there (linearly in its
logarithm), and the pitch angles follow with the fields of the requested model from *sin²(Alpha_DLC) = B_Calc / B_m_crit*
and *sin²(Alpha_DLC_Eq) = B_Eq / B_m_crit*, bounded from below by its bounce loss cone. This costs one field line trace
per time step instead of a dozen drift shell traces. IGRF is held at the table epoch (2 July) for the whole year.

*B_m_crit* is a field strength near 100 km over the South Atlantic Anomaly, set by the internal field, so it carries over
to external field models. Converting it with their *B_Calc* and *B_Eq* uses the same ratio *B_Eq / B_Calc* with which
*Alpha_Eq* is mapped, so a measurement inside the local drift loss cone is also inside the equatorial one. The effect of the
external field on the drift shell itself, which is distorted and split between day and night side at high L, is
neglected. This can lead to small deviations at L > 5. The processing notes of both variables state the method, the
table used and, for external field models, this caveat. For the full drift shell of an external field model, call
`el_paso.processing.magnetic_field_utils.get_drift_loss_cone` directly.

The tables are computed once per year with `scripts/compute_dlc_lookup_table.py`, e.g.
`python scripts/compute_dlc_lookup_table.py --years 2017 2018 --num-cores 64` (about two hours on 24 cores per year),
and stored in the *dlc_lookup_tables* subdirectory of `EL_PASO_MODEL_DATA_PATH` (default `~/.elpaso`). Years without a
table fall back to tracing in the IGRF field per time step, with a warning. Other internal field models are always traced.
The tables are not part of the magnetic field cache key, so pass `overwrite_cache=True` after recomputing one.

::: el_paso.processing.compute_magnetic_field_variables
