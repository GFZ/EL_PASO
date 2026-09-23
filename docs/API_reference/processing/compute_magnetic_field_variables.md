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

- **Alpha_LC**: the bounce loss cone. Particles below it mirror below 100 km on the local field line, in either hemisphere,
  and are lost within a bounce. It is set by the weaker of the two foot point fields at 100 km (*B_fofl*), since particles
  reach lower in that hemisphere. Where that field is below the local one, every particle seen locally is lost at the far
  end and *Alpha_LC* is 90 degrees.
- **Alpha_DLC**: the drift loss cone. Particles between the two cones survive the local bounce but mirror below 100 km
  somewhere else on their drift orbit, essentially always over the South Atlantic Anomaly, and are lost within a drift
  period. Above it they are stably trapped. *Alpha_DLC* is never smaller than *Alpha_LC*, and 90 degrees is a genuine result:
  over most of the globe at low Earth orbit, every pitch angle seen locally is inside the drift loss cone.

The drift loss cone is traced by bisection over full drift shells, which costs about a dozen drift shell traces, roughly one
second, per time step, so it is by far the most expensive quantity here. *Alpha_DLC* and *Alpha_DLC_Eq* come out of the same
calculation, so requesting both costs no more than requesting one. The IRBEM option *drift_shell_resolution* sets how finely
the drift orbit samples longitude, and with it how well the South Atlantic Anomaly is resolved; use 1 to 3. The stopping
altitude and bisection tolerance can be set by calling
`el_paso.processing.magnetic_field_utils.get_drift_loss_cone` directly.

::: el_paso.processing.compute_magnetic_field_variables
