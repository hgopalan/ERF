# FireBoundaryGuard

The fire at the edge of the fire grid. Nothing outside the grid burns, so a
front that reaches a non-periodic edge is clipped there, as in WRF-SFIRE.
Three reports tell the deck's author:

- `erf.fire.edge_reach_check` (default true): once, on the first step with a
  burning cell, the distance from the burning region to each non-periodic edge
  against the largest rate of spread times the time left in the run, and a
  warning for every edge the fire can reach.
- `erf.fire.boundary_guard_cells` (default 2): the width in fire cells of the
  band along every non-periodic edge, WRF-SFIRE's `fire_boundary_guard`.
- `erf.fire.boundary_guard_action`: `warn` (default) prints once at the first
  contact and records the time as `edge_contact_time_s` in the statistics CSV
  (with the band count on every step as `edge_band_cells`), `abort` stops the
  run as WRF-SFIRE does by default, `none` checks nothing.

A still atmosphere on 400 x 400 m with slip walls, 2 m fire cells (grid ratio
10) and a prescribed rate of spread of 1 m/s, so the front moves one cell a
second and the contact time follows from the ignition position. Three decks:

| deck | ignition centre | action | what happens |
|---|---|---|---|
| `inputs_warn` | x = 380 m, 20 m from the east wall | warn | the disc's east edge is 10 m from the wall; the centre of the first band cell is 7 m ahead, so the band is entered at 7 s |
| `inputs_far` | x = 200 m, 190 m from every wall | warn | no edge within the 19.5 m reach of a 20 s run; no warning, no contact |
| `inputs_abort` | x = 395 m, the disc overlaps the wall | abort | burning cells inside the band on the first step; the run stops there |

```
MPIRUN="mpirun -np 2" ./run_guard.sh /path/to/erf_exec
```

`check_guard.py` reads `fire_stats_<variant>.csv` and the run log and checks,
for `warn`, that the band count is zero before the contact and positive on
the last row, that the contact time is 7 s within a cell, -1 before it,
constant after it and equal to the first band row's time, and that the log
warns once at ignition for the east edge only and once at the contact; for
`far`, that no row and no line reports anything. The script fails on a binary
without the two CSV columns, on `boundary_guard_action = none` (no contact),
on `edge_reach_check = false` (no estimate) and on the `far` deck with the disc
moved to the wall. The CSV is identical on one and two ranks.

The CTests `FireBoundaryGuard_warn`, `FireBoundaryGuard_far` (both 40 steps,
one rank, `check_guard.py`) and `FireBoundaryGuard_abort` (one step, expects
"reached the boundary guard band") run these; the unit test
`ERF_GTestFireBoundaryGuard` checks the kernels on a four-box grid.

## Expected results

`inputs_warn`, log at ignition (t = 0):

```
[FIRE] Reach at ignition (t=0 s): largest rate of spread 1 m/s, 19.5 s left in the run, reach 19.5 m
[FIRE]   west (x lo) edge: 371 m from the burning region, reached in about 371 s at that rate
[FIRE]   east (x hi) edge: 11 m from the burning region, reached in about 11 s at that rate
[FIRE] WARNING: the fire can reach the east (x hi) edge of the fire grid before the run ends (11 m at 1 m/s, about 11 s). ...
[FIRE]   south (y lo) edge: 191 m from the burning region, reached in about 191 s at that rate
[FIRE]   north (y hi) edge: 191 m from the burning region, reached in about 191 s at that rate
```

and at the contact:

```
[FIRE] WARNING: the fire entered the boundary guard band (2 fire cells) at the east (x hi) edge of the fire grid at t = 7 s with 4 burning cells in it. ...
```

`fire_stats_warn.csv`, the last two columns:

| step | time_s | edge_band_cells | edge_contact_time_s |
|---|---|---|---|
| 14 | 6.5 | 0 | -1 |
| 15 | 7 | 4 | 7 |
| 40 | 19.5 | 48 | 7 |

The 11 m at ignition is to the nearest burning cell centre (389 m), the 7 s
contact is when the front crosses the centre of the first band cell (397 m).
`inputs_far` prints the estimate with 191 m to every edge and no warning, and
its columns stay 0 and -1. `inputs_abort` stops on step 1 with

```
amrex::Abort::0::[FIRE] The fire reached the boundary guard band (2 fire cells) at the east (x hi) edge of the fire grid at t = 0 s. ...
```
