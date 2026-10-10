# Level_Set_Advection

## Purpose
Exercises the level-set fire-front solver: the Godunov upwind gradient with
first-order, WENO5-Z or hybrid WENO5-Z/first-order one-sided derivatives, a
three-stage SSP-RK3 step subcycled on its own CFL condition, and
signed-distance reinitialization (WRF-Fire's scheme by default).

## Physics / Model Features Exercised
- Level-set advection of the fire front (`propagation_method = "levelset"`)
- Godunov upwind gradient with artificial viscosity on the Laplacian
- `erf.fire.levelset.gradient`: `upwind` (first order everywhere), `weno5z`
  (HJ-WENO5-Z everywhere) and the default `weno5z_front` (WENO within
  `weno_band_cells` = 4 cells of the front, first order elsewhere, the
  WRF-Fire / CFBM hybrid); `run_compare.sh` runs the three decks and
  `inputs_fire_levelset_lowvisc`, the hybrid with the two-value artificial
  viscosity (`eps_visc_front` = 0.1 within two cells of the front, the
  default since 2026-09-05; the other decks pin a single viscosity)
- CFL-based subcycling within one atmospheric step
- Reinitialization of the signed-distance property (`erf.fire.levelset.reinit_scheme`,
  default `"wrf"`)

## Status

| Case | State |
|---|---|
| `inputs_fire_levelset_baseline` | Passing. Advection with signed-distance reinitialization. |
| `inputs_fire_levelset_no_reinit` | Passing. Control: advection only. |

## The level-set path uses a true signed distance

`phi` is in **metres** for this path — negative inside the burned region,
positive outside, `|grad phi| = 1`. The FARSITE path keeps its own normalized
`[-1, 1]` indicator convention, and `initialize_ignition` takes a flag to choose
between them.

This matters. A level-set method's advection, its Godunov Hamiltonian and the
reinitialization are all derived for a signed distance in metres.
Running the solver on a `[-1, 1]`-clamped field flattens everything outside the
band, so the front eventually advances into ground carrying no gradient
information. That produced discontinuous jumps in burned area — 44 to 528 cells
in a single step — and widening the band made it worse rather than better, which
is what ruled out band width as the cause and indicted the normalization itself.

## Expected Results

Burned-cell count at t = 150 / 300 / 450 / 600 s on the 10 m fire grid of the
decks, two ranks, re-measured 2026-10-09 (`run_compare.sh`):

| | 150 s | 300 s | 450 s | 600 s |
|---|---|---|---|---|
| analytic, `r = r0 + R*t` | 34 | 41 | 48 | 56 |
| `inputs_fire_levelset_baseline` | 34 | 40 | 44 | 46 |
| `inputs_fire_levelset_weno5z` | 34 | 40 | 46 | 48 |
| `inputs_fire_levelset_weno5z_front` | 34 | 40 | 46 | 48 |
| `inputs_fire_levelset_lowvisc` | 34 | 40 | 46 | 48 |

The first deck pins `erf.fire.levelset.gradient = upwind`; the next two use
WENO5-Z everywhere and the default hybrid, which reproduces WENO everywhere
to the cell here because the whole burn sits inside the band; the last
lowers the near-front viscosity to 0.1. All four track the analytic count
to 300 s and fall behind it from 450 s, 46 to 48 cells against 56 at 600 s:
the burn is only a few cells across, and the signed-distance nudge of the
reinitialisation rounds its corners inward. The first-order scheme loses
two cells more than WENO; the viscosity option changes nothing this grid
can show (it is for fine fire grids: on the 5 m grid of the WUI wildland
case, `WUI_Subdivision`, 1200 s, the near-front value of 0.1 leaves the
head at 0.250 m/s on every segment and lets the flanks spread a little
more, 150 m wide at x = 450 m against 140 m, 4.96 ha burned against
4.83 ha). The table this README carried until 2026-10 (64 cells for the
first-order scheme and 52 for WENO at 600 s) was measured on a 25 m grid
with the previous reinitialisation and never on these decks.

Both cases track the analytic front. `phi` behaves as a signed distance:
`phi_min` about -38 m near the centre of the burn, `phi_max` about the
far-corner distance, unclamped.

The table above is measured at `OMP_NUM_THREADS=1`.

### Determinism

Repeat runs agree exactly, and both cases are clean under
`amrex.init_snan=1 amrex.fpe_trap_invalid=1`.

Neither was true before the fire `Geometry` was given the atmospheric
periodicity. `create_fire_grid()` hard-coded `{false, false, false}`, so every
`FillBoundary` on the fire grid was a no-op: the fire grid is a single box
spanning the domain, which makes all of its ghost cells domain-boundary ghosts.
The stage fields carry three ghost cells past the box edge and were reading
whatever the allocator supplied, so the burned area varied run to run — 64, 82,
106, 118 and 126 cells at 600 s across five identical runs.

The FARSITE path was unaffected, which matches operational experience with
periodic atmospheres. It rebuilds `phi` from `fire_arrival_time` every step so
nothing accumulates, and its neighbour access is guarded with in-box bounds
(`i+1 <= hi.x`) rather than reaching into ghosts. It exits cleanly under the
signaling-NaN trap that made the level-set path abort.

## Reinitialization

`erf.fire.levelset.reinit_scheme` selects the scheme: `"wrf"` (default),
WRF-Fire's `reinit_ls_rk3` (Wicker-Skamarock RK3, flux-form WENO5 near the
front, `dtau = 0.01*dx`), or `"jiang_peng"`, Jiang and Peng's HJ-WENO5 with
SSP-RK3. Neither has a subcell correction; both end with
`phi = min(phi_out, phi_in)`, so the burned area never shrinks and `phi` is
restored toward a unit gradient only where that lowers it. The decks here keep
`reinit_iters = 100` (WRF-Fire uses one step per call, at the same `dtau`),
which costs three right-hand-side evaluations per step, against one for the
previous forward-Euler iteration.

## Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `erf.fire.propagation_method` | `"levelset"` | Selects the PDE solver over the FARSITE Lagrangian default. |
| `erf.fire.levelset.cfl` | `0.4` / `0.25` | Subcycle CFL number. Must be `> 0`; a non-positive value is rejected at startup. |
| `erf.fire.levelset.eps_visc` | `0.4` / `0.2` | Artificial viscosity coefficient on the Laplacian term. |
| `erf.fire.levelset.reinit_every` | `1000000` / `1` | Reinitialize every N subcycles. Must be `>= 1`; it is a modulus divisor. Set high in the baseline to disable it. |
| `erf.fire.levelset.reinit_iters` | `100` | Outer RK3 pseudo-time steps per reinitialization (default `1`). |
| `erf.fire.levelset.reinit_dtau` | `-1.0` / `2.0` | Pseudo-time step [m]; `<= 0` selects `0.01*dx`. |

## References
- Osher & Sethian 1988, Fronts propagating with curvature-dependent speed.
- Sussman, Smereka & Osher 1994, A level set approach for computing solutions to incompressible two-phase flow.
- Jiang & Peng 2000, Weighted ENO schemes for Hamilton-Jacobi equations, SIAM J. Sci. Comput. 21, 2126-2143.
- Wicker & Skamarock 2002, Time-splitting methods for elastic models using forward time schemes, Mon. Wea. Rev. 130, 2088-2097.
