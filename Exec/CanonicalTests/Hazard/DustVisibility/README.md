# DustVisibility

## Purpose
The Koschmieder visibility from the airborne dust: `visibility_diag.csv` reports the minimum visual range `3.912 / (k_ext C)` with `erf.dust.visibility_k_ext = 600` m2/kg (inside the 300-1000 m2/kg literature range for mineral dust; 4000 until October 2026, 4-13x above it) and the counts of cells below the warning (1000 m) and road-closure (300 m) ranges. The run is a start-up check of the wiring and the units (10 s).

## Setup
| Item | Value |
|------|-------|
| Domain, grid | 3000 x 3000 x 1024 m, 8 x 8 x 64 cells (375 x 375 x 16 m), periodic in x and y, flat |
| Run length | `max_step = 20` at `erf.fixed_dt = 0.5` s (10 s): a start-up regression run, seconds on one rank |
| Dust | three bins {7, 2.5, 50} um carried as one scalar (`erf.dust.grid_ratio = 1`, the dust grid is the atmosphere's columns); Shao-Lu threshold at 75 um; deposition with `E_0 = 3e-3` |
| Sources | the surface-layer u* against the threshold, plus the haul road in `road_schedule.csv` (AP-42 PM-10 mass rate over the covered cells, stamped on bin 0) |
| Diagnostic | `erf.dust.visibility_enable = true`, `visibility_k_ext = 600`, `visibility_warning_m = 1000`, `visibility_road_closure_m = 300` |

## What to look at
- `visibility_diag.csv`: the minimum visual range falls as the surface concentration rises; halving `visibility_k_ext` doubles it (the relation is exact, so this is the check to run).
- `dust_diag.dat` and the dust plotfile: `conc_sfc_max_kg_m3` and `dust_conc_sfc` are the concentrations the range is built on.

## Outputs
The diagnostic CSVs (`stel_diag.csv`, `silica_diag.csv`, `visibility_diag.csv`, `dust_naaqs.csv`, `msha_exposure.csv`) are written by the run into the working directory; none is committed, because a committed copy from an older build cannot be reproduced and misleads (the copies removed in October 2026 carried concentrations at step 1 that the code never produces). Every row carries the end-of-step time, the same stamp as `dust_diag.dat`.

## References
- Shao and Lu 2000, A simple expression for wind erosion threshold friction velocity, J. Geophys. Res. 105, 22437.
- Marticorena and Bergametti 1995, Modeling the atmospheric dust cycle, J. Geophys. Res. 100, 16415.
- EPA AP-42 section 13.2.2 (unpaved roads), the haul-road emission factor.
- Koschmieder 1924 (the visual-range relation); Hand and Malm 2007, Review of aerosol mass scattering efficiencies, J. Geophys. Res. 112, D16203 (the extinction range).
