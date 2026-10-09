# DustSTEL

## Purpose
A short-term exposure limit (STEL) on the surface dust: the 15-minute mean of the PM10 concentration (`erf.dust.stel_averaging_s`, as a block mean of 15 slots since October 2026) against `erf.dust.stel_threshold_mg_m3`, counted per cell and per step in `stel_diag.csv`. The run is a start-up check that the diagnostic is wired and stamped right, not an exposure study: 20 steps of 0.5 s cover 10 s of the 15-minute window, so the reported mean is the mean over the time covered so far.

## Setup
| Item | Value |
|------|-------|
| Domain, grid | 3000 x 3000 x 1024 m, 8 x 8 x 64 cells (375 x 375 x 16 m), periodic in x and y, flat |
| Run length | `max_step = 20` at `erf.fixed_dt = 0.5` s (10 s): a start-up regression run, seconds on one rank |
| Dust | three bins {7, 2.5, 50} um carried as one scalar (`erf.dust.grid_ratio = 1`, the dust grid is the atmosphere's columns); Shao-Lu threshold at 75 um; deposition with `E_0 = 3e-3` |
| Sources | the surface-layer u* against the threshold, plus the haul road in `road_schedule.csv` (AP-42 PM-10 mass rate over the covered cells, stamped on bin 0) |
| Diagnostic | `erf.dust.stel_enable = true`, `stel_averaging_s` and `stel_threshold_mg_m3` as in `inputs`; the MSHA shift summary runs alongside with a 5 s shift so that a shift boundary falls inside the run |

## What to look at
- `stel_diag.csv`: one row per step with the end-of-step time, the maximum 15-minute mean and the count of cells above the threshold; the mean rises with the road emission and is below the instantaneous maximum.
- `msha_shift_summary.csv`: two shift rows at 5 s and 10 s (the boundary is found on the step that ends on it; it was found one step late until October 2026).
- `dust_diag.dat`: `emission_total_kg_s` is the AP-42 road rate plus the wind source; the deposited mass column is in kg.

## Outputs
The diagnostic CSVs (`stel_diag.csv`, `silica_diag.csv`, `visibility_diag.csv`, `dust_naaqs.csv`, `msha_exposure.csv`) are written by the run into the working directory; none is committed, because a committed copy from an older build cannot be reproduced and misleads (the copies removed in October 2026 carried concentrations at step 1 that the code never produces). Every row carries the end-of-step time, the same stamp as `dust_diag.dat`.

## References
- Shao and Lu 2000, A simple expression for wind erosion threshold friction velocity, J. Geophys. Res. 105, 22437.
- Marticorena and Bergametti 1995, Modeling the atmospheric dust cycle, J. Geophys. Res. 100, 16415.
- EPA AP-42 section 13.2.2 (unpaved roads), the haul-road emission factor.
- 29 CFR 1910.1000 (OSHA PELs; the STEL form), 30 CFR 56.5001 (MSHA).
