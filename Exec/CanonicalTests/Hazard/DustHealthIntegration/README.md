# DustHealthIntegration

## Purpose
Every dust health diagnostic in one run: NAAQS PM2.5/PM10 with the 24-hour window means, the MSHA receptor dose and shift TWA, the 15-minute STEL, the silica fraction and the Koschmieder visibility, all from the same surface dust field, so their CSVs can be read side by side. The run is a start-up check (10 s) that the diagnostics agree on the concentration and the time stamp; the 24-hour flags cannot fire in it (they compare once 24 hours are covered).

## Setup
| Item | Value |
|------|-------|
| Domain, grid | 3000 x 3000 x 1024 m, 8 x 8 x 64 cells (375 x 375 x 16 m), periodic in x and y, flat |
| Run length | `max_step = 20` at `erf.fixed_dt = 0.5` s (10 s): a start-up regression run, seconds on one rank |
| Dust | three bins {7, 2.5, 50} um carried as one scalar (`erf.dust.grid_ratio = 1`, the dust grid is the atmosphere's columns); Shao-Lu threshold at 75 um; deposition with `E_0 = 3e-3` |
| Sources | the surface-layer u* against the threshold (the 15 m/s geostrophic wind gives u* about 1.1 m/s, 5x the threshold: this source dominates, about 8e2 kg over the run), the blast in `blast_schedule.csv` (about 1e2 kg), and the haul road in `road_schedule.csv` (AP-42 PM-10 mass rate over the covered cells, on bin 0; 2e-2 kg, 2e-5 of the emission) |
| Diagnostics | `dust_naaqs.csv`, `msha_exposure.csv` + `msha_shift_summary.csv` (5 s shifts), `stel_diag.csv`, `silica_diag.csv`, `visibility_diag.csv`; `visibility_k_ext = 600` m2/kg |

## What to look at
- The PM10 maximum in `dust_naaqs.csv`, the STEL instantaneous maximum, the silica concentration divided by `silica_fraction_rcs` and the visibility range all follow one concentration field at one time stamp (the end of the step).
- The PM shares: the wind and blast sources split their mass equally over the three bins and the road (bin 0 only) is 2e-5 of the emission, so the emitted shares stay near a third each and PM2.5 is about a third of the scalar, PM10 two thirds (the shares follow the emitted mass since October 2026; a road-dominated deck would give all PM10 and no PM2.5).

## Outputs
The diagnostic CSVs (`stel_diag.csv`, `silica_diag.csv`, `visibility_diag.csv`, `dust_naaqs.csv`, `msha_exposure.csv`) are written by the run into the working directory; none is committed, because a committed copy from an older build cannot be reproduced and misleads (the copies removed in October 2026 carried concentrations at step 1 that the code never produces). Every row carries the end-of-step time, the same stamp as `dust_diag.dat`.

## References
- Shao and Lu 2000, A simple expression for wind erosion threshold friction velocity, J. Geophys. Res. 105, 22437.
- Marticorena and Bergametti 1995, Modeling the atmospheric dust cycle, J. Geophys. Res. 100, 16415.
- EPA AP-42 section 13.2.2 (unpaved roads), the haul-road emission factor.
- 40 CFR 50 (NAAQS), 30 CFR 56.5001 (MSHA), 29 CFR 1910.1053 (silica).
