# DustSilica

## Purpose
The respirable crystalline silica diagnostic: `silica_diag.csv` reports the PM10 concentration times `erf.dust.silica_fraction_rcs` in mg/m3 against `erf.dust.silica_osha_pel_mg_m3` (0.05 mg/m3, 29 CFR 1910.1053). What is compared is an instantaneous value built on PM10, an upper bound on the 8-hour respirable (PM4) measurement the PEL is written for; the MSHA dose and TWA carry the time average. The run is a start-up check of the wiring and the units (10 s), not a health assessment.

## Setup
| Item | Value |
|------|-------|
| Domain, grid | 3000 x 3000 x 1024 m, 8 x 8 x 64 cells (375 x 375 x 16 m), periodic in x and y, flat |
| Run length | `max_step = 20` at `erf.fixed_dt = 0.5` s (10 s): a start-up regression run, seconds on one rank |
| Dust | three bins {7, 2.5, 50} um carried as one scalar (`erf.dust.grid_ratio = 1`, the dust grid is the atmosphere's columns); Shao-Lu threshold at 75 um; deposition with `E_0 = 3e-3` |
| Sources | the surface-layer u* against the threshold (the 15 m/s geostrophic wind gives u* about 1.1 m/s, 5x the threshold: this source dominates, about 8e2 kg over the run), the blast in `blast_schedule.csv` (about 1e2 kg), and the haul road in `road_schedule.csv` (AP-42 PM-10 mass rate over the covered cells, on bin 0; 2e-2 kg, 2e-5 of the emission) |
| Diagnostic | `erf.dust.silica_enable = true`, `silica_fraction_rcs` and `silica_osha_pel_mg_m3` as in `inputs`; the MSHA exposure diagnostics run alongside |

## What to look at
- `silica_diag.csv`: the maximum silica concentration scales linearly with `silica_fraction_rcs` (run twice with the fraction doubled: the column doubles); the count above the PEL follows the PM10 field of the dust plotfile (`dust_pm10_ug_m3` times the fraction times 1e-3).
- `msha_exposure.csv`: the domain maxima of the TWA [mg/m3] and the dose [mg/m3 h] and the count of cells above the PEL (the deck names no receptor), rising monotonically over the 10 s.

## Outputs
The diagnostic CSVs (`stel_diag.csv`, `silica_diag.csv`, `visibility_diag.csv`, `dust_naaqs.csv`, `msha_exposure.csv`) are written by the run into the working directory; none is committed, because a committed copy from an older build cannot be reproduced and misleads (the copies removed in October 2026 carried concentrations at step 1 that the code never produces). Every row carries the end-of-step time, the same stamp as `dust_diag.dat`.

## References
- Shao and Lu 2000, A simple expression for wind erosion threshold friction velocity, J. Geophys. Res. 105, 22437.
- Marticorena and Bergametti 1995, Modeling the atmospheric dust cycle, J. Geophys. Res. 100, 16415.
- EPA AP-42 section 13.2.2 (unpaved roads), the haul-road emission factor.
- 29 CFR 1910.1053 (OSHA respirable crystalline silica), NIOSH REL 0.05 mg/m3.
