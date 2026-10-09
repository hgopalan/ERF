# DustHealthIntegration

## Purpose
This hazard case integrates several dust-health diagnostics, such as silica, visibility, or occupational exposure, within a single workflow.

## Physics / Model Features Exercised
- Dust emission / transport / deposition controls
- Exposure or air-quality diagnostics as configured
- Coupled hazard-module configuration
- Cross-module diagnostics or interaction controls

## Expected Results
See the input-file header comments in this directory for the specific validation target. In general, these cases should reproduce the documented analytical trend, qualitative regime change, or engineering diagnostic associated with the scenario.

## Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `erf.prob_name` | `"ABL"` | Primary configuration value taken from `inputs`. |
| `max_step` | `20` | Primary configuration value taken from `inputs`. |
| `amr.max_level` | `0` | Primary configuration value taken from `inputs`. |
| `amrex.fpe_trap_invalid` | `0` | Primary configuration value taken from `inputs`. |
| `fabarray.mfiter_tile_size` | `1024 1024 1024` | Primary configuration value taken from `inputs`. |
| `geometry.prob_extent` | `3000 3000 1024` | Primary configuration value taken from `inputs`. |
| `amr.n_cell` | `8 8 64` | Primary configuration value taken from `inputs`. |
| `geometry.is_periodic` | `1 1 0` | Primary configuration value taken from `inputs`. |

## Outputs
The diagnostic CSVs (`stel_diag.csv`, `silica_diag.csv`, `visibility_diag.csv`, `dust_naaqs.csv`, `msha_exposure.csv`) are written by the run into the working directory; none is committed, because a committed copy from an older build cannot be reproduced and misleads (the copies removed in October 2026 carried concentrations at step 1 that the code never produces). Every row carries the end-of-step time, the same stamp as `dust_diag.dat`.

## References
- Bagnold 1941, The Physics of Blown Sand and Desert Dunes.
- Marticorena and Bergametti 1995, Modeling the atmospheric dust cycle.
