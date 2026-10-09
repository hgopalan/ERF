# DustTerrainSlopeEffect

## Purpose
This case examines how a simple slope modifies near-surface winds and dust response compared with the flat-terrain baseline.

## Physics / Model Features Exercised
- Dust emission / transport / deposition controls
- Exposure or air-quality diagnostics as configured
- Coupled hazard-module configuration
- Cross-module diagnostics or interaction controls
- Idealized terrain effects

## Expected Results
The raster slope is uniform (10 degrees rising along +x) and the geostrophic wind blows along +x, so every dust cell is windward and the case is a smoke test of a spatially uniform slope factor, not of a windward/lee contrast: the threshold is 1.11 times the flat value everywhere (Iversen and Rasmussen 1994, sqrt(cos 10 + sin 10 / tan 35)), the FARSITE wind factor is `k_ridge` everywhere when `use_terrain_wind` is on, and `max/min` of `dust_emission_flux` over the domain is 1. A windward/lee contrast needs the slope in the atmosphere as well (`DustGaussianHill`, `DustGaussianPit`).

## Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `erf.prob_name` | `"ABL"` | Primary configuration value taken from `inputs`. |
| `max_step` | `20` | Primary configuration value taken from `inputs`. |
| `amr.max_level` | `0` | Primary configuration value taken from `inputs`. |
| `amrex.fpe_trap_invalid` | `0` | Primary configuration value taken from `inputs`. |
| `geometry.prob_extent` | `3000 3000 1024` | Primary configuration value taken from `inputs`. |
| `amr.n_cell` | `8 8 64` | Primary configuration value taken from `inputs`. |
| `geometry.is_periodic` | `1 1 0` | Primary configuration value taken from `inputs`. |
| `amr.max_grid_size` | `32` | Primary configuration value taken from `inputs`. |

## References
- Bagnold 1941, The Physics of Blown Sand and Desert Dunes.
- Marticorena and Bergametti 1995, Modeling the atmospheric dust cycle.
- Analytical Gaussian terrain idealization used for topographic verification.
