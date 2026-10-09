# Spatial_Fuel

## Purpose
These cases validate reading and use of heterogeneous fuel maps, including blended fuels and firebreak regions that should locally suppress spread.

## Physics / Model Features Exercised
- Fire spread / ignition configuration
- Atmospheric forcing and boundary-condition setup

## Firebreak pair
`inputs_fire_phase10_firebreak` places a 100 m wide north-south firebreak at
x = 800-900 m across a westerly grass fire ignited at x = 300 m. With
`erf.fire.firebreak.use_mask = true` (the default since 2026-10) the firebreak
cells are held in the non-burnable mask, so the fire stops at x = 800 m.
`inputs_fire_phase10_firebreak_sentinel` sets `use_mask = false`, the form
before 2026-10: the firebreak is only stamped into phi as the sentinel, which
the FARSITE front-cell update rebuilds away on its first subcycle, so the fire
crosses it (the start-up prints a warning for this setting). Run both and
compare the burned extent in the last `plt_fire_*` plotfile or the burned
area in the two `fire_stats_phase10_firebreak*.csv` files.

## Expected Results
See the input-file header comments in this directory for the specific validation target. In general, these cases should reproduce the documented analytical trend, qualitative regime change, or engineering diagnostic associated with the scenario.

## Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `geometry.prob_lo` | `0.0 0.0 0.0` | Primary configuration value taken from `inputs_fire_phase10_blending`. |
| `geometry.prob_hi` | `2000.0 2000.0 500.0` | Primary configuration value taken from `inputs_fire_phase10_blending`. |
| `amr.n_cell` | `40 40 50` | Primary configuration value taken from `inputs_fire_phase10_blending`. |
| `geometry.is_periodic` | `1 1 0` | Primary configuration value taken from `inputs_fire_phase10_blending`. |
| `amr.max_level` | `0` | Primary configuration value taken from `inputs_fire_phase10_blending`. |
| `amr.max_grid_size` | `100` | Primary configuration value taken from `inputs_fire_phase10_blending`. |
| `amr.max_grid_size_z` | `50` | Primary configuration value taken from `inputs_fire_phase10_blending`. |
| `zlo.type` | `"surface_layer"` | Primary configuration value taken from `inputs_fire_phase10_blending`. |

## References
- Rothermel 1972, A Mathematical Model for Predicting Fire Spread in Wildland Fuels.
- Andrews 2018, The Rothermel surface fire spread model and associated developments.
