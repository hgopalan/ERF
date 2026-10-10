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
The firebreak pair, measured 2026-10-09 on four ranks:
`inputs_fire_phase10_firebreak` (the break at x = 800 to 900 m in the
non-burnable mask, `erf.fire.firebreak.use_mask = true`, the default since
2026-10; 2000 mask cells) and `inputs_fire_phase10_firebreak_sentinel`
(`use_mask = false`, the break only stamped into phi, which the FARSITE
front-cell update rebuilds away; the start-up warns). Both burn from 0.32 ha
at the ignition disc to 1.47 ha at 900 s with the head at 0.038 m/s and the
burned region spanning x = 265 to 385 m from the ignition at x = 300 m: the
break is 400 m downwind and the run ends long before the fire reaches it, so
the two decks give the same `fire_stats` row by row. That a masked line holds
and a sentinel does not is proven by `FarsiteShape.AFireLineBlocksTheDirectSources`
and the `FireSuppression_*_farsite` CTests; a run of a few hours at this rate
is needed for the pair to differ here.

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
