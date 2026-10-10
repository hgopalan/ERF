# Crown_Fire

## Purpose
These cases contrast canopy-enabled and surface-only fire behavior to verify crown-initiation criteria, canopy fuel participation, and diagnostic outputs.

## Physics / Model Features Exercised
- Fire spread / ignition configuration
- Atmospheric forcing and boundary-condition setup

## Expected Results
`check_crown_fire.py` on the two decks (`inputs_fire_phase9_surface_only`,
`inputs_fire_phase9_crown`), four ranks, measured 2026-10-09 on the validated
code: both reach `plt_fire_04400` at t = 893.8 s with every field finite,
33 burned cells, a rate of spread within [0.00620, 0.00638] m/s, arrival
times in [0, 693] s with the sentinel on the unburned cells, the fuel load
at least 0.445 of its start and never rising, and a burned area that grows
from 0.32 to 0.33 ha without a decrease over the 4431 rows of
`fire_stats_phase9_*.csv` (7 of 7 checks). The two decks give the same
numbers: at this surface rate the crown-initiation criterion is never met,
so the canopy of the crown deck never takes part; the case checks that the
crown module leaves a surface fire untouched, not a transition.

## Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `geometry.prob_lo` | `0.0 0.0 0.0` | Primary configuration value taken from `inputs_fire_phase9_crown`. |
| `geometry.prob_hi` | `2000.0 2000.0 500.0` | Primary configuration value taken from `inputs_fire_phase9_crown`. |
| `amr.n_cell` | `40 40 50` | Primary configuration value taken from `inputs_fire_phase9_crown`. |
| `geometry.is_periodic` | `1 1 0` | Primary configuration value taken from `inputs_fire_phase9_crown`. |
| `amr.max_level` | `0` | Primary configuration value taken from `inputs_fire_phase9_crown`. |
| `amr.max_grid_size` | `100` | Primary configuration value taken from `inputs_fire_phase9_crown`. |
| `amr.max_grid_size_z` | `50` | Primary configuration value taken from `inputs_fire_phase9_crown`. |
| `zlo.type` | `"surface_layer"` | Primary configuration value taken from `inputs_fire_phase9_crown`. |

## References
- Rothermel 1972, A Mathematical Model for Predicting Fire Spread in Wildland Fuels.
- Andrews 2018, The Rothermel surface fire spread model and associated developments.
