# Synchronous_Coupling

## Purpose
This case exercises the tighter synchronous coupling pathway between fire and atmosphere, where fire uses the updated atmospheric state more directly than in the lagged configuration.

## Physics / Model Features Exercised
- Fire spread / ignition configuration
- Atmospheric forcing and boundary-condition setup

## Atmosphere
The atmosphere is neutral (300 K) and at rest. It starts from `input_sounding` in hydrostatic balance with `erf.use_gravity = true`, so the fire's heat drives a buoyant plume, and the plume's indraft feeds back on the wind the fire sees.

Until 2026-09-10 the deck used `erf.init_type = "uniform"`, which runs with gravity off: `erf.use_gravity` defaults to false, and uniform init allows gravity only when anelastic. The heat warmed the air near the ground but raised no plume. In a 120 s run, the largest vertical velocity was 3e-4 m/s with gravity off and 0.67 m/s with it on. With gravity on, the near-surface indraft reached 0.37 m/s and the fire's effective wind 0.12 m/s. The head rate of spread rose by 1.8%, and the burned area (0.12 ha) did not change. Over those 120 s this deck and Lagged_Coupling agree to within 0.3% in every one of these numbers, so the small grass fire does not yet separate the two coupling orders.

## Expected Results
See the input-file header comments in this directory for the specific validation target. In general, these cases should reproduce the documented analytical trend, qualitative regime change, or engineering diagnostic associated with the scenario.

## Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `geometry.prob_lo` | `0.0 0.0 0.0` | Primary configuration value taken from `inputs_fire_phase7`. |
| `geometry.prob_hi` | `2000.0 2000.0 500.0` | Primary configuration value taken from `inputs_fire_phase7`. |
| `amr.n_cell` | `40 40 50` | Primary configuration value taken from `inputs_fire_phase7`. |
| `geometry.is_periodic` | `1 1 0` | Primary configuration value taken from `inputs_fire_phase7`. |
| `amr.max_level` | `0` | Primary configuration value taken from `inputs_fire_phase7`. |
| `amr.max_grid_size` | `100` | Primary configuration value taken from `inputs_fire_phase7`. |
| `amr.max_grid_size_z` | `50` | Primary configuration value taken from `inputs_fire_phase7`. |
| `zlo.type` | `"surface_layer"` | Primary configuration value taken from `inputs_fire_phase7`. |

## References
- Rothermel 1972, A Mathematical Model for Predicting Fire Spread in Wildland Fuels.
- Andrews 2018, The Rothermel surface fire spread model and associated developments.
