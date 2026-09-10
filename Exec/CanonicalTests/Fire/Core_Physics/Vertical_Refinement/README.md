# Vertical_Refinement

## Purpose
This case documents a fire configuration with enhanced vertical resolution, used to confirm that the model setup and boundary conditions remain valid under refined z discretization.

## Physics / Model Features Exercised
- Fire spread / ignition configuration
- Atmospheric forcing and boundary-condition setup

## Atmosphere
The atmosphere is neutral (300 K) and at rest. It starts from `input_sounding` in hydrostatic balance with `erf.use_gravity = true`. There is no ambient wind, so the only wind the fire sees is the indraft of its own plume, resolved here by 4 m cells at the ground.

Until 2026-09-10 the deck used `erf.init_type = "uniform"`, which runs with gravity off: `erf.use_gravity` defaults to false, and uniform init allows gravity only when anelastic. The heat warmed the air but raised no plume, and the fire saw no wind. In a 90 s run, the largest vertical velocity was 2.5e-3 m/s with gravity off and 2.7 m/s with it on. With gravity on, the near-surface indraft reached 0.77 m/s and the fire's effective wind 0.29 m/s; with gravity off, the effective wind was 1e-3 m/s. The head rate of spread rose from 0.0203 to 0.0287 m/s (41%), and the peak heat flux from 5.2 to 7.5 kW/m2. The burned area (0.8 ha, mostly the 50 m ignition disc) did not change yet.

## Expected Results
See the input-file header comments in this directory for the specific validation target. In general, these cases should reproduce the documented analytical trend, qualitative regime change, or engineering diagnostic associated with the scenario.

## Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `geometry.prob_lo` | `0.0 0.0 0.0` | Primary configuration value taken from `inputs_fire_vertical_refinement`. |
| `geometry.prob_hi` | `2000.0 2000.0 1000.0` | Primary configuration value taken from `inputs_fire_vertical_refinement`. |
| `amr.n_cell` | `40 40 64` | Primary configuration value taken from `inputs_fire_vertical_refinement`. |
| `geometry.is_periodic` | `1 1 0` | Primary configuration value taken from `inputs_fire_vertical_refinement`. |
| `amr.max_level` | `0` | Primary configuration value taken from `inputs_fire_vertical_refinement`. |
| `amr.max_grid_size` | `100` | Primary configuration value taken from `inputs_fire_vertical_refinement`. |
| `amr.max_grid_size_z` | `64` | Primary configuration value taken from `inputs_fire_vertical_refinement`. |
| `erf.initial_dz` | `4.0` | Primary configuration value taken from `inputs_fire_vertical_refinement`. |

## References
- Rothermel 1972, A Mathematical Model for Predicting Fire Spread in Wildland Fuels.
- Andrews 2018, The Rothermel surface fire spread model and associated developments.
