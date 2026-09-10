# Atmospheric_Stability

## Purpose
These cases isolate the role of background atmospheric stability in modulating fire behavior, plume response, and near-surface winds.

## Physics / Model Features Exercised
- Fire spread / ignition configuration
- Atmospheric forcing and boundary-condition setup

## The two decks
The decks differ only in the potential-temperature profile. In `input_sounding_stable`, theta rises 0.01 K/m from 300 K; in `input_sounding_unstable`, it falls 0.01 K/m. Both start at rest in hydrostatic balance with `erf.use_gravity = true`, and `zhi.theta_grad` holds each lapse rate at the lid. Neither sets `erf.most.surf_temp` or a surface flux, so the surface is adiabatic and the stratification acts through buoyancy on the fire's plume and the indraft it drives. The unstable column overturns wherever it is disturbed; in this still, periodic domain, the plume is the first disturbance.

Until 2026-09-10 no stability was applied. The decks set `erf.dtheta_ref` and `erf.most.use_monin_obukhov`, which no code reads, together with `erf.init_type = "uniform"` and gravity off, so both decks ran the same neutral atmosphere. The unstable deck also set `erf.use_wind_limit` in place of `erf.fire.use_wind_limit`.

## Expected Results
See the input-file header comments in this directory for the specific validation target. In general, these cases should reproduce the documented analytical trend, qualitative regime change, or engineering diagnostic associated with the scenario.

## Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `geometry.prob_lo` | `0.0 0.0 0.0` | Primary configuration value taken from `inputs_fire_stable_atmosphere`. |
| `geometry.prob_hi` | `2000.0 2000.0 500.0` | Primary configuration value taken from `inputs_fire_stable_atmosphere`. |
| `amr.n_cell` | `40 40 50` | Primary configuration value taken from `inputs_fire_stable_atmosphere`. |
| `geometry.is_periodic` | `1 1 0` | Primary configuration value taken from `inputs_fire_stable_atmosphere`. |
| `amr.max_level` | `0` | Primary configuration value taken from `inputs_fire_stable_atmosphere`. |
| `amr.max_grid_size` | `100` | Primary configuration value taken from `inputs_fire_stable_atmosphere`. |
| `amr.max_grid_size_z` | `50` | Primary configuration value taken from `inputs_fire_stable_atmosphere`. |
| `zlo.type` | `"surface_layer"` | Primary configuration value taken from `inputs_fire_stable_atmosphere`. |

## References
- Rothermel 1972, A Mathematical Model for Predicting Fire Spread in Wildland Fuels.
- Andrews 2018, The Rothermel surface fire spread model and associated developments.
