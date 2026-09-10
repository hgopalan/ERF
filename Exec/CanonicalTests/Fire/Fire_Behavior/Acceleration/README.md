# Acceleration

## Purpose
These cases compare disabled, size-based, and temporal acceleration options for fire growth, verifying the transient adjustment from ignition toward the steady spread regime.

## Physics / Model Features Exercised
- Fire spread / ignition configuration
- Atmospheric forcing and boundary-condition setup

## Expected Results
See the input-file header comments in this directory for the specific validation target. In general, these cases should reproduce the documented analytical trend, qualitative regime change, or engineering diagnostic associated with the scenario.

## Measured head advance (2026-09-10)

The temporal deck uses the default `erf.fire.accel.clock = "legacy"`. The front
clock is `erf.fire.accel.clock=front` on the command line; see
`Docs/sphinx_doc/theory/fire_acceleration.rst`. The head is the largest
distance from the ignition point, read from `fire_arrival_time`, on 2 ranks.
"pred." is `∫ v_off (1 - exp(-s)) dt`: the head rate with acceleration off
times the factor of a fire carrying its ignition clock, with `s = ∫ A dt`
taken from that run's own `A` history (`A_line` beyond 500 m of perimeter).
The geostrophic wind decays over the run, so the unaccelerated head slows down
too.

Default FARSITE `front_cell` update, head advance [m]:

| t [s] | disabled | size-based | temporal, legacy clock | temporal, front clock | front pred. |
|------:|---------:|-----------:|-----------------------:|----------------------:|------------:|
|   120 |     20.9 |        5.5 |                   20.9 |                   2.9 |         1.8 |
|   240 |     33.6 |        8.5 |                   33.6 |                   5.8 |         5.7 |
|   420 |     45.8 |       12.9 |                   45.8 |                  10.8 |        12.1 |
|   660 |     58.8 |       16.0 |                   58.8 |                  17.7 |        19.5 |
|   880 |     67.0 |       18.3 |                   67.0 |                  22.3 |        26.6 |

With the legacy clock the head is identical to the disabled case: the
`front_cell` update takes a front cell's own equilibrium rate. The front clock
stays within one 10 m fire cell of its prediction.

Legacy FARSITE update (`erf.fire.farsite.front_update=legacy`), head advance [m]:

| t [s] | disabled | size-based | legacy clock | legacy pred. | front clock | front pred. |
|------:|---------:|-----------:|-------------:|-------------:|------------:|------------:|
|   120 |     41.1 |        6.8 |         41.0 |          3.8 |         7.1 |         3.8 |
|   240 |     64.2 |       13.6 |         65.8 |         18.9 |        14.1 |        11.6 |
|   420 |     86.6 |       23.3 |         91.2 |         43.7 |        24.2 |        23.7 |
|   660 |    110.6 |       31.1 |        116.5 |         64.3 |        37.1 |        37.1 |
|   880 |    127.8 |       37.6 |        135.7 |         83.8 |        56.0 |        55.6 |

Here the legacy clock runs slightly ahead of the disabled case, because the
first unburned row advances at the equilibrium rate. The front clock follows
its prediction. The legacy clock is bit-identical between the builds before and
after the front clock was added.

## Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `geometry.prob_lo` | `0.0 0.0 0.0` | Primary configuration value taken from `inputs_fire_phase12_disabled`. |
| `geometry.prob_hi` | `2000.0 2000.0 500.0` | Primary configuration value taken from `inputs_fire_phase12_disabled`. |
| `amr.n_cell` | `40 40 50` | Primary configuration value taken from `inputs_fire_phase12_disabled`. |
| `geometry.is_periodic` | `1 1 0` | Primary configuration value taken from `inputs_fire_phase12_disabled`. |
| `amr.max_level` | `0` | Primary configuration value taken from `inputs_fire_phase12_disabled`. |
| `amr.max_grid_size` | `100` | Primary configuration value taken from `inputs_fire_phase12_disabled`. |
| `amr.max_grid_size_z` | `50` | Primary configuration value taken from `inputs_fire_phase12_disabled`. |
| `zlo.type` | `"surface_layer"` | Primary configuration value taken from `inputs_fire_phase12_disabled`. |

## References
- Rothermel 1972, A Mathematical Model for Predicting Fire Spread in Wildland Fuels.
- Andrews 2018, The Rothermel surface fire spread model and associated developments.
