# FireDustMassConservation

## Purpose
This hazard case closes the dust mass budget of a coupled fire-dust run: the airborne dust mass printed by `erf.sum_interval` equals the mass emitted (from `dust_diag.dat`) minus the mass deposited. `erf.dust.deposition_E0 = 0` removes the surface collection term only; dust still leaves the air by settling at `v_s`, so the airborne mass is not monotone.

## Physics / Model Features Exercised
- Fire spread / ignition configuration
- Atmospheric forcing and boundary-condition setup
- Coupled hazard-module configuration
- Cross-module diagnostics or interaction controls

## Expected Results
Run the deck and the checker:

```
erf_exec inputs > run.log && python3 check_mass_conservation.py run.log dust_diag.dat
```

It passes when `|M_air - (emitted - deposited)| / emitted < 2 %` at every printed step (the flux of step n enters the air in step n+1, the documented lag), and both the emitted and the deposited totals are positive. A run with `erf.dust.atm_feedback = 0.5` fails it (half the emitted mass never reaches the air). Until October 2026 the case asserted nothing.

## Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `erf.prob_name` | `"ABL"` | Primary configuration value taken from `inputs`. |
| `max_step` | `100` | Primary configuration value taken from `inputs`. |
| `amr.max_level` | `0` | Primary configuration value taken from `inputs`. |
| `amrex.fpe_trap_invalid` | `0` | Primary configuration value taken from `inputs`. |
| `geometry.prob_extent` | `3000 3000 1024` | Primary configuration value taken from `inputs`. |
| `amr.n_cell` | `8 8 64` | Primary configuration value taken from `inputs`. |
| `geometry.is_periodic` | `1 1 0` | Primary configuration value taken from `inputs`. |
| `amr.max_grid_size` | `32` | Primary configuration value taken from `inputs`. |

## References
- Rothermel 1972, A Mathematical Model for Predicting Fire Spread in Wildland Fuels.
- Andrews 2018, The Rothermel surface fire spread model and associated developments.
