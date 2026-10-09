# FireDustMassConservation

## Purpose
This hazard case closes the dust mass budget of a coupled fire-dust run: the airborne dust mass printed by `erf.sum_interval` equals the mass emitted (from `dust_diag.dat`) minus the mass deposited. `erf.dust.deposition_E0 = 0` removes the surface collection term only; dust still leaves the air by settling at `v_s`, so the airborne mass is not monotone.

## Expected Results
Run the deck and the checker:

```
erf_exec inputs > run.log && python3 check_mass_conservation.py run.log dust_diag.dat
python3 check_mass_conservation.py --self-test   # the checker on a closed and a leaky synthetic budget (CTest FireDustMassConservation_SelfTest)
```

It passes when `|M_air - (emitted - deposited)| / emitted < 2 %` at every printed step (the flux of step n enters the air in step n+1, the documented lag), and both the emitted and the deposited totals are positive. A run with `erf.dust.atm_feedback = 0.5` fails it (half the emitted mass never reaches the air). Until October 2026 the case asserted nothing.


## References
- Rothermel 1972, A Mathematical Model for Predicting Fire Spread in Wildland Fuels.
- Andrews 2018, The Rothermel surface fire spread model and associated developments.
