# FireReactionVelocityFormula

Do `erf.fire.reaction_velocity_formula` ("albini" | "rothermel") and
`erf.fire.wrf_bmst_compat` (false | true) reach the rate of spread, alone and
together, on the uniform-fuel path and through a fuel map's per-fuel table?

- `reaction_velocity_formula` picks the exponent A of Rothermel's optimum
  reaction velocity: Albini's (1976) `A = 133 sigma^-0.7913` (the default) or
  Rothermel's (1972) Eq. 39, `A = 1/(4.774 sigma^0.1 - 7.27)`, the form
  WRF-Fire uses.
- `wrf_bmst_compat` deflates the fuel load fed to the Rothermel coefficients
  by `1 - M/(1+M)`, as WRF-Fire does to its load. It is a comparison option,
  not full WRF-Fire parity (see the theory page).

## The case

A line fire 250 m long in short grass (Anderson FM1) at 6 % 1-h moisture,
one-way coupled (`fire_atm_feedback = 0`), under a prescribed wind along the
spread direction, with the wind limit off (`use_wind_limit = false`). The
head rate is algebraic: `head_ros_ms` in the fire statistics CSV holds its
final value from the first step, so each run is five steps on a 60 x 30 x 10
grid (12.5 m fire cells).

Five decks, each run calm (`prescribed_wind_x = 0`) and at 4.005 m/s:

| deck | reaction_velocity_formula | wrf_bmst_compat | fuel |
|---|---|---|---|
| `inputs_albini` | "albini" (default) | false (default) | uniform FM1 |
| `inputs_albini_bmst` | "albini" | true | uniform FM1 |
| `inputs_rothermel` | "rothermel" | false | uniform FM1 |
| `inputs_rothermel_bmst` | "rothermel" | true | uniform FM1 |
| `inputs_rothermel_bmst_map` | "rothermel" | true | `fuel_map_fm1.asc` (FM1 everywhere), `rothermel_per_fuel = true` |

Both winds are needed. With wind, the lower load of `wrf_bmst_compat` slows
the no-wind rate but raises the wind factor through `(beta/beta_op)^-E`, and
for FM1 at 4.005 m/s the two nearly cancel: the albini pair differs by only
0.006 %. Without wind every pair differs by more than 1 %. That is with the
wind limit off, as these decks run. Under the default limit U <= 0.9 I_R the
lower I_R of `wrf_bmst_compat` also lowers the limit, which binds at this
wind: the albini pair's head rate drops 12.8 % (1.50926 to 1.31559 m/s;
0.09 % under `wind_limit = fuel_class`, 0.250151 to 0.24993 m/s). To see it,
run `inputs_albini` and `inputs_albini_bmst` with
`erf.fire.prescribed_wind_x=4.005 erf.fire.use_wind_limit=true` (and
`erf.fire.wind_limit=fuel_class`) appended and read `head_ros_ms` from the
fire statistics CSV (measured 2026-10-10).

## Running

```
[MPIRUN="mpirun -np 1"] ./run_regtest.sh /path/to/erf_exec
```

`SKIP_RUN=1 ./run_regtest.sh x` reruns the check on existing output. CTest
runs it as `FireReactionVelocityFormula` (labels `regression;fire`), the
checker's own pass/fail logic as `FireReactionVelocityFormula_SelfTest`, and
the warning a non-Rothermel model gives for these keys as
`FireReactionVelocityFormula_behave_warning`.

## Check

`check_regtest.py` compares each run's `head_ros_ms` with the closed-form
Rothermel rate of its combination, written from the papers in ERF's
single-class layout (1-h SAV as the bed SAV, `w_n = w_0 (1 - S_T)`). Every
run must be within 0.01 % (the CSV carries six significant digits).
`check_regtest.py --self-test` feeds it made-up CSVs: the right rates pass,
and a run that ignored `wrf_bmst_compat`, the formula, or both options on the
fuel-map deck fails.

Measured on 2026-10-10 (Release, one rank):

| deck | calm head rate (m/s) | 4.005 m/s head rate (m/s) | largest error |
|---|---|---|---|
| `inputs_albini` | 0.0233949 | 1.70098 | 0.0002 % |
| `inputs_albini_bmst` | 0.0231243 | 1.70108 | 0.0001 % |
| `inputs_rothermel` | 0.022323 | 1.62304 | 0.0002 % |
| `inputs_rothermel_bmst` | 0.0219922 | 1.61781 | 0.0003 % |
| `inputs_rothermel_bmst_map` | 0.0219922 | 1.61781 | 0.0003 % |

The fuel-map deck gives the uniform deck's rates to every printed digit.
With both options dropped in the per-fuel table builder, that deck is
6.4 % (calm) and 5.1 % (wind) off; with both options dropped on the
uniform-fuel path, five runs fail, by 1.2 % to 6.4 % (the windy albini_bmst
run is not among them, which is why the calm runs are there).
