# FireLevelsetReinit

Does level-set reinitialization (`Source/Fire/ERF_Reinitialize.H`,
`fire_levelset::reinitialize_phi`) actually improve front-tracking accuracy,
or is it not worth its cost?

## The case

A short (120 s), one-way-coupled, single-fuel-model windy line fire:

- Fuel: FM1 (short grass) only.
- Wind: `erf.fire.prescribed_wind`, a constant 4.005 m/s -- the fire never
  reads the atmosphere's wind, so results don't depend on atmospheric state.
- Coupling: `erf.fire.fire_atm_feedback = 0.0` -- one-way. The fire doesn't
  feed back onto the atmosphere either, so this isolates the level-set
  numerics completely.
- Everything else is ERF's own compiled-in default: no WRF-Fire-matching
  options (`reaction_velocity_formula`, `wrf_bmst_compat`,
  `directional_wind_coupling`) are touched anywhere in this test. The
  theoretical Rothermel rate of spread used by `check_regtest.py` is ERF's
  own native formula (Albini reaction-velocity exponent, no WRF deflation),
  computed independently in Python.

Two decks, differing only in `erf.fire.levelset.reinit_iters`:

| deck | reinit_iters | notes |
|---|---|---|
| `inputs_noreinit` | 0 | reinitialization off |
| `inputs_reinit` | (default: 10, every 5 subcycles) | ERF's compiled-in defaults, untouched |

## Running

```
./run_regtest.sh /path/to/erf_exec
```

(needs `FI_PROVIDER=tcp` exported first if MPICH's default OFI provider
fails to init in your environment -- `run_regtest.sh` does this for you).
`SKIP_RUN=1 ./run_regtest.sh x` re-checks already-run decks.

## Check

`check_regtest.py` reads each deck's t=0 and final (t=120s) `fire_phi`
plotfiles, finds the head position on the ignition line's centerline
(y=1500) in both, and compares the observed displacement against the
theoretical `Rf*t`. The baseline is the *actual* t=0 head position (x=525),
not the nominal ignition coordinate (x=500, from `line_finite.csv`) --
`erf.fire.ignition.polyline_width = 25` means the initial phi=0 front
already sits a half-line-width ahead of the nominal ignition line, and
using the nominal coordinate as the baseline introduces a fixed ~25 m
offset that swamps the actual reinit-vs-noreinit signal at this short a
run length.

Pass/fail is a single comparison: **the reinit deck's error must be
smaller than the no-reinit deck's.** As of the current source
(`Source/Fire/ERF_Reinitialize.H`), it is: reinit tracks the theoretical
head position to ~0.03% at t=120s, while no-reinit lags by ~2%.
