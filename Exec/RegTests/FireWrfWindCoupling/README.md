# FireWrfWindCoupling

A finite ignition line in a strong uniform wind, spread by the directional
level set with `erf.fire.directional_wind_coupling = "projection"` (the
default) and `"wrf"`, against Rothermel's head rate.

```
MPIRUN="mpirun -np 2" ./run_firewrfwindcoupling.sh /path/to/erf_exec   # three decks, then the checks
SKIP_RUN=1 ./run_firewrfwindcoupling.sh x                              # checks only
```

## The case

6000 x 3000 x 500 m, periodic sideways, a uniform 4.005 m/s westerly
(`erf.fire.prescribed_wind`, bypassing atmospheric interpolation so all three
decks see exactly the same wind) with `erf.fire.use_wind_limit = false` so
the code's 300 ft/min midflame-wind cap for fine fuels does not clip it. Flat
ground, Anderson fuel model 1 at 6 % moisture, 25 m fire cells, a 1 km
ignition line at x=500 m spanning y=1000-2000 m (short of the 3 km periodic
extent, so the fire develops real lateral flanks, not two
translationally-invariant 1D fronts, per FireLineIgnition's
Munoz-Esparza-et-al.-style case), 2100 s.

| deck | `directional_wind_coupling` |
|---|---|
| `default` | unset (must reproduce `projection` bit for bit) |
| `projection` | `"projection"`, written out |
| `wrf` | `"wrf"` |

## Why the projection's head falls short

At U = 4.005 m/s, FM1 at 6 % moisture gives phi_w = 71.7, B = 2.07, so
phi_w(B - 1) = 76.8. The default formula evaluates phi_w from the wind
already projected onto the front normal, R(n) = R0(1 + phi_w(max(U.n,0))^B).
Once phi_w(B - 1) > 1 this is not the support function of a convex set, so
the level-set head (the exact viscosity solution, and independently of the
numerical scheme) falls from Rothermel's head rate Rf = R0(1 + phi_w) =
1.701 m/s toward the Wulff-shape tip rate
R0 B/(B-1) (phi_w(B-1))^(1/B) = 0.368 m/s (22 % of Rf) as the front
develops facets -- the same degradation FireDirectionalShape documents for a
point ignition, here for a finite line.

`"wrf"` instead matches WRF-Fire's `fire_ros` (module_fr_fire_phys.F):
exponentiate phi_w from the raw, unprojected wind speed, then scale the
whole wind/slope factor by cos(theta) to the front normal afterward,
R(n) = R0(1 + phi_w(U) max(cos theta, 0)). Linear in cos(theta) -- the
convex support function of a stadium (a disc of radius R0 swept along the
wind vector) -- so no wedge forms and the head tracks Rf.

`check_firewrfwindcoupling.py` reads the centerline row (nearest y=1500, the
ignition line's midpoint, farthest from the flank curvature at its ends)
from every `fire_phi` plotfile, tracks the head (+x) and back (-x) front
position over time, and fits a rate over the second half of the run
(t >= 1050 s), by which point the projection deck's head has settled onto
its asymptotic behaviour.

## Expected Results

One rank per deck. Rf = 1.70098 m/s; the Wulff-shape tip of the projection
formula is 0.36793 m/s (22 % of Rf). Rates fitted over t >= 1050 s:

| deck | x_head(2100 s) | head rate | vs Rf |
|---|---|---|---|
| `default` | 3616.09 m | 1.20891 m/s | 71.1 % (well below Rf, well above the Wulff tip) |
| `projection` | 3616.09 m | 1.20891 m/s | same as `default`, bit for bit (`max\|phi_default - phi_projection\| = 0`) |
| `wrf` | 4091.58 m | 1.70098 m/s | 100.0 % |

`wrf` lands 28.9 points of Rf-fraction closer to Rf than `projection`. All 8
checks pass.

Back rate is reported for reference (0.02342 m/s vs R0 = 0.02339 m/s,
essentially identical across all three decks, barely one fire cell over the
run -- not checked quantitatively) and confirms the fix only changes how the
wind/slope factor couples to the front-normal direction on the advancing
side, not the backing rate, where the projected component is clipped to
zero under either formulation.
