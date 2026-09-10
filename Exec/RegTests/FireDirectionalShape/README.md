# FireDirectionalShape

A point ignition in a uniform wind, spread by the directional level set with
the projected rate (the default, `erf.fire.directional_shape = "projection"`)
and with the spread ellipse (`"ellipse"`), against Rothermel's head rate and the
exact solution of each formulation's own equation.

```
MPIRUN="mpirun -np 2" ./run_firedirectionalshape.sh /path/to/erf_exec   # four decks, then the checks
SKIP_RUN=1 ./run_firedirectionalshape.sh x                             # checks only
```

## The case

600 x 200 x 100 m, periodic sideways, a uniform 1.5 m/s westerly from
`input_sounding` with no diffusion, one-way coupling and a surface layer with
zero bulk transfer coefficients (with the default MOST drag the wind at 6.1 m
fell from 1.33 to 0.79 m/s over the run), so the fire sees the same wind
everywhere and at all times; the checker verifies that from the plotfiles. Flat
ground, Anderson fuel model 1 at 5.5 % moisture, 2 m fire cells, a 10 m
ignition disc at (100, 100), 1500 s (300 s for the isotropic disc, which would
otherwise wrap the periodic domain). The wind stays below the 300 ft/min
(1.52 m/s) cap the code applies to the midflame wind of fine fuels.

| deck | level set |
|---|---|
| `isotropic` | `directional_ros = false`: the whole front at R0 (1 + phi_w) |
| `projection` | the default: R(n) = R0 (1 + phi_w(max(U . n, 0))) |
| `projection_key` | the same with `directional_shape = "projection"` written out |
| `ellipse` | `directional_shape = "ellipse"`: head R0 (1 + phi_w), back and flanks R0 |

## Why the projection's head falls short

The level set moves each point of the front along its normal at R(n). From a
point, the exact (viscosity) solution of phi_t + R(n) |grad phi| = 0 is the Wulff
shape {x : x . n <= t R(n) for every n}, whose extent along the wind is the
minimum over n of R(n) / n_x, not R along the wind. The two agree only when R
is the support function of a convex set. Rothermel's projected rate
R0 (1 + phi_w (U n_x)^B) is not once phi_w (B - 1) > 1 (here phi_w = 9.3,
B = 2.07): its head is a wedge of oblique facets whose tip runs at
R0 B/(B - 1) (phi_w (B - 1))^(1/B), 57 % of the head rate. A better scheme
cannot fix this, since it is the exact solution that falls short. A straight
line fire keeps n along the wind and is unaffected (FireLineFire).

The ellipse option takes R(n) from the support function of an ellipse with the
model's head, back and flank rates. An ellipse is convex, so it is its own
Wulff shape and a point fire's head runs at the head rate, while the back and
flanks keep the projection's rates.

`check_firedirectionalshape.py` evaluates the exact solution of each deck's
equation with the Hopf formula T(x) = max over n of ((x - c) . n - r0) / R(n)
at the plotfile's cell centres, and fits it the same way as the run: the time
the front first reaches each column (head, back) or row (flanks) against its
distance from the ignition point.

## Expected Results

One rank per deck. U = 1.5 m/s gives phi_w = 9.38 and phi_w (B - 1) = 10.0.
Rothermel's head rate R0 (1 + phi_w) is 0.24943 m/s, and the tip of the
projection's exact solution (the Wulff shape) 0.14155 m/s, 57 % of it. Rates
[m/s]:

| deck | head | back | flanks |
|---|---|---|---|
| `isotropic` (300 s) | 0.24855 (-0.35 % of Rothermel) | 0.24855 | 0.24855 |
| `projection` | 0.18487 (-25.9 %; 40 % of the way from the Wulff tip to Rothermel) | 0.02396 | 0.02400 |
| `projection_key` | arrival times identical to `projection` bit for bit | | |
| `ellipse` | 0.24775 (-0.67 %) | 0.02388 | 0.02403 |

Back and flanks match the exact rates (R0 = 0.02404 m/s) to 0.7 %; 42 of the
44 checks below pass, all of them for the four decks of the script.

The head against the level-set scheme, with extra arguments on the same decks:

| scheme | `projection` head | `ellipse` head |
|---|---|---|
| default (`weno5z_front`, `eps_visc_front` 0.1) | 0.18487 (40 % of the way) | 0.24775 (-0.67 %) |
| `erf.fire.levelset.eps_visc_front=0.01` | 0.19798 (52 %) | 0.24926 (-0.07 %) |
| `erf.fire.levelset.gradient=upwind` | 0.15279 (10 %) | 0.24020 (-3.70 %) |

The projection's head moves with the scheme because the scheme, not the
equation, decides where between the Wulff tip and Rothermel's rate it lands;
the more dissipative the scheme, the closer to the exact (short) Wulff tip. The
ellipse's head approaches Rothermel's rate as the dissipation falls; only the
first-order upwind gradient, which rounds the narrow head, misses the 3 %
tolerance.
