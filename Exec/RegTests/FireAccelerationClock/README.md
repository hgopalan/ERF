# FireAccelerationClock

The temporal fire acceleration, `R = R_E (1 - exp(-A t))`, with each
`erf.fire.accel.clock`, on the FARSITE and level-set paths.

- `legacy` (default): a clock per burned cell, started when the cell burns and
  restarted on any change of its equilibrium rate, written to burned cells only.
- `front`: the time since ignition, carried with the front into the cells it
  burns and applied to the rate that moves it.

A fire that carries its ignition clock has a head rate equal to the
unaccelerated head rate times `1 - exp(-A t)`. By time `t` its head has covered
the share `S(t) = 1 - (1 - exp(-A t)) / (A t)` of the unaccelerated distance,
whatever the path makes of `R_E`.

## Setup

`inputs_base`: a 50 m radius burning disc at (500, 960) m on flat ground in a
uniform 3 m/s westerly. The atmosphere is one-way coupled, with no drag and no
diffusion. The fuel is short grass (Anderson 1) at 5.5 % moisture, on 10 m fire
cells, run for 600 s. `A = 1 /min` (a 60 s time constant) so that the whole
ramp fits in the run, and `perim_limit` is out of reach so `A` never switches.

| deck | path | acceleration |
|------|------|--------------|
| `inputs_farsite_off` | FARSITE | off |
| `inputs_farsite_legacy` | FARSITE | temporal, `clock = "legacy"` |
| `inputs_farsite_front` | FARSITE | temporal, `clock = "front"` |
| `inputs_levelset_off` | level set (directional rate, the default) | off |
| `inputs_levelset_legacy` | level set | temporal, `clock = "legacy"` |
| `inputs_levelset_front` | level set | temporal, `clock = "front"` |

```bash
MPIRUN="mpirun -np 1" ./run_fireaccelerationclock.sh /path/to/erf_exec
```

`check_fireaccelerationclock.py` reads the head from `fire_arrival_time` along
the rows either side of y = 960 m. It checks that each `front` deck's
`H(t) / H_off(t)` is within 0.03 plus one cell of `S(t)` at every whole minute
from 2 minutes on. The `legacy` decks are reported but not checked.
`Tests/CTestList.cmake` runs `inputs_levelset_front` for 40 steps as a smoke
test.

## Measured (2026-09-10, one rank, double precision)

Head advance as a share of the unaccelerated advance, with the FARSITE path on
its default `front_cell` update:

| t [s] | S(t) | FARSITE legacy | FARSITE front | level-set legacy | level-set front |
|------:|-----:|---------------:|--------------:|-----------------:|----------------:|
|    60 | 0.368 | 1.000 | 0.370 | 1.000 | 0.428 |
|   120 | 0.568 | 1.000 | 0.569 | 1.000 | 0.621 |
|   240 | 0.755 | 1.000 | 0.755 | 1.000 | 0.795 |
|   360 | 0.834 | 1.000 | 0.834 | 1.000 | 0.857 |
|   600 | 0.900 | 1.000 | 0.897 | 1.000 | 0.909 |

Head rate over each window as a ratio to the unaccelerated head. "ideal" is the
window mean of `1 - exp(-A t)`:

| window [s] | ideal | FARSITE off [m/s] | legacy | front | level-set off [m/s] | legacy | front |
|-----------:|------:|------------------:|-------:|------:|--------------------:|-------:|------:|
|    0-120 | 0.568 | 0.971 | 1.000 | 0.569 | 0.929 | 1.000 | 0.621 |
|  120-240 | 0.941 | 0.971 | 1.000 | 0.941 | 0.709 | 1.000 | 1.024 |
|  240-360 | 0.992 | 0.971 | 1.000 | 0.992 | 0.683 | 1.000 | 1.004 |
|  360-600 | 0.999 | 0.960 | 1.000 | 0.991 | 0.673 | 1.000 | 0.999 |

- With the legacy clock the fire does not accelerate on either path, and both
  runs are identical to acceleration off. The `front_cell` update takes the
  larger of a front cell's own rate, `R_E`, and its burned neighbours' rates.
  The directional level-set path rebuilds its rate from the model in every
  Runge-Kutta stage and never reads the accelerated field.
- With the front clock the FARSITE head follows `S(t)` to within 0.003, at 0.07
  of the tolerance. The level-set head is within 0.06, at 0.52 of the tolerance:
  its unaccelerated head is faster over its first two minutes, which the ratio
  does not remove.
- On the legacy FARSITE update (`erf.fire.farsite.front_update=legacy` on the
  command line), the head runs at about twice the rate of spread, 1.75 to
  1.83 m/s. The legacy clock gives shares of 1.039, 1.039, 1.017, 1.010 and
  1.000 at the times above, and the front clock 0.399, 0.599, 0.797, 0.866 and
  0.907.
