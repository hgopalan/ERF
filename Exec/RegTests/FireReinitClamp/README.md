# FireReinitClamp

`erf.fire.levelset.reinit_clamp`, old against new. The reinitialisation of the
level set ends with the clamp phi = min(phi, phi_ref), which keeps burned cells
burned. `"step_start"` (the default, WRF-Fire's form) takes phi_ref as the level
set at the start of the subcycle, before its advection; `"post_advection"` (the
form before 2026-10) takes the advected field, which never lets a too-steep
burned side relax and moves the front forward with every correction.

The deck is the LevelSetPropagation fire (flat ground, 5 m/s wind, 10 m fire
cells, a 30 m ignition disc, passive coupling) with a reinitialisation stronger
than the default: 2 pseudo-time steps of 2.5 m (0.25 dx) every 5th subcycle.
It uses first-order differences and a 4 m viscosity (0.4 dx) everywhere, so the
advection raises phi where the viscous term beats the Godunov |grad phi| (at the
burned minimum of the disc), which a clamp would cut back.
`run_reinit_clamp.sh` runs it for 24 steps of 5 s (120 s) at `levelset.cfl = 0.04`, so
each step takes 4 level-set subcycles, four times:

| run | `reinit_clamp` | `reinit_iters` |
|-----|----------------|----------------|
| `step_start` | `"step_start"` | 2 |
| `post_advection` | `"post_advection"` | 2 |
| `iters0_step_start` | `"step_start"` | 0 |
| `iters0_post_advection` | `"post_advection"` | 0 |

`check_reinit_clamp.py` reads `fire_phi` of each run's last fire plotfile and
checks three things (numbers measured on 2026-10-11, one rank):

| check | measured | required |
|-------|----------|----------|
| largest \|phi\| difference between the two clamps | 4.69 m | 0.5 to 7 m |
| `post_advection` head minus `step_start` head | 0.906 m | >= 0.3 m |
| largest \|phi\| difference between the two `reinit_iters = 0` runs | 0 | <= 1e-6 m |

The first two checks fail when FireLayer does not pass its start-of-subcycle
level set to the reinitialisation (the two clamps then give the same field: 0 m
and 0 m). The first also fails when the reference is copied once per
atmospheric step instead of once per subcycle (9.94 m: the clamp then holds the
field to one up to 4 subcycles old). The third fails when the clamp runs although `reinit_iters = 0` skips
the reinitialisation (0.31 m). The burned areas are printed too (4308.9 and
4330.6 m^2) but not checked: on other settings of this deck their difference
changed sign during the run. `python3 check_reinit_clamp.py --self-test` checks the checker on
synthetic fields.

```
MPIRUN="mpirun -np 1" ./run_reinit_clamp.sh /path/to/erf_exec
SKIP_RUN=1 ./run_reinit_clamp.sh x      # run nothing; check the existing output
```
