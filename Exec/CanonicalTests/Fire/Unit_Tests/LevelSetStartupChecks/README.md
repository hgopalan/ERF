# LevelSetStartupChecks

Start-up checks of the level-set front inputs. Each deck includes
`inputs_base` (a small grass fire on 25 m fire cells, the FireRestart setup)
and sets one input; the run must stop at start-up naming the input. Every deck
runs one step at most.

| Deck | Input | Expected |
|------|-------|----------|
| `inputs_reinit_clamp_bad_value` | `erf.fire.levelset.reinit_clamp = "step_begin"` | abort: `is not one of "post_advection", "step_start"` |
| `inputs_eps_visc_unstable` | `erf.fire.levelset.eps_visc = 20` | abort: unstable for every front direction (stable up to 10.79 m) |
| `inputs_eps_visc_front_unstable` | `erf.fire.levelset.eps_visc_front = 20` | abort: unstable for every front direction (stable up to 14.92 m) |
| `inputs_cfl_unstable_without_viscosity` | `erf.fire.levelset.cfl = 1.5` | abort: no viscosity is stable at this cfl |
| `inputs_eps_visc_diagonal_unstable` | `erf.fire.levelset.eps_visc = 12` | abort: unstable for fronts near the cell diagonal, which every curved front has (stable up to 10.79 m) |
| `inputs_eps_visc_front_blend_unstable` | `eps_visc_front = 14.75` with a 3-cell WENO band ending inside a 12-cell blend to `eps_visc = 2.5` | abort: the blended 13.53 m just beyond the band, with first-order differences |
| `inputs_eps_visc_front_wall_first_order` | `eps_visc_front = 12` with `wall_extrapolate = true` | abort: first-order differences next to a non-burnable cell (stable up to 10.79 m; 14.92 m without wall extrapolation) |

The check is the linear stability of the substep's three-stage Runge-Kutta
step: the largest amplification factor over every wave number and front
direction (`levelset_substep_gain` in `Source/Fire/ERF_NumericalSchemes.H`)
must not exceed 1. For first-order differences the binding wave is the two-cell
wave, and the limit is `cfl h (2 (n_x/dx + n_y/dy) + 4 eps (1/dx^2 + 1/dy^2))
<= 2.513`; for WENO5 a complex wave binds above cfl of about 0.55 (see
`levelset_viscous_stability` and the theory page). The gtest
`LevelSetAdvection.ViscousStabilityLimitIsTheSubstepsLimit` checks the limit
against the substep itself.
