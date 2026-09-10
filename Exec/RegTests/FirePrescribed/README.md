# FirePrescribed

The two options the analytic fire cases rely on, each against an exact answer.

- `erf.fire.ros_model = "prescribed"`: a rate of spread set in the deck,
  `prescribed.ros` plus an optional linear `prescribed.gradient` about
  `prescribed.origin`, or per fuel code with `prescribed.by_fuel`. No wind, slope
  or moisture dependence and no direction, so `directional_ros` is turned off and
  the level set solves |grad T| = 1/R as posed.
- `erf.fire.prescribed_heat.flux`: a constant heat flux over a disc
  (`center`, `radius`, `start_time`, `end_time`) added to the fire heat flux
  whatever the fuel and the front, and injected by the coupling like fire heat.
  A run with only this source needs no ignition.

```
MPIRUN="mpirun -np 4" ./run_fireprescribed.sh /path/to/erf_exec
```

Both run on a still atmosphere, 400 x 400 m with 20 m cells and a 2 m fire grid,
slip walls on the sides and top and the surface layer (with zero diffusion
coefficients) at the ground.

| deck | what it checks |
|---|---|
| `ros_circle` | a 10 m disc at a prescribed 1 m/s for 100 s: `fire_ros` exactly 1 m/s, the radius from the burned area 10 + t, the arrival time at distance d equal to d - 10 |
| `heat_patch` | 1e4 W/m2 over a 30 m disc, lagged coupling, nothing burning, 60 s in a 500 m deep box: `fire_heat_flux` the flux in the disc's cells and zero elsewhere, no burned cell, the coupling's `energy_in` equal to the disc power and `energy_out / energy_in` equal to 1 - exp(-H / alfg), and Cp times the change of the integral of rho theta equal to that power times the heated time |

The domain of `heat_patch` is 500 m deep because the coupling spreads the flux
over height with an e-folding scale of `erf.fire.heat_flux_alfg` = 45 m: a column
of height H carries 1 - exp(-H/45 m) of it, 89 % for 100 m. The same checks with a
100 m box find exactly that 11 % short.

CTest runs both decks for 40 steps (`FirePrescribed_ros_circle`,
`FirePrescribed_heat_patch`); `check_fireprescribed.py` reads the output of
either the full runs or those.

## Expected Results

On four ranks:

- `ros_circle`: the rate field is 1 m/s exactly; the radius from the area is
  0.06 to 0.12 cells short of 10 + t at 20 to 100 s (109.76 m at 100 s); the
  arrival time is within 0.13 to 0.16 s of d - 10 on average (95th percentile
  0.28 to 0.38 s, against a 2 s cell crossing).
- `heat_patch`: 716 cells at 1e4 W/m2, 28.64 MW (pi r^2 q = 28.27 MW for the
  exact disc); `energy_in` equal to it to round-off and `energy_out / energy_in`
  within 2e-10 of 0.999985; the heat budget within 0.01 % (1.7038 against
  1.7041 GJ at 60 s), with the air above the disc 5.2 K warmer.
