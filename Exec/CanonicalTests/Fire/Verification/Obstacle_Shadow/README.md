# Obstacle_Shadow

A point fire passing a round non-burnable obstacle, against the geometry of the
shortest path around it. With a constant rate R every point burns at (L - r_ig)/R,
where L is the distance from the ignition centre: a straight line where the
centre can see the point, and behind the obstacle the taut string (tangent to the
disc, the arc, tangent to the point). It checks the non-burnable mask and the
level set's diffraction around it with nothing but geometry as the reference.

```
python3 gen_obstacle.py                                          # fuel_obstacle.asc (committed)
MPIRUN="mpirun -np 2" ./run_obstacle_shadow.sh /path/to/erf_exec  # both decks, then the checks
SKIP_RUN=1 ./run_obstacle_shadow.sh x                             # checks only
```

## The case

400 x 200 m, still air, 2 m fire cells, `erf.fire.ros_model = "prescribed"` at
1 m/s. A 6 m ignition disc at (60, 100) and a 30 m disc of fuel code 0 at
(160, 100), made non-burnable with `fuel_map.nonburnable_codes = 0`. Both decks
run 350 s.

| deck | what differs |
|---|---|
| `disc30` | the default level set |
| `disc30_wallx` | `erf.fire.levelset.wall_extrapolate = true` |

`check_obstacle_shadow.py` requires every obstacle cell to stay unburned, and
compares the arrival time three cells clear of the obstacle and the domain edge,
separately where the ignition is in view and in the shadow.

The mask is whole fire cells, those whose centre is inside the disc, and the rate
is zero across each of them, so the obstacle the front meets is about half a cell
wider than the nominal disc. The length of the taut string grows with the radius
at the rate of the wrap angle, so the shadow is compared with the path around a
disc of radius a + h/2; the lag against the nominal disc is printed as well.

## Expected Results

On two ranks:

| deck | lit region, mean \|e\| / 95th pct | shadow vs a + h/2, mean e / 95th pct | shadow vs nominal disc |
|---|---|---|---|
| `disc30` | 0.340 / 0.905 | +1.963 / 2.738 (fails) | +2.102 |
| `disc30_wallx` | 0.327 / 0.808 | +1.369 / 1.755 (fails) | +1.507 |

in cell-crossing times (h/R = 2 s), over 9716 lit and 7608 shadow cells. None of
the 716 obstacle cells burns. Re-measured 2026-10-09: 4 of the 6 checks pass.
The code before this one fails the same two, at +1.834 / 2.645 and +1.246 /
1.662 (lit 0.236 / 0.789 and 0.223 / 0.689); the +0.435 / 0.757 and +0.521 /
0.959 this README carried until then predate both.

- Where the ignition is in view the level set is within a third of a cell.
- The front that wraps the obstacle arrives late by one and a half to two
  cells beyond the half cell the cell-resolved obstacle explains; the wall
  extrapolation removes half a cell of it. The shadow check allows three
  quarters of a cell on average (1.5 at the 95th percentile) and fails on both
  decks; the lag of the wrapped front is open.
