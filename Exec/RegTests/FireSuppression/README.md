# FireSuppression

Fire lines, retardant drops, the hold test and burnout from a watched action
file (`erf.fire.suppression.*`, theory page `fire_suppression.rst`). A still
atmosphere on 400 x 400 m with slip walls, 2 m fire cells (grid ratio 10)
and a prescribed rate of spread of 1 m/s, so the front moves one cell a
second and every arrival time follows from the geometry: a disc of radius
10 m at (200, 200) ignited at t = 0 has its east edge at x = 230 m after the
20 s (40 step) run. Every variant differs only in its action file; the base
deck runs the level-set path and `run_suppression.sh` runs each variant on
both paths (`erf.fire.propagation_method=farsite` on the command line).

| deck | action file | what happens at 20 s |
|---|---|---|
| `inputs_line_early` | L1 at x = 221 m, y = 100..300 m, from t = 0 at 50 m/s (complete at 4 s) | head stopped at x = 219 m; 100 line cells with mask 1 and ordinal 1 |
| `inputs_line_late` | the same line from t = 15 s | the front passed 221 m at 11 s; its cells there are burned and skipped, the head reaches 229 m; the line holds 85 of its 102 cells |
| `inputs_drop_hold` | D1 over x = 216..300 m, `ros_factor = 0`, 0..12 s | held at 215 m from 6 s, released at 12 s, head at 223 m (level set) or 225 m (FARSITE) |
| `inputs_drop_slow` | the same polygon, `ros_factor = 0.3`, permanent | head at 221 m; `fire_ros_factor` 0.3 in the drop, 1 outside |
| `inputs_hold` | L1 with `flame_limit_m = 3.0` under a rate gradient `1 + 0.03 (y - 200)` m/s | the line fails (mask -1) where the hotter cells north of y = 198 m arrive and the fire crosses there; it holds (mask 1) to the south |
| `inputs_burnout` | L1 at x = 241 m complete at 2 s; B1 from L1 at t = 5 s, offset 4 m | one ignition point per cell along the line on the west side; the strip at x = 237..239 m burns by 8 s, 20 s before the main front would arrive; nothing beyond the line |
| `inputs_poll` | an empty file, polled every 5 steps; `run_poll.sh` appends L1 at x = 241 m after the start-up read | the re-read applies L1 by a poll (step > 1) and the head stops at 239 m at 40 s (80 steps); also on two ranks |
| `inputs_restart_*` | L1 built at 5 m/s from y = 100 m (half built at 20 s) | straight, checkpoint at step 23, restart to 40: every fire plotfile field identical |

```
MPIRUN="mpirun -np 1" ./run_suppression.sh /path/to/erf_exec
METHOD=farsite MPIRUN="mpirun -np 2" sh run_poll.sh /path/to/erf_exec
```

`check_suppression.py` reads the last fire plotfile (`fire_arrival_time`,
`fire_suppression_mask`, `fire_ros_factor`, `fire_line_progress`) and the
suppression log, prints one PASS/FAIL line per check and exits non-zero on
any failure; a plotfile without the suppression fields (a binary without
the feature) fails. Each check names the defect it guards. The two abort
action files (`actions_bad.txt`, a rate of zero; `actions_duplicate.txt`, a
repeated id) must stop the run at start-up quoting the line.

Measured with the CTest build (Debug, one rank) on 2026-09-16:

| variant | level set | FARSITE |
|---|---|---|
| line_early, max burned x in the line's rows | 219 m | 219 m |
| line_late, max burned x at y = 200 m | 229 m | 229 m |
| drop_hold, max burned x at y = 200 m | 223 m | 225 m |
| drop_slow, max burned x at y = 200 m | 221 m | 221 m |
| hold, failed line cells (y range) / rows crossed beyond the line | 18 cells, y = 199..233 m / y = 199..231 m | 17 cells, y = 199..231 m / y = 199..229 m |
| poll, re-read and applied at fire step | 5 | 5 |
| restart, unburned line cells at 20 s / fields differing | 43 / 0 of 25 | 43 / 0 of 25 |
| burnout, arrival at x = 239 m | 5..8 s | 5..8 s |
