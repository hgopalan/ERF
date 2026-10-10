# FireMergingFronts

The three ways fires meet, each against the geometry of its ignition: spot
fires coalescing, two lines meeting at an angle (a junction fire) and two
parallel lines burning towards each other. A still atmosphere on 300 x 150 m
with slip walls, 2 m fire cells (10 m atmosphere cells, grid ratio 5) and a
prescribed rate of spread of 1 m/s, so the front moves one cell a second and
every point burns when its distance to the nearest ignition, less the
ignition's own half width, has been covered: T = (d - w) / R. The decks run
40 steps of 0.5 s (20 s); nothing but the ignition differs between the three
merging decks. `inputs_firebreak` adds a firebreak and is checked at start-up only.

| deck | ignition | what happens by 20 s |
|---|---|---|
| `inputs_coalescing` | four 6 m discs at t = 0 from `coalescing.csv`: three 40 m apart on y = 60 m and one 28 m north of the middle one | the north pair touch at 8 s and the three on the line at 14 s; the necks fill at the geometric rate |
| `inputs_junction` | one 8 m wide polyline, `junction_v60.csv`: a V of two 70 m arms at 60 degrees, apex (30, 60) m, opening towards +x | the inner fronts meet on the bisector and their meeting point runs along it at R / sin(30 deg) = 2 m/s, from x = 38 m to 78 m |
| `inputs_parallel` | two 8 m wide polylines in two files, `parallel_south.csv` (y = 40 m) and `parallel_north.csv` (y = 80 m), x from 40 to 200 m | the strip between the lines closes on y = 60 m at 16 s |
| `inputs_missing_file` | the south line and a second file that does not exist | stops at start-up: `Cannot open polygon vertex file: parallel_missing.csv` |
| `inputs_one_vertex` | the south line and `one_vertex.csv`, a single vertex | stops at start-up: a polyline needs at least 2 vertices (a polygon 3); the stamp would otherwise mark nothing |
| `inputs_firebreak` | the south line and a 20 m wide firebreak at x = 240 to 260 m | the break is held in the non-burnable mask, all 750 of its 2 m cells (the start-up line with `erf.fire.fire_debug = 1`); until 2026-10 the mask read the break back from phi after the line's distance field had been merged over it, and held none |

The parallel pair is the reason `erf.fire.ignition.polygon_file` takes a
list: each file is one perimeter, every file is stamped with the merging rule
min(phi, new), and all of them share `polygon_type` and `polyline_width`.

```
MPIRUN="mpirun -np 2" ./run_merging_fronts.sh /path/to/erf_exec
```

`check_merging_fronts.py` reads the last fire plotfile of each variant with
the pure-Python reader (no yt) and checks, each naming the defect it guards:

- **prescribed rate**: the plotfile's `fire_ros` is one value on every
  burnable cell; it is the R the exact times are built from.
- **ignition stamped**: every cell well inside an ignition has the stamp time
  as its arrival time. A file or schedule row that is not read leaves its
  cells unburned: with the first file only, as the code read before the key
  took a list, 164 of the 328 cells of the parallel pair are unburned and the
  check fails (as do *fronts reached*, 1052 cells, *arrival time*, and
  *merge region*, late by half a crossing on the north side of the strip).
- **fronts reached**: no cell the fronts should have passed is unburned.
- **arrival time**: over every cell the fronts reached, the mean |error| is at
  most half a cell crossing (h / R = 2 s), the 95th percentile at most one,
  and the signed mean within a quarter of a crossing.
- **merge region**: the same on the cells where the fronts meet (the necks
  between the discs, the bisector inside the wedge from where the inner edges
  first meet to where the meeting point is at the end, the strip between the
  lines). The signed mean is what tells a fire running ahead from one held
  back; the one-file mutant has a signed bias of +0.5 there.
- **not yet reached**: no cell more than two crossings beyond the final front
  is burned.

## Expected Results

On two ranks, errors in cell crossings (2 s):

| deck | cells reached | arrival: signed mean / mean abs. / 95th pct | merge cells | merge: signed mean / mean abs. / 95th pct |
|---|---|---|---|---|
| `coalescing` | 1096 | -0.078 / 0.086 / 0.201 | 24 | -0.016 / 0.016 / 0.023 |
| `junction` | 1042 | -0.091 / 0.099 / 0.227 | 24 | -0.067 / 0.067 / 0.067 |
| `parallel` | 2448 | -0.011 / 0.015 / 0.131 | 120 | +0.000 / 0.000 / 0.000 |

The signed mean is slightly negative: the arrival stamp is the start of the
substep in which the level set crosses zero, up to a quarter of a crossing
early, and the artificial viscosity of the default level set slows the curved
parts of the fronts by a few hundredths of a crossing. The necks fill on time;
in all three decks the line where the fronts meet lies on a cell face, where
the Godunov gradient of the merged level set is exactly one, so the merge
errors are smaller than a geometry with the meeting line through cell
centres would give (a few tenths of a crossing, still inside the tolerances).
The longer-running verification cases `Verification/Merging_Fires` and
`Verification/Junction_Fire` under `Exec/CanonicalTests/Fire` check the same
geometry over 80 to 150 s and three junction angles; the canonical case
`Fire_Behavior/Interacting_Fires` runs the three setups in a wind on grass.

Unit test: `erf_unit_tests --gtest_filter='PolygonIgnition.*'` (the key
takes a list, every file is read, two polylines stamp the distance to the
nearer line, two polygons burn their union).
