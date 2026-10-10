# Polygon_Growth

Fires started from a square and from a cross, spreading at a constant rate. The
burned region at time t is every point within r = R t of the ignition polygon, so
outside it the arrival time is the distance to the polygon over R, and the area
is exact:

- the convex square follows Steiner's formula, A0 + P0 r + pi r^2;
- the cross has eight convex corners, each adding a quarter disc, and four
  inner corners, each losing the r x r square the two edge strips share:
  A0 + P0 r + (2 pi - 4) r^2, while r is under half the arm length.

The inner corners of the cross must stay sharp. That is the entropy condition of
the level-set equation, and a scheme that violates it, or smears it, rounds them.

```
python3 gen_polygons.py                                          # square.csv, cross.csv (committed)
MPIRUN="mpirun -np 2" ./run_polygon_growth.sh /path/to/erf_exec  # both decks, then the checks
```

## The case

400 x 400 m, still air, 2 m fire cells, a prescribed 1 m/s for 60 s. A 60 m
square and a plus sign 160 m across with 40 m arms, both centred at (200, 200),
stamped as closed polygons. `check_polygon_growth.py` compares the burned area
(sub-cell, from the signed distance) with the formula at every plotfile, the
arrival time over the grid outside the polygon, and for the cross the arrival
time along the diagonals out of the four inner corners, T = d / (sqrt(2) R).

## Expected Results

On two ranks, re-measured 2026-10-09:

| deck | area error at 10 s ... 60 s | arrival mean \|e\| / 95th pct | inner-corner diagonals |
|---|---|---|---|
| `square` | -0.07 ... -0.28 cell widths of perimeter | 0.179 / 0.506 | |
| `cross` | -0.06 ... -0.37 | 0.198 / 0.508 | mean e -0.043, 95th pct 0.140 |

The code before this one (2026-10) gives the same areas, arrival errors of
0.143 / 0.423 and 0.160 / 0.425 and inner corners at -0.218 / 0.250; the
numbers this README carried until then (-0.06 ... -0.21, 0.144 / 0.338)
predate both. The arrival time is now the crossing interpolated within the
substep, half a substep later on average than the start-of-substep stamp,
which is what moves the mean error from +0.09 to +0.18 crossings and takes
the early bias off the inner corners.

Arrival errors are in cell-crossing times (2 s). The area falls slowly behind
the formula, by up to a third of a cell times the perimeter at 60 s, the
artificial viscosity of the default level set on the rounded outer corners; the
inner corners arrive within 0.05 cell of the exact time, so they are kept
sharp, not rounded. The checks allow half a cell width of perimeter on the area and half a
cell (one at the 95th percentile) on the arrival time.
