# FarsiteFrontUpdate

The two front updates of the FARSITE path, `erf.fire.farsite.front_update`,
against the Richards (1990) rates they are built on.

A 50 m radius disc of short grass (Anderson fuel model 1, 5.5 % moisture) burns
for 10 minutes in a uniform 3 m/s westerly on flat ground, with 10 m fire cells
and one-way coupling. The FARSITE normal speed is the head rate R times
`a cos + b |sin|` ahead of the wind and `b |sin| - c cos` behind, with `a = 1`,
`c = 0.2` and `b = 1.2 / (2 L/W)` from Anderson's length-to-width ratio. That is
the support function of the rectangle `[-c R, a R] x [-b R, b R]`, so the exact
burned region is the disc swept by that rectangle: head, back and flanks run at
`a R`, `c R` and `b R`, and the region stays convex.

- `inputs_front_cell`: the default. The unburned cells next to the burned
  region are the front cells; each burns when the rectangle grown from its
  burned neighbours' arrival times reaches it (a Hopf-Lax update), so the front
  moves one row at a time.
- `inputs_legacy`: the update before 2026-09. Every cell at or below
  `farsite.phi_threshold` stamped a target one cell ahead, and since phi was
  rebuilt as 0 on unburned cells, the first unburned row stamped too: two rows
  per cell of travel.

```bash
[MPIRUN="mpirun -np 2"] ./run_farsitefrontupdate.sh /path/to/erf_exec
```

`check_farsitefrontupdate.py` fits the head, back and flank positions over the
plotfiles and requires, for `front_cell`, the head within 3 % of `a R`, the
back and flanks within 10 % of `c R` and `b R`, and the burned area at least
0.95 of its convex hull; `legacy` is reported.

Measured on 2026-09-14 with Anderson's ratio carrying its second term (R = 0.971 m/s, L/W = 5.00; 46 s for both runs on 2 ranks):

| update | head [m/s] | back [m/s] | flanks [m/s] | area at 600 s [ha] | area / hull |
|---|---|---|---|---|---|
| Richards | 0.971 | 0.194 | 0.116 | | 1 |
| `front_cell` | 0.975 (+0.4 %) | 0.191 (-1.7 %) | 0.112 (-3.7 %) | 19.2 | 0.954 or more |
| `legacy` | 1.828 (1.88 a R) | 0.382 (1.97 c R) | 0.354 (3.04 b R) | 50.1 | 0.655 at 60 s |

`Tests/Unit/Fire/ERF_GTestFarsiteSpreadAccumulation.cpp` checks the same
rates cell by cell, and that the arrival times do not depend on the box
decomposition.
