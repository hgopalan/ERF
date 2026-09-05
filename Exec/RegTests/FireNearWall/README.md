# FireNearWall

How the fire behaves next to a masked building: the level set along a wall
and the wind the fire reads beside one. The FireHybridObstacles grass fire
is run with Balbi 2020 on the reference wind, and the variants switch on the
non-burnable mask, the level-set wall extrapolation
(`erf.fire.levelset.wall_extrapolate`), immersed-forcing buildings, and the
open-column wind weights (`erf.fire.structures.wind_open_columns`). It is a
demonstration deck, not a validation.

```
./run_nearwall.sh /path/to/erf_exec
```

`MPIRUN="mpirun -np 4" ./run_nearwall.sh ...` runs each variant in parallel;
`SKIP_RUN=1` only re-tabulates existing logs.

## The scenario

320 x 160 x 160 m at 5 m, fire grid at 1.25 m, neutral 8 m/s sounding with
a MOST surface layer, FM1 short grass at 6% moisture, passive coupling,
direction-dependent level set, 240 s. Three 20 m wide, 10 m tall boxes stand
across the wind at x = 180-200 m; the fire is ignited as a 15 m disc on the
middle box's centreline, 35 m upwind of its face. The probe that matters is
`u3`, the upwind face of the third box: the fire reaches it by running its
flank along the wall of the middle box, so its arrival time measures the
lateral spread next to a wall. `u2` is the head-fire arrival at the middle
box and `g1`, `g2` the gap midpoints.

## The variants

| deck | mask | wall extrapolation | buildings in atmosphere | open-column wind |
| --- | --- | --- | --- | --- |
| `noib` | off | off | none | off |
| `noib_mask` | on | off | none | off |
| `noib_mask_wall` | on | on | none | off |
| `ib_mask` | on | off | immersed forcing | off |
| `ib_mask_wall` | on | on | immersed forcing | off |
| `ib_mask_wind` | on | off | immersed forcing | on |
| `ib_mask_wall_wind` | on | on | immersed forcing | on |

## What the two options do

**Wall extrapolation.** A masked cell keeps the level-set value it had
before the front arrived, so once its open neighbour burns the two differ by
many metres across one cell. The Godunov norm takes the larger one-sided
difference, which is then the one into the wall, and the cells along a wall
burn down far faster than the spread rate; the central-difference front
normal points into the wall too, so the directional model evaluates a
head-fire rate there. With the option on, a masked stencil point takes the
centre cell's value in the gradient, the Laplacian, the front normal and the
reinitialisation, which leaves masked cells alone. The wall becomes a
zero-gradient boundary of the distance function and the flank runs at the
flank rate.

**Open-column wind.** With immersed-forcing buildings the atmospheric
columns inside a footprint carry the relaxed in-building velocity, and the
bilinear blend hands a share of it to every fire cell within one
atmospheric cell of the wall. With the option on, each column's bilinear
weight is multiplied by its open fraction whenever its sampled roof is
above the cell's wind height, and the four weights are renormalised.

## What to expect

- `noib_mask` reproduces the FireHybridObstacles `balbi_noib_mask` row and
  `noib` its `balbi_noib` row: the defaults are untouched.
- `noib_mask_wall` should move `u3` back toward the unmasked reference
  while `u2`, `g1` and `g2`, which do not involve a wall, stay where they
  are and nothing burns inside the mask.
- The immersed-forcing rows separate the two effects: the wall option acts
  on the level set only, the wind option on the wind the fire reads.

## Reference table

```
variant               exit   cells  max_ROS in_mask max_wind  u1 u2 u3 | g1 g2 g3 | d1 d2 d3
-------------------- ----- ------- -------- ------- --------  ------------------------------------------
noib                     0   17899   0.7710       -   4.2877      -    27   113 |    55    48     - |     -    55   105
noib_mask                0   16121   0.7710       0   4.2877      -    27   113 |    55    49     - |     -   135   144
noib_mask_wall           0   16277   0.7710       0   4.2877      -    27   113 |    55    49     - |     -   129   139
ib_mask                  0   12839   1.0155       0   6.3440      -    40   102 |    58    65     - |     -   172   185
ib_mask_wall             0   12834   1.0155       0   6.3440      -    40   102 |    58    65     - |     -   169   183
ib_mask_wind             0   12882   1.0155       0   6.3440      -    39   101 |    58    64     - |     -   168   182
ib_mask_wall_wind        0   12876   1.0155       0   6.3440      -    39   101 |    58    64     - |     -   166   180
```

Four ranks, 240 s, with the default hybrid WENO5-Z/first-order level set
(`erf.fire.levelset.gradient = weno5z_front`, 2026-09-05). Things to read
out of it.

**The wall effect this suite was built to expose is gone at its root.**
The suite was written when the flank along a masked wall reached the third
box at 88 s against 112 s with nothing in the way, and
`levelset.wall_extrapolate` was the cure. The cause was the gradient norm
taking the larger-magnitude one-sided difference, which at a wall is the
one into the untouched mask value; that branch was replaced by the
Osher-Sethian choice in the WUI validation work (#353), and since then
`noib_mask` reaches `u3` at 113 s with or without the extrapolation, the
same as `noib`. The option stays: it still keeps a masked cell's value out
of every stencil, which moves the downwind-face arrivals by a few seconds
(`d2` 135 against 129 s) and the burned area by 1% (16121 against 16277
cells), and next to a wall the WENO stencil falls back to the first-order
difference through it.

**The mask itself costs area, as it should.** `noib` burns 17899 cells and
`noib_mask` 16121: the three boxes and the streets are out of the fuel, and
the front has to go round them (`d2`, `d3` later by 80 and 39 s).

**With buildings in the atmosphere the wind does the rest.** The immersed
forcing slows the wind along the walls and speeds it in the gaps (largest
reference wind 6.34 against 4.29 m/s), the fire reaches the head probe
`u2` later (40 against 27 s) and the downwind faces much later (172
against 135 s), and burns 12839 cells. The open-column wind option makes
a small difference the other way: the fire cells beside a wall read the
open columns only, a slightly stronger wind than the blend with the
in-building velocity, and the downwind faces are reached a few seconds
earlier (12882 cells).

Before the hybrid scheme, with first-order derivatives everywhere and the
Godunov branch already fixed, the rows burned about 8% more cells at the
same arrival times; the table before #353 (first order, old branch) is in
the history of this file.

