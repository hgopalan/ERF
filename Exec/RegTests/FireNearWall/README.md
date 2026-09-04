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
noib                     0   18255   0.7710       -   4.2877      -    27   112 |    53    47     - |     -    55   103
noib_mask                0   17583   0.7710       0   4.2877      -    27    88 |    53    47     - |     -   121   131
noib_mask_wall           0   16859   0.7710       0   4.2877      -    27   112 |    53    47     - |     -   124   135
ib_mask                  0   13404   1.0155       0   6.3440      -    39   101 |    55    61     - |     -   162   174
ib_mask_wall             0   13405   1.0155       0   6.3440      -    40   101 |    55    61     - |     -   162   175
ib_mask_wind             0   13433   1.0155       0   6.3440      -    39   100 |    55    60     - |     -   159   172
ib_mask_wall_wind        0   13433   1.0155       0   6.3440      -    39   100 |    55    60     - |     -   159   172
```

Four ranks, 240 s. Things to read out of it.

**The defaults are untouched.** `noib` and `noib_mask` reproduce the
`balbi_noib` and `balbi_noib_mask` rows of FireHybridObstacles to the digit,
and the FireRestart suite is unchanged.

**The wall extrapolation removes the wall effect exactly.** With the mask
alone the flank reaches the third box at 88 s against 112 s with nothing in
the way; with the extrapolation it arrives at 112 s, the unmasked flank
rate, while the head-fire arrival `u2` and the gap probes `g1`, `g2` do not
move and nothing burns inside the mask. The burned area drops from 17583 to
16859 cells, which is the over-spread along the walls that the effect had
added, and the downwind faces are reached a few seconds later for the same
reason.

**With buildings in the atmosphere the wall option changes almost nothing**
(13404 against 13405 cells, `u3` 101 s in both). The immersed forcing slows
the wind along the walls, so the flank there was never running on the
level-set artefact. The open-column wind option makes a small difference the
other way: the fire cells beside a wall now read the open columns only, a
slightly stronger wind than the blend with the in-building velocity, and the
fire reaches the downwind faces a few seconds earlier (13433 cells). The
largest reference wind on the grid is the same in every row because it sits
in the gaps, away from the walls.
