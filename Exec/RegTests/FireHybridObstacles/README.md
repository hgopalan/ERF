# FireHybridObstacles

A wind-driven grass fire approaching three box obstacles, run with each
rate-of-spread model and with or without the boxes present in the atmosphere.
It exercises the hybrid `structure` selector, the non-burnable structure
mask (`erf.fire.structures.enable`), the arrival-time probes, and the
interaction between the fire model's wind extraction and immersed-forcing
buildings. It is a demonstration deck, not a validation.

## The scenario

320 x 160 x 160 m at 5 m, fire grid at 1.25 m (`grid_ratio = 4`; the 32-cell
atmosphere boxes must divide by the ratio), neutral 8 m/s sounding, FM1 short
grass at 6% moisture, passive coupling, direction-dependent level set. Three
20 m wide, 10 m tall boxes stand across the wind at x = 180-200 m with gaps of
30 m and 15 m between them, and a 5 m non-burnable street (fuel code 0) runs
along the wind at y = 125-130 m. The fire is ignited as a 15 m disc at
x = 130 m, so the upwind faces of the boxes are 35 m from its edge, and runs
for 240 s. `make_inputs.py` writes the building heightmap and the fuel map;
both are committed.

Nine `erf.fire.probes` sit at the upwind face, the gap midpoints and the
downwind face of the boxes (u1-u3, g1-g3, d1-d3 in the table; u2 and d2 are
the middle box, on the ignition centreline).

Three models cross two atmosphere configurations:

- `rothermel`, `balbi` (2020 form, reference-height wind) and `hybrid`
  (Rothermel primary, Balbi secondary, `selector = structure`, secondary
  within 10 m of a box).
- `*_noib`: flat ground, the boxes exist only in the fire grid's structure
  field. `*_ib`: the same heightmap also drives immersed-forcing buildings,
  so the wind the fire sees is blocked and channelled by the boxes.
- `*_mask`: the same six decks with `erf.fire.structures.enable = true` and
  fuel code 0 listed as non-burnable, so the boxes and the street are
  obstacles the fire has to go around.

```
./run_obstacles.sh /path/to/erf_exec
MPIRUN="mpirun -np 8" ./run_obstacles.sh /path/to/erf_exec   # parallel launcher
SKIP_RUN=1 ./run_obstacles.sh x     # rebuild the table from existing logs
```

Each variant takes seven to ten minutes on one MPI rank, so the twelve take
about an hour and a half in sequence; `MPIRUN` and running variants side by
side bring that down.

## Reference results

Burned fire cells after 240 s, the head ROS on the last step, the number of
cells the hybrid hands to Balbi, the number of burned cells inside the
non-burnable mask (`*_mask` rows, must be 0), and the arrival time [s] at
each probe ("-" means the probe never burned). The face probes sit 1.5 m
outside the sampled footprint: the heightmap is on 5 m nodes and is sampled
by nearest node, so a box spans 177.5-202.5 m on the fire grid, not 180-200.

| variant | cells | max ROS (m/s) | sec | in mask | u1 u2 u3 | g1 g2 g3 | d1 d2 d3 |
|---|---|---|---|---|---|---|---|
| `rothermel_noib` | 2256 | 0.250 | - | - | - 124 - | - - - | - 239 - |
| `balbi_noib` | 18255 | 0.771 | - | - | - 27 112 | 53 47 - | - 55 103 |
| `hybrid_noib` | 4159 | 0.771 | 3396 | - | - 98 219 | 213 124 - | - 131 216 |
| `rothermel_ib` | 1673 | 0.250 | - | - | - - - | - - - | - - - |
| `balbi_ib` | 14320 | 1.016 | - | - | - 39 101 | 55 61 - | - 161 172 |
| `hybrid_ib` | 3056 | 0.936 | 3396 | - | - 122 - | 206 174 - | - - - |
| `rothermel_noib_mask` | 1824 | 0.250 | - | 0 | - 124 - | - - - | - - - |
| `balbi_noib_mask` | 17583 | 0.771 | - | 0 | - 27 88 | 53 47 - | - 121 131 |
| `hybrid_noib_mask` | 3384 | 0.771 | 3396 | 0 | - 98 199 | 213 124 - | - 200 210 |
| `rothermel_ib_mask` | 1673 | 0.250 | - | 0 | - - - | - - - | - - - |
| `balbi_ib_mask` | 13404 | 1.016 | - | 0 | - 39 101 | 55 61 - | - 162 174 |
| `hybrid_ib_mask` | 2528 | 0.936 | 3396 | 0 | - 122 - | 206 174 - | - - - |

Things to read out of it.

**Without the mask the fire burns through the boxes.** Rothermel reaches the
upwind face of the middle box at 124 s (35 m at 0.25 m/s from the disc edge,
plus the 1.5 m to the probe) and its downwind face at 239 s: the front
crosses the footprint as if it were grass. That is the behaviour the
`*_mask` rows remove.

**With the mask, nothing burns inside a footprint** (`in mask` is 0 in every
row, checked every step), the burned area is smaller in every pair where the
fire reaches a box, and the downwind faces are reached later, through the
gaps, or not at all. Rothermel never reaches its downwind face in 240 s;
Balbi reaches it at 121 s instead of 55 s, arriving from the gap it passed at
47 s.

**The mask changes nothing it should not.** Arrival at the upwind faces
(`u2`) and at the gap midpoints (`g1`, `g2`) is identical with and without
the mask in every pair, and `rothermel_ib`, where the immersed-forcing drag
keeps the fire from reaching the boxes at all, is identical to the cell.

**The structure selector changes only the last 10 m.** The hybrid reaches
the middle box at 98 s instead of 124 s because it runs at the Balbi rate
within 10 m of a box; `sec` is the number of fire cells inside that band.

**Balbi with the reference wind is fast and decays.** Its head rate starts
at 1.22 m/s on the uniform 8 m/s initial wind and settles to 0.77 m/s as the
surface layer spins up.

**Immersed forcing slows the approach and speeds the gaps.** With the boxes
in the atmosphere the Rothermel fire never reaches them and burns 26% fewer
cells; the hybrid's arrival at the middle box moves from 98 s to 122 s. Balbi
ends with a higher head rate (1.02 against 0.77 m/s) because the reference
wind it consumes is channelled between the boxes.

**One thing to keep an eye on.** In the two flat-ground level-set rows where
the fire runs along a wall, the flank reaches the upwind face of the third
box earlier with the mask than without (`u3`: 88 s against 112 s for Balbi,
199 s against 219 s for the hybrid). Cells next to a masked wall keep
evaluating a head-fire rate along their front normal, which points into the
wall, so their level-set value keeps falling although the front cannot move;
between reinitialisations that steepens the gradient along the wall and
pushes the flank faster than the flank rate. It is a level-set wall effect of
about 20% on the lateral spread, does not occur in the immersed-forcing rows
(`u3` 101 s in both), and belongs with the near-wall work that follows.

## What this is not

The `*_mask` rows stop the fire at the footprints; the unmasked rows are kept
to show the difference. Nothing here is compared with observations, and the
two-way heat coupling is off (`coupling_type = passive`), so the atmosphere
does not respond to the fire. Heat placement around the footprints and the
wind extraction next to walls are separate work.
