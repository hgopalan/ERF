# SEB/BuildingSet

Canonical case for the balance on a set of buildings from a height map:
mutual shadowing, the building part of the view fractions, several
materials, the per-building report, and the cost of the face list, with
the immersed forcing's wall law snapped to whole cells and the wall
function beyond neutral switched on.

```
./run_buildingset.sh /path/to/erf_exec                   # NP=4, about two hours
python3 plot_buildingset.py ibseb_set.csv faces/set --plotfile plt43200
```

## The scenario

Four buildings on a 480 m periodic domain, 320 m deep, at 10 m, from a
nodal height map (`make_buildings.py`), numbered as the balance numbers
them (scan order):

1. a 60 m concrete slab, 30 x 60 m, at x = 150-180, y = 200-260;
2. a 20 m timber block, 40 x 40 m, north of the slab at y = 300-340;
3. a 40 m brick cube, 40 x 40 m, east of the slab at x = 230-270, in
   its morning shadow;
4. a 20 m timber block off on its own at x = 320-360, y = 150-190.

Three materials from `materials.csv` by building; a 3 m/s westerly,
neutral at 300 K with a MOST ground, held by nudging the plane-mean u and
v above 100 m toward the sounding (`erf.nudging_from_input_sounding`, 10
minute time scale; theta and moisture are not nudged); Boulder on the June
solstice from 05:00 solar time for six hours with the prescribed clear-sky
provider and a gray sky; the convective velocity scale (bulk Richardson
depth) and the stability functions on; `erf.if_snap_partial_cells = true`,
which a height-map set needs
(`Exec/RegTests/ImmersedForcingTest/PartialCells`).

## What is checked

1. Four buildings with the expected face counts and materials.
2. The balance residual stays below 1e-3 W/m2 on every building.
3. Mutual shadowing: at sunrise the slab is the most shaded building,
   because the 40 m cube east of it throws its shadow onto the slab's
   east wall (the slab shades the cube in the afternoon, outside this
   run), and it clears by late morning; the 20 m blocks are free of
   shadow by then. Every building also shades its own stepped rim at low
   sun.
4. The building view fraction: the facing walls of the slab and the cube
   see more building than the far block's walls (every wall sees some,
   its own stepped rim and, across the periodic boundary, the others).
5. Materials show: the timber roofs end warmer than the concrete roof.
6. The wall function beyond neutral is active: w* positive on the sunlit
   faces, most roofs unstable by the end (the shaded rim roofs may be
   slightly stable).
7. Timing: faces per rank and the cost of the balance per step, reported
   for estimating a city-scale case.
8. The wind is held: the horizontal-mean wind between the tallest roof
   and mid-depth stays between 2 and 4 m/s, with a fitted trend under
   0.2 m/s per hour over the second half of the run.

## What happens through the morning

- **Sunrise (04:35).** The run starts at 05:00 with the sun 9 degrees up in
  the east-north-east. The cube's shadow lies across the slab's east wall,
  so the slab is the most shaded building (17 percent of its faces, 6
  percent by late morning); the 20 m blocks shade only their own stepped
  rims, and are free of shadow after 10:30. The cube keeps a little
  self-shadow at 11:00 from its 20 m rim step.
- **Materials.** The two timber blocks (light cladding, 15 cm) warm
  fastest and end at 324.6 and 324.8 K on their roofs; the brick cube
  (25 cm) reaches 321.2 K and the concrete slab (30 cm) 317.6 K. The
  timber blocks' mean skin runs 1 K above the slab's from 05:46, and the
  two identical blocks stay within 0.74 K of each other all morning
  (0.18 K at 11:00) although one sits north of the slab and one on its own.
  They shed 189 and 185 W/m2 of sensible heat at 11:00, the cube 87 and
  the slab 64 W/m2.
- **View fractions.** The slab's east wall sees 28 percent building, the
  cube's west wall 48 percent (the taller slab fills its view), the far
  block's walls 18 percent (their own rim and the periodic neighbours).
- **The wall function.** The convective scale is 0.15 to 1.37 m/s on the
  268 sunlit faces at 11:00, and 94 percent of the roofs are unstable with
  their own Obukhov length by then; a few shaded rim roofs run stable.
- **Cost.** 0.21 ms per step on the slowest rank for 157 faces, 2 ms to
  build the list and sample the view fractions; the balance is a
  negligible part of the 2 h 15 min the six hours took on four ranks
  (measured with other builds running on the same machine).
- **The atmosphere.** The height-map set runs with the immersed forcing
  snapped to whole cells and the implicit drag; without the snap the slab
  drives the density negative at two minutes (see
  `Exec/RegTests/ImmersedForcingTest/PartialCells`), and the balance's
  residual stays at 3e-8 W/m2 throughout.
- **The wind.** The nudging holds the mean wind between 70 and 150 m at
  2.5 to 3.0 m/s (2.81 m/s at 11:00, trend -0.01 m/s per hour); at 25 m it
  settles near 2.3 to 2.4 m/s within the first hour. The bulk Richardson
  depth that sets the convective scale stays between 215 and 320 m
  (245 m at 11:00). Without a driver, in the earlier 160 m closed box, the
  westerly decayed to 0.2 m/s by 08:00, the depth shrank to 45 m and the
  timber roofs ran 6 K hotter (331 K).

## Reference output (4 ranks)

```
== building set, 6 h from 05:00 (4 ranks)
  four buildings with their materials: PASS (building 1: 172 faces, material 1, building 2: 72 faces, material 3, building 3: 116 faces, material 2, building 4: 72 faces, material 3)
  balance residual on every building all morning: PASS (max 3.4e-08 W/m2 over 1444 rows)
  at sunrise the slab is the most shaded building (the cube's shadow on its east wall) and clears by late morning: PASS (slab 0.168 at 05:30-06:10 vs 0.061 after 10:30; others early 0.108, 0.084, 0.112)
  the 20 m blocks are free of shadow by late morning: PASS (north block 0.000, far block 0.000 after 10:30)
  the facing walls of the slab and the cube see more building than the far block's walls: PASS (slab east 0.28, cube west 0.48, far block walls 0.179)
  timber roofs end warmer than the concrete roof (light cladding warms faster): PASS (roof means at 11.0 h: slab 317.6, north block 324.6, cube 321.2, far block 324.8 K)
  w* positive on the sunlit faces and most roofs unstable by the end (the shaded rim roofs may be stable): PASS (w* 0.15-1.37 m/s on 268 sunlit faces, 94 % of 140 roofs unstable)
  cost line reported: PASS (lev=0 advance_ms_per_step_max=0.2096 faces_per_rank_max=157 ranks=4 init_s=0.002196)
  the nudging holds the westerly above the roofs all morning, steady: PASS (mean u at 70-150 m 2.49-3.00 m/s over 37 profiles, 2.81 m/s at 11.0 h, trend -0.01 m/s per hour over the second half)
building set: PASS
ALL PASS
```
