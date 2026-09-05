# WUI_Subdivision

A wind-driven grass fire running from open wildland into three rows of
houses: the case that exercises the wildland-urban interface features
together and checks each against something independent. There is no field
dataset behind it. The head spread rate is checked against Rothermel's model,
the houses against the rule that they never burn, the subdivision against
the delay a row of obstacles must impose, defensible space against the
exposure it must remove, embers against the count that must land, and the
fuel against conservation.

```
python3 gen_wui.py                                   # rasters (already committed)
MPIRUN="mpirun -np 2" ./run_wui.sh /path/to/erf_exec  # five variants, then the checks
SKIP_RUN=1 ./run_wui.sh /path/to/erf_exec             # checks only
```

## The scenario

960 x 480 x 240 m, 10 m atmosphere cells, 5 m fire cells; a neutral 10 m/s
sounding entering at the west face and leaving at the east face, periodic in
y; Smagorinsky closure, MOST surface layer with z0 = 0.1 m. Fuel model 1
(short grass) at 6% moisture everywhere that is not a street or cleared, a
40 m ignition disc at x = 340 m. Three rows of eight 20 m square, 8 m tall
houses at x = 520, 600 and 680 m. Each 60 m of y holds a
house, a 20 m grass lane and a 20 m street; the streets run east-west, with
the wind, so they block nothing, and the lanes carry the fire through the
rows while the houses take a third of the width out of it. `gen_wui.py` writes the heightmap `houses_10m_96x48.txt` (nodal, the
ERF terrain text format; the fire's structure mask and the immersed forcing
read the same file), the two fuel maps and the sounding.

| variant | what is on |
|---|---|
| `wildland` | uniform grass, no houses, no spotting: the reference for the spread rate and the fuel budget |
| `wildland_spotting` | uniform grass with the seeded spotting of the subdivision variants: the reference for the delay through the subdivision, so that the houses are the only difference |
| `subdivision` | houses as non-burnable structures, level set extrapolated into them, streets non-burnable, exposure diagnostics every 10 s, seeded Albini spotting; passive atmosphere |
| `defensible` | the subdivision with a 30 m fuel break at x = 480-510 m and 10 m cleared around every house, which removes the lanes: only embers can reach a house |
| `coupled` | the subdivision with immersed-forcing houses in the atmosphere, the fire's wind from the open columns beside them, lagged heat coupling with the additive source in the open part of the columns; an atmosphere plotfile at 700 s intervals |

The runs are two boxes, so at most two ranks; each variant runs 2100 s.

## The checks (`check_wui.py`)

1. **Spread rate.** Head ROS along the centreline between x = 400 and 470 m
   of the `wildland` run, from the arrival-time field, within 15% of
   Rothermel's FM1 rate at 6% moisture and the midflame wind that the
   Andrews wind adjustment factor gives from 10 m/s at 6.1 m, capped at the
   model's 300 ft/min maximum effective wind for fine fuels. The reference is
   Rothermel (1972) written out in `check_wui.py`, independent of the code;
   it gives 0.2501 m/s where the model reports 0.2501505838 m/s.
2. **Fuel conservation.** Fuel consumed over the burned area of `wildland`
   equals the initial load over the burned area within 5% (cells the front
   reached in the last minute are still burning).
3. **Structures never burn.** No footprint cell has a negative level set or
   has lost fuel since the start, in all three subdivision variants.
4. **The subdivision delays the front.** `subdivision` reaches x = 780 m,
   beyond the last row, later than `wildland_spotting`; at least one house
   is reached by the front.
5. **Embers land.** At least one brand lands on a footprint in `subdivision`;
   the seed is fixed, so the count is reproducible.
6. **Defensible space works.** Fewer houses reached by the front and a lower
   maximum heat load at a house in `defensible` than in `subdivision`.
7. **The coupled run stands up.** No NaN, a plume (maximum w above 0.5 m/s in
   the last atmosphere plotfile), and the fire reaches x = 780 m.

The exposure columns are also printed against the threshold usually quoted
for the ignition of wood by radiation, about 20 kW/m² (Cohen 2004), as a
reading rather than a check: a fireline intensity per metre of front is not
a flux on a wall, and the wall energy balance that would turn it into one is
future work.

## Reference results

Two ranks, 2100 s, level set with the default hybrid WENO5-Z/first-order
derivatives and the near-front artificial viscosity of 0.1
(`erf.fire.levelset.gradient = weno5z_front`, `eps_visc_front = 0.1`,
2026-09-05):

| variant            | x = 780 m at [s] | burned cells | houses reached | peak intensity [kW/m] | max heat load [MJ/m²] | ember landings |
|--------------------|-----------------:|-------------:|---------------:|----------------------:|----------------------:|---------------:|
| wildland           |             1610 |         4352 |              - |                     - |                     - |              - |
| wildland_spotting  |             1255 |         5986 |              - |                     - |                     - |              - |
| subdivision        |             1575 |         4837 |          12/24 |                   773 |                  3.11 |             30 |
| defensible         |            never |         1752 |           0/24 |                     0 |                  0.00 |              0 |
| coupled            |             1618 |         5464 |          15/24 |                   773 |                  3.11 |             10 |

The wildland head moves at 0.250 m/s between x = 400 and 470 m against
Rothermel's 0.2501 m/s for FM1 at 6% moisture and the 300 ft/min wind cap;
the fuel consumed over its burned area is within 1.5% of the initial load.
No footprint cell burns or loses fuel in any variant. The subdivision's first
contacts are at 383, 669 and 953 s for the three rows; the coupled run's at
371, 659 and 832 s, with a plume of 20 m/s at the end. `wui_spread.png` in
the docs figures shows the four arrival-time maps.

With a single viscosity of 0.4 everywhere (`eps_visc_front = -1`) the same
runs gave x = 780 m at 1613 / 1181 / 1352 / never / 1855 s and 4352 / 5856
/ 4916 / 1660 / 5069 burned cells: the head rate is the same and the lower
near-front viscosity lets the flanks spread a little more (about 7% more
area in the wildland run). With first-order derivatives everywhere
(`gradient = upwind`) they gave 1612 / 1176 / 1403 / never / 1441 s and
5242 / 6676 / 5470 / 2210 / 5981 cells: the flanks are wider still, so the
first-order fire burns about a fifth more area and, in the coupled run,
gets through the lanes sooner on that wider front. The spotting runs
differ between the three by more than the flanks alone because the ember
launches are sampled from the burned cells.

## What this case found

**The fire read unfilled ghost cells outside non-periodic domain faces.** The
fire grid inherits the atmosphere's periodicity, and with this case's inflow
and outflow in x nothing filled the level set's ghost cells beyond the west
and east faces: the gradient stencil read whatever memory the allocator had left
there. Whether that was benign depended on the box layout. With one box per
rank the case ran; with two boxes on one rank, or two per rank on four ranks,
the level set blew up at the inflow face on the third step and the fire
"burned" twenty hectares in fifty seconds. Every fire-grid exchange now goes
through `fire_fill_boundary` (`Source/Fire/ERF_FireGrid.H`), which
extrapolates the nearest interior value into the ghost cells of a
non-periodic face, and five box layouts on one to four ranks give the same
fire to roundoff. Periodic cases are unchanged.

**The perimeter statistic depended on the box layout.** `perimeter_km` in the
stats CSV, and the ellipse axes derived from it, differed between a two-box
and an eight-box decomposition of the same run while the fields agreed to
roundoff: the edge crossings were counted inside each box only, so the edges
between two boxes were never counted. The neighbour tests now read a copy of
the arrival time with a filled ghost cell and stop at the domain instead of
the box; the three layouts give the same perimeter, and a single-box run is
unchanged.

**The level set ran ahead of its own rate of spread.** With `fire_ros` at
0.250 m/s everywhere, the head of the `wildland` fire moved at 0.250 m/s for
the first hundred metres and then at 0.29-0.33 m/s, and dropping the
artificial viscosity brought the overshoot forward rather than removing it.
The gradient norm in `Source/Fire/ERF_NumericalSchemes.H` took whichever
one-sided difference was larger in magnitude, regardless of sign. That is not
the Godunov scheme: wherever the level set is convex ahead of the front the
downwind slope wins, the update becomes anti-dissipative and the front runs
ahead of R. The reinitialisation had already been fixed for the same defect.
The norm now uses the Osher-Sethian choice for an expanding front, the
backward difference where positive and the forward one where negative, and
the head speed is 0.250 m/s over every 40 m segment from x = 400 to 640 m,
where the old branch gave 0.25 to 0.45 m/s. The scheme was also described, in the
code, the docs and the level-set canonical test, as a fifth-order WENO-Z
reconstruction; the reconstruction was defined but never called, so the
description now says what the code does.

**A 10 m gap in a 10 m atmosphere grid is a wall.** The first layout had
10 m lanes and streets between 20 m houses. The immersed forcing blanks every
atmosphere cell that touches a house node, so each row became a continuous
8 m fence across the domain: the lowest-layer wind behind x = 510 m fell to
a few centimetres per second in the lanes as well as at the houses, the
fire's open-column wind followed it, and the coupled fire crawled through the
lanes at the no-wind rate and never cleared the first row. The lanes and
streets are now 20 m, which leaves one open atmosphere cell in each. The
passive variants do not see the difference because their wind ignores the
houses.

**The Python Rothermel in `Unit_Tests/test_rothermel_unit.py` is not a
reference.** Its wind coefficient is `7.47 exp(-0.8711 sigma^-0.55)` where
Rothermel's is `7.47 exp(-0.133 sigma^0.55)`, and it caps the wind factor at
0.9 I_R, so for short grass at any wind it returns 17 m/s. The check here
carries its own Rothermel; the unit test is left as it is.

## References

- Rothermel, R. C. (1972). A mathematical model for predicting fire spread in wildland fuels. USDA Forest Service Research Paper INT-115.
- Andrews, P. L. (2012). Modeling wind adjustment factor and midflame wind speed for Rothermel's surface fire spread model. RMRS-GTR-266.
- Albini, F. A. (1983). Potential spotting distance from wind-driven surface fires. USDA Forest Service Research Paper INT-309.
- Cohen, J. D. (2004). Relating flame radiation to home ignition using modeling and experimental crown fires. Canadian Journal of Forest Research, 34, 1616-1626.
- NFPA 1144 (2018). Standard for Reducing Structure Ignition Hazards from Wildland Fire.
