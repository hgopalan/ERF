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
MPIRUN="mpirun -np 2" ./run_wui.sh /path/to/erf_exec  # six variants, then the checks
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
| `ignition` | the subdivision with structure ignition on (`erf.fire.structures.ignition.*`, 2026-09-16) with the thresholds at a third of the documented defaults: houses ignite from a 2 MJ/m² heat load in their wall band, 15 landed brands, or a minute above 300 kW/m; burn along the EN 1991-1-2 dwelling curve (250 kW/m², 780 MJ/m², 600 s to the peak, the defaults); radiate 30% of it within 100 m and launch brands; passive atmosphere. At the documented thresholds no house ignites here (see the results) |

The runs are two boxes, so at most two ranks; each variant runs 2100 s.

## The checks (`check_wui.py`)

1. **Spread rate.** Head ROS along the centreline between x = 400 and 470 m
   of the `wildland` run, from the arrival-time field, no more than 15% above
   Rothermel's FM1 rate at 6% moisture and no lower than the Wulff tip of the
   projection formula at the same wind. The wind is the effective (midflame)
   wind the fire samples on the head's path (y = 240 m, x = 400 to 470 m), read
   from the fire plotfile nearest the middle of the arrival window, and bounded
   by Rothermel's limit 0.9 I_R. The reference is Rothermel (1972) written out
   in `check_wui.py`, independent of the code, and the model's own rate is
   never used. The wind is the model's, so the checker also holds its ratio
   to the reference wind to the Andrews WAF (0.362104) at every cell of the
   path: a defect in the rate kernel or in the WAF moves the head but not the
   bracket. A doubled head fails the check, and so does a halved WAF. The
   lower bound is loose: it fails only a head below 36 % of the Rothermel
   rate, 2.65 times slower than the measured head, so a halved rate passes.
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
8. **Houses ignite and spread.** In `ignition` at least one house ignites,
   a later one ignites after the first, and at least one ignited house had
   not been reached by the front when it ignited (house-to-house spread by
   radiation or brands); the plotfile state of every ignited house agrees
   with the CSV; no footprint cell burns or loses fuel (a burning house is a
   heat source, not a burning fuel cell).

The exposure columns are also printed against the threshold usually quoted
for the ignition of wood by radiation, about 20 kW/m² (Cohen 2004), as a
reading rather than a check: a fireline intensity per metre of front is not
a flux on a wall, and the wall energy balance that would turn it into one is
future work.

## Reference results

Two ranks, 2100 s, level set with the default hybrid WENO5-Z/first-order
derivatives and the near-front artificial viscosity of 0.1
(`erf.fire.levelset.gradient = weno5z_front`, `eps_visc_front = 0.1`). All six
variants were re-measured 2026-10-09 on the validated code, where the
Rothermel wind limit (0.9 I_R) replaces the 300 ft/min fuel-class cap and a
burning cell's Byram intensity uses its full fuel load:

| variant            | x = 780 m at [s] | burned cells | houses reached | peak intensity [kW/m] | max heat load [MJ/m²] | ember landings on footprints |
|--------------------|-----------------:|-------------:|---------------:|----------------------:|----------------------:|-----------------------------:|
| wildland           |             1025 |         3452 |              - |                     - |                     - |                            - |
| wildland_spotting  |              571 |         5074 |              - |                     - |                     - |                            - |
| subdivision        |             1318 |         4025 |          12/24 |                  4057 |                  3.09 |                            0 |
| defensible         |             1241 |         2995 |           0/24 |                     0 |                  0.00 |                            0 |
| coupled            |             1498 |         4818 |          13/24 |                  2362 |                  3.09 |                            8 |
| ignition           |              661 |         5427 |          13/24 |                  4057 |                114.22 |                           40 |

What the table shows, and why it differs from the 2026-09 table:

- **Brands now fly as far as the fuel allows.** A brand rises with the Byram
  intensity of its cell's full load, so in this 10 m/s grass fire every brand
  reaches FM1's 200 m landing cap (the run logs give 200 m for every launch:
  20 in `subdivision`, 10 in `defensible`, 29 in `coupled`, 59 in
  `ignition`). The front therefore jumps ahead of itself: `wildland_spotting`
  reaches x = 780 m at 571 s instead of 1255 s, and in `defensible` brands
  carry the fire across the fuel break, so the front reaches x = 780 m at
  1241 s (it never did before) while still reaching no house.
- **A brand lands on a footprint only by chance.** Each brand lands on the
  200 m-shifted copy of its launch band, so a footprint is hit only when it
  sits exactly there: 0 landings in `subdivision`, 8 in `coupled`, 40 in
  `ignition` (where the burning houses launch brands too). `check_wui.py`
  therefore requires a footprint landing over the three structure variants,
  not in `subdivision` alone (it had 47 there in 2026-09).
- **The first contact comes from a brand.** At about 4 s the ignition disc
  throws one brand 200 m to the wall of house 4 (x = 530 m, y = 190 m); the
  spot fire burns 3 of its 21 wall cells at 4057 kW/m. That is the first
  contact of row 1 at 4 s in `subdivision` and `coupled` (rows 2 and 3 at 48
  and 275 s in `subdivision`, 253 and 518 s in `coupled`). The peak intensity
  is the full-load Byram intensity of such a spot fire (773 kW/m in 2026-09).

With the thresholds at a third of the defaults (the committed
`inputs_ignition`), 20 of 24 houses ignite, all by the heat-load criterion,
7, 7 and 6 in the three rows. The first is house 4 at 8.25 s, when the heat
load of the spot fire at its wall passes 2 MJ/m²; the last ignites at 1954 s,
and none has burned out by 2100 s. Thirteen houses ignited before the front
reached their wall band, from a neighbour's radiation or from a spot fire:
house-to-house spread at the scale of the subdivision. The largest wall heat
load rises to 114 MJ/m² under the radiation of burning neighbours. In 2026-09
the same deck ignited 19 houses, the first at 250 s, also from a spot fire at
its wall; the earlier start now comes from the disc's first brand. The
2026-09 binary was not rerun on this deck: the attribution above comes from
the run logs and the changes listed in the fire validation, not from a
side-by-side run.

At the documented default thresholds (6 MJ/m², 50 brands, 1000 kW/m for
60 s) no house ignited in 2100 s in the 2026-09 run (not re-measured). A
short-grass fire does not reach those placeholders by radiation at the ground
next to a wall, which is in line with the field finding that homes in grass
fires ignite from embers and adjacent fuels rather than from the flame front
(Cohen 2004); it says nothing about the thresholds' calibration. The fire
itself is unchanged by an ignition: no footprint cell burns or loses fuel in
any variant.

The wildland head moves at 0.599 m/s between x = 400 and 470 m. Rothermel's
rate for FM1 at 6% moisture is 1.385 m/s at the sounding's 10 m/s (the run's
value at t = 0, below the 0.9 I_R limit of 743 ft/min) and 0.632 m/s at the
2.455 m/s midflame wind on the head's path at t = 100 s, once the surface
layer has slowed the 6.1 m wind (0.632 to 0.642 m/s for every plotfile from
100 to 700 s). The head is 95 % of that on average, between the Wulff tip of
the projection formula at that wind (0.226 m/s, 36 %) and the Rothermel
rate. It is not steady. The window opens at 19 s, while the wind is still
falling from its start-up value: at 20 s the log's largest and mean rates
are both 1.005 m/s, Rothermel's rate at the wind of that moment. From there
the head slows the whole way toward the Wulff tip. Its local speed along y = 240 m (over
the 10 m centred on each point, from the arrival times) against Rothermel at
the local wind of the 100 s plotfile is 1.01 against 0.67 m/s at x = 402.5 m,
0.72 against 0.65 at 422.5 m, 0.55 against 0.62 at 442.5 m, 0.48 against
0.60 at 462.5 m, and 0.25 against 0.47 at 642.5 m, where the tip is 0.19.
That slowing of a curved front's head comes from
the default directional coupling that FireAdvectiveWindCoupling documents
(`erf.fire.directional_wind_coupling = advective` holds it at the model's
rate). The 0.250 m/s of the 2026-09 table was the 300 ft/min cap, which any
wind above 1.52 m/s reached. The fuel consumed over the burned area is within
0.5% of the initial load (14251 against 14326 kg).
The coupled run ends with a plume of 17 m/s (20 m/s in 2026-09). `wui_spread.png` in
the docs figures shows the four arrival-time maps; it predates the fuel map
fix below.

The 2026-09 table was regenerated on 2026-09-11 after the fuel map fix
(hgopalan/ERF#411). The reader had put the first row of each fuel map on the
south edge, which moved the 20 m streets of `fuel_map_subdivision.asc` under
the house rows and left grass where the streets belong. With the streets in
place the subdivision reaches x = 780 m 130 s later (1705 against 1575 s) and
the coupled run 303 s later, they burn 9 % and 2 % fewer cells, and the ember
counts follow the burned cells the launches are sampled from. `wildland`,
`wildland_spotting` and `defensible` are unchanged. The single-viscosity and
first-order numbers below were measured before the fix.

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
