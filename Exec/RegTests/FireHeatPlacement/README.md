# FireHeatPlacement

How the fire's heat enters the atmosphere when buildings are present. The
FireHybridObstacles scenario is run with lagged heat coupling and the
structure mask on, and the variants change only how the fire tendency is
written into the cell source (`erf.fire.source_mode`), whether
immersed-forcing buildings stand in the atmosphere, on which step that
forcing acts, whether the heat is confined to the open part of each column
(`erf.fire.heat_open_fraction`), and which tendency form is used
(`erf.fire.heat_tendency_density`). It is a demonstration deck, not a
validation.

```
./run_heat.sh /path/to/erf_exec
```

`MPIRUN="mpirun -np 4" ./run_heat.sh ...` runs each variant in parallel;
`SKIP_RUN=1` only re-tabulates existing logs.

## The scenario

320 x 160 x 160 m at 5 m, fire grid at 1.25 m, neutral 8 m/s sounding with a
MOST surface layer, FM1 short grass at 6% moisture, Balbi 2020 on the
reference wind so the fire reaches the boxes in about 30 s, run to 120 s.
The three 20 m wide, 10 m tall boxes at x = 180-200 m are non-burnable
through `erf.fire.structures.enable`, the 5 m street is fuel code 0 and
non-burnable through `erf.fire.fuel_map.nonburnable_codes`. The coupling
runs with `erf.fire.fire_debug`, so every RK stage prints the
column-integrated heating against the flux handed in and, for the columns
that straddle a footprint edge, the share of the heating that lands below
the roof. The script tabulates the last print, the burned-cell count and
the arrival times at the middle box's upwind face and the two gap probes.

## The variants

| deck | source_mode | buildings in atmosphere | open fraction | tendency |
| --- | --- | --- | --- | --- |
| `overwrite_noib` | overwrite | none | off | default |
| `add_noib` | add | none | off | default |
| `add_noib_open` | add | none | on | default |
| `add_noib_open_legacy` | add | none | on | historical (density factor) |
| `overwrite_ib` | overwrite | immersed forcing, substeps | off | default |
| `add_ib` | add | immersed forcing, substeps | off | default |
| `add_ib_open` | add | immersed forcing, substeps | on | default |
| `add_ib_open_legacy` | add | immersed forcing, substeps | on | historical (density factor) |
| `overwrite_ib_slow` | overwrite | immersed forcing, slow step | off | default |
| `add_ib_slow` | add | immersed forcing, slow step | off | default |

## What to expect

- **`overwrite` and `add` agree whenever nothing else writes into the
  potential-temperature source.** That includes immersed forcing applied on
  the acoustic substeps (the compressible default), which never meets the
  slow-step fire source. With the forcing on the slow step the overwrite
  discards the building relaxation and the two rows separate.
- **The energy ratio is `1 - exp(-z_top / alfg)`**, 0.9714 for this 160 m
  domain and 45 m decay height, and one for any realistic domain top. The
  historical form (`heat_tendency_density = true`, the `_legacy` rows)
  multiplies the tendency by the local density, so its ratio is about 1.11
  at sea level: until 2026-09-03 the default injected about ten percent
  more energy than the fire supplied.
- **Fully covered columns receive no heat in any variant**, because their
  fire cells are masked and carry no flux. The open-fraction option acts in
  the ring of columns that straddle a footprint edge: the below-roof share
  of their heating drops from `1 - exp(-H / alfg)` (0.20 here) to that times
  the open fraction, and the rest is lifted above the roof. On this grid
  that is a small change in the burned-cell count; it grows as the building
  size approaches the atmosphere spacing.
- **The fire spreads less with buildings in the atmosphere** (about 5000
  against 6600 cells) because the immersed forcing slows the near-surface
  wind the reference-wind Balbi model reads, and the arrival at the middle
  box's upwind face moves from 25 s to 37 s. The historical form burns
  slightly more because it injects more heat.

## Reference table

```
variant                 exit   cells    E_ratio   below_roof  theta_blk  u2 g1 g2
---------------------- ----- ------- ---------- ------------ ----------  ----------
overwrite_noib             0    6611   0.971434   0.20512201   307.8475     25   54   47
add_noib                   0    6611   0.971434   0.20512201   307.8475     25   54   47
add_noib_open              0    6625   0.971434   0.16403038   307.9225     25   54   47
add_noib_open_legacy       0    6635   1.112157   0.16550991   308.8266     25   54   47
overwrite_ib               0    4971   0.971434   0.20512201   327.8431     37   54   59
add_ib                     0    4971   0.971434   0.20512201   327.8431     37   54   59
add_ib_open                0    4969   0.971434   0.16403038   322.9018     37   54   59
add_ib_open_legacy         0    5019   1.114533   0.16409439   324.5610     37   54   59
overwrite_ib_slow          0    4980   0.971434   0.20512201   328.7950     35   55   60
add_ib_slow                0    4991   0.971434   0.20512201   328.9740     35   55   60
```

Four ranks, `heat_tendency_density` at its default (energy-consistent)
except in the `_legacy` rows. `u2`, `g1`, `g2` are arrival times in seconds
at the middle box's upwind face and the two gap midpoints; `theta_blk` is
the largest potential temperature [K] of the state in a cell below a roof
on the last stage. The `_legacy` rows reproduce, to the cell, the
`add_*_open` rows of the table generated before the default changed, and
the default `add_*_open` rows reproduce its `_energy` rows.

The table was regenerated on 2026-09-11 after the fuel map fix
(hgopalan/ERF#411): the reader had put the first row of
`fuel_map_street.asc` on the south edge, so the 5 m street ran at
y = 30-35 m instead of 125-130 m. With the street in its place the
flat-ground rows burn about 180 fewer cells (6611 against 6789), the
immersed-forcing rows under 1 % fewer, the gap probes move by a second, and
the energy ratio of the default rows is unchanged.
