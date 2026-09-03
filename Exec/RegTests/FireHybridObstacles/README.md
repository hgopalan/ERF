# FireHybridObstacles

A wind-driven grass fire approaching three box obstacles, run with each
rate-of-spread model and with or without the boxes present in the atmosphere.
It exercises the hybrid `structure` selector, the arrival-time probes, and
the interaction between the fire model's wind extraction and immersed-forcing
buildings. It is a demonstration deck, not a validation: the fire is not yet
stopped by the footprints (see "What this is not").

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

```
./run_obstacles.sh /path/to/erf_exec
SKIP_RUN=1 ./run_obstacles.sh x     # rebuild the table from existing logs
```

Each no-IB variant takes about seven minutes on a laptop; the IB variants
take about ten.

## Reference results

Burned fire cells after 240 s, the head ROS on the last step, the number of
cells the hybrid hands to Balbi, and the arrival time [s] at each probe
("-" means the probe never burned):

| variant | cells | max ROS (m/s) | sec | u1 u2 u3 | g1 g2 g3 | d1 d2 d3 |
|---|---|---|---|---|---|---|
| `rothermel_noib` | 2256 | 0.250 | - | - 139 - | - - - | - 224 - |
| `balbi_noib` | 18255 | 0.771 | - | - 31 111 | 53 47 - | - 51 104 |
| `hybrid_noib` | 4159 | 0.771 | 3396 | - 102 219 | 213 124 - | - 126 217 |
| `rothermel_ib` | 1673 | 0.250 | - | - - - | - - - | - - - |
| `balbi_ib` | 14320 | 1.016 | - | - 55 111 | 55 61 - | - 156 165 |
| `hybrid_ib` | 3056 | 0.936 | 3396 | - 137 - | 206 174 - | - 239 - |

Things to read out of it.

**Rothermel arrives at the middle box on schedule.** 35 m at 0.250 m/s is
140 s; the probe reports 139 s. It then crosses the box unhindered and
reaches the downwind face at 224 s, which is the behaviour this deck is
documenting rather than endorsing (see below).

**The structure selector changes only the last 10 m.** The hybrid runs at
the Rothermel rate until the front is within 10 m of a box, then at the
Balbi rate, and reaches the middle box at 102 s instead of 139 s. Its
`sec` count of 3396 is the number of fire cells within 10 m of a box.

**Balbi with the reference wind is fast and decays.** Its head rate starts
at 1.22 m/s on the uniform 8 m/s initial wind and settles to 0.77 m/s as
the surface layer spins up, which is why the arrival at u2 (31 s) is
earlier than the final rate implies.

**Immersed forcing slows the approach.** With the boxes in the atmosphere
the Rothermel run never reaches the upwind faces in 240 s and burns 26%
fewer cells; its mean ROS over burning cells drops from 0.25 to about
0.16 m/s. The immersed-forcing drag decelerates the near-surface flow ahead
of the boxes, and the fire's wind extraction at 6.1 m sees that. The
hybrid's arrival at u2 moves from 102 s to 137 s for the same reason.

**And speeds up the gaps.** Balbi with immersed forcing ends with a higher
head rate (1.02 against 0.77 m/s) because the reference wind it consumes is
channelled between the boxes, and its gap probes (g1, g2) now arrive before
or with the faces rather than after.

## What this is not

The fire still burns through the footprints: `rothermel_noib` reaches the
downwind face of the middle box, which is only possible by crossing it. The
hybrid `structure` selector chooses which model spreads near a building and
nothing more. Treating structure cells as non-burnable (zero ROS, phi held
positive, landings rejected, no fuel or heat) is the next change; when it
lands, the downwind probes here should only be reached around the ends of
the boxes and through the gaps, and this README's numbers will change.

Nothing here is compared with observations. The two-way heat coupling is off
(`coupling_type = passive`), so the atmosphere does not respond to the fire.
