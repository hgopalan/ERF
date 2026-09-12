# FireExposure

What each building experiences as the fire passes: the per-structure
exposure diagnostics (`erf.fire.exposure.*`) on the FireHybridObstacles
scenario. Balbi 2020 on the reference wind, the three boxes and the street
non-burnable, the level-set wall extrapolation on, one exposure row per box
every 25 s. It is a demonstration deck, not a validation.

```
./run_exposure.sh /path/to/erf_exec
```

`MPIRUN="mpirun -np 4" ./run_exposure.sh ...` runs each variant in
parallel; `SKIP_RUN=1` only re-tabulates existing CSVs.

## The scenario

320 x 160 x 160 m at 5 m, fire grid at 1.25 m, neutral 8 m/s sounding with
a MOST surface layer, FM1 short grass at 6% moisture, passive coupling,
240 s. Three 20 m wide, 10 m tall boxes stand across the wind at
x = 180-200 m, at y = 30, 80 and 115 m (ids 1-3 in scan order); the fire is
ignited as a 15 m disc on the middle box's centreline, 35 m upwind of its
face. The wall band is the ring of burnable cells one fire cell (1.25 m)
wide around each footprint.

## The variants

- `noib`: flat ground, the boxes exist only in the fire grid.
- `ib`: the same heightmap drives immersed-forcing buildings, with the
  open-column wind weights on.
- `noib_spotting`: flat ground with Albini spotting on a fixed seed, so
  brands land on the footprints and the embers column is exercised.
- `noib_spotting_front`: the same with `erf.fire.spotting.launch_from =
  front`, so brands come from the fireline rather than the whole burned
  area.

## What the columns mean

For each box the script prints the last exposure row: the fraction of its
wall band burned, the first and last arrival of the front in the band and
their difference (how long the front spent passing the box), the largest
peak fireline intensity in the band, the mean and largest accumulated heat
load there in MJ/m², and the number of embers that landed on the footprint.
Arrival times are -1 for a box the front never reached.

## What to expect

- The middle box (id 2), on the ignition centreline, is reached first and
  carries the largest heat load; its band burns completely as the front
  wraps around it. The outer boxes are reached later by the flanks and
  their residence times are longer because the front passes them obliquely.
- With immersed-forcing buildings everything arrives later (the wind the
  fire reads is slowed) and the intensities are lower.
- Embers appear only in the spotting rows; the counts depend on the seed.
  Launching from the front gives far fewer brands at the same probability,
  because the fireline is about a thousand cells against tens of thousands
  behind it.

## Reference table

```
variant              id      x      y  burned t_first  t_last  resid  peak_kWm   HL_mean    HL_max  embers  landed
------------------- --- ------ ------ ------- ------- ------- ------ --------- --------- --------- ------- -------
noib                  1    190     30    0.62     108     224    116    2711.6     1.904     3.160       0       0
noib                  2    190     80    1.00      28     128    100    3208.0     3.160     3.160       0       0
noib                  3    190    115    1.00      40     191    151    3111.3     3.160     3.160       0       0
ib                    1    190     30    0.73     106     224    118    1591.4     2.202     3.158       0       0
ib                    2    190     80    1.00      40     162    122    1532.1     3.137     3.160       0       0
ib                    3    190    115    0.98      50     218    168    1557.5     3.048     3.160       0       0
noib_spotting         1    190     30    0.76      78     224    146    2859.8     2.314     3.160       0      24
noib_spotting         2    190     80    1.00       2      99     97    3600.8     2.516     3.160       5      24
noib_spotting         3    190    115    1.00      21     169    148    3266.6     3.160     3.160       5      24
noib_spotting_front   1    190     30    0.70      78     224    146    2859.8     2.189     3.160       1      11
noib_spotting_front   2    190     80    1.00       2     113    111    3600.8     2.802     3.160       0      11
noib_spotting_front   3    190    115    1.00      40     191    151    3111.3     3.160     3.160       0      11
```

Four ranks, 240 s, last row per box (written at 224.9 s), regenerated on
2026-09-11 after the fuel map fix. Things to read out of it.

**The street moved on 2026-09-11.** Until hgopalan/ERF#411 the fuel map
reader put the first row of `fuel_map_street.asc` on the south edge, so the
non-burnable street ran at y = 30-35 m, across the first box, instead of at
125-130 m. That street cut the first box's band: 45 % of it burned, and the
front had passed it by 164 s. With the street in its place 62 % has burned
when the run ends and the front is still passing.

**The middle box is the one the head fire hits.** It is reached at 28 s, a
second after the arrival-time probe on its upwind face, its whole wall band
burns as the front wraps around it, and the front leaves its downwind side
at 128 s, two seconds after the downwind probe reports. The outer boxes are
reached by the flanks, later and obliquely, so their residence times are
longer; the flank is still passing the first box when the run ends.

**The heat load saturates at the fuel's energy.** The largest heat load on
every box is 3.16 MJ/m², which is the FM1 fuel load (0.166 kg/m²) times its
heat content (18.6 MJ/kg): a band cell that burned released all its fuel.
The mean is lower where part of the band never burned.

**Immersed-forcing buildings halve the intensity at the walls** (1.5-1.6
against 2.7-3.2 MW/m) because the wind the Balbi model reads is slowed at
the walls, and every arrival is 10 s or so later.

**Spotting reaches the walls first.** With brands in the air the middle box
is reached at 2 s by a brand from the ignition disc that landed at its
wall, 10 embers land on the three footprints in 240 s, and the spot fires
ahead of the front burn half as much again (25349 cells against 16980). The mean heat load on
the middle box drops because cells ignited by a spot have their fuel capped
at 5% of the initial load.

**Launching from the front changes what the brand count measures.** At the
same launch probability the front set is 1537 cells against 25349 burned
cells on the last step, yet 11 brands land against 24: brands from deep in
the burned area mostly fall on consumed fuel and are discarded, brands from
the front land in fuel. Fewer landings still mean fewer spot fires: 23157
cells burn against 25349, one ember reaches a footprint against 10, and the
middle box's band is passed later (113 against 99 s).
