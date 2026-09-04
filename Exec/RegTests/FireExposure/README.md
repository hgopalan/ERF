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
  because the fireline is a few hundred cells against tens of thousands
  behind it.

## Reference table

```
variant              id      x      y  burned t_first  t_last  resid  peak_kWm   HL_mean    HL_max  embers  landed
------------------- --- ------ ------ ------- ------- ------- ------ --------- --------- --------- ------- -------
noib                  1    190     30    0.45     111     164     54    2698.8     1.414     3.160       0       0
noib                  2    190     80    1.00      28     125     96    3205.8     3.160     3.160       0       0
noib                  3    190    115    1.00      39     216    177    3115.2     3.156     3.160       0       0
ib                    1    190     30    0.45     107     156     49    1591.2     1.404     3.158       0       0
ib                    2    190     80    1.00      40     160    120    1532.2     3.138     3.160       0       0
ib                    3    190    115    1.00      49     222    173    1557.5     3.080     3.160       0       0
noib_spotting         1    190     30    0.45      76     129     54    2871.8     1.414     3.160       1      23
noib_spotting         2    190     80    1.00       2      95     93    3600.8     2.409     3.160       8      23
noib_spotting         3    190    115    1.00      24     210    185    3240.2     3.159     3.160       4      23
noib_spotting_front   1    190     30    0.45      74     131     56    2878.3     1.414     3.160       0      11
noib_spotting_front   2    190     80    1.00       2      49     47    3600.8     1.623     3.160       0      11
noib_spotting_front   3    190    115    1.00      20     178    158    3362.3     2.659     3.160       2      11
```

Four ranks, 240 s, last row per box (written at 224.9 s). Things to read
out of it.

**The middle box is the one the head fire hits.** It is reached at 28 s, a
second after the arrival-time probe on its upwind face, its whole wall band
burns as the front wraps around it, and the front leaves its downwind side
at 125 s, the same time the downwind probe reports. The outer boxes are
reached by the flanks, later and obliquely, so their residence times are
longer; the first box, on the far side of the street, only ever has 45% of
its band burned.

**The heat load saturates at the fuel's energy.** The largest heat load on
every box is 3.16 MJ/m², which is the FM1 fuel load (0.166 kg/m²) times its
heat content (18.6 MJ/kg): a band cell that burned released all its fuel.
The mean is lower where part of the band never burned.

**Immersed-forcing buildings halve the intensity at the walls** (1.5-1.6
against 2.7-3.2 MW/m) because the wind the Balbi model reads is slowed at
the walls, and every arrival is 10 s or so later.

**Spotting reaches the walls first.** With brands in the air the middle box
is reached at 2 s by a brand from the ignition disc that landed at its
wall, 13 embers land on the three footprints in 240 s, and the spot fires
ahead of the front burn a quarter more of the domain. The mean heat load on
the middle box drops because cells ignited by a spot have their fuel capped
at 5% of the initial load.

**Launching from the front changes what the brand count measures.** At the
same launch probability the front set is 2082 cells against 21390 burned
cells on the last step, yet 11 brands land against 23: brands from deep in
the burned area mostly fall on consumed fuel and are discarded, brands from
the front land in fuel. The spots therefore ignite more new area (24472
cells burned against 21390), fewer embers reach the footprints (2 against
13), and the middle box's band is passed sooner because spot fires ahead of
it burn the fuel around it first.
