# idealized_fire

## Purpose

The 30 December 2021 Marshall Fire in an idealized atmosphere over its real terrain, run
with two boundary-layer schemes, MRF and YSUNew. The atmosphere is not a reconstruction
of the windstorm. It is a settled neutral boundary layer under a capping inversion,
brought in through the west and north faces and turned to the wind direction HRRR
analysed at 18 UTC. The case shows how the pieces behave together at event scale:
- the terrain-following inflow of `<face>.inflow_profile` with the interior started
  above the local ground;
- a PBL scheme driving the near-surface wind over 650 m of relief;
- the level set on the LANDFIRE fuel map with non-burnable urban and water cells.

`../Real_Terrain/Marshall_Fire` is the same event with a compressible atmosphere, a
log-law wind from 285 degrees and uniform grass.

## Setup

| | |
|---|---|
| Domain | 25.6 x 25.6 x 11.6 km: the SRTM terrain of `../Real_Terrain/Marshall_Fire` (read from that folder) with smoothed metrics |
| Grid | 80 x 80 x 50 cells: 320 m horizontally; 10 m first cell stretched by 1.1 to 11.6 km; fire grid 640 x 640 at 40 m |
| Boundaries | mass inflow on the west and north faces (`inflow_profile = file`), pressure outflow on the east and south; surface layer (z0 = 0.1 m); slip-wall top with Rayleigh damping of w over the top 3 km |
| Atmosphere | anelastic; MRF or YSUNew in the vertical and 2D Smagorinsky in the horizontal; Coriolis at 39.95 N with the precursor's geostrophic wind |
| Start | the precursor column as an input sounding, wind and theta read above the local ground |
| Fire | level set with the ellipse directional rate and Anderson length-to-breadth ratio; LANDFIRE fuel map with per-cell Rothermel coefficients; dead moisture 6/7/8 %; spotting with a fixed seed; one-way coupling |
| Time | 20 min of wind spin-up, then 45 min of fire from the first ignition (39.953 N, 105.232 W) |

The terrain-fitted anelastic projection is GMRES with an FFT preconditioner, so the decks
need a build with `ERF_ENABLE_FFT=ON`.

## Inflow: the precursor

`precursor/` holds a flat, periodic 8 x 8 column on the same vertical grid, surface and
solver as the fire decks, one per PBL scheme. It is forced by a 67 m/s geostrophic wind and
starts neutral up to 1500 m, under a +15 K capping inversion 100 m deep, with 3 K/km above.
It runs for 24 h, longer than the 18.7 h inertial period at 39.95 N.
`precursor/make_inflow.py` takes the horizontal mean of the last plotfile. It rotates the
column so the 10 m wind blows from 267 degrees, writes `inflow_PBL.txt` (`# z u v T`, heights
above the ground) and `sounding_PBL.txt`, and prints the rotated geostrophic wind for the
fire deck's `erf.abl_geo_wind`. On an f-plane a horizontally uniform column is invariant under
rotation, so this equals running the precursor with the rotated geostrophic wind.

The two columns settle differently under the same 67 m/s forcing:

| | MRF | YSUNew |
|---|---|---|
| 10 m wind after 24 h | 24.61 m/s | 28.40 m/s |
| 80 m wind | 35.88 m/s | 37.94 m/s |
| friction velocity | 2.186 m/s | 2.524 m/s |
| diagnosed PBL height | 3392 m | 2851 m |
| rotated geostrophic wind for the deck | `61.23815 -27.18252` | `60.18186 -29.44731` |
| wall time, 1 rank | 170 s | 177 s |

Both are far windier and deeper than the HRRR analysis below. YSUNew mixes momentum down
harder, which is why its 10 m wind is 15 % stronger and its fire runs faster.

The 267-degree direction is the HRRR analysis at the west face, at the latitude of the
ignition, at 18 UTC:

```
python3 ../../../Tools/hrrr_point_winds.py --date 2021-12-30 --hours 17 18 19 20 \
    --domain ../Real_Terrain/Marshall_Fire/marshall_domain.json
```

That analysis gives 16.7 m/s at 10 m, 29.4 m/s at 80 m and a PBL height of 650 m. The
precursor keeps only its direction. Its 10 m wind is 30-40 % stronger than HRRR's, and its
PBL several times deeper.

## Fuel

```
python3 ../../../Tools/make_landfire_fuel_map.py --fetch LF2016 --dx 40 \
    --domain ../Real_Terrain/Marshall_Fire/marshall_domain.json
```

The tool downloads one window of LANDFIRE 2.0.0 FBFM40 (2016, before the fire; about 2 MB
from the USGS LANDFIRE Product Service) and maps it nearest-neighbour onto the fire grid as
`fuel_marshall_fbfm40.asc`, rows from the north. The map is:

| Code | Model | Share |
|---|---|---|
| GR2 | low-load grass | 38 % |
| NB1 | developed | 31 % |
| NB3 | agricultural | 7 % |
| GS2 | grass-shrub | 4 % |
| GR1 | short grass | 4 % |
| NB8 | open water | 3 % |

The ignition point itself falls on a developed (NB1) cell. The 100 m ignition disc still
reaches the surrounding grass. Neither the raster nor the map is committed.

## Running

```
cd precursor
mpirun -np 1 erf_exec inputs_precursor_mrf              # about 3 min; the same for ysunew
python3 make_inflow.py . mrf 267 && mv inflow_mrf.txt sounding_mrf.txt ..
cd ..
python3 ../../../Tools/make_landfire_fuel_map.py --fetch LF2016 --dx 40 --domain ../Real_Terrain/Marshall_Fire/marshall_domain.json
mpirun -np 4 erf_exec inputs_marshall_ideal_mrf         # about 18 min; ysunew about 26 min
python3 check_idealized_fire.py .
```

The inflow and sounding files of both schemes are committed, so the precursor step is only
needed to regenerate them. Run each fire deck in its own directory next to this one,
because the decks read the terrain as `../Real_Terrain/Marshall_Fire/terrain_marshall.txt`.
`make_overlay_movie.py MRF_DIR YSUNEW_DIR TAG` draws both runs over USGS imagery; the imagery
comes from `Exec/Tools/make_map_movie.py --fetch usgs`, run in `../Real_Terrain/Marshall_Fire`.

## Checks

`check_idealized_fire.py RUN_DIR` prints the burned area and the head distance every 5
minutes. The head distance is measured from the ignition along 87 degrees, downwind. The
script then checks:
- every fire field is finite;
- the fire has ignited;
- the rate of spread stays in [0, 50) m/s;
- arrival times are consistent with the ignition and the plotfile time;
- fuel never increases;
- no cell outside the ignition disc has burned on a non-burnable code of the fuel map,
  which also catches a map read with its rows flipped;
- the burned area never decreases.

`--acres LO HI` and `--head-km LO HI` add ranges for the final values.

## Expected results

Burned area and head distance from the ignition, from `check_idealized_fire.py`. Both decks
carry a fixed spotting seed, so these are reproducible on the same build and rank count.

| fire minutes | MRF acres | MRF head | YSUNew acres | YSUNew head |
|---|---|---|---|---|
| 5 | 20 | 0.42 km | 28 | 0.54 km |
| 15 | 89 | 1.54 km | 126 | 1.74 km |
| 30 | 210 | 2.55 km | 311 | 3.31 km |
| 45 | 404 | 3.71 km | 678 | 4.66 km |

Both pass all seven checks. YSUNew burns 1.7 times the area of MRF and its head runs 0.95 km
further, entirely because of the stronger 10 m wind its column settles to.

A useful sanity check on the pair: the Superior Costco sits about 5.5 km east-southeast of the
ignition and was being evacuated roughly 75 minutes in. Extending the MRF deck to a 90 minute
fire (`stop_time = 6600`) puts its head at 5.28 km at 75 minutes, within 4 % of that distance,
after which it stops at the Superior urban edge, which the fuel map marks non-burnable.

## Limitations

- **The atmosphere is idealized.** It is a settled neutral column 30-40 % windier and far
  deeper than the HRRR boundary layer, not the downslope windstorm with gusts above 40 m/s.
  Only the wind direction comes from the event.
- **Coupling is one way.** The fire does not heat the atmosphere.
- **The built environment is non-burnable.** There are no houses as fuel, and the spotting
  here is too weak to carry a front across developed cells. So a front that reaches the edge of
  Superior or Louisville stops there, whereas in the event the fire burned through them.
- **Runs need FFT.** No CI job builds with FFT, so no CTest runs these decks.
