# Marshall_Fire

## Purpose
A fire in a windstorm on real terrain, at the scale of an event: the Marshall Fire of 30 December 2021 in Boulder County, Colorado, which ran east from the foot of the Front Range through Superior and Louisville in a downslope windstorm with gusts above 40 m/s and burned about 1,100 homes in an afternoon. The case exercises the fitted-mesh atmosphere on 650 m of relief with a 10 m first cell, mass inflow and pressure outflow on real terrain, the level set on a million fire cells, multiple ignitions on terrain of different curvature, and spotting in a strong wind, all with MPI. It is a demonstration, not a reconstruction: the wind is an idealised neutral profile, the fuel is uniform, and the fire does not feed back on the atmosphere.

## Terrain
SRTM1 (1 arc second) elevation from the four tiles around 40 N, 105 W, box-averaged onto a 50 m raster over a 25.6 x 25.6 km square in UTM zone 13N. The square is placed so that the first Marshall Fire ignition, Eldorado Springs Drive at CO-93 (39.953 N, 105.232 W), sits 5 km from the west face and 11 km from the south face. The floor is the lowest point of the raster, 1510 m above sea level; the relief is 1038 m before and 657 m after the edge treatment. The outer 1.5 km of every face is blended with a raised cosine to that face's mean height, so the west face is a flat shelf 488 m above the floor on the foothill front, the north face a plain 76 m above it, and nothing steep sits on an outflow face. `gen_marshall.py` builds the raster from the `elevation` package's tile cache and writes every other input file.

`marshall_terrain.png` maps the raster with the three ignition sites.

## Wind
A neutral log law over z0 = 0.1 m, 20 m/s at 10 m above ground and held at 35 m/s above the height where the law reaches it, from 285 degrees: a westerly with the small northerly component that carried the fire east-south-east. ERF reads the inflow file's heights as absolute, so each inflow face has its own file with the profile anchored at that face's mean ground (`inflow_xlo.txt`, `inflow_yhi.txt`). The interior starts from the same profile as a sounding, neutral to 1500 m above the floor and 3 K/km above that, at a surface pressure of 840 hPa. Rayleigh damping of w in the top 500 m absorbs mountain waves; there is no inflow sponge.

## Fire
Short grass (Anderson fuel model 1) at 4 % dead moisture everywhere, Rothermel with the Andrews wind adjustment factor and the model's wind limit, on a 25 m fire grid that reads the same 50 m raster. The wind limit is what keeps a grass fire in a 35 m/s wind at the observed order of 1 m/s. Three ignitions:

| start | where | why |
|---|---|---|
| 0 s at (5000, 11000) m | the historical origin on Marshall Mesa, curvature -2.8e-4 /m, slope 0.03 | the event |
| 300 s at (2100, 8250) m | a convex crest on the foothill front, curvature -1.3e-3 /m | a fire on a ridge in the accelerating flow over it |
| 600 s at (6550, 11400) m | a concave drainage east of the origin, curvature +7.6e-4 /m, slope 0.09 | a fire in the sheltered, channelled flow of a valley |

The second and third come from `ignitions_marshall.csv` through `erf.fire.ignition.schedule_file`; `gen_marshall.py` chose them as the extremes of the Laplacian of a 150 m smoothed raster within 4 km of the origin, at least 1.5 km apart. Spotting is on with the fixed seed 20211230, so the run is reproducible.

## Running
```bash
python3 gen_marshall.py                  # once; needs rasterio, pyproj, scipy and the SRTM tile cache
mpirun -np 8 ../../../../../build/Exec/erf_exec inputs_marshall_fire
python3 make_gif.py                      # marshall_fire.gif from the fire plotfiles
python3 ../../../../Tools/make_map_movie.py --fetch usgs   # the same frames over USGS imagery
```
The atmosphere has 256 x 256 x 40 cells (10 m first cell stretched by 1.1 to 4.4 km) and the fire grid 1024 x 1024; the slow step is 0.3 s, the acoustic substepping automatic. The deck stops at 30 minutes, which is enough for a short animation of the first spread and costs about 90 minutes on eight ranks (0.85 s per step); raise `stop_time` for the fire to run across the plains. `make_gif.py` writes one frame per fire plotfile (every 30 s) over an x = 0 to 10 km, y = 6 to 14 km window around the fires; `--crop ""` shows the whole domain. The SRTM tiles come from `elevation.clip(bounds=(-105.33, 39.80, -104.97, 40.10), product="SRTM1")`, which leaves them in `~/Library/Caches/elevation/SRTM1/cache`; the GDAL clip that follows is not needed since the generator reads the tiles itself. `Exec/Tools/make_map_movie.py` draws the fire over map imagery instead of the shaded relief: `--fetch usgs` downloads the 210 USGS imagery tiles covering the domain once (zoom 14, about 7 m per pixel, 5 MB, kept in `basemap_tiles/`), stitches them into `basemap_marshall_z14.tif`, warps that onto the domain's UTM 13N square and writes `marshall_fire_map.gif` and `.mp4` over the same window; `--basemap` takes any RGB GeoTIFF instead.

The deck's fuel is uniform grass. `Exec/Tools/make_landfire_fuel_map.py` puts LANDFIRE's Scott and Burgan (2005) 40-model fuel map (FBFM40) on the fire grid instead. In this directory, `python3 ../../../../Tools/make_landfire_fuel_map.py --fetch LF2016 --dx 25` downloads the LANDFIRE 2.0.0 raster (2016, before the fire) for the domain plus 500 m from the USGS LANDFIRE Product Service: one request, 961 x 975 cells of 30 m, about 2 MB, kept as `LF2016_FBFM40_marshall.tif`. It resamples the raster nearest-neighbour onto the 1024 x 1024 fire grid, writes `fuel_marshall_fbfm40.asc` and prints the deck lines that load it (`fuel_set = "scott_burgan40"`, `load_from_map`, `rothermel_per_fuel`); `--raster` takes a raster already on disk. The map is 38 % GR2 grass, 31 % NB1 developed, 7 % NB3 agricultural, 4 % GS2, 4 % GR1 and 3 % NB8 water. The historical origin falls on a developed (NB1) cell, with 41 of the 81 fire cells within 100 m burnable; the convex and concave ignitions sit in GR2 and TU1. Neither file is committed. `Exec/Tools/hrrr_point_winds.py` reads the HRRR surface analyses (Herbie, only the needed GRIB messages from NOAA's public AWS archive) and prints the 10 m and 80 m wind speed and direction, gust and PBL height at the domain's west face, upstream, the ignitions and any `--point`, with the winds rotated from HRRR's grid to true north: `python3 ../../../../Tools/hrrr_point_winds.py --date 2021-12-30 --hours 17 18 19 20 --domain marshall_domain.json`. At 18 UTC it gives 16.7 m/s at 10 m from 267 degrees at the west face (29.4 m/s at 80 m, gusts to 37 m/s, PBL height 650 m), so the inflow blew from about 18 degrees further west than a wind from 285 degrees.

## Expected Results
From the 30-minute run on eight ranks (0.85 to 1.0 s per step, 6000 steps):

- The atmosphere runs stable at 0.3 s steps with 8 acoustic substeps; the fire-grid wind reaches 4 to 5 m/s at midflame near the origin and follows the terrain, accelerating over the mesa rims and turning in the drainages.
- The three fires burn 370 ha at 30 minutes (22 ha at 6 min, 146 ha at 18 min), with a perimeter of 18.7 km and one to three spot fires ahead of the fronts at any time.
- The origin fire's head advances at 0.44 m/s including its spot fires (0.3 to 0.6 m/s on the ROS field), the order the event averaged over its 10 km run.
- Those figures are from a run with `erf.fire.directional_ros = false`, the isotropic level set, which also backed the fires into the wind at 0.18 m/s; the committed deck uses the directional default, which keeps the head rate, holds the backing fire at the no-wind rate, and grows each fire as a downwind lobe with a smaller area.
- `make_gif.py` writes a 61-frame animation (30 s at 2 fps) of the burned area over the shaded elevation; the animation is not committed.
- `make_map_movie.py` writes the same 61 frames over the USGS imagery: the origin fire starts just north-west of Marshall Lake and US 36 crosses the north-east corner of the window. Neither the imagery nor the animation is committed.
