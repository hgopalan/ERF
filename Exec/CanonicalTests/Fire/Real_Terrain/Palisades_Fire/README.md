# Palisades_Fire

## Purpose
The NLR Palisades stress test, rebuilt as a case anyone can run. The demo took the
7 January 2025 Palisades Fire region, put a 300 m atmosphere over a 30 m fire and
terrain grid, drove it with a simplified stratified atmosphere and a
southwesterly at 38 mph, coupled a Rothermel fire back into the flow, and ran
3.13 hours on four H100 GPUs. This deck reproduces that configuration on public
elevation data. Like `Marshall_Fire` it is a demonstration, not a reconstruction:
the wind is an idealised profile rather than the Santa Ana event that actually
drove the fire, and the fuel is one model over all the vegetated land.

What it exercises that `Marshall_Fire` does not: a fire grid ten times finer than
the atmosphere rather than four, its own digital elevation model under that fire
grid, a coastline as a non-burnable boundary, and two-way coupling, so the fire's
heat enters the atmosphere and the fire reads the wind that results.

## Terrain
SRTM1 (1 arc second) elevation from tile N34W119, box-averaged onto a 30 m raster
over a 24 x 24 km square in UTM zone 11N. The square is placed so the reported
origin above Temescal Canyon (34.0797 N, 118.5265 W, near the Skull Rock
trailhead) sits 9 km from the west face and 8 km from the south face. The floor
is mean sea level; SRTM carries a few metres of noise below datum over Santa
Monica Bay, so the raster is clamped there rather than at its raw minimum, which
would lift the whole ocean above the floor. The relief is 724 m before and 644 m
after the edge treatment, and 5.4 % of the fire grid is sea.

The outer 1.5 km of every face is blended with a raised cosine to that face's
**median** height, not its mean: the south and west faces are part ocean and part
coastal hillside, and a mean would lift the sea tens of metres and drop the
hillside onto it. The south face settles at 16 m and the west face at 290 m.

Two rasters come out of the generator, and this is the point of the case:

| file | spacing | read by | why |
|---|---|---|---|
| `terrain_palisades.txt` | 30 m raster box-averaged over one 300 m atmospheric cell | `erf.terrain_file_name`, the fitted mesh | at full sharpness the canyon walls fold the near-ground cells over each other and the surface layer cannot find its query height |
| `terrain_palisades_fire.txt` | the sharp 30 m raster | `erf.fire.terrain_file_name` | the slope that drives the Rothermel slope factor has to be the real one at 30 m |

The smoothing takes the maximum slope the mesh sees from 1.27 to 0.75; the fire
still sees 1.27. `erf.most.zref` is raised to 20 m so the surface layer finds its
query height on every column of the steep ground.

`palisades_terrain.png` maps the raster with the ignition site, the shoreline and
the wind.

## Fuel
`fuel_palisades.asc` is an ESRI ASCII map on the 800 x 800 fire grid: Anderson
model 4, chaparral, on every vegetated cell, and 0 over the Pacific and the
beach. 0 is listed in `erf.fire.fuel_map.nonburnable_codes`, so the front stops
at the shoreline instead of running out to sea. Chaparral is the Santa Monica
Mountains vegetation and is why this fire behaves nothing like the Marshall
grass fire; set `erf.fire.fuel_model_id = 1` and rewrite the land code for the
grass comparison. Dead fuel moisture is 3 / 4 / 6 %, a January Santa Ana.

## Wind
A neutral log law over z0 = 0.1 m anchored at the demo's 38 mph (16.99 m/s) 10 m
above ground and held at 1.75 times that above the height where the law reaches
it, from 225 degrees. The wind is from the southwest, so the west and south faces
take the inflow profile (`inflow_xlo.txt`, `inflow_ylo.txt`, each anchored at its
own face's mean ground) and the east and north faces are outflow. The interior
starts from the same profile as a sounding, neutral to 1500 m and 3 K/km above,
at 1013 hPa. Rayleigh damping of w in the top 1500 m absorbs mountain waves.

## Fire
Level set with the directional (Richards ellipse) rate of spread, so head, flank
and backing come from the ellipse rebuilt on the front normal at every
Runge-Kutta stage, with the hybrid WENO5-Z gradient and the near-front viscosity
that are now the defaults. Rothermel with the Andrews wind adjustment factor and
the model's wind limit. One ignition, a 100 m disc at the origin at t = 0.
Coupling is `lagged`: the fire's sensible heat enters the atmosphere on the next
step and the fire reads the wind that comes back. Spotting is on with the fixed
seed 20250107, so the run is reproducible.

## Running
```bash
python3 gen_palisades.py                 # once; needs rasterio, pyproj, scipy and the SRTM tile cache
mpirun -np 4 ../../../../../build/Exec/erf_exec inputs_palisades_fire
```
The atmosphere has 80 x 80 x 102 cells (652,800; 10 m first cell stretched by 1.03
to 6463 m) and the fire grid 800 x 800 (640,000). The demo quoted approximately
650,000 and 680,000. `amr.max_grid_size = 40` gives four boxes, one per rank on a
four-GPU node, and box lengths that divide by the grid ratio of 10 as the fire
module requires.

The slow step is 0.2 s. The compressible limit on this fitted mesh is about
0.217 s, so 0.3 s, which `Marshall_Fire` uses on its gentler 100 m mesh, is
unstable here. `stop_time = 11268` is the demo's 3.13 hours, which is 56,340
steps: this is an overnight run on CPUs and the reason the demo used GPUs. For a
first look set `stop_time = 1800`.

The SRTM tile comes from `elevation.clip(bounds=(-119, 34, -118, 35),
product="SRTM1")`, which leaves it in `~/Library/Caches/elevation/SRTM1/cache`.
SRTM1 is a 30 m product; the demo used a 10 m DEM, so `--dxr` is there for when a
finer one is substituted.

## Expected Results
From a 30-minute run (`stop_time = 1800`, 9000 steps, `erf.fire_plot_int = 200`)
on ten ranks, and the same deck with grass instead of chaparral:

| fuel | head ROS | burned at 30 min | perimeter |
|---|---|---|---|
| Anderson 4, chaparral, 3 % moisture (the deck) | 3.9 m/s | 1310 ha, 3236 acres | 31.4 km |
| Anderson 1, short grass, 4 % moisture | 1.0 m/s | 69 ha, 170 acres | 7.7 km |

- The atmosphere is stable at 0.2 s steps on the fitted mesh; the fire runs as a
  single downwind lobe northeast from the origin, and the shoreline holds it on
  the coastal side, which is the non-burnable fuel code doing its job.
- `make_movie.py` writes a 46-frame animation over an x = 4 to 18 km,
  y = 3 to 17 km window, and an H.264 MP4 of the same frames when ffmpeg is on
  the PATH; neither is committed.

**On the demo's 600 acres.** The demo reported about 600 acres after 3.13 hours.
This case matches its configuration, the domain, both resolutions, the cell
counts, the wind, the coupling and the run length, but burns more than that, and
the reason is the one input the slide does not state: the fuel. Chaparral is the
real Santa Monica Mountains vegetation and Rothermel spreads it at 3.9 m/s in a
38 mph wind, which is an order of magnitude more area than the demo. Short grass,
the fuel `Marshall_Fire` uses, gives 1.0 m/s and 170 acres in the first half
hour, still ahead of the demo's average rate. Reproducing the 600 acres would
need the demo's own fuel map, most likely one with the urban and irrigated parts
of the Palisades marked non-burnable rather than a single model over all the
vegetated land. Set `erf.fire.fuel_model_id = 1` with
`erf.fire.rothermel_per_fuel = 0` for the grass row above; a real fuel map drops
into `erf.fire.fuel_map.file` in the same ESRI ASCII form `gen_palisades.py`
writes.

## What building it found
**The fire never read `erf.fire.terrain_file_name`.** `ERF_FireParams.H` queried
`terrain_file_name` under the `erf` prefix only, so the fire silently used the
atmosphere's raster and the deck parameter did nothing. It went unnoticed because
every terrain case so far, `ROS_Slope_Effects`, `Terrain_Wind_Coupling` and
`Marshall_Fire`, sets both keys to the same file. This case is the first where
they differ, and the difference is the whole point: a 300 m fitted mesh cannot
carry a 30 m DEM, and a 30 m fire grid should not be given a 300 m one. The
parameter is now read under `erf.fire` and falls back to the atmosphere's file
when it is unset, so every existing deck behaves as before.

## References
- Anderson, H. E. (1982). Aids to determining fuel models for estimating fire behavior. USDA Forest Service General Technical Report INT-122.
- Rothermel, R. C. (1972). A mathematical model for predicting fire spread in wildland fuels. USDA Forest Service Research Paper INT-115.
- Andrews, P. L. (2012). Modeling wind adjustment factor and midflame wind speed for Rothermel's surface fire spread model. USDA Forest Service General Technical Report RMRS-GTR-266.
