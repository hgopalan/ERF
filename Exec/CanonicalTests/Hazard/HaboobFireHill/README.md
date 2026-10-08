# HaboobFireHill

## Purpose
A haboob gust front runs over a grass fire and then over a Gaussian hill. A
cold pool collapses into a density current. The case shows how the hill
changes the front, and how the front and the burned ground change dust
emission. HaboobFireFlat and HaboobFirePit run the same cold pool and fire
over flat ground and over a pit; only the terrain differs.

## Setup
| Item | Value |
|------|-------|
| Domain, grid | 8000 x 4000 x 1500 m, 128 x 64 x 64 cells (62.5 x 62.5 x 23.4 m), periodic in x and y |
| Time step | `erf.fixed_dt = 0.25` s |
| Background | `sounding_neutral_abl`: theta 300 K to 468 m, inversion to 308 K at 551 m, u = 5 m/s; `erf.abl_geo_wind = 5 0 0`; MRF PBL |
| Cold pool | `erf.prob_name = "Bubble"`: -10 K air temperature (`prob.T_pert_is_airtemp = true`), cos^2 profile centred at (1500, 2000, 0) m with radii (1000, 4000, 600) m |
| Fire | Fuel model 1 (short grass), 1-h moisture 0.04, ignition disc r = 150 m at (3500, 2000) m; level set with directional ROS, lagged coupling, `source_mode = add`, `heat_flux_partition = cfbm`, smoke |
| Dust | Three bins, fire-dust coupling (crust reduction, fire wind, plume lofting); `erf.dust.use_terrain_wind = true` takes u* from the local wind at `erf.dust.zref` |
| Terrain | `haboob_hill_129x65.txt`: 200 m Gaussian hill, sigma 600 m, centred at (5000, 2000) m |

The bubble uses the computational height, not the height above ground, so it
is kept over flat ground upwind of the hill. On this terrain-fitted mesh ERF
switches the surface layer to local averaging (`erf.most.average_policy = 1`
with the normal-vector rotation and the interpolation to `erf.most.zref`).

`erf.fire.use_wind_limit` keeps its default (true). The midflame wind cap of
fuel model 1 then holds the head rate of spread at 0.29 m/s for the whole run,
so the gust front does not speed the fire up (see Results).

## Running
The deck is a short regression run: `max_step = 20` (5 s). On 4 ranks it takes
about 10 s in Release and 6 min in a Debug build. The cold pool is already in
place at step 0 (theta minimum 290.0 K at the first cell, 10 K below the far
field).

The full-length run is 600 s:

    mpiexec -n 4 erf_exec inputs max_step=2400 erf.plot_int_1=120 erf.fire_plot_int=120 erf.dust.dust_plot_int=120

It takes about 20 min on 4 ranks and writes about 1.1 GB (21 atmosphere, fire
and dust plotfiles each).

To rebuild the terrain file:

    python3 ../GaussianTerrain/make_gaussian_terrain.py --nx 129 --ny 65 --dx 62.5 --dy 62.5 \
        --height 200.0 --sigma 600.0 --cx 5000.0 --cy 2000.0 --output haboob_hill_129x65.txt

## Results (full-length run)
The front position is the furthest x downwind of the cold pool where the
first-cell theta along y = 2 km is more than 1 K below the far field. The
dust ratio is the maximum of `dust_emission_flux` along y = 1 km divided by
its median there.

| t [s] | Front x [m] | Max first-cell u [m/s] | Dust max/median at y = 1 km | Highest emission |
|-------|-------------|------------------------|-----------------------------|------------------|
| 60    | 2656 | 9.2  | 5.5  | burned ground |
| 120   | 3156 | 11.1 | 11.2 | burned ground |
| 180   | 3594 | 10.6 | 11.6 | burned ground |
| 300   | 4656 | 9.4  | 36.3 | burned ground |
| 420   | 5531 | 9.0  | 16.6 | burned ground |
| 600   | 6406 | 8.8  | 12.2 | upwind slope of the hill, (4953, 2141) m |

- The front moves at 8.4 m/s from 60 to 300 s, the same as over flat ground
  (8.3 m/s). From 300 to 600 s, over the hill, it slows to 5.7 m/s (flat:
  7.2 m/s). At 600 s it is 440 m behind the flat-ground front.
- Along y = 1 km the emission peak follows 80 to 330 m behind the front
  (front measured at y = 2 km) until 330 s. From 240 to 330 s, as it reaches
  the hill's upwind flank, it rises to 19 to 36 times the median, against at
  most 12 over flat ground. The dust model multiplies the wind on slopes
  steeper than 0.05 that face into it by `erf.dust.k_ridge = 1.5`; that is
  consistent with the rise, though these runs do not separate it from the
  flow itself. After 330 s the
  peak on that line stays on the flank (x = 4.5 to 4.6 km) as the front
  moves on, and moves downwind again from 480 s.
- The highest emission is on burned ground until 420 s. There the fire
  removes 80 % of the crust (`erf.fire_dust_crust_reduction = 0.8`), which
  lowers the threshold friction velocity, and the fire wind can raise u*.
  From 450 s it is on the hill's upwind slope just below the crest.
- The fire grows from 7.4 to 13.7 ha. Its head rate of spread stays at
  0.29 m/s, the wind cap, while the front passes. With
  `erf.fire.use_wind_limit=false` on the command line the head rate rises to
  4.4 m/s at 240 s as the front crosses the fire, and 19.9 ha burns by 600 s.

These are single-run diagnostics, not a validation against observations.

## References
- Rothermel 1972, A Mathematical Model for Predicting Fire Spread in Wildland Fuels.
- Andrews 2018, The Rothermel surface fire spread model and associated developments.
- Finney 1998, FARSITE: Fire Area Simulator - model development and evaluation
  (the terrain wind factors used by the dust model).
