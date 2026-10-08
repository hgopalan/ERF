# HaboobFirePit

## Purpose
A haboob gust front runs over a grass fire and then across a Gaussian pit. A
cold pool collapses into a density current. The case shows how a depression
changes the front and where it raises dust. HaboobFireFlat and HaboobFireHill
run the same cold pool and fire over flat ground and over a hill; only the
terrain differs.

## Setup
| Item | Value |
|------|-------|
| Domain, grid | 8000 x 4000 x 1500 m, 128 x 64 x 64 cells (62.5 x 62.5 x 23.4 m), periodic in x and y |
| Time step | `erf.fixed_dt = 0.25` s |
| Background | `sounding_neutral_abl`: theta 300 K to 468 m, inversion to 308 K at 551 m, u = 5 m/s; `erf.abl_geo_wind = 5 0 0`; MRF PBL |
| Cold pool | `erf.prob_name = "Bubble"`: -10 K air temperature (`prob.T_pert_is_airtemp = true`), cos^2 profile centred at (1500, 2000, 0) m with radii (1000, 4000, 600) m |
| Fire | Fuel model 1 (short grass), 1-h moisture 0.04, ignition disc r = 150 m at (3500, 2000) m; level set with directional ROS, lagged coupling, `source_mode = add`, `heat_flux_partition = cfbm`, smoke |
| Dust | Three bins, fire-dust coupling (crust reduction, fire wind, plume lofting); `erf.dust.use_terrain_wind = true` takes u* from the local wind at `erf.dust.zref` |
| Terrain | `haboob_pit_129x65.txt`: 200 m deep Gaussian pit, sigma 600 m, centred at (5000, 2000) m |

The bubble uses the computational height, not the height above ground, so it
is kept over flat ground upwind of the pit. On this terrain-fitted mesh ERF
switches the surface layer to local averaging (`erf.most.average_policy = 1`
with the normal-vector rotation and the interpolation to `erf.most.zref`).

`erf.fire.use_wind_limit` keeps its default (true): the midflame wind cap of
fuel model 1 holds the head rate of spread at 0.29 m/s for the whole run.

## Running
The deck is a short regression run: `max_step = 20` (5 s), about 8 s on 4
ranks in Release and 3 min in a Debug build. The cold pool is in place at
step 0 (theta minimum 290.0 K at the first cell, 10 K below the far field).

The full-length run is 600 s:

    mpiexec -n 4 erf_exec inputs max_step=2400 erf.plot_int_1=120 erf.fire_plot_int=120 erf.dust.dust_plot_int=120

It takes about 20 min on 4 ranks and writes about 1.1 GB.

To rebuild the terrain file:

    python3 ../GaussianTerrain/make_gaussian_terrain.py --nx 129 --ny 65 --dx 62.5 --dy 62.5 \
        --height -200.0 --sigma 600.0 --cx 5000.0 --cy 2000.0 --output haboob_pit_129x65.txt

## Results (full-length run)
The front position is the furthest x downwind of the cold pool where the
first-cell theta along y = 2 km (which follows the pit floor) is more than
1 K below the far field. The dust ratio is the maximum of
`dust_emission_flux` along y = 1 km divided by its median there.

| t [s] | Front x [m] | Max first-cell u [m/s] | Dust max/median at y = 1 km | Highest emission |
|-------|-------------|------------------------|-----------------------------|------------------|
| 60    | 2656 | 9.5  | 5.6  | burned ground |
| 120   | 3156 | 11.3 | 10.5 | behind the front, (2891, 2016) m |
| 180   | 3656 | 11.5 | 12.0 | burned ground |
| 300   | 4719 | 10.2 | 8.0  | burned ground |
| 420   | 5844 | 10.1 | 21.8 | downwind side of the pit, (5703, 2203) m |
| 600   | 7344 | 9.3  | 25.9 | downwind side of the pit, (6078, 2828) m |

- The front moves at 8.7 m/s both from 60 to 300 s and from 300 to 600 s.
  Over flat ground it slows to 7.2 m/s after 300 s, so at 600 s the front
  here is 500 m ahead of the flat-ground front (and 940 m ahead of the hill
  case).
- From 330 s the highest emission is in the pit: on its floor at 330 s,
  then on the downwind side, moving out from 0.2 to 1.4 km from the centre
  by 510 s. Along y = 1 km the peak is on that side too (1.0 to 1.3 km from
  the centre), at 21 to 28 times the median from 390 s on (flat ground: at
  most 12). The dust model multiplies the wind on slopes steeper than 0.05
  that face into it by `erf.dust.k_ridge = 1.5`; that is consistent with
  these peaks, though these runs do not separate it from the flow itself.
- Before that the highest emission is on burned ground (30 to 60 s and 150
  to 300 s) or just behind the front (90 to 120 s). On burned ground the
  fire removes 80 % of the crust (`erf.fire_dust_crust_reduction = 0.8`),
  which lowers the threshold friction velocity, and the fire wind can raise
  u*.
- The fire grows from 7.4 to 11.9 ha.

These are single-run diagnostics, not a validation against observations.

## References
- Rothermel 1972, A Mathematical Model for Predicting Fire Spread in Wildland Fuels.
- Andrews 2018, The Rothermel surface fire spread model and associated developments.
- Finney 1998, FARSITE: Fire Area Simulator - model development and evaluation
  (the terrain wind factors used by the dust model).
