# HaboobFireFlat

## Purpose
A haboob gust front runs over a grass fire on flat ground: a cold pool
collapses into a density current that crosses the fire and raises dust behind
it. This is the no-terrain reference of the haboob set; HaboobFireHill and
HaboobFirePit run the same cold pool and fire over a 200 m hill and a 200 m
deep pit.

## Setup
| Item | Value |
|------|-------|
| Domain, grid | 8000 x 4000 x 1500 m, 128 x 64 x 64 cells (62.5 x 62.5 x 23.4 m), periodic in x and y |
| Time step | `erf.fixed_dt = 0.25` s |
| Background | `sounding_neutral_abl`: theta 300 K to 468 m, inversion to 308 K at 551 m, u = 5 m/s; `erf.abl_geo_wind = 5 0 0`; MRF PBL |
| Cold pool | `erf.prob_name = "Bubble"`: -10 K air temperature (`prob.T_pert_is_airtemp = true`), cos^2 profile centred at (1500, 2000, 0) m with radii (1000, 4000, 600) m |
| Fire | Fuel model 1 (short grass), 1-h moisture 0.04, ignition disc r = 150 m at (3500, 2000) m; level set with directional ROS, lagged coupling, `source_mode = add`, `heat_flux_partition = cfbm`, smoke |
| Dust | Three bins, fire-dust coupling (crust reduction, fire wind, plume lofting); `erf.dust.use_terrain_wind = true` takes u* from the local wind at `erf.dust.zref` |
| Surface layer | `erf.most.average_policy = 1` (local) |

The deck differs from HaboobFireHill only in the terrain lines and in
`erf.most.average_policy = 1`. The hill and pit decks get local surface-layer
averaging by default on their terrain-fitted mesh; on this flat mesh the
default is a plane average, which gives every column the same u* (0.349 m/s
everywhere at 300 s in a run without the key), so neither the surface drag
nor the dust emission would see the front. With no terrain the slopes are
zero and `erf.dust.use_terrain_wind` changes nothing but the source of u*.

`erf.fire.use_wind_limit` keeps its default (true): the midflame wind cap of
fuel model 1 holds the head rate of spread near 0.28 m/s for the whole run.

## Running
The deck is a short regression run: `max_step = 20` (5 s), about 6 s on 4
ranks in Release and 4 min in a Debug build. The cold pool is in place at
step 0 (theta minimum 290.0 K at the first cell, 10 K below the far field).

The full-length run is 600 s:

    mpiexec -n 4 erf_exec inputs max_step=2400 erf.plot_int_1=120 erf.fire_plot_int=120 erf.dust.dust_plot_int=120

It takes 10 to 20 min on 4 ranks and writes about 0.8 GB.

## Results (full-length run)
The front position is the furthest x downwind of the cold pool where the
first-cell theta along y = 2 km is more than 1 K below the far field. The
dust ratio is the maximum of `dust_emission_flux` along y = 1 km divided by
its median there.

| t [s] | Front x [m] | Max first-cell u [m/s] | Dust max/median at y = 1 km | Highest emission |
|-------|-------------|------------------------|-----------------------------|------------------|
| 60    | 2656 | 9.2  | 5.6  | behind the front, (2328, 2016) m |
| 120   | 3156 | 10.9 | 10.7 | behind the front, (2828, 2016) m |
| 180   | 3594 | 10.6 | 11.3 | burned ground |
| 300   | 4656 | 9.0  | 9.1  | burned ground |
| 420   | 5594 | 8.2  | 7.4  | behind the front, (5453, 2078) m |
| 600   | 6844 | 8.0  | 9.7  | behind the front, (6391, 2078) m |

- The front moves at 8.3 m/s from 60 to 300 s and 7.2 m/s from 300 to
  600 s. The first-cell wind behind it peaks at 10.9 m/s at 120 s.
- Along y = 1 km the emission peak follows 140 to 330 m behind the front
  (front measured at y = 2 km) until 540 s, at up to 11.6 times the median.
- The highest emission is on burned ground at 30 s and from 150 to 390 s.
  There the fire removes 80 % of the crust
  (`erf.fire_dust_crust_reduction = 0.8`), which lowers the threshold
  friction velocity, and the fire wind can raise u*. At the other times it
  is near the front.
- The fire grows from 7.4 to 13.1 ha.

These are single-run diagnostics, not a validation against observations.

## References
- Rothermel 1972, A Mathematical Model for Predicting Fire Spread in Wildland Fuels.
- Andrews 2018, The Rothermel surface fire spread model and associated developments.
