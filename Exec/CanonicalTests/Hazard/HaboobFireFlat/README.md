# HaboobFireFlat

## Purpose
A haboob gust front runs over a grass fire on flat ground: a cold pool
collapses into a density current that crosses the fire and raises dust
behind it. This is the no-terrain reference of the haboob set;
HaboobFireHill and HaboobFirePit run the same cold pool and fire over a
200 m hill and a 200 m deep pit.

## Setup
| Item | Value |
|------|-------|
| Domain, grid | 8000 x 4000 x 1500 m, 128 x 64 x 64 cells (62.5 x 62.5 x 23.4 m), periodic in x and y, terrain-fitted mesh |
| Time step | `erf.fixed_dt = 0.25` s |
| Background | `sounding_neutral_abl`: theta 300 K to 468 m, inversion to 308 K at 551 m, u = 5 m/s at all heights; `erf.abl_geo_wind = 5 0 0`; MRF PBL |
| Cold pool | `erf.prob_name = "Bubble"`: -10 K air temperature (`prob.T_pert_is_airtemp = true`), cos^2 profile centred at (1500, 2000, 0) m with radii (1000, 4000, 600) m |
| Fire | Fuel model 1 (short grass), 1-h moisture 0.04, ignition disc r = 150 m at (3500, 2000) m; level set with directional ROS, lagged two-way coupling, `source_mode = add`, `heat_flux_partition = cfbm`; `use_wind_limit = false`, `use_terrain_wind = false` |
| Smoke, dust | Passive tracers. Three dust bins; emission from the surface-layer u*, raised by the fire wind and by plume lofting in burning cells (fire-dust coupling) |
| Terrain | none: `erf.terrain_type = StaticFittedMesh` with no terrain file, so the mesh type (VariableDz) and surface-layer treatment are those of the hill and pit decks |

The surface layer uses ERF's default on a terrain-fitted mesh: local
averaging, sampling along the surface normal and interpolation to
`erf.most.zref`. The three decks differ only in the terrain lines.

Two fire settings differ from the code defaults on purpose:
- `erf.fire.use_wind_limit = false`. With the midflame wind cap, fuel model 1
  spreads at 0.29 m/s from the first step and the fire never responds to the
  gust front (Andrews et al. 2013 recommend against the original limit).
- `erf.fire.use_terrain_wind = false` (and `erf.dust.use_terrain_wind`
  stays at its default, false). These switches apply FARSITE-style factors
  (x1.5 on slopes facing the wind) to the sampled wind, but the mesh
  resolves the terrain, so its speed-up is already in that wind. With both
  on, the y = 1 km dust peak on the hill's flank at 300 s was 36 times the
  median; without them it is 12.

## Running
The deck is a short regression run: `max_step = 20` (5 s), a few seconds on
4 ranks in Release. It covers the cold-pool start on the terrain-fitted mesh
and the start-up of the fire and dust modules (first-cell theta minimum
290.0 K, 10 K below the far field), not the front's interaction with them:
the front reaches the fire after about 150 s.

The full-length run is 600 s:

    mpiexec -n 4 erf_exec inputs max_step=2400 erf.plot_int_1=120 erf.fire_plot_int=120 erf.dust.dust_plot_int=120

It took 12 to 21 min of wall time on 4 ranks of a shared machine and writes
about 1.1 GB (21 atmosphere, fire and dust plotfiles each).

## Results (full-length run)
The front position is the furthest x downwind of the cold pool where the
first-cell theta on a line of constant y is more than 1 K below the far
field. The cold pool is a lobe, not a straight front, so it is given along
the centre line y = 2 km and along y = 1 km. The dust ratio is the maximum of
`dust_emission_flux` along y = 1 km divided by its median there.

| t [s] | Front x, y = 2 km / 1 km [m] | Max first-cell u [m/s] | Dust max/median at y = 1 km | Highest emission |
|-------|------------------------------|------------------------|-----------------------------|------------------|
| 60    | 2656 / 2594 | 9.4  | 5.6  | burning cells |
| 120   | 3156 / 3094 | 11.2 | 11.1 | behind the front, (2891, 1953) m |
| 180   | 3656 / 3594 | 11.1 | 11.9 | burning cells |
| 300   | 4656 / 4656 | 9.3  | 9.4  | burning cells |
| 420   | 5656 / 5594 | 8.5  | 7.9  | burning cells |
| 600   | 6906 / 6844 | 8.3  | 8.8  | behind the front, (6578, 3828) m |

- Along y = 2 km the front moves at 8.2 m/s from 60 to 300 s and 7.4 m/s
  from 300 to 600 s. The first-cell wind behind it peaks at 11.6 m/s at
  210 s.
- The fire's head rate of spread rises from 0.40 m/s to 2.11 m/s at 200 s as
  the front crosses it, and falls to 0.26 m/s by 600 s. It burns 16.1 ha by
  600 s (7.4 ha at ignition).
- Along y = 1 km the emission peak follows 80 to 270 m behind the front on
  the same line until 510 s, at up to 12 times the median. Outside the fire
  the highest emission is just behind the front.

Caveats:
- The initial wind is a uniform 5 m/s down to the ground, with no surface
  layer profile. It spins down: the first-cell u at x > 7.5 km falls from
  5.0 m/s to about 3.9 m/s at 300 s and 3.0 to 3.5 m/s at 600 s, so the
  later front speeds and dust ratios include that decay.
- The dust threshold friction velocity (0.04 to 0.05 m/s) is below u* almost
  everywhere, so emission follows u* (about u*^3); the crust reduction of
  burned cells changes little. In burning cells (fire heat flux above
  550 W/m2) plume lofting multiplies the emission by 1 + `k_loft` = 3, which
  is why the highest emission is so often inside the fire.
- The lid is a slip wall at 1.5 km with no damping layer, and turbulence is
  the MRF column scheme with no LES closure at 62.5 m, so the mixing at the
  head of the density current is not resolved.
- These are single-run diagnostics, not a validation against observations.

## References
- Rothermel 1972, A Mathematical Model for Predicting Fire Spread in Wildland Fuels.
- Andrews, Cruz and Rothermel 2013, Examination of the wind speed limit function in the Rothermel surface fire spread model.
- Andrews 2018, The Rothermel surface fire spread model and associated developments.
- Benjamin 1968, Gravity currents and related phenomena.
- Straka et al. 1993, Numerical solutions of a non-linear density current: a benchmark solution and comparisons.
