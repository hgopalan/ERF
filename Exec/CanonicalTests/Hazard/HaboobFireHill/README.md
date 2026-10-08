# HaboobFireHill

## Purpose
A haboob gust front runs over a grass fire and then over a Gaussian hill. A
cold pool collapses into a density current. The case shows how the hill
changes the front, the fire's response and where dust is raised.
HaboobFireFlat and HaboobFirePit run the same cold pool and fire over flat
ground and over a pit; only the terrain differs.

## Setup
| Item | Value |
|------|-------|
| Domain, grid | 8000 x 4000 x 1500 m, 128 x 64 x 64 cells (62.5 x 62.5 x 23.4 m), periodic in x and y, terrain-fitted mesh |
| Time step | `erf.fixed_dt = 0.25` s |
| Background | `sounding_neutral_abl`: theta 300 K to 468 m, inversion to 308 K at 551 m, u = 5 m/s at all heights; `erf.abl_geo_wind = 5 0 0`; MRF PBL |
| Cold pool | `erf.prob_name = "Bubble"`: -10 K air temperature (`prob.T_pert_is_airtemp = true`), cos^2 profile centred at (1500, 2000, 0) m with radii (1000, 4000, 600) m |
| Fire | Fuel model 1 (short grass), 1-h moisture 0.04, ignition disc r = 150 m at (3500, 2000) m; level set with directional ROS, lagged two-way coupling, `source_mode = add`, `heat_flux_partition = cfbm`; `use_wind_limit = false`, `use_terrain_wind = false` |
| Smoke, dust | Passive tracers. Three dust bins; emission from the surface-layer u*, raised by the fire wind and by plume lofting in burning cells (fire-dust coupling) |
| Terrain | `haboob_hill_129x65.txt`: 200 m Gaussian hill, sigma 600 m, centred at (5000, 2000) m |

The bubble uses the computational height, not the height above ground, so it
is kept over flat ground upwind of the hill.

The surface layer uses ERF's default on a terrain-fitted mesh: local
averaging, sampling along the surface normal and interpolation to
`erf.most.zref`. The three decks differ only in the terrain lines.

Two fire settings differ from the code defaults on purpose:
- `erf.fire.use_wind_limit = false`. ERF's midflame wind cap (300 ft/min,
  1.52 m/s, for fine fuels) holds fuel model 1 at 0.28 to 0.29 m/s from the
  first step, so the fire never responds to the gust front (Andrews et al.
  2013 recommend against applying a wind limit).
- `erf.fire.use_terrain_wind = false` (and `erf.dust.use_terrain_wind`
  stays at its default, false). These switches apply FARSITE-style factors
  (x1.5 on slopes facing the wind) to the sampled wind, but the mesh
  resolves the terrain, so its speed-up is already in that wind. In the hill
  case with both on, the y = 1 km dust peak on the hill's flank at 300 s was
  36 times the median; without them it is 12.

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

To rebuild the terrain file:

    python3 ../GaussianTerrain/make_gaussian_terrain.py --nx 129 --ny 65 --dx 62.5 --dy 62.5 \
        --height 200.0 --sigma 600.0 --cx 5000.0 --cy 2000.0 --output haboob_hill_129x65.txt

## Results (full-length run)
The front position is the furthest x downwind of the cold pool where the
first-cell theta on a line of constant y is more than 1 K below the far
field. The cold pool is a lobe, not a straight front, so it is given along
the centre line y = 2 km and along y = 1 km. The dust ratio is the maximum of
`dust_emission_flux` along y = 1 km divided by its median there.

| t [s] | Front x, y = 2 km / 1 km [m] | Max first-cell u [m/s] | Dust max/median at y = 1 km | Highest emission |
|-------|------------------------------|------------------------|-----------------------------|------------------|
| 60    | 2656 / 2594 | 9.2  | 5.4  | burning cells |
| 120   | 3156 / 3094 | 11.1 | 10.8 | behind the front, (2891, 1953) m |
| 180   | 3594 / 3594 | 10.7 | 11.1 | burning cells |
| 300   | 4594 / 4594 | 9.4  | 11.6 | burning cells |
| 420   | 5469 / 5656 | 9.1  | 12.9 | south flank of the hill, lee side, (5516, 1016) m |
| 600   | 6469 / 6844 | 8.8  | 12.2 | behind the front, (6828, 3828) m |

- Along y = 2 km the front moves at 7.9 m/s from 60 to 300 s and 6.2 m/s
  from 300 to 600 s (flat: 8.2 and 7.4). It is slowest after crossing the
  crest at about 360 s: 4 to 6 m/s per 30 s on the lee side from 450 s. At
  600 s it is 440 m behind the flat-ground front. Along y = 1 km, 1.7 sigma
  from the hill axis, it is not delayed (6844 m in both cases).
- The fire's head rate of spread rises from 0.40 m/s to 1.96 m/s at 203 s as
  the front crosses it (flat: 2.11 m/s at 200 s). It burns 15.0 ha by 600 s,
  against 16.1 ha over flat ground.
- Outside the fire the highest emission is just behind the front until
  450 s (at about 210 s it is at the unburned edge of the fire head, 50 m
  ahead of the front); from 270 s that is on the hill's south flank, about 1 km from the
  axis (upwind of the crest line until 360 s, on the lee side after). From
  480 to 540 s it stays just downwind of the crest, 640 to 770 m behind the
  front: a lee feature, not a frontal one. The highest emission outside the
  fire at 390 s is 44 % above the flat case's (9.26e-7 against
  6.42e-7 kg/m2/s). The time-averaged domain-total emission is within 1 %
  of the flat case's; step by step the ratio ranges from 0.93 to 1.08.

Caveats:
- The initial wind is a uniform 5 m/s down to the ground, with no surface
  layer profile. It spins down: ahead of the front (x > 7.5 km) the
  first-cell u falls from 5.0 m/s to 3.8 to 4.0 m/s by 300 s, so the later
  front speeds and dust ratios include that decay.
- The dust threshold friction velocity (0.040 to 0.054 m/s) is below u* in
  more than 90 % of the dust cells at every output (all of them at 30 s),
  so emission follows u* (about u*^3); the crust reduction of burned cells
  changes little. In burning cells (fire heat flux above
  550 W/m2) plume lofting multiplies the emission by 1 + `k_loft` = 3, which
  is why the highest emission is so often inside the fire.
- The lid is a slip wall at 1.5 km with no damping layer, and turbulence is
  the MRF column scheme with no LES closure at 62.5 m, so the mixing at the
  head of the density current is not resolved.
- Smoke and dust undershoot to small negative values at sharp edges (at
  600 s about -1.6e-6 kg/m3 smoke and -2.7e-7 kg/m3 dust, against maxima
  of 9e-6 and 2e-5). Deposition clamps them at zero, but linear colour maps
  show them.
- These are single-run diagnostics, not a validation against observations.

## References
- Rothermel 1972, A Mathematical Model for Predicting Fire Spread in Wildland Fuels.
- Andrews, Cruz and Rothermel 2013, Examination of the wind speed limit function in the Rothermel surface fire spread model.
- Andrews 2018, The Rothermel surface fire spread model and associated developments.
- Benjamin 1968, Gravity currents and related phenomena.
- Straka et al. 1993, Numerical solutions of a non-linear density current: a benchmark solution and comparisons.
