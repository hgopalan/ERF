# Terrain_Wind_Coupling

## Purpose
The only fire case that exercises terrain and wind together. Every other case with terrain has no wind, and every case with wind is flat, so the terrain-following wind paths had no end-to-end coverage: the ground datum for the extraction height, the per-column ground anchoring of the bilinear horizontal interpolation, the projection of surface spread into map view, and the descent of firebrands onto terrain.

## Terrain
Real SRTM elevation over the 2017 Tubbs Fire footprint — Sonoma/Napa, California, in the Diablo wind corridor — taken from the FastWindTerrain case catalogue at 38.6500, -122.4833. The source is a warped lat/lon grid at roughly 24 m spacing; it is box-averaged onto a regular 50 m raster over the 5 x 5 km domain, and the minimum elevation is subtracted so the field starts at the domain floor rather than at 202 m above sea level.

The outer 500 m is tapered to the domain-edge mean with a raised cosine, which makes the field periodic in y and gives the inflow boundary a flat approach. The interior 4 km is untouched real terrain: 297 m of relief with slopes reaching 0.87, about 41 degrees.

## Wind
A neutral log-law profile enters at `xlo` as mass inflow and leaves at `xhi` as pressure outflow, following the Askervein real-terrain case, so the wind is driven through the domain rather than recycled and the terrain does not have to match across the x boundary. The profile is anchored at 12 m/s at 10 m with z0 = 0.1 m, held constant above 400 m.

## Physics / Model Features Exercised
- Wind extraction at a height above true ground over varying terrain
- Bilinear horizontal interpolation with per-column ground anchoring
- Terrain slopes on the fire grid, and the ROS slope factor
- Projection of surface spread into map view on the level-set path
- Firebrand descent onto terrain rather than to a flat datum
- Mass inflow / pressure outflow boundaries with a fire present

## Expected Results
- The extraction height tracks the terrain: the reported range spans roughly 12 m over the lowest ground to 284 m over the highest, each being that column's ground plus `wind_ref_ht`.
- The reference wind reaches about 22 m/s, since the highest columns sample the capped part of the profile, and the effective midflame wind is about 8 m/s after the Wind Adjustment Factor.
- The fire spreads from a 100 m ignition disk on a slope, at 0.4 to 0.7 m/s, and crosses terrain without instability.
- Spotting launches occasionally and brands land at terrain elevation. Landing distances saturate at the 200 m Scott cap for FM1.

## Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `geometry.prob_hi` | `5000.0 5000.0 1200.0` | 5 x 5 km footprint, 1200 m deep. |
| `amr.n_cell` | `40 40 100` | 125 m horizontal, 12 m vertical. Box lengths must divide by `grid_ratio`. |
| `geometry.is_periodic` | `0 1 0` | Inflow/outflow in x, periodic in y. |
| `erf.terrain_type` | `StaticFittedMesh` | Terrain-fitted atmospheric mesh. |
| `erf.terrain_file_name` | `"terrain_tubbs.txt"` | Tubbs Fire SRTM raster. |
| `erf.fire.terrain_file_name` | `"terrain_tubbs.txt"` | Same raster at fire-grid resolution. |
| `erf.fire.grid_ratio` | `4` | 31.25 m fire cells. |
| `erf.fire.propagation_method` | `"levelset"` | Continuous front advance; see the note below. |
| `erf.fire.wind_interp` | `"bilinear"` | Blend the four surrounding atmospheric columns. |
| `erf.fire.coupling_type` | `"passive"` | Isolates the terrain and wind paths from feedback. |
| `erf.fire.use_terrain_wind` | `false` | Terrain flow is resolved, so the empirical corrections would double count. |
| `erf.fire.spotting.enable` | `true` | Exercises the terrain-aware firebrand descent. |

## Notes

**Propagation method.** The case runs the level-set path, which advances the front continuously and shows 300 s of spread on a 31.25 m fire grid. The FARSITE path accumulates displacement per front cell and stamps a whole cell only once that displacement reaches a full cell width, so at this rate of spread and cell size it advances in jumps of `dx_fire / ROS`, about 45 s of fire time, and needs a much longer run to show anything. Set `propagation_method = "farsite"` to cover that path.

**Spotting re-entry filter.** `spotting.reentry_fuel_thresh` is set to zero. Any positive value silences spotting here: the residual-fuel field reads exactly zero at the landing cells, so even 0.01 kg/m² rejects every brand while 0.0 accepts all of them. That behaviour is independent of terrain and of the fuel model and wants its own investigation.

## References
- Balbi et al. 2020, A convective-radiative propagation model for wildland fires.
- Rothermel 1972, A Mathematical Model for Predicting Fire Spread in Wildland Fuels.
- Albini 1983, Potential Spotting Distance from Wind-Driven Surface Fires, USDA INT-309.
- Terrain from the FastWindTerrain case catalogue (SRTM), Tubbs Fire, California 2017.
