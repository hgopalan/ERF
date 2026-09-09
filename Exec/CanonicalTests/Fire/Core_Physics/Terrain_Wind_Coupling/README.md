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
- The fire spreads from a 100 m ignition disk on a slope, at 0.4 to 0.7 m/s, reaching 320 fire cells at 300 s, and crosses terrain without instability.
- Spotting launches occasionally and brands land at terrain elevation. Landing distances saturate at the 200 m Scott cap for FM1.
- The fire cells at 300 s figure above is for the level-set path; see the note on anisotropy below before comparing it with the FARSITE path.
- The fire-grid slopes stay within the raster's (0.70 along x, 0.99 along y) in every column, the outflow column included, and the rate of spread stays below 1.2 m/s everywhere.
- No cell west of the ignition disc burns before the backing fire can reach it: the wind is westerly, so brands land downwind of the cell that launched them, and a burned cell `d` metres upwind of the disc's western edge has an arrival time of at least `(d - 30 m) / ROS_max`. At 300 s the fire has backed 110 m upwind.
- No cell on the inflow or outflow column burns.

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

## Checks (`check_terrain_wind.py`)

Run from the case directory after the case itself:

```bash
python3 check_terrain_wind.py            # reads the last plt_fire_????? plotfile
```

It needs yt and reads the terrain raster as an independent reference. It fails when a fire-grid slope exceeds the raster's along either axis, when the rate of spread exceeds 5 m/s anywhere, when a cell west of the ignition disc burns before the backing fire (at the largest ROS on the grid, plus one stamp radius of 30 m) can reach it, when a cell on the inflow or outflow column burns, or when the extraction height minus `wind_ref_ht` leaves the raster's elevation range.

The upwind check exists because this case once burned in vertical stripes 500 to 1200 m west of the ignition from the first seconds of the run. Two defects were behind it. The fire grid's terrain reader filled only the valid region of a field that holds the height at each cell's lower-left node, so the slope stencil of the column on the outflow face read an unfilled ghost entry, saw a 163 m cliff and gave that column a rate of spread of 26 m/s; the reader now fills the ghost entries from the raster. And a binary older than the fire-grid ghost fill of 2026-09-04 (`fire_fill_boundary`) left the level-set stencil reading uninitialised memory outside both x faces. Nothing downwind is bounded this way: there the fire advances by chains of spot fires, up to the Scott cap every spotting interval.

## Notes

**Propagation method.** The case runs the level-set path. Over the same 300 s the two paths burn very different areas:

| Path | first advance | cells at 300 s |
|---|---|---|
| level set | continuous | 284 |
| FARSITE | step 396 (99 s) | 53 |

That gap is **anisotropy, not a defect in either path**. FARSITE applies the Anderson length-to-width ellipse: at this wind the ratio saturates at its cap of 8, so the head advances at the full rate of spread while the flanks run at 7.5% of it and the backing fire at 20%. The burned area therefore grows as a downwind lobe rather than a disc. The level-set path gets its direction-dependence from the model itself: by default (`erf.fire.directional_ros = true`, the WRF-Fire form) Rothermel is evaluated with the wind and slope projected on the front normal, so the head sees the full wind and the backing fire the no-wind rate. With `directional_ros = false` the head rate is applied in every direction and the fire grows a disc that covers several times the area; the figures in this README are for the default.

FARSITE also advances in whole-cell quanta: a front cell converts one cell only once it has accumulated a full cell width of displacement, about 60 s of simulated time here, and the default `farsite.gaussian_sigma = -1` converts exactly one cell per stamp. Over the full run 27 stamps produced 21 new cells, so the quantization is granular but not lossy. Set `propagation_method = "farsite"` to exercise that path.

## References
- Balbi et al. 2020, A convective-radiative propagation model for wildland fires.
- Rothermel 1972, A Mathematical Model for Predicting Fire Spread in Wildland Fuels.
- Albini 1983, Potential Spotting Distance from Wind-Driven Surface Fires, USDA INT-309.
- Terrain from the FastWindTerrain case catalogue (SRTM), Tubbs Fire, California 2017.
