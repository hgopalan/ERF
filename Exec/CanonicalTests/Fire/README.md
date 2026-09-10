# Fire

## Overview
Canonical fire tests covering analytical ROS checks, atmospheric coupling, ignition geometry, fuel heterogeneity, and advanced fire-behavior options.

## Subcases
| Subdirectory | Description | Character |
|---|---|---|
| `Verification` | Cases with an independent answer: exact solutions of the equations the fire module solves (geometry of the level set with a prescribed rate, Snell refraction, a speed gradient, Rothermel on a slope, the moisture time lag) and plume theory for the injected heat. Each has a check script. | verification |
| `Core_Physics` | Rothermel spread on flat and sloped ground, fuel models and moisture, the wind adjustment factor and wind speed, terrain-wind coupling, vertical grid refinement. | empirical / regression |
| `Fire_Behavior` | Behaviour options: rate-of-spread models, the FARSITE elliptical spread, acceleration, crown fire, ignition patterns, spatial fuel, spotting. | empirical / regression |
| `Fire_Atmosphere_Coupling` | Passive, lagged and synchronous coupling and the smoke tracer. | empirical / regression |
| `Atmospheric_Boundary_Layer` | Fires under stability and MRF boundary-layer settings. | empirical / regression |
| `Heat_Flux_Diagnostics` | Flame temperature, heat flux and fireline intensity diagnostics. | empirical / regression |
| `WUI_Subdivision` | A wind-driven grass fire running into three rows of houses: the wildland-urban interface features together, each checked against an independent expectation. | verification |
| `Real_Terrain` | Fires on real (SRTM) terrain at event scale with inflow/outflow boundaries: the Marshall and Palisades fires. | demonstration |
| `Unit_Tests` | Python unit tests for the Rothermel kernel, the FARSITE ellipse, the ROS models, the fuel map, ignition schedules, spotting, crown fire, acceleration, wind interpolation and terrain projection. | unit |
| `Supporting_Files` | Shared fuel maps, ignition files, guides and the input validator. | assets |

Every case folder has a `check_*.py`; the regression suites that pin quantitative
results are under `Exec/RegTests/Fire*`.

## Notes
- See each subdirectory README for case-specific purpose, expected results, and key parameters.
- See `inputs_fire_master_reference` for module-level reference settings.
