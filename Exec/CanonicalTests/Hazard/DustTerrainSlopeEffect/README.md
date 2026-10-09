# DustTerrainSlopeEffect

## Purpose
This case examines how a simple slope modifies near-surface winds and dust response compared with the flat-terrain baseline.

## Expected Results
The raster slope is uniform (10 degrees rising along +x) and the geostrophic wind blows along +x, so every dust cell is windward and the case is a smoke test of a spatially uniform slope factor, not of a windward/lee contrast: the threshold is 1.11 times the flat value everywhere (Iversen and Rasmussen 1994, sqrt(cos 10 + sin 10 / tan 35)), the FARSITE wind factor is `k_ridge` everywhere when `use_terrain_wind` is on, and `max/min` of `dust_emission_flux` over the domain is 1. A windward/lee contrast needs the slope in the atmosphere as well (`DustGaussianHill`, `DustGaussianPit`).


## References
- Shao and Lu 2000, A simple expression for wind erosion threshold friction velocity, J. Geophys. Res. 105, 22437.
- Marticorena and Bergametti 1995, Modeling the atmospheric dust cycle.
- Analytical Gaussian terrain idealization used for topographic verification.
