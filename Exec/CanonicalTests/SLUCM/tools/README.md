# ERF-SLUCM CSV Generator Toolchain

## Purpose

This toolchain generates canonical `building_layout.csv` + `materials.csv` files required by the ERF-SLUCM (Single-Layer Urban Canopy Model) for urban physics simulations. It supports two workflows:

1. **Synthetic patterns** — Generate idealized urban grids for physics validation and regression testing.
2. **Real-city ingestion** — Build urban morphology from OpenStreetMap (OSM) building footprints and local climate zone (LCZ) classification.

## Installation

### Synthetic patterns only (no external dependencies)

```bash
cd Exec/CanonicalTests/SLUCM/tools
pip install -r requirements.txt  # Actually installs nothing; pure stdlib
python3 examples/gen_uniform.py
```

### Real-city ingestion (with geospatial stack)

```bash
cd Exec/CanonicalTests/SLUCM/tools
pip install -r requirements-gis.txt
python3 examples/gen_real_boston.py
```

## Examples

All examples generate `building_layout.csv` and `materials.csv` in the current directory.

### Homogeneous grid

```bash
cd examples
python3 gen_uniform.py
# Produces: building_layout.csv (16×16 uniform urban), materials.csv (1 material)
```

### Checkerboard material pattern

```bash
cd examples
python3 gen_checkerboard.py
# Produces: 16×16 grid with alternating roof/wall materials (good for
# testing heterogeneous conduction).
```

### Domain with non-urban patch

```bash
cd examples
python3 gen_nonurban_patch.py
# Produces: 16×16 grid with a 4×4 non-urban rectangle in the middle
# (park, water body, etc.).
```

### Real-city: Boston

```bash
cd examples
python3 gen_real_boston.py
# Requires: pip install -r ../requirements-gis.txt
# Fetches OSM buildings, generates 32×32 UCM grid (75m cells),
# produces building_layout.csv + materials.csv for downtown Boston.
```

## ⚠ UTM Requirement (Critical)

**The UCM grid is Cartesian in meters.** All morphology aggregates (plan area fraction, building height, frontal area) depend on constant-scale area weighting in meters. Using a geographic CRS (lat/lon degrees) or a non-UTM projected CRS **silently introduces area errors of 10–30% at mid-latitudes and much worse at high latitudes.**

### What you must do

Ensure the domain CRS is **UTM** (EPSG codes: 32601–32660 for northern hemisphere, 32701–32760 for southern).

### How to find your UTM zone

```python
from ucm_from_gis import suggest_utm_zone_epsg

lon_center, lat_center = -71.06, 42.36  # Boston
epsg = suggest_utm_zone_epsg(lon_center, lat_center)
print(f"Use EPSG:{epsg}")  # Output: Use EPSG:32619
```

### What happens if you use the wrong CRS

If you pass EPSG:4326 (lat/lon) or another non-UTM CRS to `build_ucm_from_location()`, the function aborts with:

```
========================================================================
[ucm_from_gis] ABORTING: CRS is not UTM.
========================================================================
  You passed CRS: WGS 84 (EPSG:4326)
  The UCM grid is Cartesian in meters, so the domain MUST be in UTM.

  HOW TO FIX:
    1. Determine your domain's UTM zone. Rule of thumb:
         zone = int((longitude + 180) / 6) + 1
         For Boston (long -71): zone 19. EPSG = 32619 (N hemi).
         ...
```

This is a **feature, not a bug.** It prevents silent area-weighting errors.

## Module reference

### `ucm_csv.py` — Low-level CSV writers

Canonical functions for writing ERF-SLUCM CSV files. Zero third-party dependencies.

**Key functions:**

- `write_layout(path, nx_ucm, ny_ucm, cell_fn)` — Write `building_layout.csv` from a per-cell generator.
- `write_materials(path, materials)` — Write `materials.csv` from an iterable of material dicts.

**Example:**

```python
from ucm_csv import write_layout, write_materials
from ucm_generators import uniform_urban

write_layout("building_layout.csv", 16, 16, uniform_urban())
write_materials("materials.csv", [
    dict(mat_id=1, name="concrete", albedo=0.20, emissivity=0.90,
         k_therm_W_per_mK=1.5, rho_cp_J_per_m3K=2.0e6,
         thickness_m=0.3, description="generic urban"),
])
```

### `ucm_generators.py` — Synthetic pattern factories

Pure-stdlib generators for common urban patterns. Each returns a `cell_fn(i, j)` callable.

**Functions:**

- `uniform_urban(H_bldg, plan_frac, ...)` — Homogeneous grid.
- `checkerboard_materials(base, mat_a, mat_b)` — Alternate materials in checkerboard.
- `with_nonurban_box(base, i0, i1, j0, j1)` — Punch out a non-urban rectangle.
- `two_halves_heights(H_short, H_tall, split_axis, ...)` — Split domain with different heights (for scale-aware tests).

### `ucm_from_gis.py` — Real-city ingestion

Build urban morphology from OpenStreetMap + WUDAPT LCZ classification.

**Key functions:**

- `suggest_utm_zone_epsg(lon, lat)` — Look up UTM zone EPSG for a given lat/lon.
- `build_ucm_from_location(domain_bbox_m, crs, ucm_dx_m, ...)` — Main entry point. Fetches OSM buildings, maps to UCM grid, writes CSVs.

**Example:**

```python
from ucm_from_gis import build_ucm_from_location, suggest_utm_zone_epsg

epsg = suggest_utm_zone_epsg(-71.06, 42.36)  # 32619
build_ucm_from_location(
    domain_bbox_m=(330000 - 1200, 4691000 - 1200,
                   330000 + 1200, 4691000 + 1200),
    crs=f"EPSG:{epsg}",
    ucm_dx_m=75.0,
    output_dir=".",
    lcz_source="manual",
    lcz_manual_class=2,  # LCZ 2 = compact mid-rise
)
```

## Materials and LCZ

The toolchain ships `materials_lcz.csv`, a hand-curated mapping of the 10 urban Local Climate Zone (LCZ) classes (Stewart & Oke, 2012) to material properties based on Oke et al. (2017).

### LCZ classes in materials_lcz.csv

| LCZ | Name | Description |
|-----|------|-------------|
| 1 | Compact high-rise | Dense downtown core; low albedo, high thermal mass |
| 2 | Compact mid-rise | Dense residential; moderate albedo and mass |
| 3 | Compact low-rise | Brownstones, small residential |
| 4 | Open high-rise | Dispersed tall buildings |
| 5 | Open mid-rise | Dispersed medium-rise |
| 6 | Open low-rise | Suburban residential; moderate albedo |
| 7 | Lightweight low-rise | Mobile homes, prefab; low mass |
| 8 | Large low-rise | Industrial / commercial single-story |
| 9 | Sparsely built | Scattered buildings; high albedo |
| 10 | Heavy industrial | Manufacturing; high thermal mass |

### Extending materials_lcz.csv

To add a new material (e.g., "green roof" with lower albedo):

1. Choose a unique `mat_id` (≥ 1).
2. Add a row with properties (albedo ∈ [0, 1], emissivity ∈ [0, 1], thermal conductivity, heat capacity, thickness).
3. Reference the new ID in a custom LCZ or generator.

Example:

```python
# materials_lcz.csv
13,green_roof,0.25,0.80,0.5,1.2e6,0.2,Vegetated roof (albedo suppressed)

# my_generator.py
materials = [
    dict(mat_id=13, name="green_roof", albedo=0.25, emissivity=0.80, ...),
]
```

## Testing

### Synthetic tests (no external deps, runs in CI)

```bash
pytest tests/test_ucm_csv.py -v
```

Validates:
- CSV row/column counts and headers.
- Urban/non-urban material rules.
- Albedo, emissivity, thickness validation.

### GIS smoke tests (requires GIS stack, skipped by default)

```bash
# To enable, set environment variable:
SLUCM_GIS_TESTS=1 pytest tests/test_ucm_from_gis_smoke.py -v
```

Validates:
- UTM abort behavior (wrong CRS raises with fix-it message).
- UTM zone suggestion for known cities.

## Future extensions (Phase 3.x)

- **Microsoft ML Footprints API** — Higher-accuracy building footprints for regions with poor OSM coverage.
- **Google Open Buildings** — Global ML-derived footprints.
- **Real WUDAPT LCZ raster download** — Automatically fetch LCZ tiles (currently manual fallback).
- **LiDAR fusion** — Height refinement from open LiDAR datasets.
- **Vegetation layer** — Tree CSV generation (Phase 5.1).

## CSV schema

The C++ reader in `Source/UrbanCanopy/ERF_UCMBuildingLayoutReader.cpp` is the source of truth for schema and validation. All CSVs produced by this toolchain pass through the C++ reader without errors.

**building_layout.csv:**

```
i,j,bldg_id,height_m,plan_area_frac,W_road_m,W_roof_m,roof_mat_id,wall_mat_id,road_mat_id,orientation_deg,ah_profile_id,is_urban
```

- `i, j` — UCM cell indices (0-indexed).
- `bldg_id` — Building ID (currently =1 for all; future: heterogeneous morphology per building).
- `height_m` — Mean building height in meters.
- `plan_area_frac` — Plan area fraction (building footprint / cell area, ∈ [0, 1]).
- `W_road_m, W_roof_m` — Road and roof widths in meters.
- `roof_mat_id, wall_mat_id, road_mat_id` — Material IDs (≥1 for urban, can be 0 for non-urban).
- `orientation_deg` — Building orientation (currently unused in Phase 2).
- `ah_profile_id` — Anthropogenic heat profile ID (reserved for Phase 2.7).
- `is_urban` — 1 if urban cell, 0 if non-urban (park, water, etc.).

**materials.csv:**

```
mat_id,name,albedo,emissivity,k_therm_W_per_mK,rho_cp_J_per_m3K,thickness_m,description
```

- `mat_id` — Unique material ID (≥1).
- `name` — Material name (e.g., "concrete", "LCZ_2_compact_midrise").
- `albedo` — Solar reflivity ∈ [0, 1].
- `emissivity` — Thermal emissivity ∈ [0, 1].
- `k_therm_W_per_mK` — Thermal conductivity (W/m/K).
- `rho_cp_J_per_m3K` — Heat capacity per volume (J/m³/K).
- `thickness_m` — Material thickness in meters.
- `description` — Free text.

## References

- Stewart, I. D., & Oke, T. R. (2012). Local Climate Zone classification and its performance over the urban-rural scale. *Journal of Applied Meteorology and Climatology*, 51(8), 1506–1526.
- Oke, T. R., Mills, G., Christen, A., & Voogt, J. A. (2017). Urban Climates. Cambridge University Press.
- WUDAPT project: https://www.wudapt.org/

## Contact

ERF Urban Canopy team. See `Exec/CanonicalTests/SLUCM/` for test cases.
