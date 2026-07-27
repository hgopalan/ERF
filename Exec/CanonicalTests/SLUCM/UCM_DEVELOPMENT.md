# ERF-SLUCM development notes

## Phase 2.9 close-out

The close-out architecture adds lightweight extension points without changing the
Phase 2.9 core CSV schema, C++ reader, or fetch framework:

- `tools/ucm_sources.py` provides fail-safe source adapters for OSM, Microsoft,
  Google Open Buildings, Copernicus GHSL Built-H, and WUDAPT LCZ.
- `tools/ucm_fusion.py` fuses footprint layers with deterministic source
  priority (`OSM > MICROSOFT > GOOGLE`) and 70% overlap deduplication.
- `tools/ucm_ah_from_lucy.py` parses LUCY NetCDF anthropogenic heat fields and
  returns a time-averaged 2-D `W/m²` array.
- `tools/ucm_plot.py` generates six non-interactive QA PNGs for the generated
  CSV products.

All new helpers are intentionally minimal and degrade gracefully when optional
GIS/scientific dependencies are unavailable.
