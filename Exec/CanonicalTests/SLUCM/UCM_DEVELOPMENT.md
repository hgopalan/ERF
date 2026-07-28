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

## Phase 3.10 Multi-Level Two-Way Heat Coupling (Phase 3 Finale)

**Status**: ✅ COMPLETE (this PR)  
**Canonical**: `Exec/CanonicalTests/SLUCM/UCMBostonMultiLevelTwoWay/`

### Overview
Phase 3 finale: combines multi-level refinement (Phase 3.6) with two-way heat feedback (Phase 3.2).
Tests that UCM operates on refined AMR level and feeds back atmospheric heating, producing UHI signal.

### Configuration
- Base: 20×20×64; refined level over downtown (5–15 km × 5–15 km, ref_ratio=2)
- Anchor_level = 1 (UCM on refined level)
- **atm_feedback_heat = 1.0** (differs from 3.6: enables heat feedback)
- atm_feedback_momentum = 1.0 (inherited)
- Duration: 600 steps (~14 min simulated)

### Validation Criteria
1. All fields finite on both levels (no NaN/Inf in θ, u, v)
2. θ bounded in [280, 320] K on both levels
3. UHI signal at k=0: Δθ_urban > 0.01 K (heat feedback active)
4. Rural contamination: std(θ_rural, k=0) < 0.01 K (minimal spurious heating)
5. Wind reduction at k=1 > 10% (momentum drag persists)

### Key Differences
- vs. Phase 3.6: atm_feedback_heat OFF → ON (+1.0)
- vs. Phase 3.5c: single-level → multi-level (refined level 1)
- Both changes enable proper testing of multi-level two-way coupling

### References
- Phase 3.2 two-way heat baseline: UCMBostonTwoWayHeat (single-level, heat ON)
- Phase 3.6 multi-level baseline: UCMBostonMultiLevel (multi-level, heat OFF)
- Design contracts: all nine from Phase 3 preserved (level-aware indexing, MPI safety, etc.)
