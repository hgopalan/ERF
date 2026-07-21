# SLUCM Phase 1.1 Scaffold Test

## Test Purpose

Verify that the ERF-SLUCM module can be compiled, linked, and initialized without errors. This is a **no-physics test** — the entire module is a stub in Phase 1.1, so no urban surface computations occur.

## What This Test Verifies

1. **Build integration** — All source files compile and link correctly
2. **ParmParse wiring** — All `erf.ucm.*` parameters are read from inputs file
3. **Prerequisite checks pass** — No configuration constraint violations
4. **Initialization completes** — ERF initializes successfully with UCM enabled
5. **Time stepping completes** — Simulation advances 2 time steps without crash
6. **Debug output is correct** — Expected console messages appear:
   - `[UCM]` startup banner with all parameter values
   - `[UCM DEBUG] create_ucm_grid stub called (Phase 1.1 no-op)`
   - **NOT:** `[UCM] Terrain-following mode active` (because `erf.use_terrain = false`)

## Expected Exit Code

**0** (success) after advancing 2 time steps

## Expected Console Output

```
[UCM] =========================================================
[UCM] SLUCM Module Initialization Summary (Phase 1.1 Scaffold)
[UCM] =========================================================
[UCM]   enable              = true
[UCM]   ucm_debug           = true
[UCM]   anchor_level        = 0
[UCM]   static_refinement   = true
[UCM]   grid_ratio          = 1
[UCM]   allow_steep_terrain = false
[UCM]   atm_feedback        = 0
[UCM]   zref [m]            = 2
[UCM]   alpha_ucm [m]       = 10
[UCM]   H_bldg_uniform [m]  = 10
[UCM]   W_road_uniform [m]  = 10
[UCM]   W_roof_uniform [m]  = 10
[UCM]   albedo_roof         = 0.2
[UCM]   albedo_wall         = 0.2
[UCM]   albedo_road         = 0.15
[UCM]   emissivity_roof     = 0.9
[UCM]   emissivity_wall     = 0.9
[UCM]   emissivity_road     = 0.94
[UCM]   ucm_plot_int        = -1
[UCM]   ucm_diag_file       = ucm_diag.dat
[UCM] =========================================================

[UCM DEBUG] create_ucm_grid stub called (Phase 1.1 no-op)
```

## Domain Configuration

- **Spatial:** 8×8×16 cells, 3000 m × 3000 m × 1024 m domain (small for fast execution)
- **PBL:** MYNN2.5 boundary layer (simplest available)
- **Time:** 2 steps at 1 s dt (2 seconds total simulation time)
- **Boundary conditions:**
  - Bottom: Surface layer (no flux)
  - Top: Slip wall (free)
  - Sides: Periodic
- **Physics:**
  - No moisture
  - No terrain (flat domain)
  - No radiation
  - No LSM (surface layer only)
- **UCM:**
  - One-way coupling only (`atm_feedback = 0.0`)
  - Homogeneous building height 10 m
  - Grid ratio 1 (no refinement)

## Sounding

Neutral atmosphere with:
- Surface potential temperature: 293.15 K
- Lapse rate: 0 K/100m (neutral)
- Wind: 5 m/s from west
- Humidity: 0 (disabled)

See `sounding_neutral_abl` for full profile.

## Future Regression Tests (Phase 1.4+)

In Phase 1.4, this test will be extended to verify **bit-for-bit reproducibility**:
- Run same inputs with `erf.ucm.enable = false`
- Compare plotfile output with enabled run
- Assert bit-for-bit identity (proving Phase 1.1 is a true no-op)

## Building and Running

```bash
cd Exec/CanonicalTests/SLUCM/UCMScaffold
mkdir build && cd build
cmake -DERF_ENABLE_UCM=ON .. && make
./erf_ucm_scaffold inputs
```

Expected final output:
```
Job finished
```

## References

- `Source/UrbanCanopy/UCM_DEVELOPMENT.md` — Phase 1.1 section
- `Source/UrbanCanopy/ERF_UCM.H` — Design contracts and roadmap
- `Source/UrbanCanopy/ERF_UCMPrerequisites.cpp` — Prerequisite checks enforced
