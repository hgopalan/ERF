# ERF-SLUCM Phase 1.2 — Homogeneous Grid Test

## Purpose

This canonical test verifies the Phase 1.2 implementation of the SLUCM grid refinement and MultiFab allocation infrastructure:

1. **UCM grid construction** with uniform refinement (`grid_ratio = 2`)
2. **MultiFab allocation** on the UCM grid with correct ghost cells (`IntVect(1,1,0)`)
3. **Homogeneous fill** of all URBPARM fields from ParmParse values
4. **Urban mask allocation** (`is_urban = 1` everywhere — fully urban patch)
5. **Debug output verification** — all Phase 1.2 debug banners and messages printed
6. **Bit-for-bit ATM regression** — atmospheric state unchanged by UCM (Phase 1.2 is one-way no-op)

## Atmospheric Setup

Inherited from the neutral ABL canonical test on branch `ERF-SLUCM`:

- **Domain:** `3000 × 3000 × 1024` m
- **ATM grid:** `8 × 8 × 64` cells (coarse grid test)
- **PBL:** MRF with neutral stability
- **Wind forcing:** Geostrophic `15 m/s` in x-direction
- **Initial state:** Input sounding (`sounding_neutral_abl`)
- **Terrain:** None
- **Time steps:** 2 steps @ 1.0 s `fixed_dt`

## UCM Configuration

- **Grid ratio:** `2` (ATM 8×8 → UCM 16×16×1)
- **Morphology:** Homogeneous buildings 10 m tall, 10 m road/roof widths
- **Albedos:** 0.20 (roof, wall), 0.15 (road)
- **Emissivities:** 0.90 (roof, wall), 0.94 (road)
- **Debug mode:** Enabled (`ucm_debug = true`)
- **Feedback:** One-way only (`atm_feedback = 0.0`)

## Pass Criteria

### Criterion 1: Exit Code
The simulation must complete without error:
```bash
./erf_ucm_homogeneous_grid inputs
echo $?  # Must be 0
```

### Criterion 2: UCM Grid Extents
Debug output must show UCM grid dimensions `16 × 16 × 1`:
```
[UCM][1.2][create_ucm_grid] UCM output: ... domain=[(0,0,0) (15,15,0)]
```
(Domain hi-index `15` for both x,y indicates `16 cells`)

### Criterion 3: Phase 1.2 Debug Messages
All required debug banners must appear in stdout:
- `[UCM][1.2][ERF] calling create_ucm_grid for lev=0`
- `[UCM][1.2][create_ucm_grid] ATM input: ba.size=... domain=...`
- `[UCM][1.2][create_ucm_grid] grid_ratio = 2`
- `[UCM][1.2][create_ucm_grid] UCM output: ... domain=[(0,0,0) (15,15,0)]`
- `[UCM][1.2][ERF] calling allocate_ucm_fields for lev=0`
- `[UCM][1.2][allocate_ucm_fields]` per-field messages (16 MultiFabs)
- `[UCM][1.2][allocate_ucm_fields] allocated 16 MultiFabs on UCM grid at lev=0`
- `[UCM][1.2][ERF] calling fill_ucm_fields_homogeneous for lev=0`
- `[UCM][1.2][fill_ucm_fields_homogeneous]` per-field value messages
- `[UCM][1.2][fill_ucm_fields_homogeneous] is_urban = 1 everywhere on UCM grid at lev=0`
- Phase 1.2 grid-check banner: `UCM grid extents = 16 × 16 × 1 (cells)`

### Criterion 4: MultiFab Field Values
Spot-checks on UCM MultiFab values (via test harness or manual post-processing):
- `H_bldg` = 10.0 m
- `albedo_roof` = 0.20
- `albedo_wall` = 0.20
- `albedo_road` = 0.15
- `emissivity_road` = 0.94
- `is_urban` = 1 (everywhere in domain)

### Criterion 5: ATM Bit-for-Bit Regression
Run test twice — once with `erf.ucm.enable = true` and once with `erf.ucm.enable = false`.
Plot-file values for `Rho`, `RhoTheta`, `U`, `V`, `W` at step 2 must be identical (UCM is a no-op).

### Criterion 6: Parameter Banner
Startup banner must show correct parameter values:
```
[UCM] =========================================================
[UCM] SLUCM Module Initialization Summary (Phase 1.1 Scaffold)
[UCM] =========================================================
[UCM]   enable              = true
[UCM]   ucm_debug           = true
[UCM]   grid_ratio          = 2
[UCM]   H_bldg_uniform [m]  = 10.0
...
```

## How to Run

### Build (from ERF root)
```bash
cd /home/runner/work/ERF/ERF
mkdir -p build && cd build
cmake -DEREn_ENABLE_UCM=ON ..
make -j 4
```

### Execute
```bash
cd Exec/CanonicalTests/SLUCM/UCMHomogeneousGrid
cmake -B build -DERF_ENABLE_UCM=ON
cmake --build build
./build/erf_ucm_homogeneous_grid inputs
```

### Expected Output (first 50 lines)
```
...
[UCM] =========================================================
[UCM] SLUCM Module Initialization Summary (Phase 1.1 Scaffold)
[UCM] =========================================================
[UCM]   enable              = true
[UCM]   ucm_debug           = true
[UCM]   grid_ratio          = 2
...
[UCM] =========================================================

[UCM][1.2][ERF] calling create_ucm_grid for lev=0
[UCM][1.2][create_ucm_grid] ATM input: ba.size=1 boxes, domain=[(0,0,0) (7,7,63)]
[UCM][1.2][create_ucm_grid] ATM physical extent: x=[0, 3000], y=[0, 3000], z=[0, 1024] m
[UCM][1.2][create_ucm_grid] grid_ratio = 2
[UCM][1.2][create_ucm_grid] UCM output: ba.size=1 boxes, domain=[(0,0,0) (15,15,0)]
[UCM][1.2][create_ucm_grid] UCM physical extent: x=[0, 3000], y=[0, 3000], z=[0, 1] m
[UCM][1.2][ERF] calling allocate_ucm_fields for lev=0
[UCM][1.2][allocate_ucm_fields] H_bldg: 1 boxes, ngrow=(1,1,0), ncomp=1
...
[UCM][1.2][allocate_ucm_fields] is_urban (iMultiFab): 1 boxes, ngrow=(1,1,0), ncomp=1
[UCM][1.2][allocate_ucm_fields] allocated 16 MultiFabs on UCM grid at lev=0
[UCM][1.2][ERF] calling fill_ucm_fields_homogeneous for lev=0
[UCM][1.2][fill_ucm_fields_homogeneous] H_bldg = 10 m
[UCM][1.2][fill_ucm_fields_homogeneous] albedo_roof = 0.2
...
[UCM][1.2][fill_ucm_fields_homogeneous] is_urban = 1 everywhere on UCM grid at lev=0
[UCM] =========================================================
[UCM] Phase 1.2 — Grid and Fields Check
[UCM] =========================================================
[UCM]   UCM grid extents   = 16 × 16 × 1 (cells)
[UCM]   Refinement ratio   = 2
[UCM]   Ghost cells        = IntVect(1, 1, 0)
[UCM]   All fields allocated: true
[UCM]   is_urban set to 1 everywhere (homogeneous patch)
[UCM] =========================================================
```

## References

- Phase 1.2 specification: `Source/UrbanCanopy/UCM_DEVELOPMENT.md`
- Phase 1.1 foundation: `Exec/CanonicalTests/SLUCM/UCMScaffold/README.md`
- Dust grid pattern: `Source/Dust/ERF_DustGrid.H/cpp`
- MRF PBL setup: `Exec/CanonicalTests/ABL/MRF_Enhancements/canonical/inputs`
