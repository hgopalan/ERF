# LNG_WindExtraction — Phase 4: Live ATM Field Extraction

## Purpose

Validates Phase 4 extraction of live atmospheric fields from the ERF 3D solver to the 2D LNG grid:
- **`u*`** — friction velocity from `SurfaceLayer::get_u_star(0)`
- **Wind at `zref`** — vertical interpolation of face-staggered `xvel`/`yvel` to `z_surf + zref`
- **`T_sfc`** — surface skin temperature from `SurfaceLayer::get_t_surf(0)`
- **`PBLH`** — PBL height from `SurfaceLayer::get_pblh(0)`
- **`c_LNG_sfc`** — near-surface LNG vapor concentration from 3D conserved state at k=0

## Atmospheric Configuration

Identical to `LNG_ScalarInjection` and other neutral ABL tests (inherited from `DustCriticalMaterials`):
- **Domain**: 3000×3000×1024 m
- **Grid**: 8×8×64 cells
- **Sounding**: Neutral ABL (see `sounding_neutral_abl`)
- **Surface layer**: MRF (Mellor-Yamada-Nakanishi-Niino) with:
  - `z0 = 0.1 m` (momentum roughness)
  - `zref = 24 m` (reference height, **must match `erf.lng.zref`**)
- **Geostrophic wind**: 15 m/s (easterly, u-component)
- **Coriolis**: enabled at 45° N latitude
- **Timestep**: 0.5 s
- **Duration**: 5 steps (2.5 s total)

## Analytic Check

With 15 m/s geostrophic wind and neutral ABL, the MRF surface layer produces:
- **`u*` ≈ 0.5–0.6 m/s** at the surface (friction velocity scaled by Coriolis)
- **Wind at 24 m ≈ 12–13 m/s** (reduced from geostrophic due to surface drag)
- **`T_sfc` ≈ 293 K** (initialized at domain center from sounding)

Phase 4 extraction must recover these live values from the SurfaceLayer and interpolation kernel.

## Pass Criteria

All 8 criteria must be met:

1. **Exit code 0**: ERF3d runs 5 steps and exits successfully.
2. **Phase 4 extraction active**: `[LNG DEBUG] Phase 4: live ATM extraction active` appears 5 times in stdout.
3. **`u*` extracted**: `[LNG DEBUG] Phase 4: u* extracted  ustar_max > 0 m/s` appears 5 times with positive values.
4. **Wind extracted**: `[LNG DEBUG] Phase 4: wind extracted  u_max > 0 m/s` appears 5 times with positive values.
5. **Phase 3 not regressed**: `[LNG DEBUG] Phase 3: apply_to_cc_source` still appears 5 times (one per step).
6. **Evaporation driven by live u***: `evap_flux_max > 0` in all 5 CSV data rows (not placeholders).
7. **No NaNs**: `[LNG DEBUG] NaN check PASSED` appears 5 times.
8. **Fallback behavior (optional)**: If SurfaceLayer is unavailable at test time, the placeholder path triggers: `[LNG DEBUG] Phase 4: placeholder path` with `test_ustar=0.5 m/s`, `test_T_sfc=293.15 K`, `test_wind=15.0 m/s`. This is acceptable in minimal test environments.

## Expected Stdout Excerpt

```
[LNG] ===== ERF-LNG Phase 1 initialized =====
[LNG DEBUG] Phase 1: pool_centre=(...) m  area=500 m^2  depth=0.05 m
[LNG DEBUG] Phase 1: LNGGrid created ... grid_ratio=1
[LNG DEBUG] Phase 1: MultiFabs allocated (pool_depth, pool_mask, evap_flux, ...)
[LNG DEBUG] Phase 1: lng_diag.csv header written
[LNG DEBUG] Phase 3: lng_scalar_comp=... (RhoScalar_comp+1)
[LNG DEBUG] Phase 2: pool evaporation model initialized
[LNG DEBUG] Phase 2:   pool_area_init=500 m^2  pool_mass_init=... kg
[LNG DEBUG] Phase 2: using placeholder u*=... OR Phase 4: live ATM extraction active

[LNG DEBUG] advance: step=1  time=5.000e-01 s  dt=0.5 s  pool_mass=... kg  evap_flux_max=... kg/m^2/s
[LNG DEBUG] Phase 4: live ATM extraction active  u*_max=0.5-0.6 m/s  u_ref_max=... m/s  PBLH_max=... m
[LNG DEBUG] Phase 4: u* extracted  ustar_max=0.5-0.6 ustar_min=... m/s
[LNG DEBUG] Phase 4: wind extracted  u_max=... v_max=... m/s at zref=24 m
[LNG DEBUG] Phase 4: T_sfc extracted  T_max=... T_min=... K
[LNG DEBUG] Phase 4: PBLH extracted  PBLH_max=... PBLH_min=... m
[LNG DEBUG] Phase 2: step=1  pool_mass=... kg  pool_area=500 m^2  active_cells=1
[LNG DEBUG] Phase 3: apply_to_cc_source step=1  F_evap_atm_max=... kg/m^2/s  scalar_comp=...  feedback=1
[LNG COUPLING] Phase 3: F_evap_max=... kg/m^2/s RhoLNG_tend_max=... sum=...
[LNG DEBUG] Phase 4: conc_sfc extracted  conc_sfc_max=... kg/m^3  conc_sfc_sum=...
[LNG DEBUG] NaN check PASSED step=1

... (steps 2–5 repeat similar pattern) ...

[LNG DEBUG] write_output step=5  time=2.500e+00 s  pool_mass=... kg
```

## References

### Wind Interpolation Algorithm

- **Source**: `Source/Fire/ERF_FireWindExtract.cpp` (original)
- **Phase 9 analog**: `Source/Dust/ERF_DustWindExtract.cpp`
- **Algorithm**: Vertical interpolation of face-staggered wind to cell centers at target height `z_target = z_surf + zref`

### Surface Layer Model

- **Scheme**: Mellor-Yamada-Nakanishi-Niino (MYNN) Level 2.5, also known as MRF
- **Reference**: Hong & Pan (1996). Mon. Wea. Rev., 124, 2322–2339.
  - https://doi.org/10.1175/1520-0493(1996)124<2322:NBLVDI>2.0.CO;2

### LNG Physics

- **Pool evaporation**: Zeman & Tennekes (1977). J. Fluid Mech., 79, 233–256.
- **Latent heat**: Hentze & Richter (2016). J. Hazard. Mater., 300, 627–638.

## Running the Test

### Command Line

```bash
cd Exec/CanonicalTests/LNG/LNG_WindExtraction
${ERF_BUILD_DIR}/Exec/ERF3d inputs_lng_windextraction > run.log 2>&1
```

### CMake / CTest

```bash
cd build
ctest -R LNG_WindExtraction -VV
```

### Expected Behavior

- All Phase 4 debug prints appear 5 times
- CSV file `lng_diag.csv` is created with 6 lines (1 header + 5 data rows)
- No NaN or Inf values in any LNG MultiFab
- Evaporation flux values are driven by extracted `u*` (not test placeholders)
- Surface layer fields (`u*`, `T_sfc`, `PBLH`) are extracted successfully

## Debugging Tips

1. **Missing Phase 4 prints**: Check that SurfaceLayer is properly allocated in ERF.cpp and passed to `m_lng_layer->advance()`.
2. **Zero `u*` values**: Verify MRF surface layer type is enabled (`zlo.type = "surface_layer"`) and `erf.pbl_type = "MRF"`.
3. **Divergence in conc_sfc**: Confirm `S_cons` is passed correctly and `lng_scalar_comp` is set to `RhoScalar_comp + 1`.
4. **Lingering placeholder messages**: If both placeholder and live extraction messages appear, check the `have_atm` branching logic in Phase 4.
