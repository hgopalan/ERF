# LNG_GravityCurrent Test

## Purpose

This canonical test verifies all three Phase 5 physics components of the ERF-LNG module:

1. **Shallow-water gravity current PDEs** — 2D depth-averaged velocity and depth evolution
2. **Richardson transition criterion** — Automatic handoff from 2D gravity current to 3D ERF dispersion
3. **Flammability zone tracking** — LFL/UFL exceedance detection and area computation

## Physical Configuration

- **Atmospheric sounding**: Neutral boundary layer (constant potential temperature θ = 300 K, linearly stratified above)
- **Initial LNG pool**: 500 m² area, 0.05 m depth, at domain center
- **Release**: Continuous spill at 20 kg/s for 100 seconds
- **Wind**: 15 m/s test wind speed (constant placeholder in Phase 5)
- **Friction velocity**: 0.5 m/s (constant placeholder in Phase 5)

## Physics Verification

### Gravity Current Spreading

**Expected behavior**: Dense LNG vapor spreads radially outward on the 2D slab due to pressure gradient.

**Analytical check** (Didden 1982, similarity solutions for instantaneous release):
- Initial front radius: R₀ = sqrt(500/π) ≈ 12.6 m
- Gravity current speed: u ~ sqrt(g'*h_init) = sqrt(9.81*(1.76-1.225)/1.225 * 0.05) ≈ 0.46 m/s
- Richardson number: Ri = g'*h / u*² = 9.81*0.427*0.05 / 0.5² ≈ 0.84
- Since Ri (0.84) > Ri_crit (0.25), gravity current regime should be active initially

**Verification in output**:
- `[LNG DEBUG] Phase 5: gravity_current` lines appear with h_max > 0 and u_max > 0
- `gc_active_cells > 0` for initial timesteps (Ri_crit initially exceeded)
- h_max and u_max should evolve physically (depth decreases as source stops, velocity adjusts to wind shear)

### Richardson Transition

**Expected behavior**: As the gravity current spreads and weakens, Richardson number falls below Ri_crit, transitioning control to 3D ERF.

**Verification in output**:
- ri_flag transitions from 0 (active GC) to 1 (mixed regime) as simulation progresses
- `mixed_cells` count should increase over time as gravity current dissipates
- By step 20, many or all cells may be in mixed regime (depending on wind shear and evaporation)

### Flammability Tracking

**Expected behavior**: As concentration builds up in the 3D atmosphere, it returns to the 2D LNG grid via `fill_lng_conc_from_atm()`.
LFL/UFL exceedance zones are detected and areas computed.

**Verification in output**:
- `[LNG DEBUG] Phase 5: extract_atm_return_fields` appears each step
- `conc_sfc_max > 0` by step 10+ (3D transport fills k=0 slab)
- `[LNG DEBUG] Phase 5: flammability` lines appear with lfl_area and ufl_area
- lfl_area and ufl_area columns in CSV should be non-negative

## Test Configuration

| Parameter | Value | Unit | Rationale |
|-----------|-------|------|-----------|
| `max_step` | 20 | steps | Sufficient for GC and Ri transition development |
| `domain_nx/ny` | 32 | cells | Coarse grid for fast iteration |
| `grid_ratio` | 2 | - | 2× refinement on LNG grid (64×64) |
| `pool_area_m2` | 500 | m² | Modest spill for short timescale |
| `pool_depth_init_m` | 0.05 | m | Thin pool; fast evaporation and GC spreading |
| `spill_rate_kg_s` | 20 | kg/s | Continuous source to replenish pool |
| `gc_drag_coeff` | 2e-3 | - | Webber & Brighton (1987) reference value |
| `gc_ri_crit` | 0.25 | - | Benjamin (1968) transition threshold |

## Pass Criteria

✓ **Build**: Compiles with `-DERF_USE_LNG=ON` (no linker errors, no warnings)

✓ **Execution**: Runs 20 steps, exit code 0

✓ **Gravity current step output**:
- `[LNG DEBUG] Phase 5: gravity_current` appears in stdout
- `gc_h_max > 0` and `gc_u_max > 0` after step 1 (dense vapor has non-zero depth and speed)
- `gc_active_cells > 0` for initial steps (Ri > Ri_crit with given parameters)

✓ **Concentration extraction**:
- `[LNG DEBUG] Phase 5: extract_atm_return_fields` appears in each step's output
- Debug message includes `conc_sfc_max` and `conc_sfc_sum` values

✓ **Flammability computation**:
- `[LNG DEBUG] Phase 5: flammability` appears in each step's output
- `lfl_area` and `ufl_area` are non-negative scalars

✓ **CSV output**:
- `lng_diag.csv` has header row with all standard columns
- `vapor_conc_max_kg_m3 > 0` by step 10+ (real atmosphere-extracted value, not placeholder)
- `lfl_area_m2` column present, all entries non-negative
- `ufl_area_m2` column present, all entries non-negative
- 20 data rows written (one per step)

✓ **NaN check**:
- `[LNG DEBUG] NaN check PASSED` appears 20 times (once per step)
- No NaN abort triggered

✓ **Disable gravity current**:
- Edit `inputs_lng_gravitycurrent`: change `erf.lng.enable_gravity_current = false`
- Re-run: no `[LNG DEBUG] Phase 5: gravity_current` lines appear
- `lfl_area` and `ufl_area` columns still populate (flammability independent of GC)

✓ **Disable flammability**:
- Edit: `erf.lng.track_flammability = false`
- Re-run: no `[LNG DEBUG] Phase 5: flammability` lines appear
- But gravity current debug lines still present (GC independent of flammability)

✓ **Regression**:
- All 4 prior LNG canonical tests still pass:
  - `LNG_BuildOnly` (Phase 1 stub)
  - `LNG_ScalarInjection` (Phase 3 coupling)
  - `LNG_WindExtraction` (Phase 4 extraction)
  - (Any new test from Phase 2/3/4 development)

## Debugging Notes

If **gravity current is inactive** (all mixed_cells, no gc_active_cells):
- Check `Ri_crit` value; increase threshold if physically unrealistic
- Verify `test_ustar` is set; high u* depresses Ri = g'*h/u*²
- Check initial pool depth; deeper pools have higher Ri

If **concentration is zero** at step 10+:
- Ensure `transport_scalar = true` in inputs (ERF must advect LNG scalar)
- Check ATM grid and LNG grid have compatible alignments
- Verify `lng_scalar_comp` is correctly mapped (debug prints should show it)

If **flammability areas are zero**:
- Check `track_flammability = true`
- Verify `lfl_vol_fraction` and `ufl_vol_fraction` are reasonable (5% and 15% by default)
- Ensure concentration extraction is working (check `conc_sfc_max > 0` first)

## References

- **Webber, D.M. & Brighton, P.W.M.** (1987). UKAEA SRD R390. Gravity current models for hazardous gas vapours.
- **Didden, N. & Maxworthy, T.** (1982). J. Fluid Mech. 121:43. The dynamics of the head of a gravity current advancing over a horizontal surface.
- **Benjamin, T.B.** (1968). J. Fluid Mech. 31:209. Gravity currents and related phenomena.
- **DEGADIS** (Spicer & Havens 1986). Dense gas dispersion model — near-field physics reference.
- **NFPA 59A** — Recommended Practice for the Siting of Liquefied Natural Gas Facilities. Flammability thresholds: LFL = 5%, UFL = 15% for natural gas.
