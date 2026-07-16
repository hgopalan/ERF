# ERF-LNG Phase 3: 2D→3D Scalar Injection Coupling Test

## Purpose

This test validates Phase 3 of the ERF-LNG module: one-way coupling from the 2D LNG evaporation layer to the 3D atmospheric model via injection of LNG vapor mass into a passive scalar at the surface layer (k=0).

**Key validation goal:** Confirm that 2D evaporation flux is correctly coarsened from the LNG grid to the atmospheric level-0 grid, then distributed into the lowest atmospheric layer as a mass source term.

## Physics

**One-step explicit lag injection:**
- Flux computed in LNG layer at step n
- Injected into atmosphere at step n+1, before `advance_dycore()`
- One-way coupling: LNG → ATM only (no ATM→LNG feedback in Phase 3)

**Tendency formula at k=0:**
```
d(RhoLNG)/dt = F_evap * feedback / dz_k0  [kg/m^3/s]
```
where:
- `F_evap` = evaporation flux [kg/m²/s] from LNG layer
- `feedback` = coupling strength [0, 1]
- `dz_k0` = thickness of lowest atmospheric layer [m]
- Applied only at k=0; zero everywhere else

## Atmospheric Configuration

**Domain:**
- 3D Cartesian box: 3000 m × 3000 m × 1024 m
- Grid: 8 × 8 × 64 cells (coarse resolution for fast testing)
- Flat terrain (no EB)
- Doubly-periodic in x,y; slip-wall at top (z_hi)

**Physics:**
- Neutral ABL (stable stratification: dθ/dz = 0.003 K/m)
- MRF PBL model with geostrophic wind forcing (15 m/s West, u = +15 m/s)
- No moisture, no microphysics, no radiation
- No molecular diffusion, no LES turbulence
- Passive scalar transport enabled (`erf.transport_scalar = true`)

**Boundary conditions:**
- Surface layer BC at z_lo (z0 = 0.1 m)
- Reference height for wind: zref = 24.0 m
- No surface heat flux (neutral, dry)

**Initial condition:**
- Read from sounding file (`sounding_neutral_abl`)
- Neutral profile: θ = 298.15 K, ρ ≈ 1.2 kg/m³

**Temporal:**
- Fixed dt = 0.5 s
- Run 5 timesteps → total 2.5 s

## LNG Configuration

**Pool:**
- Area: 500 m² (radius ≈ 12.6 m)
- Initial depth: 0.05 m
- Centered at domain center
- Spill rate: 20 kg/s (maintains active pool throughout 5 steps)

**Evaporation model:**
- Test friction velocity: u* = 0.5 m/s
- Test surface temperature: T_sfc = 293.15 K
- Roughness: z0_lng = 0.01 m
- Vapor saturation density: ρ_vapor = 1.76 kg/m³
- Latent heat: Hv = 509,000 J/kg

**Coupling:**
- **`atm_feedback = 1.0`** (full coupling strength; key for Phase 3)
- Grid ratio: 1 (LNG grid matches ATM level-0)
- Scalar component: RhoScalar_comp + 1

## Pass Criteria

All criteria must be satisfied to pass the test:

1. **Exit code 0:** Program completes successfully (no crash, no Abort).

2. **5 apply_to_cc_source calls:** `[LNG DEBUG] Phase 3: apply_to_cc_source step=` appears exactly 5 times in stdout (one per timestep).

3. **5 coupling messages:** `[LNG COUPLING] Phase 3:` appears exactly 5 times with:
   - `F_evap_max > 0` (evaporation is active)
   - `RhoLNG_tend_max > 0` (tendency is positive, vapor injected)
   - `sum > 0` (total mass change is positive)

4. **RhoLNG_tend_sum > 0 every step:** Integrated tendency is always positive (monotonic mass increase in scalar at k=0).

5. **5 NaN checks PASSED:** `[LNG DEBUG] NaN check PASSED step=` appears 5 times (no NaN corruption).

6. **lng_diag.csv:** Created with exactly 6 lines: 1 header + 5 data rows (one per step).

7. **With `atm_feedback = 0.0`:** Re-running with feedback gated off should produce **zero** `[LNG COUPLING]` prints (entire injection mechanism bypassed).

## Implementation Notes

### Files Updated
- **`Source/LNG/ERF_LNGAtmCoupling.H`:** Real coarsen function + function declaration
- **`Source/LNG/ERF_LNGAtmCoupling.cpp`:** Implementation of `apply_lng_tendency_to_cc_source`
- **`Source/LNG/ERF_LNGLayer.cpp`:** 
  - Set `m_lng_scalar_comp = RhoScalar_comp + 1` in `initialize()`
  - Implement `apply_to_cc_source()` with coarsening + injection
- **`Source/TimeIntegration/ERF_Advance.cpp`:** Wire call before `advance_dycore()`
- **`Source/LNG/Make.package`:** Register `ERF_LNGAtmCoupling.cpp`

### Reference Pattern
All implementations follow the Dust module pattern:
- `apply_dust_tendency_to_cc_source()` in `Source/Dust/ERF_DustAtmCoupling.cpp`
- One-to-one correspondence: coarsen via `amrex::average_down()`, apply to k=0 only

### Debug Output
When `lng_debug = true`, every step prints:
```
[LNG DEBUG] Phase 3: apply_to_cc_source step=1  F_evap_atm_max=<value> kg/m^2/s ...
[LNG COUPLING] Phase 3: F_evap_max=<value> kg/m^2/s  RhoLNG_tend_max=<value> kg/m^3/s ...
```

## Expected Behavior

**With `atm_feedback = 1.0` (normal run):**
- Every step: pool evaporates at ~0.05 kg/m²/s
- Vapor injected into scalar at k=0 with full strength
- Scalar mass increases monotonically

**With `atm_feedback = 0.0` (validation):**
- Pool evaporates as normal
- **No injection** (check line is skipped)
- No `[LNG COUPLING]` messages in output
- Scalar remains unchanged

## Running the Test

```bash
# Run with coupling enabled (should pass)
${ERF3d} inputs_lng_scalarinjection

# Run with coupling disabled (validation check)
# Modify inputs: erf.lng.atm_feedback = 0.0
${ERF3d} inputs_lng_scalarinjection
```

## Verification Checklist

- [ ] Compiles with `-DERF_USE_LNG=ON`
- [ ] Runs to completion (exit 0)
- [ ] 5 `[LNG DEBUG] Phase 3: apply_to_cc_source` messages
- [ ] 5 `[LNG COUPLING] Phase 3:` messages with `F_evap_max > 0`
- [ ] 5 `RhoLNG_tend_sum > 0` values
- [ ] 5 `NaN check PASSED` messages
- [ ] `lng_diag.csv` has 6 lines
- [ ] No warnings or Aborts
- [ ] Disable feedback → no coupling messages
