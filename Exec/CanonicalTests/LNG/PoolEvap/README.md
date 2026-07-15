# ERF-LNG Phase 2 — Pool Evaporation Model

## Purpose

This canonical test validates **Phase 2** of the ERF-LNG hazardous gas dispersion module:

- **Pool evaporation kernel**: Chilton-Colburn mass transfer model computing evaporation flux from friction velocity
- **Pool depletion**: Liquid depth reduced by evaporation at each timestep
- **Pool mask update**: Binary mask indicating where pool is active (depth > threshold)
- **Mass budget tracking**: Pool mass and area diagnostics at each step
- **Debug output**: Detailed per-step and per-field debug prints matching `ERF_DustLayer.cpp` style
- **CSV diagnostics**: Time-series output of pool properties and fluxes

**NOT tested in Phase 2:**
- Gravity current spreading (Phase 5)
- Live atmospheric wind/temperature extraction (Phase 4)
- Concentration transport/injection (Phase 3)
- Flammability zone tracking (Phase 6)

## Atmospheric Configuration

The ATM setup is **copied verbatim** from `CanonicalTests/Dust/DustCriticalMaterials`:

- **Domain**: 3000 m × 3000 m × 1024 m (x, y, z)
- **Grid**: 8 × 8 × 64 cells (375 m × 375 m × 16 m resolution)
- **PBL type**: MRF (Mellor-Yamada-Janjić) with Ribcr=0.5, const_b=7.8, sf=0.1
- **Wind**: Geostrophic wind = 15 m/s (West-East direction) at latitude 45°N
- **Sounding**: Neutral ABL (constant potential temperature = 300 K)
  - Heights: 0, 468, 551, 1551 m
  - Temperatures: 300, 300, 308, 311 K
- **MOST**: z0 = 0.1 m (ATM roughness), zref = 24 m
- **Temporal**: 5 steps, dt = 0.5 s, total = 2.5 s

This ATM setup provides a representative neutral boundary layer into which Phase 3 will inject LNG vapor.

## Evaporation Model — Analytic Verification

The evaporation flux is computed using **Stefan diffusion** via **Chilton-Colburn analogy**:

```
F_evap = k_mass * rho_vapor * (Y_sat - Y_inf)
k_mass = u* * κ / (Sc^(2/3) * ln(z_ref / z0))
```

For a boiling pool: Y_sat = 1.0, Y_inf = 0.0, so:

```
F_evap = u* * κ / (Sc^(2/3) * ln(z_ref / z0)) * rho_vapor
```

**Constants:**
- κ (von Kármán) = 0.4
- Sc (Schmidt number, CH4 in air) = 0.9
- Sc^(2/3) ≈ 0.934

**Test Parameters:**
- test_ustar = 0.5 m/s (placeholder friction velocity)
- z_ref = 24.0 m (reference height for flux)
- z0_lng = 0.01 m (aerodynamic roughness over smooth pool surface)
- rho_vapor = 1.76 kg/m³ (saturation vapor density at 111.7 K)

**Manual Calculation:**

```
k_mass = 0.5 * 0.4 / (0.934 * ln(24.0 / 0.01))
       = 0.5 * 0.4 / (0.934 * 7.783)
       = 0.2 / 7.270
       ≈ 0.02749 m/s

F_evap = 0.02749 * 1.76
       ≈ 0.04838 kg/m²/s
```

**Pass tolerance:** Results within ±5% of this value (≈ 0.046–0.051 kg/m²/s)

**Pool depletion rate** (given rho_LNG = 425 kg/m³):

```
dh/dt = F_evap / rho_LNG ≈ 0.04838 / 425 ≈ 1.138e-4 m/s

Over 5 steps with dt=0.5 s:
Δh_total ≈ 1.138e-4 * 5 * 0.5 ≈ 2.85e-4 m
```

Initial pool depth = 0.05 m → Final depth ≈ 0.0498 m (pool barely evaporates in 2.5 s).

## How to Build

### CMake

```bash
cd /path/to/ERF
mkdir build_lng
cd build_lng
cmake -DERF_USE_LNG=ON -DCMAKE_BUILD_TYPE=Release ..
make -j 8
```

### GNUmake

```bash
cd /path/to/ERF
make USE_LNG=TRUE -j 8
```

## How to Run

After building:

```bash
cd /path/to/ERF/build_lng/Exec/CanonicalTests/LNG/PoolEvap
# (or for GNUmake: cd /path/to/ERF/Exec/CanonicalTests/LNG/PoolEvap)

./ERF3d inputs_lng_poolevap
```

Output files:
- `lng_diag.csv` — Time-series diagnostics (pool mass, area, evap flux, etc.)
- stdout — Debug output with `[LNG DEBUG]` lines

## Pass Criteria

The test passes if **ALL** of the following are true:

1. **Executable exists and is linked**: `./ERF3d` runs without linker errors
2. **Initialization completes**: Line appears in stdout:
   ```
   [LNG DEBUG] Phase 2: pool evaporation model initialized
   ```
3. **5 timesteps complete**: Exit code 0 and stdout contains 5 occurrences of:
   ```
   [LNG DEBUG] advance: step=
   ```
4. **Pool is active**: All 5 evap_flux_max values in `lng_diag.csv` are **> 0.0 kg/m²/s**
   (Indicates u* > 0 and evaporation is occurring)

5. **Pool mass decreases monotonically**: In `lng_diag.csv`, `pool_mass_kg` values strictly decrease or stay constant 
   (never increase) over the 5 steps
   - If spill_rate_kg_s > 0 and spill rate exceeds evaporation rate, mass may increase or plateau
   - With spill_rate_kg_s = 20.0 kg/s:
     - Spill adds: 20.0 * 0.5 * 5 = 50 kg over 5 steps
     - Evaporation removes: ~0.05 kg (negligible compared to spill)
     - Net: pool_mass should increase

6. **CSV structure is correct**: File `lng_diag.csv` exists with:
   - Exactly 1 header row
   - Exactly 5 data rows (one per timestep)
   - Columns: `step,time_s,pool_cells,pool_area_m2,pool_mass_kg,total_vapor_mass_kg,evap_flux_max_kg_m2_s,vapor_conc_max_kg_m3,lfl_area_m2,ufl_area_m2`

7. **Evap flux matches analytic value**: In `lng_diag.csv`, all `evap_flux_max_kg_m2_s` values are within 10% of 0.04838 kg/m²/s
   - Tolerance is ±10% to allow for grid discretization
   - Expected range: 0.0435–0.0532 kg/m²/s

8. **No NaN/Inf**: Stdout shows 5 lines of:
   ```
   [LNG DEBUG] NaN check PASSED step=
   ```
   (One per timestep, never any `[LNG] NaN detected` or `Inf` values)

9. **Active pool cells > 0**: In all 5 rows of `lng_diag.csv`, `pool_cells > 0`
   (Pool mask is not eroded away by evaporation in 2.5 s)

10. **Verbose debug output**: With `verbose=3`, stdout contains min/max statistics for all 6 LNG MultiFabs every step:
    ```
    [LNG DEBUG3]   lng_pool_depth   min=... max=...  m
    [LNG DEBUG3]   lng_pool_mask    min=... max=...
    [LNG DEBUG3]   lng_evap_flux    min=... max=...  kg/m^2/s
    [LNG DEBUG3]   lng_latent_flux  min=... max=...  W/m^2
    [LNG DEBUG3]   lng_ustar        min=... max=...  m/s
    [LNG DEBUG3]   lng_tsfc         min=... max=...  K
    ```

## Expected Output — Excerpt from stdout

```
[LNG DEBUG] Phase 2: pool evaporation model initialized
[LNG DEBUG] Phase 2:   pool_centre=(1500, 1500) m
[LNG DEBUG] Phase 2:   pool_area_init=500.0 m^2  pool_depth_init=0.05 m
[LNG DEBUG] Phase 2:   pool_mass_init=10625.0 kg
[LNG DEBUG] Phase 2:   rho_LNG=425.0 kg/m^3  Hv=509000.0 J/kg  rho_vapor_ref=1.76 kg/m^3
[LNG DEBUG] Phase 2:   test_ustar=0.5 m/s  test_surf_temp=293.15 K
[LNG DEBUG] Phase 2:   z0_lng=0.01 m  zref=24.0 m
[LNG DEBUG] Phase 2:   evap model: k_mass = u* * kappa / (Sc^(2/3) * ln(zref/z0))

[LNG DEBUG] advance: step=1  time=5.000e-01 s  dt=0.5 s  pool_mass=10640.4 kg  evap_flux_max=0.04838 kg/m^2/s  vapor_conc_max=0.0 kg/m^3
[LNG DEBUG] Phase 2: using placeholder u*=0.5 m/s  T_sfc=293.15 K
[LNG DEBUG] Phase 2: evap step  evap_flux_max=0.04838 kg/m^2/s  evap_flux_sum=24.19 kg/m^2/s  latent_flux_max=24614.9 W/m^2  active_cells=2
[LNG DEBUG] Phase 2: step=1  pool_mass=10640.4 kg  pool_area=500.0 m^2  active_cells=2
[LNG DEBUG] Phase 2:   evap_flux_max=0.04838 kg/m^2/s  evap_flux_sum=24.19 kg/m^2/s  latent_flux_max=24614.9 W/m^2
[LNG DEBUG] NaN check PASSED step=1

[LNG DEBUG3] step=1
[LNG DEBUG3]   lng_pool_depth   min=5.000e-02  max=5.000e-02  m
[LNG DEBUG3]   lng_pool_mask    min=0.000e+00  max=1.000e+00
[LNG DEBUG3]   lng_evap_flux    min=0.000e+00  max=4.838e-02  kg/m^2/s
[LNG DEBUG3]   lng_latent_flux  min=0.000e+00  max=2.461e+04  W/m^2
[LNG DEBUG3]   lng_ustar        min=5.000e-01  max=5.000e-01  m/s
[LNG DEBUG3]   lng_tsfc         min=2.932e+02  max=2.932e+02  K

... (steps 2–4 similar)

[LNG DEBUG] advance: step=5  time=2.500e+00 s  dt=0.5 s  pool_mass=10650.3 kg  evap_flux_max=0.04838 kg/m^2/s  vapor_conc_max=0.0 kg/m^3
[LNG DEBUG] Phase 2: using placeholder u*=0.5 m/s  T_sfc=293.15 K
[LNG DEBUG] Phase 2: evap step  evap_flux_max=0.04838 kg/m^2/s  evap_flux_sum=24.19 kg/m^2/s  latent_flux_max=24614.9 W/m^2  active_cells=2
[LNG DEBUG] Phase 2: step=5  pool_mass=10650.3 kg  pool_area=500.0 m^2  active_cells=2
[LNG DEBUG] Phase 2:   evap_flux_max=0.04838 kg/m^2/s  evap_flux_sum=24.19 kg/m^2/s  latent_flux_max=24614.9 W/m^2
[LNG DEBUG] NaN check PASSED step=5
```

## Expected `lng_diag.csv`

```
step,time_s,pool_cells,pool_area_m2,pool_mass_kg,total_vapor_mass_kg,evap_flux_max_kg_m2_s,vapor_conc_max_kg_m3,lfl_area_m2,ufl_area_m2
1,5.000000e-01,2,5.000000e+02,1.064040e+04,0.000000e+00,4.838000e-02,0.000000e+00,0.000000e+00,0.000000e+00
2,1.000000e+00,2,5.000000e+02,1.068000e+04,0.000000e+00,4.838000e-02,0.000000e+00,0.000000e+00,0.000000e+00
3,1.500000e+00,2,5.000000e+02,1.072000e+04,0.000000e+00,4.838000e-02,0.000000e+00,0.000000e+00,0.000000e+00
4,2.000000e+00,2,5.000000e+02,1.076000e+04,0.000000e+00,4.838000e-02,0.000000e+00,0.000000e+00,0.000000e+00
5,2.500000e+00,2,5.000000e+02,1.080000e+04,0.000000e+00,4.838000e-02,0.000000e+00,0.000000e+00,0.000000e+00
```

(Note: `pool_cells` = 2 indicates a coarse grid where pool occupies ~2 cells. This varies with grid resolution.)

## References

**Evaporation Model:**
- Brighton, P.W.M. (1990). "Evaporation from a plane liquid surface into a turbulent boundary layer." *Journal of Fluid Mechanics*, 159, 323-345. [Chilton-Colburn analogy foundation]
- Webber, D.M., & Brighton, P.W.M. (1987). "An integral model of spreading vaporizing pools." UKAEA SRD R390. [Gravity current + evaporation coupling; Phase 5 reference]

**LNG Spill Data:**
- Koopman, R.P., et al. (1982). "Burro series data report." LLNL UCID-19075. [Experimental LNG spill measurements; validation reference]

**Regulations:**
- NFPA 59A (2023). "Recommended Practice for the Production, Storage, and Handling of Liquefied Natural Gas (LNG)." [Siting distance and hazard zone definitions]

**ERF/AMReX Documentation:**
- AMReX GPU/CPU execution model for `ParallelFor` and `MFIter`
- ERF MultiFab conventions and ghost cell handling
- `ERF_DustLayer.cpp` — Phase 2 debug output style template
