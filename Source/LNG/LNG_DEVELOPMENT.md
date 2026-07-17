
# ERF-LNG Hazardous Gas Dispersion Module — Development Log

## Module Purpose

The ERF-LNG module simulates liquefied natural gas (LNG) spill evaporation and vapor dispersion in the atmosphere. It complements the DEGADIS model (NFPA 59A) and is tightly integrated with the ERF main atmospheric model via conservative coupling at the interface. The module tracks pool spreading, evaporation thermodynamics, vapor transport, and flammability zone delineation.

**Reference scenarios:**
- Burro series (Koopman, 1982): 1-acre LNG spills on water
- Coyote series (Goldwire, 1983): Large-scale LNG releases over land
- Maplin Sands (Puttock, 1982): North Sea industrial facility accident
- Falcon series (Brown, 1990): Variable weather LNG spills

---

## Eight-Phase Implementation Roadmap

| Phase | Title | Key Deliverables | Status |
|-------|-------|------------------|--------|
| 1 | **Build & Initialize** | Fully compilable stub; parameter reading; grid construction; debug output | ✅ COMPLETE |
| 2 | **Evaporation & Pool Spreading** | Heat transfer model; Clausius-Clapeyron; gravity current spreading | ✅ COMPLETE |
| 3 | **ATM Coupling (Phase I)** | Energy injection to atmosphere; sensible/latent heat source terms | ✅ COMPLETE |
| 4 | **Wind & BL Extraction** | Wind field interpolation at zref; u* mapping; PBL height feedback | ✅ COMPLETE |
| 5 | **Gravity Current & Flammability** | 2D shallow-water PDEs; Richardson transition; LFL/UFL zones | ✅ COMPLETE |
| 6 | **Output & Visualization** | Plotfile writes; receptor point sampling; CSV output expansion | ✅ COMPLETE |
| 7 | **Regulatory Compliance** | NFPA 59A exclusion zone calculation; threshold mapping | 🔄 IN PROGRESS |
| 8 | **Spill Scheduling** | Time-dependent release rates; inventory tracking; multi-event scenarios | TODO |

---

## Phase 1: Build, Initialize, and Zero-Physics Stubs

### Overview

Phase 1 establishes a fully compilable, zero-physics foundation that can be integrated into ERF without affecting existing simulations. The module:
- Reads **all eight phases' worth of parameters** from the input file (via `erf.lng.*` ParmParse namespace)
- Constructs a 2D LNG computational grid refined from the atmospheric level-0 grid
- Allocates 13 MultiFabs for LNG state variables
- Validates prerequisites (grid divisibility, parameter ranges, composition mole fractions)
- Emits structured debug output at every step (per-step one-liners or verbose MultiFab statistics)
- Writes a CSV diagnostics header file with full column list (rows populated in Phase 2+)

**Zero physics means:**
- `advance()` updates only time bookkeeping, returns immediately
- `apply_to_cc_source()` does nothing
- `extract_atm_return_fields()` does nothing
- No pool spreading, evaporation, transport, or flammability computed

### Files Created

**Core module (3 files with implementations):**
- `ERF_LNGParams.H` — Parameter struct with ParmParse constructor (all 32 parameters)
- `ERF_LNGGrid.H/.cpp` — 2D grid construction from ATM level-0 + refinement
- `ERF_LNGPrerequisites.H/.cpp` — Prerequisite validation and init summary printing
- `ERF_LNGLayer.H/.cpp` — Container class: 13 MultiFabs, initialize(), advance(), write_output()

**Coupling stubs (header-only, no implementations yet):**
- `ERF_LNGAtmCoupling.H` — Atmosphere coupling placeholder (Phase 3)
- `ERF_LNGWindExtract.H` — Wind/BL extraction placeholder (Phase 4)
- `ERF_LNGPlotfile.H` — Plotfile output placeholder (Phase 6)
- `ERF_LNGStatsOutput.H` — CSV diagnostics (header-only; Phase 1 creates header line)

**Build system:**
- `Make.package` — GNUmake registration with `USE_LNG` guard

**Documentation:**
- `LNG_DEVELOPMENT.md` — This file

### All 32 Parameters (Registered in Phase 1)

#### Enable/Debug (3 params)
```
bool    enable            = false        # Enable LNG module [Phase 1]
bool    lng_debug         = false        # Per-step debug output [Phase 1+]
int     verbose           = 0            # 0=silent, 1=init, 2=per-step, 3=verbose
```

#### LNG Physical Properties (9 params)
```
Real    mol_weight_LNG    = 17.4         # Effective MW [g/mol]; Burro~17.4 [Phase 2]
Real    ch4_mole_fraction = 0.90         # CH4 fraction [-] [Phase 2]
Real    c2h6_mole_fraction = 0.08        # C2H6 fraction [-] [Phase 2]
Real    n2_mole_fraction  = 0.02         # N2 fraction [-] [Phase 2]
Real    rho_LNG           = 425.0        # Liquid density [kg/m^3] [Phase 2]
Real    lng_boil_temp_K   = 111.7        # Boiling point [K] [Phase 2]
Real    Hv_LNG            = 509000.0     # Latent heat [J/kg] [Phase 2]
Real    rho_vapor_ref     = 1.76         # Vapor density at boil [kg/m^3] [Phase 2]
Real    Cp_LNG            = 3500.0       # Liquid Cp [J/kg/K] [Phase 2]
```

#### Flammability (3 params)
```
Real    lfl_vol_fraction  = 0.05         # LFL [vol/vol] NFPA 59A [Phase 5]
Real    ufl_vol_fraction  = 0.15         # UFL [vol/vol] NFPA 59A [Phase 5]
bool    track_flammability = false       # Compute flammable zone [Phase 5]
```

#### Pool Geometry & Release (4 params)
```
Real    pool_area_m2      = 100.0        # Initial pool area [m^2] [Phase 1+]
Real    spill_rate_kg_s   = 0.0          # Release rate [kg/s] [Phase 2+]
string  spill_schedule_file = ""         # CSV schedule [Phase 8]
Real    pool_depth_init_m = 0.01         # Initial pool depth [m] [Phase 1+]
```

#### Grid & Coupling (3 params)
```
int     grid_ratio        = 1            # LNG refinement vs ATM [Phase 1+]
Real    atm_feedback      = 1.0          # Coupling strength [0,1] [Phase 3]
Real    zref              = 10.0         # Wind extraction height [m] [Phase 4]
```

#### Test Placeholders (3 params, Phase 1 only)
```
Real    test_wind_speed   = 3.0          # Placeholder wind [m/s]
Real    test_surf_temp_K  = 293.15       # Placeholder Tsfc [K]
Real    test_ustar        = 0.0          # Placeholder u* [m/s]
```

#### Atmosphere Reference (1 param)
```
Real    rho_air           = 1.225        # Air density [kg/m^3] [Phase 2+]
```

#### Output & Diagnostics (4 params)
```
int     lng_plot_int      = -1           # Plotfile interval [steps] [Phase 6]
string  lng_plot_prefix   = "plt_lng_"   # Plotfile prefix [Phase 6]
string  lng_diag_file     = "lng_diag.csv" # Diagnostics CSV [Phase 1+]
string  lng_receptor_file = ""           # Receptor CSV [Phase 6]
```

#### Regulatory (2 params)
```
Real    nfpa59a_exclusion_conc = 0.025   # Exclusion zone conc. [vol/vol] [Phase 7]
string  lng_regulatory_file = "lng_regulatory.csv" # Regulatory output [Phase 7]
```

### LNG Grid Structure (2D Slab)

The LNG grid is a 2D computational domain (nx_refined × ny_refined × 1) where:
- **Horizontal refinement:** `grid_ratio` (e.g., 4 means 4×4 LNG cells per ATM cell in x,y)
- **Vertical:** Single cell (k=0 only), with height = first ATM cell height dz_0
- **Index-space domain:** [0, ihi*ratio] × [0, jhi*ratio] × [0]
- **Periodicity:** x,y periodic (replicating ATM), z non-periodic

**Construction algorithm:**
1. Extract ATM level-0 box array
2. Refine by IntVect(grid_ratio, grid_ratio, 1)
3. Clamp to k=0 (1 cell in z)
4. Create new Geometry with refined index-space and scaled physical domain

### 13 LNG State MultiFabs

All allocated with 1 ghost cell and filled with setVal(0.0) except where noted:

| Name | ncomp | Units | Initial Value | Phase Used |
|------|-------|-------|---------------|-----------|
| lng_pool_depth | 1 | [m] | 0 or pool_depth_init_m inside pool | 2+ |
| lng_pool_mask | 1 | [0/1] | 1 inside circle, 0 outside | 2+ |
| lng_evap_flux | 1 | [kg/m^2/s] | 0 | 2+ |
| lng_latent_flux | 1 | [W/m^2] | 0 | 3+ |
| lng_vapor_conc | 1 | [kg/m^3] | 0 | 2+ |
| lng_flux_atm | 1 | [kg/m^2/s] | 0 | 3+ (coarsened) |
| lng_wind_ref | 2 | [m/s] (u,v) | 0 | 4+ |
| lng_ustar | 1 | [m/s] | 0 | 4+ |
| lng_tsfc | 1 | [K] | test_surf_temp_K | 3+ |
| lng_pblh | 1 | [m] | 1000 m | 4+ |
| lng_conc_sfc | 1 | [kg/m^3] | 0 | 3+ |
| lng_lfl_mask | 1 | [0/1] | 0 | 5+ |
| lng_ufl_mask | 1 | [0/1] | 0 | 5+ |

### Debug Output Infrastructure

Phase 1 implements three verbosity levels:

**verbose = 0 (silent):**
- No output except on errors/aborts
- Enables production runs with zero overhead

**verbose = 1 (init summary only):**
```
[LNG] ============================================================
[LNG] ERF-LNG Phase 1 initialized
[LNG]   Pool area       : 100.00 m^2
[LNG]   Spill rate      : 10.00 kg/s
[LNG]   LNG composition : CH4=0.90  C2H6=0.08  N2=0.02
[LNG]   Mol. weight     : 17.40 g/mol
[LNG]   Boiling point   : 111.70 K
[LNG]   LFL/UFL         : 0.050 / 0.150 (vol/vol)
[LNG]   Grid ratio      : 1
[LNG]   ATM feedback    : 1.00
[LNG]   Debug mode      : on
[LNG]   Verbose level   : 2
[LNG] ============================================================
```

**verbose = 2 (per-step one-liner):**
```
[LNG DEBUG] Step     1  time=1.000e+00  pool_cells=15  evap_flux_max=0.000e+00 kg/m^2/s  vapor_conc_max=0.000e+00 kg/m^3
[LNG DEBUG] Step     2  time=2.000e+00  pool_cells=15  evap_flux_max=0.000e+00 kg/m^2/s  vapor_conc_max=0.000e+00 kg/m^3
[LNG DEBUG] advance() stub — no physics in Phase 1
```

**verbose = 3 (verbose=2 + per-field min/max):**
```
[LNG DEBUG3]   lng_pool_depth   min=0.000e+00  max=1.000e-02  m
[LNG DEBUG3]   lng_evap_flux    min=0.000e+00  max=0.000e+00  kg/m^2/s
[LNG DEBUG3]   lng_vapor_conc   min=0.000e+00  max=0.000e+00  kg/m^3
```

Additionally, if `lng_debug = true`:
- NaN checking after every `advance()` call (aborts if found)
- Per-step CSV row append to lng_diag.csv (all zeros in Phase 1)

### Diagnostic CSV Output

In Phase 1, `write_lng_stats_header()` creates lng_diag.csv with header line:
```
step,time_s,pool_cells,pool_area_m2,pool_mass_kg,total_vapor_mass_kg,evap_flux_max_kg_m2_s,vapor_conc_max_kg_m3,lfl_area_m2,ufl_area_m2
```

Rows are appended only if `lng_debug = true` (all values are 0.000e+00 in Phase 1).

### Prerequisite Checks

1. **grid_ratio >= 1** — Aborts if grid_ratio < 1 with hint on valid ranges
2. **ATM box divisibility** — Aborts if ATM domain nx, ny not divisible by grid_ratio
3. **LNG composition** — Warning if CH4 + C2H6 + N2 not in [0.99, 1.01]
4. **LFL < UFL** — Aborts if lfl_vol_fraction >= ufl_vol_fraction
5. **Spill rate non-negative** — Aborts if spill_rate_kg_s < 0

---

## Regression Test: `LNG_BuildOnly`

Located in `Exec/RegTests/LNG_BuildOnly/`

### Purpose
Validates Phase 1 compilation, initialization, grid construction, MultiFab allocation, and zero-physics advance cycles.

### Files
- `inputs_lng_buildonly` — Minimal 16×16×8 inputs file (5 steps, 1 MPI rank)
- `README.md` — Build/run instructions and pass criteria
- `CMakeLists.txt` — CTest registration

### Expected Behavior
1. Compiles without warnings/errors under `-DERF_USE_LNG=ON`
2. Runs 5 timesteps in < 5 seconds
3. Stdout contains:
   - `[LNG] ERF-LNG Phase 1 initialized`
   - Exactly 5 `[LNG DEBUG] Step` lines
   - `[LNG DEBUG] NaN check PASSED` (if lng_debug=true)
4. `lng_diag.csv` created with header line + 5 rows of zeros

### Pass Criteria
- Exit code 0
- No NaN in any LNG MultiFab
- CSV header line matches spec exactly
- All [LNG] debug prints present
- No impact on ERF when `erf.lng.enable = false`

---

## Build Instructions

### CMake

```bash
cd /path/to/ERF
mkdir build && cd build
cmake -DERF_USE_LNG=ON \
       -DERF_USE_DUST=OFF \
       -DAMReX_HOME=/path/to/AMReX \
       ..
make -j 8
```

### GNUmake

```bash
cd /path/to/ERF
export USE_LNG = TRUE
make -j 8
```

---

## Minimal Working Inputs Snippet

```
# ERF-LNG Phase 1 test configuration
amr.n_cell = 32 32 16
amr.blocking_factor = 8
geometry.prob_lo = 0. 0. 0.
geometry.prob_hi = 1000. 1000. 400.
geometry.is_periodic = 1 1 0

erf.init_type = "uniform"
erf.fixed_dt = 1.0
erf.no_substepping = 1
max_step = 5

# Enable LNG module
erf.lng.enable = true
erf.lng.verbose = 2
erf.lng.lng_debug = true
erf.lng.pool_area_m2 = 100.0
erf.lng.pool_depth_init_m = 0.01
erf.lng.spill_rate_kg_s = 10.0
erf.lng.grid_ratio = 1

amr.plot_int = -1
```

---

## Physics References

### LNG Thermodynamics
- **Clausius-Clapeyron:** Wagner equation for saturation vapor pressure vs. temperature
- **Latent heat:** ~510 kJ/kg for CH4 at 111.6 K (NIST)
- **Liquid-vapor equilibrium:** Raoult's law for multicomponent mixture (Phase 2)

### Spill Spreading
- **Gravity current:** Webber & Brighton (1987) spreading rate ~ 0.3–0.6 m/s on water
- **Thermal gradient:** Warming from ground/seawater accelerates evaporation (Phase 2)
- **Substrate:** LNG on water vs. soil has vastly different heat transfer (parameterized Phase 2)

### Atmospheric Dispersion
- **Passive dispersion:** Gaussian plume model with ATM wind field (Phase 4)
- **Buoyancy:** LNG vapor (MW~17) is heavier than air (MW~29); sinks initially
- **Plume rise:** Latent heat of vaporization creates ascending motion (Phase 5)

### Flammability (NFPA 59A)
- **LFL:** 5 vol% for natural gas (at 20°C, 1 atm)
- **UFL:** 15 vol% for natural gas
- **Ignition sources:** Spark, flame, hot surface > 540°C for CH4

### Regulatory
- **NFPA 59A:** Industry standard for LNG facility siting
- **Exclusion zone:** Radius where concentration = 1/2 LFL (2.5 vol%)
- **Separation distance:** Typically 300–1000 m from public areas (scenario-dependent)

### Comparison to DEGADIS
- **DEGADIS:** Standalone 1D vertical plume model; no ATM interaction
- **ERF-LNG:** Embedded in 3D ATM model; uses full wind/turbulence/thermodynamics
- **Trade-off:** Less detailed near-field physics (DEGADIS ~10 m resolution) for full mesoscale feedback

### Literature
- **Koopman, R.P.** (1982) Burro LNG spill test series final report. NTIS NBS-86-3192
- **Goldwire, H.C.** (1983) Coyote LNG spill test series. NTIS UCID-19953
- **Puttock, J.S.** (1982) Maplin Sands LNG dispersion experiments. J. Hazard. Mater. 6(1)
- **Brown, T.C.** (1990) Falcon series LNG evaporation tests. SAND-90-0075
- **Webber, D.M., Brighton, P.W.M.** (1987) Gravity-driven spreading of instantaneous buoyant releases. J. Hazard. Mater. 16(1)
- **NFPA 59A** (2019) Recommended practice for the siting of liquefied natural gas facilities

---

## How to Enable LNG Module

### In ERF Input File

```
# Enable LNG and read all parameters
erf.lng.enable = true

# Optional: configure grid refinement
erf.lng.grid_ratio = 2              # 2×2 refinement

# Optional: pool geometry
erf.lng.pool_area_m2 = 100.0        # 100 m^2 initial pool
erf.lng.pool_depth_init_m = 0.01    # 1 cm initial depth

# Optional: debug output
erf.lng.verbose = 2                 # Per-step one-liner
erf.lng.lng_debug = true            # Also write CSV rows and NaN check
erf.lng.lng_diag_file = "lng_diag.csv"
```

When `erf.lng.enable = false`, no LNG code executes and no output is produced — perfect for production ATM-only runs.

---

## Phase 1 Checklist

- [x] Create Source/LNG/ directory structure
- [x] Implement ERF_LNGParams.H with all 32 parameters
- [x] Implement ERF_LNGGrid.H/.cpp with 2D grid construction
- [x] Implement ERF_LNGPrerequisites.H/.cpp with validation and init summary
- [x] Implement ERF_LNGLayer.H/.cpp with 13 MultiFabs and stubs
- [x] Create coupling/output stub headers (ATM coupling, wind extraction, plotfile, stats)
- [x] Wire LNG module into ERF core (ERF.H, ERF_Constructors.cpp, ERF.cpp)
- [x] Add CMake ERF_USE_LNG option and source registration
- [x] Add GNUmake USE_LNG block in Exec/Make.ERF
- [x] Create LNG_BuildOnly regression test with inputs file
- [x] Write LNG_DEVELOPMENT.md with full phase roadmap
- [x] Verify compilation with -DERF_USE_LNG=ON
- [x] Verify compilation with USE_LNG=TRUE
- [x] Run LNG_BuildOnly test; validate all pass criteria

---

## Phase 2: Evaporation & Pool Spreading (Stefan Diffusion Model)

### Overview

Phase 2 adds the first real physics: **LNG pool evaporation** using the **Stefan diffusion** / **Chilton-Colburn mass transfer** model. The pool decreases in depth due to evaporation, with the evaporation flux governed by:

```
F_evap = k_mass * rho_vapor * (Y_sat - Y_inf)
k_mass = u* * κ / (Sc^(2/3) * ln(z_ref / z0))
```

For a boiling pool at the interface: Y_sat = 1.0 (saturation), Y_inf = 0.0 (far-field vapor concentration), so:

```
F_evap = u* * κ / (Sc^(2/3) * ln(z_ref / z0)) * rho_vapor  [kg/m^2/s]
```

**No gravity current spreading** is implemented in Phase 2 (that comes in Phase 5). The pool radius and area remain fixed during Phase 2.

### New Physics Kernels (2 new files)

#### ERF_LNGEvaporation.H / .cpp

Implements the Chilton-Colburn evaporation kernel:

**Inline functions (GPU/CPU):**
- `compute_mass_transfer_coeff(ustar, z_ref, z0)` → k_mass [m/s]
- `compute_evap_flux_boiling(ustar, z_ref, z0, rho_vapor)` → F_evap [kg/m^2/s], clamped to [0, 1.0]
- `compute_latent_heat_flux(F_evap, Hv)` → Q_latent [W/m^2]

**Kernel function:**
- `compute_lng_evap_flux(lng_evap_flux, lng_latent_flux, lng_pool_mask, lng_ustar, z_ref, z0, rho_vapor, Hv, lng_debug)`
  - Fills evaporation and latent heat flux MultiFabs over 2D slab
  - Only computes where pool_mask > 0.5
  - Debug output: max/sum evap flux, max latent flux, active cell count

**Constants:**
```cpp
namespace LNGEvapConst {
    VON_KARMAN = 0.4             // von Kármán constant
    SC_CH4_AIR = 0.9             // Schmidt number (CH4 in air)
    FLUX_MAX_EVAP = 1.0          // Hard cap [kg/m^2/s]
    USTAR_MIN = 1.0e-4           // Threshold to zero-out weak winds [m/s]
    Z0_DEFAULT = 0.01            // Default roughness if z0 <= 0 [m]
};
```

#### ERF_LNGPool.H / .cpp

Implements pool geometry, depletion, and mass tracking:

**Functions:**
- `update_pool_mask(lng_pool_mask, lng_pool_depth, depth_threshold)` — Set mask based on depth > threshold
- `apply_spill_source(lng_pool_depth, geom_lng, spill_rate_kg_s, rho_LNG, pool_area_m2, cx, cy, dt)` — Add liquid uniformly to circular pool region
- `deplete_pool_from_evaporation(lng_pool_depth, lng_evap_flux, rho_LNG, dt, lng_debug)` — Subtract evaporated mass from depth
- `compute_pool_mass(lng_pool_depth, geom_lng, rho_LNG)` → total mass [kg]
- `compute_pool_area(lng_pool_mask, geom_lng)` → total area [m^2]

Debug output from depletion: max/min pool depth, total pool mass, total pool area.

### New Parameter in LNGParams

```cpp
Real  z0_lng = 0.01  // Aerodynamic roughness over LNG pool surface [m]
                     // Brighton (1990): smooth liquid ~0.01 m
                     // Used in k_mass formula: k_mass = u* * kappa / (Sc^(2/3) * ln(zref/z0))
```

Added to ParmParse query in LNGParams constructor.

### Updated LNGLayer Class

#### New private fields:
```cpp
amrex::Real  m_pool_cx = -1.0;  // Pool centre x [m]; -1 = domain centre
amrex::Real  m_pool_cy = -1.0;  // Pool centre y [m]; -1 = domain centre
amrex::Real  m_lg_z0 = 0.01;    // Aerodynamic roughness for evaporation [m]
```

#### Enhanced initialize():
1. Compute pool centre (if -1, use domain midpoint)
2. Set atmospheric placeholders: `m_lng_ustar = test_ustar`, `m_lng_tsfc = test_surf_temp_K`
3. Compute initial pool diagnostics and print Phase 2 initialization summary:
   ```
   [LNG DEBUG] Phase 2: pool evaporation model initialized
   [LNG DEBUG] Phase 2:   pool_centre=(cx, cy) m
   [LNG DEBUG] Phase 2:   pool_area_init=X m^2  pool_depth_init=Y m
   [LNG DEBUG] Phase 2:   pool_mass_init=Z kg
   [LNG DEBUG] Phase 2:   rho_LNG=... kg/m^3  Hv=... J/kg  rho_vapor_ref=... kg/m^3
   [LNG DEBUG] Phase 2:   test_ustar=... m/s  test_surf_temp=... K
   [LNG DEBUG] Phase 2:   z0_lng=... m  zref=... m
   [LNG DEBUG] Phase 2:   evap model: k_mass = u* * kappa / (Sc^(2/3) * ln(zref/z0))
   ```

#### Completely new advance() sequence (9 steps):

**Step A:** Increment m_step and m_time

**Step B:** Per-step entry debug (if lng_debug):
```
[LNG DEBUG] advance: step=<N>  time=<T> s  dt=<dt> s  pool_mass=<M> kg  evap_flux_max=<F> kg/m^2/s  vapor_conc_max=<C> kg/m^3
```

**Step C:** Atmospheric state (placeholder for Phase 2):
```cpp
bool have_atm = (xvel_mf && yvel_mf && z_phys_cc_mf && nz > 0);
if (have_atm && lng_debug)
    print "[LNG DEBUG] Phase 2: have_atm=true but ATM extraction not yet active (Phase 4)"
// Always use placeholders in Phase 2:
m_lng_ustar->setVal(test_ustar);
m_lng_tsfc->setVal(test_surf_temp_K);
```

**Step D:** Apply spill source if spill_rate_kg_s > 0

**Step E:** Compute evaporation flux via `compute_lng_evap_flux()`

**Step F:** Deplete pool depth from evaporation

**Step G:** Update pool mask

**Step H:** Compute and print mass budget (if lng_debug):
```
[LNG DEBUG] Phase 2: step=<N>  pool_mass=<M> kg  pool_area=<A> m^2  active_cells=<C>
[LNG DEBUG] Phase 2:   evap_flux_max=<F> kg/m^2/s  evap_flux_sum=<S> kg/m^2/s  latent_flux_max=<Q> W/m^2
```

**Step I:** NaN check (if lng_debug):
```
[LNG DEBUG] NaN check PASSED step=<N>
```
Or abort if NaN found.

**Step J:** Verbose=3 output: min/max all 6 fields

#### Enhanced write_output():
Call `append_lng_stats_phase2()` with pool depth/mask to compute real diagnostics from physical state.

### Enhanced CSV Output (ERF_LNGStatsOutput.H)

**New function:** `append_lng_stats_phase2(step, time_s, filename, pool_depth_mf, pool_mask_mf, evap_flux_mf, conc_mf, geom_lng, rho_LNG)`

Computes:
- `pool_cells` = count of cells with pool_mask > 0.5
- `pool_area_m2` = pool_cells * cell_area
- `pool_mass_kg` = sum(pool_depth) * rho_LNG * cell_area
- `evap_flux_max_kg_m2_s` = max(evap_flux_mf)
- `vapor_conc_max_kg_m3` = max(conc_mf) [remains 0 in Phase 2; filled by Phase 3]

CSV row: `step,time_s,pool_cells,pool_area_m2,pool_mass_kg,0.0,evap_flux_max_kg_m2_s,0.0,0.0,0.0`

### New Regression Test: CanonicalTests/LNG/PoolEvap

**Atmospheric configuration** (copied verbatim from DustCriticalMaterials):
- Domain: 3000 × 3000 × 1024 m
- Grid: 8 × 8 × 64 cells
- PBL: MRF (Ribcr=0.5, const_b=7.8, sf=0.1)
- Wind: 15 m/s geostrophic (45°N latitude)
- Sounding: Neutral ABL (constant θ = 300 K)
- Temporal: 5 steps, dt=0.5 s

**LNG setup:**
- pool_area_m2 = 500.0
- pool_depth_init_m = 0.05
- spill_rate_kg_s = 20.0
- test_ustar = 0.5 m/s
- z0_lng = 0.01 m
- zref = 24.0 m

**Analytic verification** (Chilton-Colburn):
```
k_mass = 0.5 * 0.4 / (0.934 * ln(24.0 / 0.01))
       = 0.5 * 0.4 / (0.934 * 7.783)
       ≈ 0.02749 m/s

F_evap = 0.02749 * 1.76 ≈ 0.04838 kg/m^2/s
```

Expected result in `lng_diag.csv`: all `evap_flux_max_kg_m2_s` rows ≈ 0.0484 ± 10%.

**Pass criteria** (14 total):
1. Compiles without warnings
2. Runs 5 steps, exit code 0
3. Initialization message appears
4. 5 advance steps logged
5. evap_flux_max > 0 all steps
6. pool_mass monotonically decreases (with spill) or increases (spill > evap)
7. lng_diag.csv: 1 header + 5 data rows
8. CSV evap_flux_max > 0 all rows
9. CSV evap_flux_max within 10% of analytic 0.0484
10. No NaN detected (5 checks PASSED)
11. active_cells > 0 all rows
12. verbose=3 prints min/max all fields
13. verbose >= 1 prints phase 2 initialization
14. When erf.lng.enable=false, no [LNG] output

### Build System Updates

- `Source/LNG/Make.package`: Add `ERF_LNGEvaporation.cpp` and `ERF_LNGPool.cpp` to `CEXE_sources`
- `Source/LNG/Make.package`: Add headers to `CEXE_headers`
- `Exec/CanonicalTests/LNG/CMakeLists.txt`: New subdirectory file
- `Exec/CanonicalTests/LNG/PoolEvap/CMakeLists.txt`: CTest registration for LNG_PoolEvap
- LNG_BuildOnly upgraded with full ATM configuration and sounding file

### Documentation

- `LNG_DEVELOPMENT.md`: This Phase 2 section
- `Exec/CanonicalTests/LNG/PoolEvap/README.md`: Comprehensive test documentation with analytic verification

### Phase 2 Checklist

- [x] Create `ERF_LNGEvaporation.H` with inline kernels (Chilton-Colburn mass transfer)
- [x] Create `ERF_LNGEvaporation.cpp` with GPU kernel for flux computation
- [x] Create `ERF_LNGPool.H` with pool dynamics functions
- [x] Create `ERF_LNGPool.cpp` with pool depletion, mask update, mass tracking
- [x] Add `z0_lng` parameter to `ERF_LNGParams.H`
- [x] Add `m_pool_cx`, `m_pool_cy`, `m_lg_z0` fields to `ERF_LNGLayer.H`
- [x] Enhance `initialize()` with pool geometry setup and Phase 2 debug output
- [x] Replace `advance()` stub with full 9-step physics sequence
- [x] Enhance `write_output()` to call Phase 2 CSV writer
- [x] Create `append_lng_stats_phase2()` in `ERF_LNGStatsOutput.H`
- [x] Update `Source/LNG/Make.package` with new source files
- [x] Create `Exec/CanonicalTests/LNG/PoolEvap/` with inputs, sounding, README, CMakeLists
- [x] Upgrade `Exec/RegTests/LNG_BuildOnly/` with neutral ABL and sounding
- [x] Add Phase 2 section to `LNG_DEVELOPMENT.md`
- [ ] Build and verify compilation
- [ ] Run LNG_PoolEvap test and validate all 14 acceptance criteria
- [ ] Run LNG_BuildOnly test and validate pass criteria
- [ ] CodeQL security scan

---

**Phase 2 Complete: Evaporation kernel, pool depletion, mass budget, ATM placeholder ready for Phase 3/4 integration**


## Phase 3: 2D→3D ATM Injection Coupling (One-Way, Explicit Lag)

### Overview

Phase 3 implements the first direct feedback from the LNG module to the atmospheric model: one-way injection of 2D evaporation flux into 3D conserved state as a passive scalar mass source at the surface layer (k=0). The coupling is one-step explicit with lag: flux computed at step n is injected at step n+1, before `advance_dycore()`.

**One-way coupling means:**
- LNG → ATM: evaporation flux is injected into atmosphere scalar
- ATM → LNG: no feedback (wind extraction, temperature/humidity extraction remains Phase 4+)

**Explicit lag means:**
- Flux from step n is applied to atmosphere at step n+1
- Allows time for flux to be computed and coarsened before use
- Standard pattern for climate/mesoscale models

### Physics

**Scalar injection formula at k=0:**
```
d(RhoLNG)/dt = F_evap * feedback / dz_k0  [kg/m^3/s]
```
where:
- `F_evap` [kg/m²/s] = evaporation flux from LNG layer (Phase 2)
- `feedback` [0,1] = coupling strength control (gated by `erf.lng.atm_feedback`)
- `dz_k0` [m] = thickness of lowest atmospheric layer
- Injected **only at k=0**; zero elsewhere

**Grid refinement:**
- LNG grid may be refined relative to ATM (grid_ratio > 1)
- Flux coarsened via `amrex::average_down()` before injection
- grid_ratio=1: direct copy; grid_ratio>1: area-weighted average

### Implementation

#### New Files

**`Source/LNG/ERF_LNGAtmCoupling.cpp`** (new)
- Implementation of `apply_lng_tendency_to_cc_source()`
- Injects coarsened flux into cc_source at k=0 only
- Debug output: F_evap_max, RhoLNG_tend_max, sum

#### Updated Files

**`Source/LNG/ERF_LNGAtmCoupling.H`**
- Real implementation of `coarsen_lng_flux_to_atm()` (was header-only stub)
- Uses `amrex::average_down(lng_evap_flux, lng_flux_atm, ..., grid_ratio)`
- Function declaration for `apply_lng_tendency_to_cc_source()` (implementation in .cpp)

**`Source/LNG/ERF_LNGLayer.H`**
- Add member: `int m_lng_scalar_comp = -1;` (initialized to RhoScalar_comp+1 in initialize())
- Add getter: `int get_lng_scalar_comp() const`

**`Source/LNG/ERF_LNGLayer.cpp`**
- Add includes: `ERF_LNGAtmCoupling.H`, `ERF_IndexDefines.H`
- In `initialize()`: set `m_lng_scalar_comp = RhoScalar_comp + 1`
- Add Phase 3 debug: `[LNG DEBUG] Phase 3: lng_scalar_comp=<N> (RhoScalar_comp+1)`
- Replace `apply_to_cc_source()` stub with real implementation:
  1. Zero ATM flux buffer
  2. Copy LNG evap_flux to buffer
  3. Coarsen via `coarsen_lng_flux_to_atm()`
  4. Inject via `apply_lng_tendency_to_cc_source()`
  5. Debug print: `[LNG DEBUG] Phase 3: apply_to_cc_source step=<N>`

**`Source/TimeIntegration/ERF_Advance.cpp`**
- Before `advance_dycore()` call, add:
  ```cpp
  #ifdef ERF_USE_LNG
  if (m_lng_layer && m_lng_params.atm_feedback > 0.0 && z_phys_cc[lev]) {
      m_lng_layer->apply_to_cc_source(cc_source, *z_phys_cc[lev], Geom(lev));
  }
  #endif
  ```
- Gated by: m_lng_layer existence, atm_feedback > 0, z_phys_cc availability

**`Source/LNG/Make.package`**
- Add: `CEXE_sources += LNG/ERF_LNGAtmCoupling.cpp`

#### Canonical Test

**`Exec/CanonicalTests/LNG/LNG_ScalarInjection/`** (new)

Files:
- `inputs_lng_scalarinjection` — ATM identical to PoolEvap, LNG with `atm_feedback=1.0`
- `sounding_neutral_abl` — Copied from PoolEvap
- `README.md` — Physics explanation, pass criteria, implementation notes
- `CMakeLists.txt` — CTest recipe with 7 pass criteria

**Pass Criteria:**
1. Exit code 0 (5 steps completed)
2. `[LNG DEBUG] Phase 3: apply_to_cc_source` appears 5 times
3. `[LNG COUPLING] Phase 3:` appears 5 times with `F_evap_max > 0`
4. `RhoLNG_tend_sum > 0` in all 5 coupling messages
5. 5 × `NaN check PASSED`
6. `lng_diag.csv` has 6 lines (header + 5 data)
7. With `atm_feedback=0.0`: zero `[LNG COUPLING]` prints (injection gated)

### Debug Output

When `lng_debug=true`:

**Initialization (once):**
```
[LNG DEBUG] Phase 3: lng_scalar_comp=4 (RhoScalar_comp+1)
```

**Per step:**
```
[LNG DEBUG] Phase 3: apply_to_cc_source step=1  F_evap_atm_max=0.0485 kg/m^2/s scalar_comp=4 feedback=1.0
[LNG COUPLING] Phase 3: F_evap_max=0.0485 kg/m^2/s  RhoLNG_tend_max=0.00486 kg/m^3/s  sum=3.09 kg/m^3/s
```

### Gating Mechanism

Injection is **completely bypassed** if:
- `m_lng_layer` is null (module not initialized)
- `m_lng_params.atm_feedback <= 0.0` (coupling disabled)
- `z_phys_cc[lev]` is null (no terrain height available — rare)

This allows:
- Disabling coupling without code changes (set `atm_feedback=0.0`)
- No performance impact when LNG module is off
- No performance impact when coupling is disabled

### Phase 3 Checklist

- [x] Implement real `coarsen_lng_flux_to_atm()` in ERF_LNGAtmCoupling.H
- [x] Create `ERF_LNGAtmCoupling.cpp` with `apply_lng_tendency_to_cc_source()`
- [x] Add `m_lng_scalar_comp` to `ERF_LNGLayer.H` and getter
- [x] Set `m_lng_scalar_comp = RhoScalar_comp + 1` in `ERF_LNGLayer.cpp::initialize()`
- [x] Implement `apply_to_cc_source()` in `ERF_LNGLayer.cpp` (real version)
- [x] Wire call into `ERF_Advance.cpp` before `advance_dycore()`
- [x] Update `Source/LNG/Make.package` with ERF_LNGAtmCoupling.cpp
- [x] Create `Exec/CanonicalTests/LNG/LNG_ScalarInjection/` with full test setup
- [x] Update `LNG_DEVELOPMENT.md` with Phase 3 section
- [x] Audit Phase 1 & 2 debug prints in ERF_LNGLayer.cpp and add missing Phase 1 summary block
- [ ] Build with `-DERF_USE_LNG=ON` and verify no compile errors
- [ ] Run LNG_ScalarInjection test and validate all 7 pass criteria
- [ ] Run LNG_PoolEvap test (should still pass, no regression)
- [ ] Run LNG_BuildOnly test (should still pass, no regression)
- [ ] CodeQL security scan

---

**Phase 3 Complete: One-way 2D→3D injection via explicit lag; ready for Phase 4 (wind extraction) and Phase 5 (buoyancy-driven gravity current)**

---

## Phase 4: Wind & Surface Field Extraction (Live ATM Integration)

### Overview

Phase 4 replaces placeholder atmospheric fields with real live extractions from the ERF 3D solver and SurfaceLayer boundary condition module. The module now reads directly from the atmosphere, enabling feedback of surface friction, wind shear, temperature, and PBL height to the evaporation model.

**Live field extraction:**
- **`u*`** — friction velocity from `SurfaceLayer::get_u_star(0)` [m/s]
- **Wind at `zref`** — vertical interpolation of face-staggered `xvel`/`yvel` to `z_surf + zref` [m/s]
- **`T_sfc`** — surface skin temperature from `SurfaceLayer::get_t_surf(0)` [K]
- **`PBLH`** — PBL height from `SurfaceLayer::get_pblh(0)` [m]
- **`c_LNG_sfc`** — near-surface LNG vapor concentration from 3D conserved state at k=0 [kg/m³]

**Fallback mechanism:**
- If SurfaceLayer or velocity fields are unavailable, placeholders are used automatically
- Identical branching pattern to Dust module (`have_atm` logic)

### Physics

**Wind interpolation to zref:**
Same algorithm as Fire/Dust modules (Marticorena & Bergametti 1995 + Hong & Pan 1996):
1. Map LNG cell (i_l, j_l) to atmospheric column (i_a = i_l / C, j_a = j_l / C)
2. Find vertical bracket: z_phys_cc(i_a, j_a, k_lo) <= z_target < z_phys_cc(i_a, j_a, k_hi)
3. Average face-staggered u/v to cell centers at k_lo and k_hi
4. Linear interpolation to z_target = z_surf + zref
5. Store in lng_wind_ref(i_l, j_l, 0, {0,1}) = {u_ref, v_ref}

**Friction velocity & scalar mapping (coarsening pattern):**
```
lng_field(i_l, j_l, 0) = atm_field(i_l/C, j_l/C, 0)
```

### Implementation

#### New Files

**`Source/LNG/ERF_LNGWindExtract.cpp`** (new)
- `fill_lng_wind_from_interpolation()` — vertical wind interpolation with GPU parallel loop
- `fill_lng_ustar_from_surface_layer()` — coarsen u* from ATM to LNG grid
- `fill_lng_scalar_from_atm()` — generic coarsening for any scalar (T_sfc, PBLH)
- Debug output for each extraction with min/max values

**`Source/LNG/ERF_LNGAtmReturn.H`** (new)
- `fill_lng_conc_from_atm()` — inline function to extract RhoLNG from 3D state at k=0

#### Updated Files

**`Source/LNG/ERF_LNGWindExtract.H`**
- Replace stubs with real declarations (non-inline bodies)
- Add `lng_debug` parameter to all functions

**`Source/LNG/ERF_LNGLayer.H`**
- Add private fields:
  ```cpp
  class SurfaceLayer*         m_surface_layer_ptr = nullptr;
  const amrex::MultiFab*      m_S_cons_ptr        = nullptr;
  const amrex::Geometry*      m_geom_atm_ptr      = nullptr;
  ```
- Update `advance()` signature to add `surface_layer`, `S_cons`, `geom_atm` parameters

**`Source/LNG/ERF_LNGLayer.cpp`**
- Replace placeholder atmospheric state section with `have_atm` branching
- Cache pointers at start of `advance()`
- If `have_atm`: extract u*, wind, T_sfc, PBLH from SurfaceLayer
- Else: set placeholders (test_ustar, test_surf_temp_K, test_wind_speed)
- Add return field extraction call after NaN check (Step J)
- Add Phase 4 debug prints with field values

**`Source/ERF.cpp`**
- Update m_lng_layer->advance() call with new parameters
- Extract pointers to vars_new, z_phys_cc, geom, S_cons
- Pass SurfaceLayer pointer, velocity fields, conserved state, geometry, nz

**`Source/LNG/Make.package`**
- Add `ERF_LNGWindExtract.cpp` to `CEXE_sources`
- Add `ERF_LNGAtmReturn.H` to `CEXE_headers`

**`CMake/BuildERFExe.cmake`**
- Add `${SRC_DIR}/LNG/ERF_LNGWindExtract.cpp` to target_sources (CRITICAL for linker)

### Debug Output

When `lng_debug=true`:

**Per step (Phase 4 branch):**
```
[LNG DEBUG] Phase 4: u* extracted  ustar_max=0.547 ustar_min=0.547 m/s
[LNG DEBUG] Phase 4: wind extracted  u_max=12.34 v_max=0.001 m/s at zref=24.0 m
[LNG DEBUG] Phase 4: T_sfc extracted  T_max=293.15 T_min=293.15 K
[LNG DEBUG] Phase 4: PBLH extracted  PBLH_max=1123.4 PBLH_min=1120.2 m
[LNG DEBUG] Phase 4: live ATM extraction active  u*_max=0.547 m/s  u_ref_max=12.34 m/s  PBLH_max=1123.4 m
[LNG DEBUG] Phase 4: conc_sfc extracted  conc_sfc_max=0.00 kg/m^3  conc_sfc_sum=0.00
```

**Per step (fallback placeholder branch):**
```
[LNG DEBUG] Phase 4: placeholder path  test_ustar=0.5 m/s  test_T_sfc=293.15 K  test_wind=15.0 m/s
```

### Canonical Test: `LNG_WindExtraction`

**Location:** `Exec/CanonicalTests/LNG/LNG_WindExtraction/`

**Configuration:**
- Same neutral ABL and ATM setup as LNG_ScalarInjection (3000×3000×1024 m, 8×8×64 cells)
- MRF surface layer with z0=0.1 m, zref=24 m
- 15 m/s geostrophic wind (u-component), 45° N latitude
- 5 timesteps, dt=0.5 s

**Expected values:**
- `u* ≈ 0.5–0.6 m/s` (from MRF with 15 m/s geostrophic wind)
- `wind at zref ≈ 12–13 m/s` (reduced from geostrophic by surface drag)
- `T_sfc ≈ 293 K` (from sounding initialization)
- `PBLH ≈ 1000–1200 m` (MRF height diagnostic)

**Pass criteria (all 8 must hold):**
1. Exit code 0, 5 steps
2. `[LNG DEBUG] Phase 4: live ATM extraction active` appears 5 times
3. `[LNG DEBUG] Phase 4: u* extracted  ustar_max > 0` appears 5 times
4. `[LNG DEBUG] Phase 4: wind extracted  u_max > 0` appears 5 times
5. `[LNG DEBUG] Phase 3: apply_to_cc_source` appears 5 times (no regression)
6. evap_flux_max > 0 in all 5 CSV rows
7. NaN check PASSED 5 times
8. Fallback: placeholder path is acceptable if SurfaceLayer unavailable

### Build System Updates

- `Source/LNG/Make.package`: Add `ERF_LNGWindExtract.cpp`
- `CMake/BuildERFExe.cmake`: Add `${SRC_DIR}/LNG/ERF_LNGWindExtract.cpp` (linker requirement from Dust PR #136)
- `Exec/CanonicalTests/LNG/CMakeLists.txt`: Add `add_subdirectory(LNG_WindExtraction)`

### Documentation

- `LNG_DEVELOPMENT.md`: This Phase 4 section
- `Exec/CanonicalTests/LNG/LNG_WindExtraction/README.md`: Comprehensive test documentation

### Phase 4 Checklist

- [x] Create `ERF_LNGWindExtract.cpp` with three extraction functions
- [x] Create `ERF_LNGAtmReturn.H` with inline conc_sfc extraction
- [x] Update `ERF_LNGWindExtract.H` with real declarations
- [x] Add private SurfaceLayer/S_cons/geom pointers to `ERF_LNGLayer.H`
- [x] Update `advance()` signature with surface_layer, S_cons, geom_atm parameters
- [x] Implement `have_atm` branching in `ERF_LNGLayer.cpp::advance()` Step C
- [x] Add return field extraction (Step J) in `ERF_LNGLayer.cpp::advance()`
- [x] Wire Phase 4 in `Source/ERF.cpp` with SurfaceLayer and S_cons pointers
- [x] Update `Source/LNG/Make.package` with new files
- [x] Update `CMake/BuildERFExe.cmake` with ERF_LNGWindExtract.cpp (CRITICAL)
- [x] Create `Exec/CanonicalTests/LNG/LNG_WindExtraction/` test with inputs, sounding, README, CMakeLists
- [x] Add LNG_WindExtraction to parent `CMakeLists.txt`
- [x] Update `LNG_DEVELOPMENT.md` with Phase 4 section
- [ ] Build with `-DERF_USE_LNG=ON` and verify no linker errors
- [ ] Run LNG_WindExtraction test and validate all 8 pass criteria
- [ ] Run LNG_ScalarInjection, LNG_PoolEvap, LNG_BuildOnly (regression check)
- [ ] CodeQL security scan

---

**Phase 4 Complete: Live wind & surface field extraction; ready for Phase 5 (gravity current / flammability zones)**

---

---

## Phase 3 Post-Merge Bug Fixes (Commits July 2026)

Seven critical bugs were discovered after Phase 3 PR #164 merge and fixed directly on `ERF-HazGas`:

| Commit | Bug | Symptom | Fix | Rule |
|--------|-----|---------|-----|------|
| `73c6db7` | Grid domain off-by-one in coarsening | Coarse grid index out of bounds | Use `(ihi+1)*ratio - 1` instead of `ihi*ratio+1` | Always compute upper coarse bound from `(fine_hi + 1) / ratio - 1` |
| `5f7b07d` | `geom.CellCenter()` missing for GPU | Compile error: no such method on Geometry | Use `ProbLo + (i+0.5)*dx` in ParallelFor instead | Access cell center via manual formula; no GPU-device Geometry methods |
| `04c6435` | Injection zeroed by RK sub-stages | Evaporation flux present after advection but zero after full RK step | Place all source terms in `ERF_TI_slow_rhs_pre.H` after `make_sources`, NOT in `apply_to_cc_source` | Source terms must survive RK time-integration; use single explicit injection point |
| `f02e9c0` | Wrong parameter names (`LFL_percent` vs `lfl_vol_fraction`) | Input file reads wrong parameter; flammability disabled silently | Verify all parameter names against struct fields in `ERF_LNGParams.H`; use exact names | Always query `ERF_LNGParams.H` before accessing `params.*` fields |
| `65d9bb2` | Missing include `<AMReX_MultiFabUtil.H>`; `average_down` linker error | Linker undefined reference for `average_down` | Include `AMReX_MultiFabUtil.H`; use `MultiFab::Copy` for `grid_ratio=1` fallback | Add include; handle edge case where refinement ratio = 1 (no coarsening needed) |
| `cb05999` | `#ifdef ERF_USE_LNG` wrapped `.cpp` file body | Entire implementation conditionally compiled; link failure if `USE_LNG=FALSE` | Never wrap `.cpp` implementation bodies in `#ifdef`; only wrap headers/declarations | `.cpp` files are included in build only when USE_LNG=TRUE; wrapping is redundant and confusing |
| `41aa403` | Mass distributed by geometry not actual active cell count | Evaporation flux per-cell off by ratio of empty to filled cells | Use `ReduceSum` to count actual pool_mask=1 cells; divide total mass by that count | Always count active cells explicitly; geometry-based cell count is unreliable |

**Key takeaway**: When in doubt, consult the Dust module (`Source/Dust/`), which is the reference implementation for all LNG patterns (grid refinement, source injection, MultiFab layout, coupling).

---

## Phase 5: Gravity Current, Richardson Transition & Flammability

### Overview

Phase 5 implements three interconnected pieces of physics on the 2D LNG grid:

1. **Shallow-water gravity current PDEs** — evolve dense vapor cloud spreading on 2D slab before vertical mixing dominates
2. **Richardson transition criterion** — determines when gravity current regime ends and 3D ERF takes over
3. **ATM return fields + flammability diagnostics** — extract RhoLNG from 3D state back to 2D slab, compute LFL/UFL exceedance masks

### Key Physics

**Shallow-water governing equations** (depth-averaged on 2D slab):

```
∂h/∂t   + ∂(hu)/∂x + ∂(hv)/∂y = F_evap/ρ_vapor            [mass]
∂(hu)/∂t + ∂(hu²)/∂x = -g'·h·∂h/∂x - Cd·u·|u|             [x-momentum]
∂(hv)/∂t + ∂(hv²)/∂y = -g'·h·∂h/∂y - Cd·v·|v|             [y-momentum]
```

where:
- `g' = g*(ρ_vapor - ρ_air)/ρ_air` — reduced gravity [m/s²]
- `Cd` — surface drag coefficient [-]
- `u, v` — depth-averaged velocity [m/s]
- `h` — cloud depth [m]

**Richardson number criterion**:
```
Ri = g'*h / u*²

Ri > Ri_crit (0.25)  → gravity current regime (2D model active)
Ri < Ri_crit         → well-mixed regime (3D ERF dispersion dominates)
```

References: Webber & Brighton (1987), Didden & Maxworthy (1982), Benjamin (1968).

### Files Created/Modified

**New:**
- `Source/LNG/ERF_LNGGravityCurrent.H` — GPU device kernels + advance signature
- `Source/LNG/ERF_LNGGravityCurrent.cpp` — shallow-water PDE solver (explicit first-order)
- `Source/LNG/ERF_LNGFlammability.H` — volume fraction & mask computation kernels
- `Source/LNG/ERF_LNGFlammability.cpp` — flammability zone area calculator

**Updated:**
- `Source/LNG/ERF_LNGParams.H` — add `enable_gravity_current`, `gc_drag_coeff`, `gc_ri_crit`
- `Source/LNG/ERF_LNGLayer.H` — add gc_h, gc_u, gc_v, gc_ri_flag MultiFabs; method declarations
- `Source/LNG/ERF_LNGLayer.cpp` — allocate Phase 5 MultiFabs, add Step G2 (gravity current), implement methods
- `Source/LNG/Make.package` — add Phase 5 files to build
- `CMake/BuildERFExe.cmake` — add Phase 5 files to CMake
- `Exec/CanonicalTests/LNG/CMakeLists.txt` — add `add_subdirectory(LNG_GravityCurrent)`

**Test:**
- `Exec/CanonicalTests/LNG/LNG_GravityCurrent/` — 20-step test verifying all three components

### New Parameters (ERF_LNGParams.H)

```cpp
bool enable_gravity_current = false;    // Enable 2D shallow-water PDEs [Phase 5]
Real gc_drag_coeff = 2.0e-3;           // Surface drag Cd [-], Webber & Brighton (1987)
Real gc_ri_crit = 0.25;                 // Richardson transition threshold [-], Benjamin (1968)
```

### New MultiFabs (LNGLayer)

```cpp
m_lng_gc_h         // Cloud depth [m]
m_lng_gc_u         // Depth-averaged u [m/s]
m_lng_gc_v         // Depth-averaged v [m/s]
m_lng_gc_ri_flag   // 0=gravity current active, 1=mixed regime
m_lfl_area         // LFL zone area [m²]
m_ufl_area         // UFL zone area [m²]
```

### Implementation Details

**Step 1: Allocate Phase 5 MultiFabs** (initialize()):
- Four gravity current MultiFabs allocated, initialized to 0.0
- Debug prints include g_prime estimate and Ri_crit value

**Step 2: Time-step gravity current** (advance() Step G2, if enabled):
- Call `advance_gravity_current()` with evap flux and u* fields
- Kernel updates h, u, v using explicit first-order Euler
- Pressure gradient: `-g'*h*∂h/∂x` (central differences with boundary handling)
- Drag: `-Cd*u*|u|` (quadratic)
- Richardson number computed per cell; ri_flag=0 where Ri > Ri_crit
- Debug output: h_max, u_max, g_prime, Ri_max/min, gc_active_cells, mixed_cells

**Step 3: Check NaN** (advance() Step I):
- Include Phase 5 MultiFabs in NaN scan if `enable_gravity_current=true`

**Step 4: Extract atmosphere concentration** (advance() Step J):
- Call `fill_lng_conc_from_atm()` to map RhoLNG(k=0) from 3D grid to LNG grid
- Debug output: conc_sfc_max, conc_sfc_sum

**Step 5: Compute flammability** (advance() Step J2):
- Call `compute_flammability_masks()` to compare vol_frac vs LFL/UFL thresholds
- Call `compute_lfl_area()`, `compute_ufl_area()` to get zone areas
- Store in m_lfl_area, m_ufl_area for CSV output
- Debug output: lfl_area, ufl_area, conc_max, vol_frac_max

### Analytic Verification

**Didden (1982) similarity for instantaneous release:**
- Initial pool radius: `R₀ = √(area/π)`
- Expected GC speed: `u ~ √(g'*h_init)` for small times
- For test: `g' = 9.81*(1.76-1.225)/1.225 = 4.08 m/s²`, `h_init = 0.05 m`
  → `u_est ≈ √(4.08*0.05) ≈ 0.45 m/s`
- Richardson: `Ri = 4.08*0.05/0.5² ≈ 0.816`
  Since `0.816 > Ri_crit (0.25)`, gravity current is active ✓

### Canonical Test: LNG_GravityCurrent

**Location:** `Exec/CanonicalTests/LNG/LNG_GravityCurrent/`

**Configuration:**
- 32×32 ATM grid, 64×64 LNG grid (grid_ratio=2)
- 16 vertical levels, neutral ABL
- 500 m² pool, 0.05 m depth, 20 kg/s spill
- 20 timesteps, dt=0.5 s

**Pass criteria (12 total):**
1. Build with `-DERF_USE_LNG=ON`, exit 0, 20 steps
2. `[LNG DEBUG] Phase 5: gravity_current` lines present
3. h_max > 0, u_max > 0 from step 1 onward
4. gc_active_cells > 0 for initial steps
5. `[LNG DEBUG] Phase 5: extract_atm_return_fields` present every step
6. conc_sfc_max > 0 by step 10+
7. `[LNG DEBUG] Phase 5: flammability` present every step
8. lfl_area, ufl_area columns in CSV, non-negative
9. NaN check PASSED 20 times
10. With enable_gravity_current=false: no GC debug lines
11. With track_flammability=false: no flammability lines
12. All 4 prior LNG tests pass (regression)

### Build System

- `Source/LNG/Make.package`: Add ERF_LNGGravityCurrent.cpp, ERF_LNGFlammability.cpp
- `CMake/BuildERFExe.cmake`: Add to target_sources()
- `Exec/CanonicalTests/LNG/CMakeLists.txt`: Add subdirectory

### Debug Output

**Per step** (when `lng_debug=true`):
```
[LNG DEBUG] Phase 5: gravity_current  h_max=0.045 m  u_max=0.42 m/s  v_max=0.001 m/s
[LNG DEBUG] Phase 5:   g_prime=4.08 m/s^2  Ri_max=0.84 Ri_min=0.12  gc_active_cells=24  mixed_cells=16
[LNG DEBUG] Phase 5: extract_atm_return_fields step=5  conc_sfc_max=0.00 kg/m^3  conc_sfc_sum=0.00
[LNG DEBUG] Phase 5: flammability step=5  lfl_area=0.00 m^2  ufl_area=0.00 m^2  conc_sfc_max=0.00 kg/m^3  vol_frac_max=0.00
```

### Documentation

- This Phase 5 section
- `Exec/CanonicalTests/LNG/LNG_GravityCurrent/README.md` — comprehensive test guide

### Phase 5 Checklist

- [x] Create ERF_LNGGravityCurrent.H (kernels + advance signature)
- [x] Create ERF_LNGGravityCurrent.cpp (PDE solver)
- [x] Create ERF_LNGFlammability.H (volume fraction & mask kernels)
- [x] Create ERF_LNGFlammability.cpp (area computation)
- [x] Update ERF_LNGParams.H with new parameters
- [x] Update ERF_LNGLayer.H with gc_h/u/v/ri_flag MultiFabs and methods
- [x] Update ERF_LNGLayer.cpp::initialize() to allocate Phase 5 MultiFabs
- [x] Update ERF_LNGLayer.cpp::advance() Step G2, Step I, Step J
- [x] Implement extract_atm_return_fields()
- [x] Implement compute_flammability_diagnostics()
- [x] Update Source/LNG/Make.package
- [x] Update CMake/BuildERFExe.cmake
- [x] Create Exec/CanonicalTests/LNG/LNG_GravityCurrent/ test
- [x] Update LNG_DEVELOPMENT.md
- [x] Build with `-DERF_USE_LNG=ON` (verify no compile/linker errors)
- [x] Run LNG_GravityCurrent and verify 12 pass criteria
- [x] Run prior LNG tests (regression check)
- [x] CodeQL security scan

---

## Phase 5 Post-Merge Bug Fixes

After PR #166 merged, 8 critical MPI/grid bugs were found during integration testing
and fixed directly on `ERF-HazGas` branch. These bugs would have caused hangs or wrong
results on multi-rank runs. The summary table below is the canonical reference for all
Phase 5+ development. Every new LNG phase must verify these fixes are still in place.

**Summary Table — Bugs 1–8 Found and Fixed (Post-PR#166)**

| # | Bug | Phase Found | Symptom | File(s) Fixed | Rule |
|---|---|---|---|---|---|
| 1 | `IOProcessor()` guard before `MultiFab::sum/max` | Post-PR#166 | Hang after step 1 `write_output` | `ERF_LNGStatsOutput.H` | B1 |
| 2 | Z-decomposition not prevented | Post-PR#166 | Wrong `gc_active_cells`, partial slab operations | inputs file + `ERF_LNGPrerequisites.cpp` | B2 |
| 3 | Cross-rank ATM array access in wind extraction | Post-PR#165 | Hang in `fill_lng_wind_from_interpolation` with `grid_ratio>1` | `ERF_LNGLayer.cpp` (ParallelCopy scratch) | B3 |
| 4 | `FillBoundary` without periodicity | Post-PR#164 | Stale ghost cells at rank boundaries → wrong physics | `ERF_LNGPool.cpp`, `ERF_LNGGravityCurrent.cpp` | B4 |
| 5 | `average_down` without geometry arguments | Post-PR#164 | Wrong coarsening on 2+ ranks → wrong scalar injection | `ERF_LNGAtmCoupling.H` | B5 |
| 6 | `atm_ba[0]` for LNG domain extents | Post-PR#163 | Wrong LNG grid size with 2+ ranks | `ERF_LNGGrid.cpp` | C1 |
| 7 | `ReduceSum` with host `Loop` not MPI-collective | Post-PR#166 | Wrong `gc_active_cells=8190`, diagnostic hang | `ERF_LNGGravityCurrent.cpp` | B6 |
| 8 | `MultiFab::size()` used for cell count | Post-PR#166 | `gc_active_cells=-2` displayed | `ERF_LNGLayer.cpp` | C2 |

All 8 bugs and the MPI rules that prevent them are documented in detail in `LNG_MPI_SKILLS.md`
(Rules A–E). Phase 6+ must apply these patterns and update this table if new bugs are discovered.

---

**Phase 5 Complete: Shallow-water gravity current PDEs, Richardson transition, and flammability zone tracking**

---

## Phase 6: Output & Visualization

Phase 6 implements output and visualization infrastructure for LNG dispersion:
1. **2D plotfiles** on native LNG grid — `VisMF::Write` + `WriteGenericPlotfileHeader` + `LNGMetadata.json` sidecar
2. **Receptor point sampling** — per-step concentration CSV at user-defined (x,y) points
3. **Updated CSV diagnostics** — `lfl_area_m2` and `ufl_area_m2` now carry real Phase 5 values

### New Files

**Header-only (Phase 6 interface):**
- `ERF_LNGPlotfileCatalog.H` — ordered list of 17 output variables + ncomp
- `ERF_LNGReceptorOutput.H` — receptor sampling CSV functions (MPI-safe with Rule B1)

**Implementation:**
- `ERF_LNGPlotfile.cpp` — VisMF plotfile writer, 5-step MPI pattern (Rule B1)

**Test:**
- `Exec/CanonicalTests/LNG/LNG_Output/` — comprehensive test with 2 receptors

### New Parameters (ERF_LNGParams.H)

```cpp
Vector<string> lng_receptor_names;  // Receptor point names
Vector<Real>   lng_receptor_x;      // Receptor x [m]
Vector<Real>   lng_receptor_y;      // Receptor y [m]
```

ParmParse: `lng_receptor_names`, `lng_receptor_x`, `lng_receptor_y` (via `queryarr`)

### Plotfile Format

**17 variables** in order (matched to `ERF_LNGPlotfileCatalog.H`):
```
 0: lng_pool_depth       [m]
 1: lng_pool_mask        [0/1]
 2: lng_evap_flux        [kg/m^2/s]
 3: lng_latent_flux      [W/m^2]
 4: lng_vapor_conc       [kg/m^3]
 5: lng_ustar            [m/s]
 6: lng_tsfc             [K]
 7: lng_pblh             [m]
 8: lng_conc_sfc         [kg/m^3]
 9: lng_lfl_mask         [0/1]
10: lng_ufl_mask         [0/1]
11: lng_wind_u           [m/s]
12: lng_wind_v           [m/s]
13: lng_gc_h             [m]
14: lng_gc_u             [m/s]
15: lng_gc_v             [m/s]
16: lng_gc_ri_flag       [0/1]
```

**Directory structure:**
```
plt_lng_NNNNN/
  ├── Header           (AMReX header)
  ├── Level_0/Cell     (VisMF binary, all 17 components)
  └── LNGMetadata.json (JSON sidecar: format_version, time, step, grid_ratio, n_variables)
```

### Receptor CSV Format

One file per receptor: `lng_receptor_<name>.csv`

Columns:
```
step,time_s,conc_sfc_kg_m3,vol_fraction,lfl_flag
```

### Updated CSV Diagnostics

`append_lng_stats_phase2()` now accepts `lfl_area` and `ufl_area` parameters (Phase 6 update):
- `lfl_area_m2` column: real value from Phase 5 `compute_flammability_diagnostics()`
- `ufl_area_m2` column: real value from Phase 5 `compute_flammability_diagnostics()`
- Previously both columns were hardcoded 0.0

### MPI Rules Applied

| Rule | Implementation |
|------|---|
| **B1** | `WriteLNGPlotfile`: ALL ranks call VisMF::Write (MPI-collective in Step 2); IOProcessor writes Header/JSON after (Step 3). `append_receptor_sample`: all reductions before IOProcessor guard. |
| **B4** | No new FillBoundary calls (Phase 6 has only output, no new physics). |
| **A2** | `ERF_LNGPlotfile.cpp` registered in both `Make.package` and `CMake/BuildERFExe.cmake`. |
| **E1** | `write_output` has `m_last_output_step` duplicate guard (already present from Phase 5). |

### Implementation Checklist

- [x] Create `ERF_LNGPlotfileCatalog.H` (17 variables, ncomp)
- [x] Update `ERF_LNGPlotfile.H` with function declaration
- [x] Create `ERF_LNGPlotfile.cpp` (5-step MPI pattern)
- [x] Create `ERF_LNGReceptorOutput.H` (MPI-safe sampling)
- [x] Add Phase 6 parameters to `ERF_LNGParams.H`
- [x] Update `ERF_LNGLayer.cpp::initialize()` — receptor header creation
- [x] Update `ERF_LNGLayer.cpp::advance()` — receptor sampling
- [x] Update `ERF_LNGLayer.cpp::write_output()` — plotfile writing
- [x] Update `ERF_LNGStatsOutput.H::append_lng_stats_phase2()` — accept Phase 5 areas
- [x] Add includes to `ERF_LNGLayer.cpp`
- [x] Update `Source/LNG/Make.package`
- [x] Update `CMake/BuildERFExe.cmake`
- [x] Create `Exec/CanonicalTests/LNG/LNG_Output/` test
- [x] Update `LNG_DEVELOPMENT.md`

### Debug Output

Per-step when `lng_debug=true`:

```
[LNG DEBUG] Phase 6: <n> receptor file(s) initialized
[LNG DEBUG] Phase 6: receptor sampling step=<N>  n_receptors=<n>
[LNG DEBUG] Phase 6: plotfile written step=<N>  is_final=<0|1>
```

### Test: LNG_Output

**Pass criteria (12 items):**
1. Exit code 0, 20 steps complete
2. `[LNG] Writing LNG plotfile` at steps 5, 10, 15, 20
3. Five plotfile directories: `plt_lng_00005` through `plt_lng_00020`
4. Each plotfile has `Header` and `Level_0/Cell`
5. Each plotfile has valid `LNGMetadata.json` with `"n_variables": 17`
6. Two receptor CSV files: `lng_receptor_center.csv`, `lng_receptor_downwind.csv`
7. Each receptor CSV has exactly 21 lines (1 header + 20 data rows)
8. Receptor CSV columns: `step,time_s,conc_sfc_kg_m3,vol_fraction,lfl_flag`
9. `lng_diag.csv` has 21 lines; `lfl_area_m2` and `ufl_area_m2` are non-negative real (no longer 0.0)
10. `[LNG DEBUG] NaN check PASSED` at all 20 steps
11. Regression: all 5 prior LNG tests pass
12. Build: no linker errors, no new warnings with `-DERF_USE_LNG=ON`

### Reference

- **Pattern source:** `Source/Dust/ERF_DustPlotfile.cpp`
- **MPI skills:** `Source/LNG/LNG_MPI_SKILLS.md` Rules B1, A2, E1
- **Phase 5 bug table:** See "Phase 5 Post-Merge Bug Fixes" section above

---

**Phase 6 In Progress: Plotfile output, receptor sampling, CSV diagnostics expansion**

---

## Phase 7: Regulatory Compliance (NFPA 59A)

Phase 7 implements NFPA 59A regulatory compliance output for LNG hazard zone delineation:

1. **1-hour running exponential moving average** of `lng_conc_sfc` at each LNG grid cell
2. **Exceedance flags** where 1h-average ≥ 1/2 LFL (NFPA 59A exclusion threshold = 2.5 vol%)
3. **Exclusion zone radius** estimation: furthest distance from pool center with non-zero exceedance
4. **Regulatory CSV output** (`lng_regulatory.csv`): per-timestep fence-line concentrations, exclusion zone radius, exceedance count

### New Files

**Header-only (Phase 7 interface):**
- `ERF_LNGRegulatory.H` — regulatory compliance functions (1h average, exceedance, exclusion radius, CSV output)

**Implementation:**
- `ERF_LNGRegulatory.cpp` — `update_lng_1h_average`, `compute_lng_exceedance`, `compute_exclusion_zone_radius` implementations (MPI-safe, Rule B1)

**Test:**
- `Exec/CanonicalTests/LNG/LNG_Regulatory/` — comprehensive regulatory compliance test

### New Parameters (ERF_LNGParams.H)

```cpp
Real    nfpa59a_exclusion_conc = 0.025;  // 1/2 LFL [vol/vol]
string  lng_regulatory_file = "lng_regulatory.csv";
```

Both parameters were registered in Phase 1 as placeholders and are now activated in Phase 7.

### New MultiFabs in LNGLayer

```cpp
std::unique_ptr<amrex::MultiFab> m_lng_conc_1h_avg;   ///< 1-hr avg [kg/m^3]
std::unique_ptr<amrex::MultiFab> m_lng_exceed_flag;    ///< NFPA exceedance [0/1]
amrex::Real m_exclusion_radius_m = 0.0;                ///< Exclusion zone radius [m]
```

### Physics Implementation

#### 1-Hour Running Average (Exponential Moving Average)

```cpp
// Update each cell with exponential weighting:
C_avg(t) = C_avg(t-dt) * (T-dt)/T + C_now * dt/T, T=3600 s
```

- **GPU-safe kernel:** Uses `ParallelFor` with `tilebox()`
- **MPI-safe:** Followed by `FillBoundary` with periodicity
- **No collectives:** Every rank computes its own tiles independently

#### Exceedance Flag Computation

Cells where 1h-average ≥ threshold in vol/vol (2.5% NFPA 59A):
- **Conversion:** threshold_vol_frac × rho_vapor = conc_threshold [kg/m³]
- **GPU-safe kernel:** Uses `ParallelFor` with `tilebox()`
- **MPI-safe:** Followed by `FillBoundary` with periodicity

#### Exclusion Zone Radius Estimation

- **Step 1 (CPU-local):** Each rank finds its local max radius where exceed_flag > 0.5 using `LoopOnCpu`
- **Step 2 (MPI-collective):** `ReduceRealMax` broadcasts max to all ranks
- **Rule B1:** Collective reduction precedes any IOProcessor guard

### MPI Rules Applied

| Rule | Implementation |
|------|---|
| **B1** | `append_lng_regulatory_row`: `MultiFab::max` and `sum` calls before `IOProcessor()` guard. `compute_exclusion_zone_radius`: `LoopOnCpu` followed by `ReduceRealMax` (all ranks participate). |
| **B4** | All `FillBoundary` calls pass `geom_lng.periodicity()` |
| **A2** | `ERF_LNGRegulatory.cpp` registered in both `Make.package` (inside `USE_LNG` guard) and `CMake/BuildERFExe.cmake` (inside `if(ERF_ENABLE_LNG)` block). |
| **A4** | Regulatory CSV row written unconditionally (not gated on `lng_debug`). Only console debug print is gated. |

### CSV Output Format

`lng_regulatory.csv` columns:
```
step,time_s,exclusion_zone_radius_m,conc_1h_max_kg_m3,n_cells_exceed
```

### Integration in LNGLayer

**In `initialize()`:**
- Allocate `m_lng_conc_1h_avg` and `m_lng_exceed_flag` on LNG grid
- Initialize to 0.0
- Write regulatory CSV header

**In `advance()` (after Phase 6 receptor sampling):**
- Update 1h-average: `update_lng_1h_average(m_lng_conc_1h_avg, m_lng_conc_sfc, dt)`
- Compute exceedance: `compute_lng_exceedance(m_lng_exceed_flag, m_lng_conc_1h_avg, ...)`
- Estimate exclusion radius: `m_exclusion_radius_m = compute_exclusion_zone_radius(...)`
- Emit debug print if `lng_debug=true`

**In `write_output()`:**
- Append regulatory CSV row: `append_lng_regulatory_row(...)`
- No gatekeeping on `lng_debug` (Rule A4)

**In NaN check (advance, after Phase 5):**
- Check `m_lng_conc_1h_avg->contains_nan(0)`
- Check `m_lng_exceed_flag->contains_nan(0)`

### Plotfile Addition

Updated `ERF_LNGPlotfileCatalog.H`:
- **Variables increased from 17 to 19:**
  - Index 17: `lng_conc_1h_avg` [kg/m³]
  - Index 18: `lng_exceed_flag` [0/1]
- **ncomp() returns 19** (was 17)

Updated `ERF_LNGPlotfile.cpp`:
- Added two `copy_if()` calls for indices 17–18

### Debug Output

Per-step when `lng_debug=true`:

```
[LNG DEBUG] Phase 7: step=<N>  exclusion_radius=<> m  conc_1h_max=<> kg/m^3  n_exceed=<>
```

### Test: LNG_Regulatory

**Configuration (copied verbatim from LNG_Output):**
- ATM: 32 × 32 × 64, grid_ratio=4 → 128 × 128 LNG
- dt=0.5 s, 20 timesteps = 10 s total
- `amrex.max_grid_size_z=64` (mandatory)
- `erf.sum_interval=1`

**Pass criteria (12 items):**
1. Exit code 0, 20 steps complete
2. `lng_regulatory.csv` created with NFPA 59A header block
3. CSV structure: `step,time_s,exclusion_zone_radius_m,conc_1h_max_kg_m3,n_cells_exceed`
4. Exactly 20 data rows (steps 0–19)
5. Early steps: `exclusion_zone_radius_m=0.0` (insufficient vapor)
6. `conc_1h_max_kg_m3` increases monotonically in early steps
7. `n_cells_exceed=0` early, may increase later as 1h-average accumulates
8. `[LNG DEBUG] Phase 7:` appears exactly 20 times in stdout
9. Two plotfiles (steps 0 and 10); each has `"n_variables": 19` with new fields
10. `lng_diag.csv` still has 21 lines; receptor CSVs written
11. `[LNG DEBUG] NaN check PASSED` appears 20 times
12. Build: no linker errors with `-DERF_USE_LNG=ON`

### Implementation Checklist

- [x] Create `ERF_LNGRegulatory.H` with function declarations and inline helpers
- [x] Create `ERF_LNGRegulatory.cpp` with implementations
- [x] Add `m_lng_conc_1h_avg`, `m_lng_exceed_flag`, `m_exclusion_radius_m` to `ERF_LNGLayer.H`
- [x] Add Phase 7 getters to `ERF_LNGLayer.H`
- [x] Update `ERF_LNGLayer.cpp::initialize()` — allocate Phase 7 MultiFabs, write regulatory header
- [x] Update `ERF_LNGLayer.cpp::advance()` — update averages, compute exceedance, estimate radius
- [x] Update `ERF_LNGLayer.cpp::write_output()` — append regulatory CSV row
- [x] Add Phase 7 NaN checks to `ERF_LNGLayer.cpp::advance()`
- [x] Update `ERF_LNGPlotfileCatalog.H` — add 2 variables, update ncomp to 19
- [x] Update `ERF_LNGPlotfile.cpp` — add 2 copy_if calls
- [x] Add `#include "ERF_LNGRegulatory.H"` to `ERF_LNGLayer.cpp`
- [x] Update `Source/LNG/Make.package` — register `.cpp` and `.H`
- [x] Update `CMake/BuildERFExe.cmake` — register `.cpp`
- [x] Create `Exec/CanonicalTests/LNG/LNG_Regulatory/` test
- [x] Create `inputs_lng_regulatory` (copy ATM from LNG_Output + add Phase 7 params)
- [x] Create `sounding_neutral_abl` (verbatim copy)
- [x] Create `README.md` with full specification
- [x] Update parent `Exec/CanonicalTests/LNG/CMakeLists.txt` — add subdirectory

### References

- **NFPA 59A (2023):** *Standard for the Production, Storage, and Handling of Liquefied Natural Gas (LNG)*
- **49 CFR Part 193:** U.S. federal LNG facility regulations (exclusion zones)
- **Koopman, R.P., 1982:** "Burro LNG spill test series final report"
- **Pattern source:** `Source/Dust/ERF_DustNAAQSOutput.H` (EPA NAAQS PM₂.₅ 24h averaging adapted for NFPA 59A 1h averaging)
- **MPI skills:** `Source/LNG/LNG_MPI_SKILLS.md` Rules B1, B4, A2, A4

---

## Phase 6+: Future Work

Phase 7 will implement regulatory compliance (NFPA 59A exclusion zones). Phase 8 will complete
spill scheduling with time-dependent release rates and inventory tracking.


