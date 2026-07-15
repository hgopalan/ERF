
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
| 1 | **Build & Initialize** | Fully compilable stub; parameter reading; grid construction; debug output | **ACTIVE** |
| 2 | **Evaporation & Pool Spreading** | Heat transfer model; Clausius-Clapeyron; gravity current spreading | TODO |
| 3 | **ATM Coupling (Phase I)** | Energy injection to atmosphere; sensible/latent heat source terms | TODO |
| 4 | **Wind & BL Extraction** | Wind field interpolation at zref; u* mapping; PBL height feedback | TODO |
| 5 | **Flammability Tracking** | LFL/UFL exceedance zones; buoyancy-driven dispersion; plume rise | TODO |
| 6 | **Output & Visualization** | Plotfile writes; receptor point sampling; CSV output expansion | TODO |
| 7 | **Regulatory Compliance** | NFPA 59A exclusion zone calculation; threshold mapping | TODO |
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



