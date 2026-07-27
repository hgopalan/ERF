# Phase 3.1a — Level-aware allocation & call-site audit

**Branch:** ERF-SLUCM  
**Task type:** static analysis (no code changes)  
**Purpose:** classify every UCM code site that touches ATM AMR level selection as CORRECT, HARDCODED_LEVEL_0, AMBIGUOUS, or N/A before Phase 3.1b.

## Section 1: Files inspected

- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCM.H`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCM.cpp`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMAllocate.H`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMAllocate.cpp`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMAtmAggregation.H`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMAtmCoupling.H`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMAtmPlotfile.H`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMAtmPlotfile.cpp`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMBuildingLayoutReader.H`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMBuildingLayoutReader.cpp`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMDiagnostics.H`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMDiagnostics.cpp`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMFacet3D.H`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMFields.H`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMGrid.H`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMGrid.cpp`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMLayer.H`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMLayer.cpp`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMMaterialRegistry.H`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMMaterialRegistry.cpp`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMParams.H`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMParams.cpp`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMPlotfile.H`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMPlotfile.cpp`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMPlotfileCatalog.H`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMPrerequisites.H`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMPrerequisites.cpp`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMShadowing.H`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMSlabConduction.H`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMWindExtract.H`
- `/home/runner/work/ERF/ERF/Source/UrbanCanopy/ERF_UCMWindExtract.cpp`

## Section 2: Grep results (raw)

Raw command outputs are included below exactly as executed from `/home/runner/work/ERF/ERF`. Documentation hits in `*.md` are preserved here but excluded from Section 3 classification, which only covers code sites.

<details>
<summary><code>grep -rn "get_new_data(0)"       Source/UrbanCanopy/</code></summary>

```text
(no matches)
```

</details>

<details>
<summary><code>grep -rn "get_old_data(0)"       Source/UrbanCanopy/</code></summary>

```text
(no matches)
```

</details>

<details>
<summary><code>grep -rn "\.getLevel(0)"         Source/UrbanCanopy/</code></summary>

```text
(no matches)
```

</details>

<details>
<summary><code>grep -rn "level = 0"             Source/UrbanCanopy/</code></summary>

```text
Source/UrbanCanopy/UCM_DEVELOPMENT.md:162:- `amr.max_level = 0` (no AMR)
Source/UrbanCanopy/UCM_DEVELOPMENT.md:165:- `erf.ucm.anchor_level = 0`
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1668:amr.max_level = 0
Source/UrbanCanopy/ERF_UCMPrerequisites.cpp:49:        "[UCM] anchor_level must be >= 0. Set: erf.ucm.anchor_level = 0");
Source/UrbanCanopy/ERF_UCMPrerequisites.cpp:58:                        + "Set: erf.ucm.anchor_level = 0 or wait for Phase 3.1 multi-level support. "
```

</details>

<details>
<summary><code>grep -rn "level(0)"              Source/UrbanCanopy/</code></summary>

```text
(no matches)
```

</details>

<details>
<summary><code>grep -rn "lev = 0"               Source/UrbanCanopy/</code></summary>

```text
Source/UrbanCanopy/ERF_UCMWindExtract.H:116:                                       int lev = 0);
Source/UrbanCanopy/ERF_UCMWindExtract.H:168:                                     int lev = 0);
Source/UrbanCanopy/ERF_UCMWindExtract.H:200:                              int lev = 0);
Source/UrbanCanopy/ERF_UCMGrid.H:87:                        int                             lev = 0,
Source/UrbanCanopy/ERF_UCMShadowing.H:74:    int lev = 0,
Source/UrbanCanopy/ERF_UCMLayer.H:79:    UCMLayer(const UCMParams& params, int lev = 0);
Source/UrbanCanopy/ERF_UCMLayer.H:163:                 int lev = 0);
Source/UrbanCanopy/UCM_DEVELOPMENT.md:137:1. **ERF_UCMParams.H** – Struct with 22 parameters + `read_from_parmparse(int lev = 0)` method
Source/UrbanCanopy/ERF_UCMParams.H:222:    void read_from_parmparse(int lev = 0);
Source/UrbanCanopy/ERF_UCMMaterialRegistry.H:125:    void load_and_broadcast(const std::string& path, int lev = 0, bool ucm_debug = false);
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:98:    int                        lev = 0);
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:149:    int                        lev = 0);
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:271:    int                    lev = 0);
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:334:    int                    lev = 0);
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:386:    int                    lev = 0);
Source/UrbanCanopy/ERF_UCMDiagnostics.H:60:    UCMDiagnostics(const UCMParams& params, int lev = 0);
Source/UrbanCanopy/ERF_UCMDiagnostics.H:112:                int lev = 0);
Source/UrbanCanopy/ERF_UCMPlotfile.H:46:    UCMPlotfile(const UCMParams& params, int lev = 0);
Source/UrbanCanopy/ERF_UCMPlotfile.H:80:               int lev = 0);
Source/UrbanCanopy/ERF_UCMBuildingLayoutReader.H:133:                            int lev = 0, bool ucm_debug = false);
Source/UrbanCanopy/ERF_UCMAtmPlotfile.H:95:              int                       lev = 0);
Source/UrbanCanopy/ERF_UCMAllocate.H:47:                         int lev = 0);
Source/UrbanCanopy/ERF_UCMAllocate.H:80:                              int lev = 0,
Source/UrbanCanopy/ERF_UCMAllocate.H:113:                                  int lev = 0);
Source/UrbanCanopy/ERF_UCMAllocate.H:135:                          int lev = 0);
Source/UrbanCanopy/ERF_UCMAllocate.H:166:                               int                     lev = 0);
Source/UrbanCanopy/ERF_UCMPrerequisites.H:62:                              int lev = 0);
Source/UrbanCanopy/ERF_UCMPrerequisites.H:97:                                int lev = 0);
```

</details>

<details>
<summary><code>grep -rn "lev(0)"                Source/UrbanCanopy/</code></summary>

```text
(no matches)
```

</details>

<details>
<summary><code>grep -rn "boxArray(0)"           Source/UrbanCanopy/</code></summary>

```text
(no matches)
```

</details>

<details>
<summary><code>grep -rn "DistributionMap(0)"    Source/UrbanCanopy/</code></summary>

```text
(no matches)
```

</details>

<details>
<summary><code>grep -rn "Geom(0)"               Source/UrbanCanopy/</code></summary>

```text
(no matches)
```

</details>

<details>
<summary><code>grep -rn "anchor_level"          Source/UrbanCanopy/</code></summary>

```text
Source/UrbanCanopy/ERF_UCMParams.cpp:25:    pp.query("anchor_level", anchor_level);
Source/UrbanCanopy/UCM_DEVELOPMENT.md:32:| 3 | 3.1 | Finest-level anchoring turned on + multi-level regression | anchor_level > 0 enabled, multi-AMR-level UCM slab | 🔲 PLANNED |
Source/UrbanCanopy/UCM_DEVELOPMENT.md:57:All UCM MultiFab members held by the `ERF` class are declared as `amrex::Vector<std::unique_ptr<amrex::MultiFab>>` sized to `finest_level+1`. In Phase 1.1, only index `anchor_level` is allocated; other indices remain `nullptr`.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:63:- **`erf.ucm.anchor_level`** (int, default `0`): The single AMR level UCM runs on. Must satisfy `0 ≤ anchor_level ≤ finest_level`. In Phase 1.1, only `anchor_level=0` is exercised; higher values are gated by an assertion in the prerequisites check (Phase 3.1 relaxes this).
Source/UrbanCanopy/UCM_DEVELOPMENT.md:64:- **`erf.ucm.static_refinement`** (bool, default `true`): Required to be `true`. If AMR issues a regrid on `anchor_level` during the run, UCM must error out. In Phase 1.1, install the assertion; regrid-detection hook is a TODO comment (Phase 3.3).
Source/UrbanCanopy/UCM_DEVELOPMENT.md:73:4. Prerequisites check (Phase 1.1): if `erf.use_terrain = true`, initialize expecting `z_phys_cc[anchor_level]` populated. If `erf.use_terrain = false`, `z_phys_cc[anchor_level]` is flat by construction; same code path handles both.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:81:Declare a per-level `ucm_is_urban` `iMultiFab` member on `ERF`, allocated on the UCM 2D slab at `anchor_level` (allocation actually happens Phase 1.2 when UCM grid is created; Phase 1.1 only has declaration). In Phase 1.1, no LSM/MOST bypass hooks are wired; that is Phase 4.1. Only field declaration and TODO comments are added.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:112:| `anchor_level` | `int` | `0` | AMR level at which UCM runs |
Source/UrbanCanopy/UCM_DEVELOPMENT.md:165:- `erf.ucm.anchor_level = 0`
Source/UrbanCanopy/ERF_UCMParams.H:14: * sized to finest_level+1; only anchor_level is allocated in Phase 1.1.
Source/UrbanCanopy/ERF_UCMParams.H:17: * `anchor_level` (int, default 0): Single AMR level UCM runs on.
Source/UrbanCanopy/ERF_UCMParams.H:19: * In Phase 1.1, only anchor_level=0 is allowed; higher values gated for Phase 3.1.
Source/UrbanCanopy/ERF_UCMParams.H:89:    int     anchor_level       = 0;         ///< AMR level at which UCM runs [Phase 1.1 only 0]
Source/UrbanCanopy/ERF_UCMParams.H:90:    bool    static_refinement  = true;      ///< Static refinement required (no regrid on anchor_level)
Source/UrbanCanopy/ERF_UCM.H:46: * only anchor_level allocated in Phase 1.1.
Source/UrbanCanopy/ERF_UCM.H:49: * `anchor_level` (int, default 0): Single AMR level UCM runs on.
Source/UrbanCanopy/ERF_UCM.H:51: * Phase 1.1: anchor_level=0 only; Phase 3.1 enables higher levels.
Source/UrbanCanopy/ERF_UCMPrerequisites.cpp:46:    // Check 1: anchor_level within bounds
Source/UrbanCanopy/ERF_UCMPrerequisites.cpp:48:        params.anchor_level >= 0,
Source/UrbanCanopy/ERF_UCMPrerequisites.cpp:49:        "[UCM] anchor_level must be >= 0. Set: erf.ucm.anchor_level = 0");
Source/UrbanCanopy/ERF_UCMPrerequisites.cpp:52:        params.anchor_level <= finest_level,
Source/UrbanCanopy/ERF_UCMPrerequisites.cpp:53:        "[UCM] anchor_level must be <= finest_level. Reduce erf.ucm.anchor_level");
Source/UrbanCanopy/ERF_UCMPrerequisites.cpp:55:    // Check 2: Phase 1.1 constraint - anchor_level must be 0
Source/UrbanCanopy/ERF_UCMPrerequisites.cpp:56:    if (params.anchor_level > 0) {
Source/UrbanCanopy/ERF_UCMPrerequisites.cpp:57:        std::string msg = std::string("[UCM] anchor_level > 0 not supported in Phase 1.1. ")
Source/UrbanCanopy/ERF_UCMPrerequisites.cpp:58:                        + "Set: erf.ucm.anchor_level = 0 or wait for Phase 3.1 multi-level support. "
Source/UrbanCanopy/ERF_UCMPrerequisites.cpp:177:    amrex::Print() << "[UCM]   anchor_level        = " << params.anchor_level << "\n";
Source/UrbanCanopy/ERF_UCMLayer.cpp:32:    // Phase 1.1: enforce anchor_level == 0
Source/UrbanCanopy/ERF_UCMLayer.cpp:33:    if (lev != params.anchor_level) {
Source/UrbanCanopy/ERF_UCMLayer.cpp:35:                        + std::to_string(lev) + " but params.anchor_level = "
Source/UrbanCanopy/ERF_UCMLayer.cpp:36:                        + std::to_string(params.anchor_level)
Source/UrbanCanopy/ERF_UCMLayer.cpp:37:                        + ". Phase 1.3 supports only anchor_level=0.";
Source/UrbanCanopy/ERF_UCMPrerequisites.H:10: *  - AMR level constraints (anchor_level must be within [0, finest_level])
Source/UrbanCanopy/ERF_UCMPrerequisites.H:38: *  1. anchor_level >= 0 and anchor_level <= finest_level
Source/UrbanCanopy/ERF_UCMPrerequisites.H:39: *  2. anchor_level == 0 (higher values gated for Phase 3.1)
```

</details>

<details>
<summary><code>grep -rn "erf\."                 Source/UrbanCanopy/ | grep -v "erf.ucm"</code></summary>

```text
Source/UrbanCanopy/ERF_UCMDiagnostics.cpp:26:    // TODO: In full ERF integration, prepend erf.plot_file_base if available
Source/UrbanCanopy/UCM_MPI_SKILLS.md:65:erf.fixed_dt         = 0.5
Source/UrbanCanopy/UCM_MPI_SKILLS.md:67:erf.pbl_type         = "MYNN2.5" (or simplest available)
Source/UrbanCanopy/UCM_MPI_SKILLS.md:68:erf.transport_scalar = true
Source/UrbanCanopy/UCM_DEVELOPMENT.md:73:4. Prerequisites check (Phase 1.1): if `erf.use_terrain = true`, initialize expecting `z_phys_cc[anchor_level]` populated. If `erf.use_terrain = false`, `z_phys_cc[anchor_level]` is flat by construction; same code path handles both.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:168:- `erf.use_terrain = false` (flat terrain)
Source/UrbanCanopy/UCM_DEVELOPMENT.md:213:**Root cause:** Agent synthesized fake sounding instead of using reference file; used wrong sounding column format, wrong `n_cell`, `amr.dt_shrink` (not applicable), wrong boundary types (`slip_wall` vs `SlipWall`), omitted `erf.prob_name`, omitted surface-layer roughness, omitted Coriolis, used `MYNN2.5` instead of `MRF`.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:587:- Removed `erf.fixed_dt` (deprecated timestepping)
Source/UrbanCanopy/UCM_DEVELOPMENT.md:588:- Added `erf.cfl = 0.5` (adaptive dt)
Source/UrbanCanopy/UCM_DEVELOPMENT.md:633:- All now use `erf.cfl = 0.5` instead of deprecated `erf.fixed_dt`
Source/UrbanCanopy/UCM_DEVELOPMENT.md:645:5. ✅ No `erf.fixed_dt` in any test input
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1137:  - `erf.most.surf_temp_flux = 0.02` (unstable surface T flux, ~25 W/m²)
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1202:  - **Inputs:** `max_step=2`, `erf.cfl=0.5`, Phase 2.6 parameters enabled, `ucm_atm_plot_int=1` (write after each step), `ucm_debug=1`.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1342:  - **Inputs:** `max_step=2`, `erf.cfl=0.5`, Phase 2.7 parameters enabled (`use_facet3d_injection=1`, `use_gaussian_height_distribution=0` for sharp-mode test), `ucm_atm_plot_int=1`, `ucm_debug=1`. Phase 2.6 morphology params still present (for fallback).
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1461:- **Inputs:** `max_step=10` (need more steps to see wind decay vs instant heat); `erf.cfl=0.5`; `wall_drag_mode="explicit"` (force compressible path); `Cd_wall=0.4`, `Cd_roof=0.15`; `ucm_debug=1`, `ucm_atm_plot_int=1`, `amr.plot_int=5`.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1535:  - `erf.terrain_type = None` (no DEM terrain file; buildings from SLUCM CSV).
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1545:  - `geometry.is_periodic = 0 1 0`, `erf.terrain_type = None`.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1546:  - MRF + MOST retained, with neutral surface forcing (`erf.most.surf_temp_flux = 0.0`).
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1664:erf.cfl = 0.5
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1669:erf.terrain_type = None
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1670:erf.pbl_type = "MRF"
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1671:erf.theta_ref = 295.0
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1674:erf.plot_file_1 = "plt_"
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1675:erf.plot_int_1 = 600
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1676:erf.plot_vars_1 = x_velocity y_velocity z_velocity theta
Source/UrbanCanopy/ERF_UCMPlotfile.H:32: * **Location:** Same output directory as main ATM plotfiles (erf.plot_file_base).
Source/UrbanCanopy/ERF_UCMPlotfile.H:90:     * Combines erf.plot_file_base (or current directory if empty) with
Source/UrbanCanopy/ERF_UCMPlotfile.cpp:30:    // TODO: In full ERF integration, query erf.plot_file_base via ParmParse
Source/UrbanCanopy/ERF_UCMAtmPlotfile.H:31: * **Location:** Same output directory as main ATM plotfiles (erf.plot_file_base).
Source/UrbanCanopy/ERF_UCMAtmPlotfile.H:103:     * Combines erf.plot_file_base (or current directory if empty) with
```

</details>

<details>
<summary><code>grep -rn "amr_wind"              Source/UrbanCanopy/</code></summary>

```text
(no matches)
```

</details>

<details>
<summary><code>grep -rn "cc_source"             Source/UrbanCanopy/</code></summary>

```text
Source/UrbanCanopy/UCM_MPI_SKILLS.md:137:// Step 2: Apply exponential vertical decay to cc_source
Source/UrbanCanopy/UCM_MPI_SKILLS.md:138:apply_ucm_tendency_to_cc_source(cc_source, Q_atm, z_phys_cc, S_old, geom_atm,
Source/UrbanCanopy/UCM_MPI_SKILLS.md:159:    cc_source(i, j, k, RhoTheta_comp) += feedback * theta_tend;
Source/UrbanCanopy/UCM_MPI_SKILLS.md:438:(`apply_ucm_tendency_to_cc_source`) reads Q_atm AS-IS with NO multiplication by `f_urb_atm`.
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:7: * `apply_ucm_tendency_to_cc_source` OWNS `cc_source[RhoTheta_comp]` (and
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:8: * optionally `cc_source[RhoQ1_comp]`) for the duration of each RK stage.
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:18: *  - `apply_ucm_tendency_to_cc_source` is then called ONCE PER RK STAGE
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:25: * If any other physics module needs to write into `cc_source[RhoTheta_comp]`
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:145:    // The injection kernel `apply_ucm_tendency_to_cc_source` reads Q_atm AS-IS.
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:217:void apply_ucm_tendency_to_cc_source(
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:218:    amrex::MultiFab&        cc_source,
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:249:        amrex::Print() << "[UCM][2.11][apply_ucm_tendency_to_cc_source]\n";
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:307:    cc_source.setVal(0.0, RhoTheta_comp, 1, cc_source.nGrowVect());
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:309:        cc_source.setVal(0.0, RhoQ1_comp, 1, cc_source.nGrowVect());
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:330:    for (amrex::MFIter mfi(cc_source, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:333:        auto cc_src_a        = cc_source.array(mfi);
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:635:        amrex::Print() << "[UCM][2.7][WARN] apply_ucm_tendency_to_cc_source: "
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:656:            amrex::Print() << "[UCM][2.7][apply_ucm_tendency_to_cc_source]\n";
Source/UrbanCanopy/UCM_DEVELOPMENT.md:404:Phase 1.4 turns UCM into a **fully one-way coupled** module. The sensible and latent heat fluxes computed in Phase 1.3 are coarsened to the ATM grid and injected back into `cc_source` using the WRF-SFIRE exponential-decay pattern (Mandel 2011). This is the first phase where UCM affects atmospheric state.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:410:   - `apply_ucm_tendency_to_cc_source()` — Inject vertical exponential tendency into `RhoTheta_comp` and optionally `RhoQ1_comp`
Source/UrbanCanopy/UCM_DEVELOPMENT.md:444:   cc_source(i, j, k, RhoTheta_comp) += atm_feedback * theta_tend(k)
Source/UrbanCanopy/UCM_DEVELOPMENT.md:449:   cc_source(i, j, k, RhoQ1_comp) += atm_feedback * le_tendency(k) / L_v
Source/UrbanCanopy/UCM_DEVELOPMENT.md:469:- **`apply_ucm_tendency_to_cc_source`** — min/max RhoTheta tendency; k=0 rho and dz; alpha_ucm; atm_feedback; expected surface magnitude
Source/UrbanCanopy/UCM_DEVELOPMENT.md:741:but `apply_ucm_tendency_to_cc_source` accessed component `RhoTheta_comp = 1` on
Source/UrbanCanopy/UCM_DEVELOPMENT.md:749:- `apply_ucm_tendency_to_cc_source` now has **overwrite semantics**: it zeroes
Source/UrbanCanopy/UCM_DEVELOPMENT.md:750:  `cc_source[RhoTheta_comp]` (and `cc_source[RhoQ1_comp]` when moisture is on)
Source/UrbanCanopy/UCM_DEVELOPMENT.md:760:- Removed the redundant `apply_ucm_tendency_to_cc_source(*rhotheta_src[lev], ...)`
Source/UrbanCanopy/UCM_DEVELOPMENT.md:769:- Per-stage `apply_ucm_tendency_to_cc_source(cc_src, ...)` call now passes
Source/UrbanCanopy/UCM_DEVELOPMENT.md:776:`apply_ucm_tendency_to_cc_source` **OWNS** `cc_source[RhoTheta_comp]` (and
Source/UrbanCanopy/UCM_DEVELOPMENT.md:777:optionally `cc_source[RhoQ1_comp]`) for the duration of each RK stage:
Source/UrbanCanopy/UCM_DEVELOPMENT.md:782:3. `apply_ucm_tendency_to_cc_source` is called **once per RK stage** from
Source/UrbanCanopy/UCM_DEVELOPMENT.md:787:If any other physics writes into `cc_source[RhoTheta_comp]` per stage on the
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1056:- **Conservation convention audit** — Confirmed Phase 2.5 uses Convention A (weighted-divide): `Q_atm = sum(is_urban*Q_ucm) / f_urb`. Injection kernel (`apply_ucm_tendency_to_cc_source`) verifies it multiplies back by `f_urb_atm` for proper area-weighted tendency. Added comment above `coarsen_ucm_flux_to_atm` explaining convention with one-sentence conservation argument.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1185:- **Rewritten injection kernel** — `apply_ucm_tendency_to_cc_source` in `ERF_UCMAtmCoupling.H/cpp` completely reworked:
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1188:  - **RK-stage safety:** zeros `cc_source[RhoTheta_comp]` at entry, accumulates both terms via `+=` (never `=` post-zero).
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1197:- **Function call wiring** — `ERF_TI_slow_rhs_pre.H` updated to pass six new parameters to `apply_ucm_tendency_to_cc_source`.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1208:- **Documentation** — This Phase 2.6 section in `UCM_DEVELOPMENT.md`. Physics and design rationale documented in `apply_ucm_tendency_to_cc_source` docstring (header signature in .H file).
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1291:- **Rewritten injection kernel** — `apply_ucm_tendency_to_cc_source` in `ERF_UCMAtmCoupling.H/cpp` completely reworked:
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1312:  - **RK-stage safety:** zeros `cc_source[RhoTheta_comp]` at entry, accumulates via `+=` (identical to Phase 2.6 contract).
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1316:    [UCM][2.7][apply_ucm_tendency_to_cc_source]
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1337:  - Pass all Phase 2.7 parameters plus Phase 2.6 fallback args to `apply_ucm_tendency_to_cc_source`.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1348:- **Documentation** — This Phase 2.7 section in `UCM_DEVELOPMENT.md`. Physics docstrings in `apply_ucm_tendency_to_cc_source` (header file) include Martilli citations.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1757:   - `apply_ucm_tendency_to_cc_source(... feedback_heat, feedback_moisture ...)`
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:11: * 2. **apply_ucm_tendency_to_cc_source**: Apply exponential-decay vertical injection
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:12: *    (after Mandel 2011 / WRF-SFIRE fire_tendency pattern) to cc_source at every k.
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:15: * `apply_ucm_tendency_to_cc_source` OWNS `cc_source[RhoTheta_comp]` (and
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:16: * optionally `cc_source[RhoQ1_comp]`) for the duration of each RK stage.
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:119: * The injection kernel `apply_ucm_tendency_to_cc_source` reads Q_atm AS-IS with no
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:152: * @brief Apply exponential-decay vertical tendency to cc_source (Phase 2.6: morphology-aware).
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:180: * Total tendency: `theta_tend = theta_tend_road + theta_tend_exp` (both added to cc_source).
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:186: * [UCM][2.6][apply_ucm_tendency_to_cc_source]
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:196: * @param[in,out] cc_source              Cell-centered source term MultiFab (modified)
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:220: * @note RK-stage safety: OWNS cc_source[RhoTheta_comp]. Zeros at entry, always uses `+=`.
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:222: * @param cc_source         Coarse-grid source term to be populated [output]
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:248:void apply_ucm_tendency_to_cc_source(
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:249:    amrex::MultiFab&       cc_source,
Source/UrbanCanopy/ERF_UCMFacet3D.H:23: *  - Called once per ATM cell k in `apply_ucm_tendency_to_cc_source` via ParallelFor
```

</details>

<details>
<summary><code>grep -rn "xmom_src\|ymom_src\|zmom_src" Source/UrbanCanopy/</code></summary>

```text
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:675: * @brief Apply BEP momentum drag to xmom_src and ymom_src (Phase 2.8 compressible)
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:682:    amrex::MultiFab&       xmom_src,
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:683:    amrex::MultiFab&       ymom_src,
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:739:    for (amrex::MFIter mfi(xmom_src, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:742:        auto xmom_a      = xmom_src.array(mfi);
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:743:        auto ymom_a      = ymom_src.array(mfi);
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1417:- **Compressible mode (explicit):** New function `apply_ucm_momentum_drag_to_source()` in `ERF_UCMAtmCoupling.cpp`. Four-kernel pattern (facet3d × terrain) identical to Phase 2.7 heat injection. Adds `ρ * F_wall` and `ρ * F_roof` directly to `xmom_src` and `ymom_src` after `make_mom_sources` (phase RK-stage safety: fixed timestep, momentum sources recomputed per stage, no drift).
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1446:    xmom_src, ymom_src,
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1483:4. RK-stage safety: own `xmom_src`, `ymom_src` only within this function; zero at entry (handled by caller `make_mom_sources`), accumulate via `+=`.
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:274: * @brief Apply BEP momentum drag to xmom_src and ymom_src (Phase 2.8 compressible mode)
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:290: * @param[in,out] xmom_src              x-momentum source [output, modified]
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:291: * @param[in,out] ymom_src              y-momentum source [output, modified]
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:315:    amrex::MultiFab&       xmom_src,
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:316:    amrex::MultiFab&       ymom_src,
```

</details>

<details>
<summary><code>grep -rn "MFIter"                Source/UrbanCanopy/</code></summary>

```text
Source/UrbanCanopy/ERF_UCMAtmAggregation.H:128:    for (MFIter mfi(f_urb_atm, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:92:    for (MFIter mfi(Q_ucm_out, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:175:       for (MFIter mfi(Q_masked, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:330:    for (amrex::MFIter mfi(cc_source, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:739:    for (amrex::MFIter mfi(xmom_src, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:972:    for (amrex::MFIter mfi(S_new, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/UrbanCanopy/ERF_UCMShadowing.H:36:#include <AMReX_MFIter.H>
Source/UrbanCanopy/ERF_UCMShadowing.H:50: *   2. Iterate MFIter over H_bldg
Source/UrbanCanopy/ERF_UCMShadowing.H:86:    for (MFIter mfi(H_bldg, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/UrbanCanopy/ERF_UCMAllocate.cpp:362:    for (MFIter mfi(*(fields.H_bldg)); mfi.isValid(); ++mfi) {
Source/UrbanCanopy/ERF_UCMAllocate.cpp:490:        for (amrex::MFIter mfi(*(fields.is_urban)); mfi.isValid(); ++mfi) {
Source/UrbanCanopy/ERF_UCMAllocate.cpp:975:    for (amrex::MFIter mfi(*f.H_bldg, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/UrbanCanopy/ERF_UCMAllocate.cpp:1028:    for (amrex::MFIter mfi(AH_out, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/UrbanCanopy/ERF_UCMAllocate.cpp:1056:    for (amrex::MFIter mfi(AH_out); mfi.isValid(); ++mfi) {
Source/UrbanCanopy/ERF_UCMLayer.cpp:259:    for (amrex::MFIter mfi(*forcing.u_star, amrex::TilingIfNotGPU());
Source/UrbanCanopy/ERF_UCMPlotfile.cpp:87:    for (amrex::MFIter mfi(ucm_plot, false); mfi.isValid(); ++mfi) {
Source/UrbanCanopy/ERF_UCMWindExtract.cpp:75:    for (amrex::MFIter mfi(ucm_wind_ref, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/UrbanCanopy/ERF_UCMWindExtract.cpp:127:    for (amrex::MFIter mfi(ucm_scalar, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
```

</details>

<details>
<summary><code>grep -rn "MultiFab"              Source/UrbanCanopy/ | head -100</code></summary>

```text
Source/UrbanCanopy/ERF_UCMWindExtract.H:30:#include <AMReX_MultiFab.H>
Source/UrbanCanopy/ERF_UCMWindExtract.H:42: * All fields are 2D MultiFabs on the UCM grid with components as follows:
Source/UrbanCanopy/ERF_UCMWindExtract.H:57:    std::unique_ptr<amrex::MultiFab> u_star;
Source/UrbanCanopy/ERF_UCMWindExtract.H:61:    std::unique_ptr<amrex::MultiFab> wind_ref;
Source/UrbanCanopy/ERF_UCMWindExtract.H:66:    std::unique_ptr<amrex::MultiFab> T_atm_ref;
Source/UrbanCanopy/ERF_UCMWindExtract.H:71:    std::unique_ptr<amrex::MultiFab> q_atm_ref;
Source/UrbanCanopy/ERF_UCMWindExtract.H:76:    std::unique_ptr<amrex::MultiFab> SW_down;
Source/UrbanCanopy/ERF_UCMWindExtract.H:81:    std::unique_ptr<amrex::MultiFab> LW_down;
Source/UrbanCanopy/ERF_UCMWindExtract.H:108: * @param[out] ucm_ustar      Target UCM u* MultiFab [m/s]
Source/UrbanCanopy/ERF_UCMWindExtract.H:109: * @param[in]  atm_u_star     Source ATM u* MultiFab from SurfaceLayer [m/s]
Source/UrbanCanopy/ERF_UCMWindExtract.H:113:void fill_ucm_ustar_from_surface_layer(amrex::MultiFab& ucm_ustar,
Source/UrbanCanopy/ERF_UCMWindExtract.H:114:                                       const amrex::MultiFab& atm_u_star,
Source/UrbanCanopy/ERF_UCMWindExtract.H:139: * they come from z0_ucm and d_disp_ucm MultiFabs.
Source/UrbanCanopy/ERF_UCMWindExtract.H:146: * @param[out] ucm_wind_ref   Target UCM wind MultiFab (2 components: u, v) [m/s]
Source/UrbanCanopy/ERF_UCMWindExtract.H:158:void fill_ucm_wind_from_interpolation(amrex::MultiFab& ucm_wind_ref,
Source/UrbanCanopy/ERF_UCMWindExtract.H:159:                                     const amrex::MultiFab& xvel,
Source/UrbanCanopy/ERF_UCMWindExtract.H:160:                                     const amrex::MultiFab& yvel,
Source/UrbanCanopy/ERF_UCMWindExtract.H:161:                                     const amrex::MultiFab& z_phys_cc,
Source/UrbanCanopy/ERF_UCMWindExtract.H:162:                                     const amrex::MultiFab& H_bldg,
Source/UrbanCanopy/ERF_UCMWindExtract.H:163:                                     const amrex::MultiFab& z0_ucm,
Source/UrbanCanopy/ERF_UCMWindExtract.H:164:                                     const amrex::MultiFab& d_disp_ucm,
Source/UrbanCanopy/ERF_UCMWindExtract.H:188: * @param[out] ucm_scalar    Target UCM scalar MultiFab
Source/UrbanCanopy/ERF_UCMWindExtract.H:189: * @param[in]  atm_scalar    Source ATM scalar MultiFab (e.g., T, q)
Source/UrbanCanopy/ERF_UCMWindExtract.H:195:void fill_ucm_scalar_from_atm(amrex::MultiFab& ucm_scalar,
Source/UrbanCanopy/ERF_UCMWindExtract.H:196:                              const amrex::MultiFab& atm_scalar,
Source/UrbanCanopy/ERF_UCMFields.H:3: * @brief SLUCM MultiFab field container
Source/UrbanCanopy/ERF_UCMFields.H:5: * Declares all per-cell URBPARM parameters and state variables as MultiFabs
Source/UrbanCanopy/ERF_UCMFields.H:9: * All MultiFabs are allocated with ghost cells IntVect(1, 1, 0) for efficient
Source/UrbanCanopy/ERF_UCMFields.H:20:#include <AMReX_MultiFab.H>
Source/UrbanCanopy/ERF_UCMFields.H:21:#include <AMReX_iMultiFab.H>
Source/UrbanCanopy/ERF_UCMFields.H:26: * @brief Container for all SLUCM 2D MultiFab fields on the UCM grid
Source/UrbanCanopy/ERF_UCMFields.H:41:    std::unique_ptr<amrex::MultiFab>  H_bldg;
Source/UrbanCanopy/ERF_UCMFields.H:44:    std::unique_ptr<amrex::MultiFab>  W_road;
Source/UrbanCanopy/ERF_UCMFields.H:47:    std::unique_ptr<amrex::MultiFab>  W_roof;
Source/UrbanCanopy/ERF_UCMFields.H:54:    std::unique_ptr<amrex::MultiFab>  albedo_roof;
Source/UrbanCanopy/ERF_UCMFields.H:57:    std::unique_ptr<amrex::MultiFab>  albedo_wall;
Source/UrbanCanopy/ERF_UCMFields.H:60:    std::unique_ptr<amrex::MultiFab>  albedo_road;
Source/UrbanCanopy/ERF_UCMFields.H:67:    std::unique_ptr<amrex::MultiFab>  emissivity_roof;
Source/UrbanCanopy/ERF_UCMFields.H:70:    std::unique_ptr<amrex::MultiFab>  emissivity_wall;
Source/UrbanCanopy/ERF_UCMFields.H:73:    std::unique_ptr<amrex::MultiFab>  emissivity_road;
Source/UrbanCanopy/ERF_UCMFields.H:81:    std::unique_ptr<amrex::MultiFab>  T_skin_roof;
Source/UrbanCanopy/ERF_UCMFields.H:85:    std::unique_ptr<amrex::MultiFab>  T_skin_wall;
Source/UrbanCanopy/ERF_UCMFields.H:89:    std::unique_ptr<amrex::MultiFab>  T_skin_road;
Source/UrbanCanopy/ERF_UCMFields.H:93:    std::unique_ptr<amrex::MultiFab>  T_canyon_air;
Source/UrbanCanopy/ERF_UCMFields.H:101:    std::unique_ptr<amrex::MultiFab>  H_sensible;
Source/UrbanCanopy/ERF_UCMFields.H:105:    std::unique_ptr<amrex::MultiFab>  LE_latent;
Source/UrbanCanopy/ERF_UCMFields.H:115:    std::unique_ptr<amrex::iMultiFab> is_urban;
Source/UrbanCanopy/ERF_UCMFields.H:124:    std::unique_ptr<amrex::iMultiFab> mat_id_roof;
Source/UrbanCanopy/ERF_UCMFields.H:129:    std::unique_ptr<amrex::iMultiFab> mat_id_wall;
Source/UrbanCanopy/ERF_UCMFields.H:134:    std::unique_ptr<amrex::iMultiFab> mat_id_road;
Source/UrbanCanopy/ERF_UCMFields.H:142:    std::unique_ptr<amrex::MultiFab> k_therm_roof;
Source/UrbanCanopy/ERF_UCMFields.H:146:    std::unique_ptr<amrex::MultiFab> k_therm_wall;
Source/UrbanCanopy/ERF_UCMFields.H:150:    std::unique_ptr<amrex::MultiFab> k_therm_road;
Source/UrbanCanopy/ERF_UCMFields.H:154:    std::unique_ptr<amrex::MultiFab> rho_cp_roof;
Source/UrbanCanopy/ERF_UCMFields.H:158:    std::unique_ptr<amrex::MultiFab> rho_cp_wall;
Source/UrbanCanopy/ERF_UCMFields.H:162:    std::unique_ptr<amrex::MultiFab> rho_cp_road;
Source/UrbanCanopy/ERF_UCMFields.H:166:    std::unique_ptr<amrex::MultiFab> slab_L_roof;
Source/UrbanCanopy/ERF_UCMFields.H:170:    std::unique_ptr<amrex::MultiFab> slab_L_wall;
Source/UrbanCanopy/ERF_UCMFields.H:174:    std::unique_ptr<amrex::MultiFab> slab_L_road;
Source/UrbanCanopy/ERF_UCMFields.H:178:    std::unique_ptr<amrex::MultiFab> z0_ucm;
Source/UrbanCanopy/ERF_UCMFields.H:182:    std::unique_ptr<amrex::MultiFab> d_disp_ucm;
Source/UrbanCanopy/ERF_UCMFields.H:189:    std::unique_ptr<amrex::MultiFab> H_road;
Source/UrbanCanopy/ERF_UCMFields.H:192:    std::unique_ptr<amrex::MultiFab> H_wall;
Source/UrbanCanopy/ERF_UCMFields.H:195:    std::unique_ptr<amrex::MultiFab> H_roof;
Source/UrbanCanopy/ERF_UCMFields.H:198:    std::unique_ptr<amrex::MultiFab> AH;
Source/UrbanCanopy/ERF_UCMFields.H:201:    std::unique_ptr<amrex::MultiFab> AH_Wm2_ucm;
Source/UrbanCanopy/ERF_UCMFields.H:204:    std::unique_ptr<amrex::MultiFab> plan_area_frac;
Source/UrbanCanopy/ERF_UCMFields.H:207:    std::unique_ptr<amrex::iMultiFab> ah_profile_id;
Source/UrbanCanopy/ERF_UCMFields.H:216:    std::unique_ptr<amrex::MultiFab> SVF_wall;
Source/UrbanCanopy/ERF_UCMFields.H:221:    std::unique_ptr<amrex::MultiFab> SVF_road;
Source/UrbanCanopy/ERF_UCMFields.H:225:    std::unique_ptr<amrex::MultiFab> SVF_roof;
Source/UrbanCanopy/ERF_UCMFields.H:232:     * @brief Check that all MultiFabs are allocated
Source/UrbanCanopy/ERF_UCMDiagnostics.cpp:94:                           const amrex::MultiFab* f_urb_atm,
Source/UrbanCanopy/ERF_UCMDiagnostics.cpp:95:                           const amrex::MultiFab* H_bldg_mean_atm,
Source/UrbanCanopy/ERF_UCMDiagnostics.cpp:96:                           const amrex::MultiFab* H_bldg_std_atm,
Source/UrbanCanopy/ERF_UCMDiagnostics.cpp:97:                           const amrex::MultiFab* lambda_f_atm,
Source/UrbanCanopy/ERF_UCMDiagnostics.cpp:98:                           const amrex::MultiFab* H_atm,
Source/UrbanCanopy/ERF_UCMDiagnostics.cpp:126:    const bool local = false; // perform MPI reduction inside MultiFab::max/sum
Source/UrbanCanopy/UCM_MPI_SKILLS.md:143:- When `grid_ratio == 1`: use `amrex::MultiFab::Copy(dst, src, 0, 0, 1, 0)`
Source/UrbanCanopy/UCM_MPI_SKILLS.md:171:**TODO(UCM Phase 1.4):** MultiFab::sum(), max(), min(), contains_nan() are MPI_Allreduce calls. Every rank must execute them before any IOProcessor guard:
Source/UrbanCanopy/UCM_MPI_SKILLS.md:210:**TODO(UCM Phase 1.3):** When UCM grid reads ATM fields with `grid_ratio>1` and 2+ ranks, rank-local LNG tiles may access ATM data owned by other ranks. Fix: copy to scratch MultiFabs first:
Source/UrbanCanopy/UCM_MPI_SKILLS.md:219:Scratch MultiFabs allocated on ATM BoxArray/DistributionMapping in `initialize()`.
Source/UrbanCanopy/UCM_MPI_SKILLS.md:274:**TODO(UCM Phase 1.2):** All UCM MultiFabs use `amrex::IntVect(1, 1, 0)` ghost cells:
Source/UrbanCanopy/UCM_MPI_SKILLS.md:278:auto mf = std::make_unique<amrex::MultiFab>(ba, dm, ncomp, nghost);
Source/UrbanCanopy/UCM_MPI_SKILLS.md:338:- [ ] MultiFab ghost cells: `IntVect(1,1,0)` for 2D slabs
Source/UrbanCanopy/UCM_MPI_SKILLS.md:352:1. Used `MultiFab[mfi].array()` returning by reference → GPU/memory safety bug
Source/UrbanCanopy/UCM_MPI_SKILLS.md:361:- **MultiFab copy:** Use `amrex::MultiFab::Copy(dst, src, srccomp, dstcomp, ncomp, ngrow)` only.
Source/UrbanCanopy/UCM_MPI_SKILLS.md:367:grep -rn "amrex::Copy(" Source/UrbanCanopy/  → MUST be 0 hits (only MultiFab::Copy)
Source/UrbanCanopy/UCM_MPI_SKILLS.md:375:3. Assumed accessors return `Vector<MultiFab*>` → actually return single `MultiFab*`
Source/UrbanCanopy/UCM_MPI_SKILLS.md:381:- **Dereferencing:** All return `MultiFab*`, so dereference: `*m_SurfaceLayer->get_u_star(lev)` to use by reference.
Source/UrbanCanopy/UCM_MPI_SKILLS.md:390:// Then use u_star, t_star, q_star as MultiFab references
Source/UrbanCanopy/ERF_UCMPlotfileCatalog.H:5: * Enumerates all MultiFab components written to plt_ucm_NNNNN files
Source/UrbanCanopy/ERF_UCMAtmAggregation.H:49:#include <AMReX_MultiFab.H>
Source/UrbanCanopy/ERF_UCMAtmAggregation.H:50:#include <AMReX_iMultiFab.H>
Source/UrbanCanopy/ERF_UCMAtmAggregation.H:100:    amrex::MultiFab&           f_urb_atm,
Source/UrbanCanopy/ERF_UCMAtmAggregation.H:101:    amrex::MultiFab&           H_bldg_mean_atm,
Source/UrbanCanopy/ERF_UCMAtmAggregation.H:102:    amrex::MultiFab&           H_bldg_std_atm,
Source/UrbanCanopy/ERF_UCMAtmAggregation.H:103:    amrex::MultiFab&           lambda_p_atm,
Source/UrbanCanopy/ERF_UCMAtmAggregation.H:104:    amrex::MultiFab&           lambda_f_atm,
Source/UrbanCanopy/ERF_UCMAtmAggregation.H:105:    const amrex::MultiFab&     H_bldg_ucm,
```

</details>

<details>
<summary><code>grep -rn "CellSize"              Source/UrbanCanopy/</code></summary>

```text
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:278:    const auto  dx     = geom_atm.CellSizeArray();
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:719:    const auto  dx     = geom_atm.CellSizeArray();
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:955:    const auto  dx     = geom_atm.CellSizeArray();
Source/UrbanCanopy/ERF_UCMAtmPlotfile.cpp:89:    Real dz = geom.CellSize(2);
```

</details>

<details>
<summary><code>grep -rn "ProbLo\|ProbHi"        Source/UrbanCanopy/</code></summary>

```text
Source/UrbanCanopy/ERF_UCMAtmPlotfile.cpp:90:    RealBox slab_rb({geom.ProbLo(0), geom.ProbLo(1), geom.ProbLo(2) + klo*dz},
Source/UrbanCanopy/ERF_UCMAtmPlotfile.cpp:91:                    {geom.ProbHi(0), geom.ProbHi(1), geom.ProbLo(2) + (klo+1)*dz});
```

</details>

<details>
<summary><code>grep -rn "dx\[0\]\|dx\[1\]\|dx\[2\]"    Source/UrbanCanopy/</code></summary>

```text
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:279:    const amrex::Real dz = dx[2];
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:720:    const amrex::Real dz = dx[2];
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:956:    const amrex::Real dz = dx[2];
```

</details>

<details>
<summary><code>grep -rn "Facet3D\|facet3d"       Source/UrbanCanopy/</code></summary>

```text
Source/UrbanCanopy/ERF_UCMParams.cpp:70:    // Section 4.2: Phase 2.7 — Facet3D BEP-style geometric injection parameters
Source/UrbanCanopy/ERF_UCMParams.cpp:71:    pp.query("use_facet3d_injection", use_facet3d_injection);
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:44:#include <UrbanCanopy/ERF_UCMFacet3D.H>
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:214: * When facet3d injection is disabled, the routine falls back to the smoother
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:233:    bool                    use_facet3d_injection,
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:260:        if (use_facet3d_injection) {
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:298:            "[UCM][2.7] Terrain-following Facet3D injection requires z_phys_nd.");
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:361:        if (use_facet3d_injection) {
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:585:                // meaningful even when facet3d injection is disabled.
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:657:            amrex::Print() << "  Mode: facet3d=" << (use_facet3d_injection ? "yes" : "no")
Source/UrbanCanopy/UCM_DEVELOPMENT.md:29:| 2 | 2.7 | Facet3D injection: BEP geometric overlap + terrain-following + Gaussian height PDF | Wall/roof/road 3D geometric splitting, sharp + Gaussian modes, terrain-ready coords | ✅ COMPLETE (Phase 2.7 PR) |
Source/UrbanCanopy/UCM_DEVELOPMENT.md:931:- **Phase 2.7 (future):** Facet3D will use per-facet SVF from ray-tracing
Source/UrbanCanopy/UCM_DEVELOPMENT.md:945:At grid_ratio=4 (UCM 75 m → ATM 300 m), each ATM cell covers 16 UCM cells. When some are urban and some are not (a park inside a city), the current plain average_down averages ALL 16 including the non-urban ones, silently reducing the injected flux. Also, Phase 2.7 (Facet3D injection) needs **subgrid morphology statistics** (mean and std of H_bldg per ATM cell) to build vertical distribution kernels. Phase 2.5 computes those aggregates and fixes the horizontal coarsening.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1034:- **Phase 2.7:** Facet3D injection (uses H_bldg_mean, H_bldg_std from aggregates)
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1244:## Phase 2.7: Facet3D BEP-Continuous-TF (Geometric Overlap, Terrain-Following Coords, Gaussian Height PDF)
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1266:  bool        use_facet3d_injection              = true;    // Enable Phase 2.7 BEP injection
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1273:- **Geometry helper library** — New header-only file `ERF_UCMFacet3D.H` with three GPU-safe inline device functions:
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1292:  - **New signature:** accepts `H_wall_atm`, `H_roof_atm`, `H_bldg_std_atm`, `lambda_p_atm`, `lambda_f_atm`, `z_phys_nd` (nullable), `use_facet3d_injection`, `use_gaussian_height_distribution`, `height_std_threshold_m`, plus Phase 2.6 fallback parameters.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1296:    if (use_facet3d_injection) {
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1313:  - **Fallback:** if `use_facet3d_injection == false`, reverts to Phase 2.6 exponential kernel (or Phase 2.5 if `use_morphology_injection=false`).
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1317:      Mode: facet3d=yes/no  gaussian=yes/no  terrain=yes/no
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1339:- **Canonical test `UCMFacet3DInjection`** — New directory `Exec/CanonicalTests/SLUCM/UCMFacet3DInjection/`:
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1342:  - **Inputs:** `max_step=2`, `erf.cfl=0.5`, Phase 2.7 parameters enabled (`use_facet3d_injection=1`, `use_gaussian_height_distribution=0` for sharp-mode test), `ucm_atm_plot_int=1`, `ucm_debug=1`. Phase 2.6 morphology params still present (for fallback).
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1344:    - `check_facet3d.py` — Assert 9-component plotfile, H_bldg_mean split (left~30 m, right~5 m), flux conservation: `H_atm ≈ H_road_atm*(1-lambda_p) + H_wall_atm*lambda_f + H_roof_atm*lambda_p` within 5%.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1345:    - `check_facet3d_gaussian.py` (optional) — Rerun with `use_gaussian_height_distribution=1`, verify smoother vertical profile, totals conserved vs sharp mode.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1361:- ✅ Phase 2.5 test `UCMScaleAwareAggregation` still passes (fallback: `use_facet3d_injection=false`, `use_morphology_injection=false` → uniform alpha_ucm).
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1362:- ✅ Phase 2.6 test `UCMMorphologyInjection` still passes with `use_facet3d_injection=false` (exponential fallback, bit-for-bit match).
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1363:- ✅ New Phase 2.7 test `UCMFacet3DInjection` exits 0 at step 2; verification script passes.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1370:- ✅ `ERF_UCMFacet3D.H` is header-only, reusable by Phase 2.8 (no circular deps).
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1376:- Phase 2.5 test: `use_facet3d_injection=false`, `use_morphology_injection=false` → uniform alpha_ucm injection.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1377:- Phase 2.6 test: `use_facet3d_injection=false`, `use_morphology_injection=true` → exponential morphology injection (bit-for-bit match).
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1378:- Phase 2.7 test sharp: `use_facet3d_injection=true`, `use_gaussian_height_distribution=false` → BEP sharp mode (this PR).
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1379:- Phase 2.7 test Gaussian: `use_facet3d_injection=true`, `use_gaussian_height_distribution=true`, `H_std >= height_std_threshold_m` → BEP Gaussian mode.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1388:- **Drag momentum coupling** — Phase 2.8 will use same `ERF_UCMFacet3D.H` helpers for wind profile (wall overlap, roof placement) to avoid code duplication.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1417:- **Compressible mode (explicit):** New function `apply_ucm_momentum_drag_to_source()` in `ERF_UCMAtmCoupling.cpp`. Four-kernel pattern (facet3d × terrain) identical to Phase 2.7 heat injection. Adds `ρ * F_wall` and `ρ * F_roof` directly to `xmom_src` and `ymom_src` after `make_mom_sources` (phase RK-stage safety: fixed timestep, momentum sources recomputed per stage, no drift).
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1459:Mirror of Phase 2.7 `UCMFacet3DInjection/`:
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1491:- Phase 2.7 test `UCMFacet3DInjection` still passes with `erf.ucm.wall_drag_mode = "off"` added to inputs.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1536:  - Facet3D heat injection enabled, AH set to noon representative value (`AH_Wm2=40`).
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1547:  - Heat injection off (`use_facet3d_injection=0`, `AH=0`) to isolate momentum drag.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1622:- **Phase 2.7:** Facet3D BEP-continuous injection (wall/roof/road 3D geometric splitting)
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1683:erf.ucm.use_facet3d_injection = 1
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1711:- **Phase 2.7 Facet3D:** Referenced in Phase 2.7 PR; BEP-continuous injection geometry
Source/UrbanCanopy/ERF_UCMParams.H:130:    // Section 4.2: Phase 2.7 — Facet3D BEP-style geometric injection parameters
Source/UrbanCanopy/ERF_UCMParams.H:133:    bool        use_facet3d_injection              = true;       ///< Phase 2.7: enable facet-3D geometric injection with BEP-style overlap; false = fall back to Phase 2.6
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:223: * @param H_atm             Phase 2.5 lumped sensible flux (unused if use_facet3d_injection or use_morphology_injection)
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:237: * @param use_facet3d_injection       Phase 2.7: enable facet-3D geometric injection (default true)
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:264:    bool                   use_facet3d_injection,
Source/UrbanCanopy/ERF_UCMPrerequisites.cpp:220:    amrex::Print() << "[UCM]   --- Phase 2.7 Facet3D BEP-Continuous-TF ---\n";
Source/UrbanCanopy/ERF_UCMPrerequisites.cpp:221:    amrex::Print() << "[UCM]   use_facet3d_injection = " << (params.use_facet3d_injection ? "true" : "false") << "\n";
Source/UrbanCanopy/ERF_UCMLayer.cpp:309:           // they sum to the ATM-cell sensible flux. Phase 2.7 Facet3D injection assumes
Source/UrbanCanopy/ERF_UCMFacet3D.H:2: * @file ERF_UCMFacet3D.H
Source/UrbanCanopy/ERF_UCMFacet3D.H:3: * @brief Header-only inline device functions for Phase 2.7 Facet3D geometric injection
```

</details>

<details>
<summary><code>grep -rn "SurfaceLayer\|surface_layer\|MOST\|most_" Source/UrbanCanopy/</code></summary>

```text
Source/UrbanCanopy/ERF_UCM.cpp:48:// TODO(UCM Phase 4.1): Add is_urban mask enforcement in LSM/MOST
Source/UrbanCanopy/ERF_UCMWindExtract.H:6: * 1. `fill_ucm_ustar_from_surface_layer` — Copy u* from SurfaceLayer
Source/UrbanCanopy/ERF_UCMWindExtract.H:56:    /// Friction velocity from SurfaceLayer [m/s]
Source/UrbanCanopy/ERF_UCMWindExtract.H:100: * @brief Extract friction velocity u* from SurfaceLayer into UCM
Source/UrbanCanopy/ERF_UCMWindExtract.H:102: * Simple copy of SurfaceLayer u* (computed by MRF, YSU, MYNN2.5) onto UCM grid.
Source/UrbanCanopy/ERF_UCMWindExtract.H:106: * Emits [UCM][1.3][fill_ucm_ustar_from_surface_layer] min/max u*.
Source/UrbanCanopy/ERF_UCMWindExtract.H:109: * @param[in]  atm_u_star     Source ATM u* MultiFab from SurfaceLayer [m/s]
Source/UrbanCanopy/ERF_UCMWindExtract.H:113:void fill_ucm_ustar_from_surface_layer(amrex::MultiFab& ucm_ustar,
Source/UrbanCanopy/ERF_UCMFields.H:108:    // Urban mask (Phase 1.2 NEW; LSM/MOST bypass Phase 4.1)
Source/UrbanCanopy/ERF_UCMFields.H:112:    /// 0 = LSM/MOST owns this cell (non-urban, Phase 2.1+).
Source/UrbanCanopy/ERF_UCMFields.H:114:    /// Phase 4.1: heterogeneous LSM/MOST bypass wired.
Source/UrbanCanopy/ERF_UCMFields.H:177:    /// Phase 1.2 (homogeneous fallback): set to 0.1 m (MOST default).
Source/UrbanCanopy/UCM_MPI_SKILLS.md:66:zlo.type             = "surface_layer"
Source/UrbanCanopy/UCM_MPI_SKILLS.md:370:### Phase 1.3 – Bug 8: SurfaceLayer Accessor API Misuse (Fixed [`8c1cddb`](https://github.com/hgopalan/ERF/commit/8c1cddb))
Source/UrbanCanopy/UCM_MPI_SKILLS.md:372:**Issue:** Phase 1.3 agent misunderstood SurfaceLayer's public API:
Source/UrbanCanopy/UCM_MPI_SKILLS.md:373:1. Called `m_SurfaceLayer->get_u_star()[lev]` — missing `lev` argument
Source/UrbanCanopy/UCM_MPI_SKILLS.md:374:2. Called `m_SurfaceLayer->get_theta_star()` — function does not exist
Source/UrbanCanopy/UCM_MPI_SKILLS.md:381:- **Dereferencing:** All return `MultiFab*`, so dereference: `*m_SurfaceLayer->get_u_star(lev)` to use by reference.
Source/UrbanCanopy/UCM_MPI_SKILLS.md:386:auto& u_star = *m_SurfaceLayer->get_u_star(lev);
Source/UrbanCanopy/UCM_MPI_SKILLS.md:387:auto& t_star = *m_SurfaceLayer->get_t_star(lev);
Source/UrbanCanopy/UCM_MPI_SKILLS.md:388:auto& q_star = *m_SurfaceLayer->get_q_star(lev);
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:679: * MOST owns k=klo momentum; drag is skipped at k=klo.
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:771:                // MOST owns k=klo momentum
Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:843:                // MOST owns k=klo momentum
Source/UrbanCanopy/ERF_UCMLayer.H:93:     * - atm_u_star, atm_t_star, atm_q_star: SurfaceLayer diagnostics [m/s, K, kg/kg]
Source/UrbanCanopy/ERF_UCMLayer.H:126:     * @param[in]     atm_u_star   ATM friction velocity from SurfaceLayer [m/s]
Source/UrbanCanopy/ERF_UCMLayer.H:127:     * @param[in]     atm_t_star   ATM potential temp scale from SurfaceLayer [K]
Source/UrbanCanopy/ERF_UCMLayer.H:128:     * @param[in]     atm_q_star   ATM humidity scale from SurfaceLayer [kg/kg]
Source/UrbanCanopy/UCM_DEVELOPMENT.md:36:| 4 | 4.1 | is_urban mask enforcement (LSM + MOST bypass) | Wiring is_urban into LSM/MOST paths, mixed urban/non-urban domains | 🔲 PLANNED |
Source/UrbanCanopy/UCM_DEVELOPMENT.md:77:UCM MUST NOT call `SurfaceLayer::get_pblh(lev)` in any code path. All near-surface stability information comes from `u*`, `theta*`, `q*`, and the Obukhov length `L`, all reliably populated by MRF/YSU/MYNN2.5. This is a **hard rule** enforced by code review.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:81:Declare a per-level `ucm_is_urban` `iMultiFab` member on `ERF`, allocated on the UCM 2D slab at `anchor_level` (allocation actually happens Phase 1.2 when UCM grid is created; Phase 1.1 only has declaration). In Phase 1.1, no LSM/MOST bypass hooks are wired; that is Phase 4.1. Only field declaration and TODO comments are added.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:369:**Bug 8** – `[8c1cddb](https://github.com/hgopalan/ERF/commit/8c1cddb)` "Fix UCM advance call: SurfaceLayer accessors take lev arg and return MultiFab*"
Source/UrbanCanopy/UCM_DEVELOPMENT.md:370:- **Wrong:** Called `m_SurfaceLayer->get_u_star()[lev]` (missing `lev` argument)
Source/UrbanCanopy/UCM_DEVELOPMENT.md:371:- **Correct:** `m_SurfaceLayer->get_u_star(lev)` returns `MultiFab*` (single pointer, not a Vector)
Source/UrbanCanopy/UCM_DEVELOPMENT.md:372:- **Also wrong:** `m_SurfaceLayer->get_theta_star()` (does not exist)
Source/UrbanCanopy/UCM_DEVELOPMENT.md:374:- **Dereferencing:** All SurfaceLayer accessors return `MultiFab*`, so use `*m_SurfaceLayer->get_u_star(lev)` to pass by reference
Source/UrbanCanopy/UCM_DEVELOPMENT.md:398:4. **SurfaceLayer accessors:** All take `lev` argument and return `MultiFab*` (single pointer). Dereference with `*get_u_star(lev)` etc. Never `get_theta_star()` — use `get_t_star(lev)` instead.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:678:- Each facet: `H_facet = f_facet * H_base` where `H_base = -ρ*Cp*u*t` from MOST
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1398:Implements drag forces opposing horizontal wind at each ATM cell k > klo (MOST owns k=klo). Two components:
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1480:1. `if (k == klo_c) return;` inside wall/roof drag kernels — MOST owns k=klo momentum, UCM does NOT touch.
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1500:- ✅ No double-counting: MOST owns k=klo, drag skips k=klo (assertion in kernel).
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1546:  - MRF + MOST retained, with neutral surface forcing (`erf.most.surf_temp_flux = 0.0`).
Source/UrbanCanopy/UCM_DEVELOPMENT.md:1564:- Kanda lower boundary deviates from paper LES (neutral MOST used instead of slip lower BC) to preserve MRF `u_star` computation; canopy momentum-drag physics under test is unchanged.
Source/UrbanCanopy/ERF_UCMParams.H:28: * UCM MUST NOT call `SurfaceLayer::get_pblh(lev)`. All stability info comes from
Source/UrbanCanopy/ERF_UCMParams.H:32: * `ucm_is_urban` iMultiFab declared on ERF (allocated Phase 1.2). No LSM/MOST
Source/UrbanCanopy/ERF_UCMAllocate.cpp:608:    // Fill urban mask (Phase 1.2: all 1; Phase 4.1: heterogeneous LSM/MOST bypass)
Source/UrbanCanopy/ERF_UCMAllocate.cpp:989:               z0_a(i,j,0) = 0.1;   // MOST default
Source/UrbanCanopy/ERF_UCM.H:28: * | 4 | 4.1 | is_urban mask enforcement (LSM + MOST bypass) | 🔲 PLANNED |
Source/UrbanCanopy/ERF_UCM.H:61: * UCM MUST NOT call `SurfaceLayer::get_pblh(lev)`. All stability from u*, theta*, q*, L.
Source/UrbanCanopy/ERF_UCM.H:65: * LSM/MOST bypass hooks: Phase 4.1.
Source/UrbanCanopy/ERF_UCMAtmCoupling.H:288: * **MOST owns k=klo:** Drag is NOT applied at k=klo (surface layer); see early-exit guard.
Source/UrbanCanopy/ERF_UCMLayer.cpp:142:    // Extract u* from SurfaceLayer
Source/UrbanCanopy/ERF_UCMLayer.cpp:143:    fill_ucm_ustar_from_surface_layer(*forcing.u_star, atm_u_star, ucm_grid, lev);
Source/UrbanCanopy/ERF_UCMLayer.cpp:229:    // BUG HISTORY: Original Phase 2.3 code multiplied the plan-area MOST flux
Source/UrbanCanopy/ERF_UCMLayer.cpp:246:    // road/roof share a single MOST-driven t_star. AH is treated as already
Source/UrbanCanopy/ERF_UCMLayer.cpp:303:           // MOST bulk sensible flux -- already per unit plan area [W/m^2].
Source/UrbanCanopy/ERF_UCMBuildingLayoutReader.H:24: * - is_urban: urban/non-urban flag (0 or 1); controls LSM/MOST bypass (Phase 4.1)
Source/UrbanCanopy/ERF_UCMBuildingLayoutReader.H:48: * Phase 4.1: is_urban used to bypass LSM/MOST.
Source/UrbanCanopy/ERF_UCMBuildingLayoutReader.H:65:    int    is_urban;           ///< Urban mask [0 or 1]; 0 = LSM/MOST, 1 = UCM
Source/UrbanCanopy/ERF_UCMAllocate.H:121: *  - Non-urban cells: z0_ucm = 0.1 m (MOST default), d_disp_ucm = 0.0 m
Source/UrbanCanopy/ERF_UCMWindExtract.cpp:46:void fill_ucm_ustar_from_surface_layer(amrex::MultiFab& ucm_ustar,
```

</details>

<details>
<summary><code>grep -rn "get_pblh\|get_z0\|get_ustar" Source/UrbanCanopy/</code></summary>

```text
Source/UrbanCanopy/UCM_DEVELOPMENT.md:77:UCM MUST NOT call `SurfaceLayer::get_pblh(lev)` in any code path. All near-surface stability information comes from `u*`, `theta*`, `q*`, and the Obukhov length `L`, all reliably populated by MRF/YSU/MYNN2.5. This is a **hard rule** enforced by code review.
Source/UrbanCanopy/ERF_UCMParams.H:28: * UCM MUST NOT call `SurfaceLayer::get_pblh(lev)`. All stability info comes from
Source/UrbanCanopy/ERF_UCM.H:61: * UCM MUST NOT call `SurfaceLayer::get_pblh(lev)`. All stability from u*, theta*, q*, L.
```

</details>

## Section 3: Classification table

| File | Line number | Code snippet (1 line) | Category | Rationale | Fix hint |
|---|---:|---|---|---|---|
| `Source/ERF.cpp` | 1245 | `const int lev = m_ucm_params.anchor_level;` | CORRECT | Integration selects the working UCM level from anchor_level. | None |
| `Source/ERF.cpp` | 1265 | `create_ucm_grid(grids[lev], dmap[lev], geom[lev], m_ucm_params.grid_ratio, lev, ...)` | CORRECT | UCM grid is built from the ATM BoxArray/DM/Geometry at anchor_level. | None |
| `Source/ERF.cpp` | 1273 | `allocate_ucm_fields(*m_ucm_fields[lev], *m_ucm_grid[lev], m_ucm_params, lev);` | CORRECT | UCM field allocation is keyed off the anchor-level UCM grid. | None |
| `Source/ERF.cpp` | 1294 | `m_ucm_material_registry->load_and_broadcast(m_ucm_params.material_library_csv_path, lev, ...);` | CORRECT | CSV material load path threads the selected level explicitly. | None |
| `Source/ERF.cpp` | 1305 | `m_ucm_building_reader->read_and_broadcast(m_ucm_params.building_layout_csv_path, nx_ucm, ny_ucm, lev, ...);` | CORRECT | Building-layout ingest uses anchor-level UCM dimensions and passes lev explicitly. | None |
| `Source/ERF.cpp` | 1310 | `fill_ucm_fields_from_csv(*m_ucm_fields[lev], *m_ucm_grid[lev], *m_ucm_building_reader, *m_ucm_material_registry, ..., lev, ...);` | CORRECT | CSV scatter populates the anchor-level UCM field bundle. | None |
| `Source/ERF.cpp` | 1318 | `fill_ucm_fields_homogeneous(*m_ucm_fields[lev], m_ucm_params, lev);` | CORRECT | Homogeneous fallback is invoked on anchor_level, not hardwired 0. | None |
| `Source/ERF.cpp` | 1325 | `fill_ucm_z0_and_disp(*m_ucm_fields[lev], m_ucm_params, lev);` | CORRECT | Derived roughness/displacement are computed for the anchor-level field set. | None |
| `Source/ERF.cpp` | 1331 | `m_ucm_layer[lev] = std::make_unique<UCMLayer>(m_ucm_params, lev);` | CORRECT | Only the anchor-level UCMLayer is instantiated. | None |
| `Source/TimeIntegration/ERF_Advance.cpp` | 191 | `refine_atm_to_ucm(ustar_ucm, *m_SurfaceLayer->get_u_star(lev), gr, klo_atm);` | CORRECT | Surface-layer forcing is extracted from the current ERF level; UCM only exists at anchor_level. | None |
| `Source/TimeIntegration/ERF_Advance.cpp` | 196 | `m_ucm_layer[lev]->advance(..., *z_phys_cc[lev].get(), ..., lev);` | CORRECT | Wind/thermo forcing for UCM advance is taken from the same level indexed by lev. | None |
| `Source/TimeIntegration/ERF_Advance.cpp` | 357 | `m_ucm_grid[lev]->geom, Geom(lev),` | CORRECT | Morphology aggregation explicitly uses UCM + ATM geometries from the same selected level. | None |
| `Source/TimeIntegration/ERF_Advance.cpp` | 387 | `m_ucm_grid[lev]->geom, Geom(lev),` | CORRECT | Heat-flux coarsening uses ATM geometry from lev, which is the active anchor level. | None |
| `Source/TimeIntegration/ERF_Advance.cpp` | 415 | `m_ucm_grid[lev]->geom, Geom(lev),` | CORRECT | Road-flux coarsening stays on the current level geometry. | None |
| `Source/TimeIntegration/ERF_Advance.cpp` | 421 | `m_ucm_grid[lev]->geom, Geom(lev),` | CORRECT | Wall-flux coarsening stays on the current level geometry. | None |
| `Source/TimeIntegration/ERF_Advance.cpp` | 426 | `m_ucm_grid[lev]->geom, Geom(lev),` | CORRECT | Roof/AH coarsening stays on the current level geometry. | None |
| `Source/TimeIntegration/ERF_Advance.cpp` | 454 | `m_ucm_diagnostics[lev] = std::make_unique<UCMDiagnostics>(m_ucm_params, lev);` | CORRECT | Diagnostics writer is instantiated for the active UCM level only. | None |
| `Source/TimeIntegration/ERF_Advance.cpp` | 456 | `m_ucm_diagnostics[lev]->append(*m_ucm_fields[lev], iteration, time, ..., lev);` | CORRECT | Diagnostics append path samples UCM/ATM aggregate data for the same lev. | None |
| `Source/TimeIntegration/ERF_Advance.cpp` | 468 | `m_ucm_plotfile[lev] = std::make_unique<UCMPlotfile>(m_ucm_params, lev);` | CORRECT | UCM plot writer is allocated for the active UCM level. | None |
| `Source/TimeIntegration/ERF_Advance.cpp` | 470 | `m_ucm_plotfile[lev]->write(*m_ucm_fields[lev], *m_ucm_grid[lev], iteration, time, false, lev);` | CORRECT | UCM plotfile metadata comes from the anchor-level UCM grid. | None |
| `Source/TimeIntegration/ERF_Advance.cpp` | 492 | `Geom(lev),` | CORRECT | ATM-grid UCM plotfile uses Geometry from the current lev, not Geom(0). | None |
| `Source/TimeIntegration/ERF_TI_slow_rhs_pre.H` | 160 | `apply_ucm_tendency_to_cc_source(..., fine_geom, *m_ucm_is_urban_atm[level], ..., level);` | CORRECT | Heat/moisture source injection is called with the current level Geometry and level index. | None |
| `Source/TimeIntegration/ERF_TI_slow_rhs_pre.H` | 264 | `apply_ucm_momentum_drag_to_source(..., fine_geom, ..., level);` | CORRECT | Momentum drag write-back is called with the current level Geometry and level index. | None |
| `Source/UrbanCanopy/ERF_UCMLayer.cpp` | 33 | `if (lev != params.anchor_level) {` | CORRECT | Explicitly ties the layer instance to params.anchor_level. | None |
| `Source/UrbanCanopy/ERF_UCMParams.cpp` | 25 | `pp.query("anchor_level", anchor_level);` | CORRECT | Directly reads the configured anchor level from erf.ucm parameters. | None |
| `Source/UrbanCanopy/ERF_UCMPrerequisites.cpp` | 48 | `params.anchor_level >= 0,` | CORRECT | Validates the user-selected anchor level rather than assuming level 0. | None |
| `Source/UrbanCanopy/ERF_UCMPrerequisites.cpp` | 52 | `params.anchor_level <= finest_level,` | CORRECT | Checks requested anchor level against actual AMR hierarchy. | None |
| `Source/ERF.cpp` | 2587 | `m_ucm_params.read_from_parmparse(0);` | HARDCODED_LEVEL_0 | Parameter read path is explicitly hardwired to level 0 at startup. | Pass m_ucm_params.anchor_level after parsing, or delete the unused lev argument entirely. |
| `Source/UrbanCanopy/ERF_UCMPrerequisites.cpp` | 56 | `if (params.anchor_level > 0) {` | HARDCODED_LEVEL_0 | Startup gate rejects every anchor_level > 0, so higher-level UCM can never run. | Remove the Phase 1.1 guard and replace it with true multi-level prerequisite checks. |
| `Source/UrbanCanopy/ERF_UCMAllocate.H` | 47 | `int lev = 0);` | AMBIGUOUS | Allocation helper is safe today only because caller passes lev explicitly. | Remove default and require explicit lev. |
| `Source/UrbanCanopy/ERF_UCMAllocate.H` | 80 | `int lev = 0,` | AMBIGUOUS | CSV fill helper would silently fall back to 0 if invoked without lev. | Remove default and require explicit lev. |
| `Source/UrbanCanopy/ERF_UCMAllocate.H` | 113 | `int lev = 0);` | AMBIGUOUS | Homogeneous fill helper relies on caller discipline, not intrinsic anchor-level lookup. | Remove default and require explicit lev. |
| `Source/UrbanCanopy/ERF_UCMAllocate.H` | 135 | `int lev = 0);` | AMBIGUOUS | z0/displacement helper has a silent level-0 default even though level is threaded by caller. | Remove default and require explicit lev. |
| `Source/UrbanCanopy/ERF_UCMAllocate.H` | 166 | `int lev = 0);` | AMBIGUOUS | AH helper defaults to 0; current usage is explicit but API is still level-fragile. | Remove default and require explicit lev. |
| `Source/UrbanCanopy/ERF_UCMAtmCoupling.H` | 98 | `int lev = 0);` | AMBIGUOUS | Morphology aggregation depends on caller-supplied geometry/lev; default 0 is unsafe for future call sites. | Remove default and require explicit lev. |
| `Source/UrbanCanopy/ERF_UCMAtmCoupling.H` | 149 | `int lev = 0);` | AMBIGUOUS | Flux coarsening is caller-safe today, but the API still silently permits level-0 omission. | Remove default and require explicit lev. |
| `Source/UrbanCanopy/ERF_UCMAtmCoupling.H` | 271 | `int lev = 0);` | AMBIGUOUS | Heat/moisture injection API defaults to 0; correctness currently relies on ERF_TI_slow_rhs_pre.H passing level. | Remove default and require explicit lev. |
| `Source/UrbanCanopy/ERF_UCMAtmCoupling.H` | 334 | `int lev = 0);` | AMBIGUOUS | Momentum drag API defaults to 0; correctness currently relies on caller discipline. | Remove default and require explicit lev. |
| `Source/UrbanCanopy/ERF_UCMAtmCoupling.H` | 386 | `int lev = 0);` | AMBIGUOUS | Implicit-drag stub has no caller today, so the silent default remains latent risk. | Remove default before wiring the anelastic path. |
| `Source/UrbanCanopy/ERF_UCMAtmPlotfile.H` | 95 | `int lev = 0);` | AMBIGUOUS | ATM plotfile writer would silently target level 0 if a future caller omits lev. | Remove default and require explicit lev. |
| `Source/UrbanCanopy/ERF_UCMBuildingLayoutReader.H` | 133 | `int lev = 0, bool ucm_debug = false);` | AMBIGUOUS | CSV reader defaults lev to 0 although ERF init currently passes the anchor level. | Remove default and require explicit lev. |
| `Source/UrbanCanopy/ERF_UCMDiagnostics.H` | 60 | `UCMDiagnostics(const UCMParams& params, int lev = 0);` | AMBIGUOUS | Constructor silently defaults to level 0; current factory passes lev explicitly. | Remove default and require explicit lev. |
| `Source/UrbanCanopy/ERF_UCMDiagnostics.H` | 112 | `int lev = 0);` | AMBIGUOUS | Append path is safe today only because caller passes lev. | Remove default and require explicit lev. |
| `Source/UrbanCanopy/ERF_UCMGrid.H` | 87 | `int lev = 0,` | AMBIGUOUS | Default lev=0 would silently create a level-0 UCM grid for future omitted-arg callers. | Require explicit lev everywhere. |
| `Source/UrbanCanopy/ERF_UCMLayer.H` | 79 | `UCMLayer(const UCMParams& params, int lev = 0);` | AMBIGUOUS | Constructor defaults to level 0, though current construction passes anchor_level explicitly. | Remove default and require explicit lev. |
| `Source/UrbanCanopy/ERF_UCMLayer.H` | 163 | `int lev = 0);` | AMBIGUOUS | Advance API still silently permits omitted level selection. | Remove default and require explicit lev. |
| `Source/UrbanCanopy/ERF_UCMMaterialRegistry.H` | 125 | `void load_and_broadcast(const std::string& path, int lev = 0, bool ucm_debug = false);` | AMBIGUOUS | Material registry load helper carries a silent level-0 default. | Remove default and require explicit lev. |
| `Source/UrbanCanopy/ERF_UCMParams.H` | 222 | `void read_from_parmparse(int lev = 0);` | AMBIGUOUS | Default argument silently selects 0 if caller omits lev; current safety depends on caller behavior. | Remove the default or remove the lev parameter if unused. |
| `Source/UrbanCanopy/ERF_UCMPlotfile.H` | 46 | `UCMPlotfile(const UCMParams& params, int lev = 0);` | AMBIGUOUS | Plot writer constructor silently defaults to 0. | Remove default and require explicit lev. |
| `Source/UrbanCanopy/ERF_UCMPlotfile.H` | 80 | `int lev = 0);` | AMBIGUOUS | UCM plot write API stays safe only if caller keeps passing lev explicitly. | Remove default and require explicit lev. |
| `Source/UrbanCanopy/ERF_UCMPrerequisites.H` | 62 | `int lev = 0);` | AMBIGUOUS | Prerequisite API defaults to 0 even though current ERF init passes anchor_level explicitly. | Remove default and require explicit lev. |
| `Source/UrbanCanopy/ERF_UCMPrerequisites.H` | 97 | `int lev = 0);` | AMBIGUOUS | Grid/field post-check helper defaults to 0; safety depends on explicit caller argument. | Remove default and require explicit lev. |
| `Source/UrbanCanopy/ERF_UCMShadowing.H` | 74 | `int lev = 0,` | AMBIGUOUS | Shadowing helper is currently safe only because UCMLayer passes lev explicitly. | Remove default and require explicit lev. |
| `Source/UrbanCanopy/ERF_UCMWindExtract.H` | 116 | `int lev = 0);` | AMBIGUOUS | Surface-layer extraction helper carries a silent level-0 default. | Remove default and require explicit lev. |
| `Source/UrbanCanopy/ERF_UCMWindExtract.H` | 168 | `int lev = 0);` | AMBIGUOUS | Wind extraction helper remains safe only when callers thread lev explicitly. | Remove default and require explicit lev. |
| `Source/UrbanCanopy/ERF_UCMWindExtract.H` | 200 | `int lev = 0);` | AMBIGUOUS | Scalar extraction helper defaults to 0 although current callers pass lev. | Remove default and require explicit lev. |
| `Source/UrbanCanopy/ERF_UCMAllocate.cpp` | 25 | `const BoxArray& ba = ucm_grid.ba;` | N/A | Uses the prebuilt UCM grid object; AMR level was already chosen upstream. | None |
| `Source/UrbanCanopy/ERF_UCMAllocate.cpp` | 26 | `const DistributionMapping& dm = ucm_grid.dm;` | N/A | Uses the prebuilt UCM grid object; no independent level selection. | None |
| `Source/UrbanCanopy/ERF_UCMAtmPlotfile.cpp` | 89 | `Real dz = geom.CellSize(2);` | N/A | Reads vertical spacing from the Geometry object handed in by caller; no hardcoded AMR level. | None |
| `Source/UrbanCanopy/ERF_UCMAtmPlotfile.cpp` | 96 | `MultiFab slab_mf(slab_ba, f_urb_atm.DistributionMap(), 9, 0);` | N/A | Plotfile slab follows the already-provided ATM MultiFab layout; no extra level decision. | None |
| `Source/UrbanCanopy/ERF_UCMGrid.cpp` | 58 | `DistributionMapping dm_ucm = dm_atm;` | N/A | Pure pass-through of an already-selected DistributionMap; no new AMR level decision here. | None |
| `Source/UrbanCanopy/ERF_UCMPlotfile.cpp` | 67 | `amrex::MultiFab ucm_plot(fields.H_bldg->boxArray(), fields.H_bldg->DistributionMap(), UCMPlot_ncomp, 0);` | N/A | UCM plotfile packs already-allocated UCM fields; no cross-level dependency is introduced here. | None |
| `Source/UrbanCanopy/ERF_UCMWindExtract.cpp` | 70 | `const int klo = 0;` | N/A | This is the local slab vertical index inside already-refined per-level data, not an AMR-level selector. | None |
| `Source/UrbanCanopy/ERF_UCMWindExtract.cpp` | 122 | `const int klo = 0;` | N/A | This is the local slab vertical index inside already-refined per-level data, not an AMR-level selector. | None |

## Section 4: Ambiguous site traces

### UCMParams::read_from_parmparse default argument
- **Sites:** Source/UrbanCanopy/ERF_UCMParams.H:222
- **Function/context:** Parameter-read API advertises a default lev=0 even though the implementation ignores lev and the only current caller is ERF startup.
- **All callers:**
  - Source/ERF.cpp:2587 — m_ucm_params.read_from_parmparse(0)
- **Resolution:** Resolves to HARDCODED_LEVEL_0 today because the sole call site passes 0 explicitly.
- **Reclassification note:** Header declaration stays AMBIGUOUS; effective runtime behavior is HARDCODED_LEVEL_0 until the caller and/or API are changed.

### create_ucm_grid default argument
- **Sites:** Source/UrbanCanopy/ERF_UCMGrid.H:87
- **Function/context:** Grid factory exposes lev=0 default, but actual grid selection comes from caller-supplied ba/dm/geom.
- **All callers:**
  - Source/ERF.cpp:1265 — create_ucm_grid(grids[lev], dmap[lev], geom[lev], ..., lev, ...)
- **Resolution:** Resolves to CORRECT in current code because lev is first set from params.anchor_level and then passed explicitly.
- **Reclassification note:** Could be reclassified to CORRECT once the lev=0 default is removed.

### Allocation / fill helper defaults
- **Sites:** Source/UrbanCanopy/ERF_UCMAllocate.H:47; Source/UrbanCanopy/ERF_UCMAllocate.H:80; Source/UrbanCanopy/ERF_UCMAllocate.H:113; Source/UrbanCanopy/ERF_UCMAllocate.H:135; Source/UrbanCanopy/ERF_UCMAllocate.H:166
- **Function/context:** Five public helpers default lev to 0 even though they operate on already-selected UCM field/grid bundles.
- **All callers:**
  - Source/ERF.cpp:1273 — allocate_ucm_fields(..., lev)
  - Source/ERF.cpp:1310 — fill_ucm_fields_from_csv(..., lev, ...)
  - Source/ERF.cpp:1318 — fill_ucm_fields_homogeneous(..., lev)
  - Source/ERF.cpp:1325 — fill_ucm_z0_and_disp(..., lev)
  - Source/UrbanCanopy/ERF_UCMLayer.cpp:222 — compute_anthropogenic_heat(..., lev)
- **Resolution:** Resolves to CORRECT in present code because every caller passes lev explicitly and that lev traces back to anchor_level.
- **Reclassification note:** Could all become CORRECT after removing the defaults.

### Wind/scalar extraction helper defaults
- **Sites:** Source/UrbanCanopy/ERF_UCMWindExtract.H:116; Source/UrbanCanopy/ERF_UCMWindExtract.H:168; Source/UrbanCanopy/ERF_UCMWindExtract.H:200
- **Function/context:** Extraction helpers default lev=0, but the actual ATM data already comes in as level-specific slabs/MultiFabs.
- **All callers:**
  - Source/UrbanCanopy/ERF_UCMLayer.cpp:143 — fill_ucm_ustar_from_surface_layer(..., lev)
  - Source/UrbanCanopy/ERF_UCMLayer.cpp:146-148 — fill_ucm_wind_from_interpolation(..., lev)
  - Source/UrbanCanopy/ERF_UCMLayer.cpp:151 — fill_ucm_scalar_from_atm(..., 0, lev)
  - Source/UrbanCanopy/ERF_UCMLayer.cpp:155 — fill_ucm_scalar_from_atm(..., 0, lev)
- **Resolution:** Resolves to CORRECT in current code because UCMLayer threads lev explicitly from anchor-level construction.
- **Reclassification note:** Could be reclassified to CORRECT after deleting the defaults.

### ATM coupling API defaults
- **Sites:** Source/UrbanCanopy/ERF_UCMAtmCoupling.H:98; Source/UrbanCanopy/ERF_UCMAtmCoupling.H:149; Source/UrbanCanopy/ERF_UCMAtmCoupling.H:271; Source/UrbanCanopy/ERF_UCMAtmCoupling.H:334; Source/UrbanCanopy/ERF_UCMAtmCoupling.H:386
- **Function/context:** All public ATM-coupling helpers accept lev=0 by default, but actual level safety depends on ERF passing Geometry/MultiFabs for the intended level.
- **All callers:**
  - Source/TimeIntegration/ERF_Advance.cpp:347-359 — aggregate_ucm_morphology_to_atm(..., Geom(lev), ..., lev)
  - Source/TimeIntegration/ERF_Advance.cpp:385-427 — coarsen_ucm_flux_to_atm(..., Geom(lev), ..., lev)
  - Source/TimeIntegration/ERF_TI_slow_rhs_pre.H:160-183 — apply_ucm_tendency_to_cc_source(..., fine_geom, ..., level)
  - Source/TimeIntegration/ERF_TI_slow_rhs_pre.H:264-283 — apply_ucm_momentum_drag_to_source(..., fine_geom, ..., level)
  - No caller found for apply_ucm_implicit_drag_correction()
- **Resolution:** Aggregate/coarsen/heat/explicit-drag paths resolve to CORRECT today because callers pass lev/current geometry explicitly. The implicit-drag stub remains truly AMBIGUOUS because it has no caller yet.
- **Reclassification note:** All but the implicit stub could become CORRECT after deleting defaults; the implicit stub should stay AMBIGUOUS until wired and audited.

### Diagnostics / plot writer defaults
- **Sites:** Source/UrbanCanopy/ERF_UCMDiagnostics.H:60; Source/UrbanCanopy/ERF_UCMDiagnostics.H:112; Source/UrbanCanopy/ERF_UCMPlotfile.H:46; Source/UrbanCanopy/ERF_UCMPlotfile.H:80; Source/UrbanCanopy/ERF_UCMAtmPlotfile.H:95
- **Function/context:** Writer constructors and methods default lev=0, but runtime factories currently pass lev explicitly.
- **All callers:**
  - Source/TimeIntegration/ERF_Advance.cpp:454-462 — UCMDiagnostics ctor + append(..., lev)
  - Source/TimeIntegration/ERF_Advance.cpp:468-471 — UCMPlotfile ctor + write(..., lev)
  - Source/TimeIntegration/ERF_Advance.cpp:480-494 — UCMAtmPlotfile::write(..., Geom(lev), ..., lev)
- **Resolution:** Resolves to CORRECT in current code; all call sites use the active lev.
- **Reclassification note:** Could be reclassified to CORRECT after removing the defaults.

### Prerequisite / validation helper defaults
- **Sites:** Source/UrbanCanopy/ERF_UCMPrerequisites.H:62; Source/UrbanCanopy/ERF_UCMPrerequisites.H:97
- **Function/context:** Validation helpers default lev=0 though ERF init already computes anchor_level explicitly.
- **All callers:**
  - Source/ERF.cpp:1241-1242 — check_ucm_prerequisites(..., m_ucm_params.anchor_level)
  - Source/ERF.cpp:1328 — check_ucm_grid_and_fields(..., lev)
- **Resolution:** Resolves to CORRECT today because both callers pass explicit non-default levels.
- **Reclassification note:** Could be reclassified to CORRECT after removing the defaults.

### CSV/material helper defaults
- **Sites:** Source/UrbanCanopy/ERF_UCMBuildingLayoutReader.H:133; Source/UrbanCanopy/ERF_UCMMaterialRegistry.H:125
- **Function/context:** Reader/registry interfaces default lev=0 even though they are invoked from explicit anchor-level init code.
- **All callers:**
  - Source/ERF.cpp:1294-1295 — load_and_broadcast(..., lev, ...)
  - Source/ERF.cpp:1305-1307 — read_and_broadcast(..., lev, ...)
- **Resolution:** Resolves to CORRECT with current callers.
- **Reclassification note:** Could be reclassified to CORRECT after removing the defaults.

### Shadowing / UCMLayer defaults
- **Sites:** Source/UrbanCanopy/ERF_UCMShadowing.H:74; Source/UrbanCanopy/ERF_UCMLayer.H:79; Source/UrbanCanopy/ERF_UCMLayer.H:163
- **Function/context:** Shadowing and UCMLayer public APIs still default lev=0.
- **All callers:**
  - Source/ERF.cpp:1331 — UCMLayer(m_ucm_params, lev)
  - Source/TimeIntegration/ERF_Advance.cpp:196-202 — m_ucm_layer[lev]->advance(..., lev)
  - Source/UrbanCanopy/ERF_UCMLayer.cpp:202-204 — compute_sky_view_factors(..., lev, ...)
- **Resolution:** Resolves to CORRECT in current code because all live callers pass lev explicitly and UCMLayer constructor enforces lev == params.anchor_level.
- **Reclassification note:** Could be reclassified to CORRECT after removing the defaults.

## Section 5: Special focus findings

### 5.1 UCM state array allocation (is_urban, T_wall, T_road, T_roof, building layout)
- **File + line reference:** Source/ERF.cpp:1265-1273; Source/UrbanCanopy/ERF_UCMGrid.cpp:58; Source/UrbanCanopy/ERF_UCMAllocate.cpp:25-31, 85-121; Source/UrbanCanopy/ERF_UCMBuildingLayoutReader.cpp:247-255
- **Current behavior:** anchor_level
- **Category:** CORRECT
- **Risk assessment:** low
- **Fix hint for Phase 3.1b:** None for allocation path; keep upstream lev explicit.
- **Notes:** UCM fields allocate from ucm_grid.ba/dm. UCMGrid itself is created from grids[anchor_level]/dmap[anchor_level]/geom[anchor_level]. CSV row-count/range checks are against nx_ucm,ny_ucm derived from the same anchor-level UCM grid.

### 5.2 Heat injection into ATM (cc_source / RhoTheta)
- **File + line reference:** Source/TimeIntegration/ERF_TI_slow_rhs_pre.H:160-183; Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:307-309, 422, 489, 546, 596
- **Current behavior:** anchor_level
- **Category:** CORRECT
- **Risk assessment:** high
- **Fix hint for Phase 3.1b:** After removing the startup level-0 gate, keep explicit level plumbing and delete the default lev=0 on apply_ucm_tendency_to_cc_source().
- **Notes:** The source kernel writes whichever cc_source/fine_geom object the caller supplies. Current caller passes current level and only the anchor-level m_ucm_layer exists, so write-back lands on anchor_level.

### 5.3 Momentum drag write-back
- **File + line reference:** Source/TimeIntegration/ERF_TI_slow_rhs_pre.H:264-283; Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:833-834, 898-899
- **Current behavior:** anchor_level
- **Category:** CORRECT
- **Risk assessment:** high
- **Fix hint for Phase 3.1b:** Same as heat injection: keep explicit level threading, remove default lev=0, and wire the future implicit path before enabling anelastic drag.
- **Notes:** xmom_src/ymom_src are modified on the caller-supplied level. Current ERF caller passes level and only anchor_level has active UCM drag state.

### 5.4 Wind extraction from ATM
- **File + line reference:** Source/TimeIntegration/ERF_Advance.cpp:187-202; Source/UrbanCanopy/ERF_UCMWindExtract.cpp:52, 98-99, 136
- **Current behavior:** anchor_level
- **Category:** CORRECT
- **Risk assessment:** medium
- **Fix hint for Phase 3.1b:** Retain explicit lev in caller paths; optional cleanup is to document that local klo=0 is slab-local, not AMR-level 0.
- **Notes:** ERF first refines per-level ATM data onto UCM slabs using lev-specific state and SurfaceLayer accessors. Subsequent extraction kernels use local slab index 0 only inside those already-level-specific slabs.

### 5.5 Facet3D vertical binning (Δz)
- **File + line reference:** Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:278-279, 374-383, 448-450, 719-720, 782-791, 854-856; Source/TimeIntegration/ERF_TI_slow_rhs_pre.H:174, 275
- **Current behavior:** anchor_level
- **Category:** CORRECT
- **Risk assessment:** medium
- **Fix hint for Phase 3.1b:** No logic change required beyond preserving explicit caller-supplied geom_atm/z_phys_nd at anchor_level.
- **Notes:** Both heat and drag kernels derive dz from geom_atm or terrain metrics provided by the caller. The caller passes fine_geom/z_phys_nd[level], so the vertical binning uses the active level, not level 0.

### 5.6 UCM plotfile grid metadata
- **File + line reference:** Source/UrbanCanopy/ERF_UCMPlotfile.cpp:107-113; Source/UrbanCanopy/ERF_UCMAtmPlotfile.cpp:89-93, 133-138; Source/TimeIntegration/ERF_Advance.cpp:470-494
- **Current behavior:** anchor_level
- **Category:** CORRECT
- **Risk assessment:** low
- **Fix hint for Phase 3.1b:** Keep explicit lev in writer construction/calls; remove writer API defaults.
- **Notes:** UCM plotfiles use grid.geom from m_ucm_grid[lev]. ATM aggregate plotfiles rebuild slab_geom from Geom(lev). Neither writer uses Geom(0).

### 5.7 Building layout CSV mapping
- **File + line reference:** Source/ERF.cpp:1300-1307; Source/UrbanCanopy/ERF_UCMBuildingLayoutReader.cpp:247-255, 257-263; Source/UrbanCanopy/ERF_UCMAllocate.cpp:348-357, 394-405
- **Current behavior:** anchor_level
- **Category:** CORRECT
- **Risk assessment:** low
- **Fix hint for Phase 3.1b:** None beyond keeping lev explicit and removing helper defaults.
- **Notes:** CSV rows are validated against nx_ucm*ny_ucm from the anchor-level UCM grid, then scattered directly by UCM indices. The code explicitly rejects ATM-indexed CSVs.

### 5.8 MPI decomposition
- **File + line reference:** Source/UrbanCanopy/ERF_UCMGrid.cpp:58; Source/UrbanCanopy/ERF_UCMAllocate.cpp:25-26; Source/UrbanCanopy/ERF_UCMAtmPlotfile.cpp:96; Source/UrbanCanopy/ERF_UCMPlotfile.cpp:67
- **Current behavior:** anchor_level
- **Category:** CORRECT
- **Risk assessment:** low
- **Fix hint for Phase 3.1b:** None; just preserve the current pass-through from caller-selected DM.
- **Notes:** UCMGrid reuses dm_atm from the selected level, field allocation uses ucm_grid.dm, and plotfile staging MultiFabs inherit the corresponding DistributionMap from those already-selected data structures.

### 5.9 Diagnostics file (ucm_diag.dat)
- **File + line reference:** Source/UrbanCanopy/ERF_UCMDiagnostics.cpp:93-160; Source/TimeIntegration/ERF_Advance.cpp:452-462
- **Current behavior:** anchor_level
- **Category:** CORRECT
- **Risk assessment:** medium
- **Fix hint for Phase 3.1b:** No semantic fix needed; remove diagnostics API defaults to prevent future omitted-lev construction/calls.
- **Notes:** The file contains UCM-grid extrema plus ATM-grid aggregates passed in from m_ucm_*_atm[lev]. Current caller only allocates/appends diagnostics for the active UCM level.

### 5.10 Coarser-level treatment (levels < anchor_level)
- **File + line reference:** Source/ERF.cpp:1245-1331; Source/TimeIntegration/ERF_Advance.cpp:312-313; Source/TimeIntegration/ERF_TI_slow_rhs_pre.H:153-183, 255-283
- **Current behavior:** NO
- **Category:** CORRECT
- **Risk assessment:** low
- **Fix hint for Phase 3.1b:** None, but preserve the invariant that only m_ucm_layer[anchor_level] is non-null.
- **Notes:** No evidence of UCM feedback being applied below anchor_level. Integration guards on m_ucm_layer[lev] != nullptr, and only m_ucm_layer[anchor_level] is constructed.

## Section 6: Summary

- **Total sites inspected:** 64
- **CORRECT:** 27
- **HARDCODED_LEVEL_0:** 2
- **AMBIGUOUS:** 27
- **N/A:** 8
- **Files with most HARDCODED_LEVEL_0 hits:**
  - `Source/ERF.cpp` — 1
  - `Source/UrbanCanopy/ERF_UCMPrerequisites.cpp` — 1
- **High-risk sites:**
  - Source/ERF.cpp:2587 — m_ucm_params.read_from_parmparse(0)
  - Source/UrbanCanopy/ERF_UCMPrerequisites.cpp:56 — if (params.anchor_level > 0)
  - Source/UrbanCanopy/ERF_UCMAtmCoupling.H:271 — apply_ucm_tendency_to_cc_source(..., int lev = 0)
  - Source/UrbanCanopy/ERF_UCMAtmCoupling.H:334 — apply_ucm_momentum_drag_to_source(..., int lev = 0)
  - Source/UrbanCanopy/ERF_UCMParams.H:222 — read_from_parmparse(int lev = 0)

## Section 7: Phase 3.1b scope recommendations

1. **Ordering: which sites to fix first?**  
   Fix the true blockers first: `Source/ERF.cpp:2587` (parameter read hardwired to 0) and `Source/UrbanCanopy/ERF_UCMPrerequisites.cpp:56` (startup abort on `anchor_level > 0`). Next, remove all public `lev = 0` defaults on UCM APIs, starting with `ERF_UCMAtmCoupling.H`, `ERF_UCMWindExtract.H`, `ERF_UCMAllocate.H`, and `ERF_UCMParams.H`.
2. **Threading anchor_level: where are signature changes needed?**  
   Public UCM interfaces that currently default `lev=0` should require explicit level: parameter read/init, grid/allocation helpers, wind extraction, ATM coupling, diagnostics, plot writers, shadowing, and UCMLayer APIs. Caller-side plumbing already exists in most hot paths (`ERF.cpp`, `ERF_Advance.cpp`, `ERF_TI_slow_rhs_pre.H`).
3. **API surface: should anchor_level be encapsulated in helpers?**  
   Yes. Prefer a single integration-side `const int ucm_lev = m_ucm_params.anchor_level;` plus helper accessors for `grids[ucm_lev]`, `dmap[ucm_lev]`, `geom[ucm_lev]`, `z_phys_cc[ucm_lev]`, and UCM vectors. Inside UrbanCanopy, avoid recomputing policy; consume explicit level/Geometry/MultiFab inputs.
4. **Testing strategy: how to regression-test anchor_level=0 remains bit-identical?**  
   Re-run the existing UCM canonical tests with `anchor_level=0` before and after Phase 3.1b, compare plotfiles / diagnostics / conserved-field outputs bit-for-bit, and verify no call path changes behavior when `ucm_lev == 0`. Then add at least one targeted multi-level case with `anchor_level > 0` to prove level routing changes only the selected level and leaves coarser levels untouched.
