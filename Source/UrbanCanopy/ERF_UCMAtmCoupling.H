/**
 * @file ERF_UCMAtmCoupling.H
 * @brief Atmospheric coupling for UCM (coarsen and inject fluxes)
 *
 * Implements the one-way coupling pipeline between the UCM (on a refined 2D slab)
 * and the ATM (on a coarser 3D domain). Two main functions:
 *
 * 1. **coarsen_ucm_flux_to_atm**: Coarsen UCM fluxes from refined UCM grid to ATM grid.
 *    Uses `amrex::average_down` when grid_ratio > 1; `amrex::MultiFab::Copy` when ratio = 1.
 *
 * 2. **apply_ucm_tendency_to_cc_source**: Apply exponential-decay vertical injection
 *    (after Mandel 2011 / WRF-SFIRE fire_tendency pattern) to cc_source at every k.
 *
 * **RK-stage safety contract (Phase 2.3):**
 * `apply_ucm_tendency_to_cc_source` OWNS `cc_source[RhoTheta_comp]` (and
 * optionally `cc_source[RhoQ1_comp]`) for the duration of each RK stage.
 * It zeros those components at the top and writes with `=` (not `+=`), so
 * calling it once per RK stage after `make_sources` resets `cc_src` gives
 * the correct stage-invariant tendency.  The lagged fluxes (`m_ucm_H_atm`,
 * `m_ucm_LE_atm`) are computed ONCE per coarse step in `ERF::Advance` before
 * the MRI integrator starts, mirroring the ERF-Fire explicit-lag convention.
 * If any other physics module writes into those same slots per stage on the
 * same cells, this function must be moved to a pre-integrator path and
 * semantics changed back to `+=`.
 *
 * **Key Design Rules (Phase 1.4):**
 * - All ParallelFor kernels use: `[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept { ... }`
 * - Array4 access via `mf.array(mfi)` / `mf.const_array(mfi)` (by value), never `mf[mfi].array()` with `auto&`
 * - MultiFab copy via `amrex::MultiFab::Copy`, never `amrex::Copy`
 * - Every kernel guarded by `if (is_urban(...)==0) return;`
 * - Debug traces with `[UCM][1.4][function_name]` prefix on IO rank only
 * - Terrain-following coords: `z_phys_cc(i,j,k) - z_phys_cc(i,j,klo)` for height-above-surface
 *
 * References:
 *  - WRF Single-Layer Urban Canopy Model (Chen et al., 2011)
 *  - Mandel et al. (2011) "Coupled atmosphere-wildland fire modeling"
 *  - WRF Module: module_sf_urban.F / module_fire_tendency
 *  - Source/Dust/ERF_DustAtmCoupling.H (coarsen pattern)
 *  - Source/Fire/ERF_FireAtmCoupling.H (exponential injection pattern, ERF-Hazard branch)
 */

#ifndef ERF_UCM_ATM_COUPLING_H
#define ERF_UCM_ATM_COUPLING_H

#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Geometry.H>
#include "ERF_UCMParams.H"

/**
 * Refine an ATM-grid MultiFab onto the UCM 2D slab grid via piecewise-constant
 * injection. Handles both 3D ATM MFs (uses k = klo_atm plane) and 2D ATM slab MFs.
 */
void refine_atm_to_ucm(amrex::MultiFab&       Q_ucm_out,
                       const amrex::MultiFab& Q_atm_in,
                       int                    grid_ratio,
                       int                    klo_atm);

/**
 * @brief Compute per-ATM-cell morphology aggregates from UCM 2D slab (Phase 2.5).
 *
 * For each ATM cell (covering grid_ratio² UCM cells), aggregates:
 * - Urban fraction (count of urban cells / total)
 * - Mean building height (area-weighted average over urban cells)
 * - Std dev of building height (variance over urban cells)
 * - Mean plan-area density (building footprint ratio)
 * - Mean frontal-area density (wall area ratio)
 *
 * @param[out] f_urb_atm             Urban fraction per ATM cell [0,1]
 * @param[out] H_bldg_mean_atm       Mean building height per ATM cell [m]
 * @param[out] H_bldg_std_atm        Std dev of building height per ATM cell [m]
 * @param[out] lambda_p_atm          Mean plan-area density per ATM cell [-]
 * @param[out] lambda_f_atm          Mean frontal-area density per ATM cell [-]
 * @param[in]  H_bldg_ucm            Building height on UCM grid [m]
 * @param[in]  W_road_ucm            Road width on UCM grid [m]
 * @param[in]  plan_area_frac_ucm    Plan-area density on UCM grid
 * @param[in]  is_urban_ucm          Urban mask on UCM grid (1 = urban, 0 = non-urban)
 * @param[in]  geom_ucm              UCM geometry
 * @param[in]  geom_atm              ATM geometry
 * @param[in]  grid_ratio            Refinement factor (UCM vs ATM); typically 2 or 1
 * @param[in]  ucm_debug             Print debug info (default false)
 * @param[in]  lev                   AMR level (default 0)
 */
void aggregate_ucm_morphology_to_atm(
    amrex::MultiFab&           f_urb_atm,
    amrex::iMultiFab&          is_urban_atm,
    amrex::MultiFab&           H_bldg_mean_atm,
    amrex::MultiFab&           H_bldg_std_atm,
    amrex::MultiFab&           lambda_p_atm,
    amrex::MultiFab&           lambda_f_atm,
    amrex::iMultiFab&          is_tree_atm,
    amrex::MultiFab&           H_tree_atm,
    amrex::MultiFab&           H_crown_base_atm,
    amrex::MultiFab&           LAD_atm,
    amrex::MultiFab&           crown_area_frac_atm,
    amrex::MultiFab&           Cd_leaf_atm,
    const amrex::MultiFab&     H_bldg_ucm,
    const amrex::MultiFab&     W_road_ucm,
    const amrex::MultiFab&     plan_area_frac_ucm,
    const amrex::iMultiFab&    is_urban_ucm,
    const amrex::MultiFab&     H_tree_ucm,
    const amrex::MultiFab&     H_crown_base_ucm,
    const amrex::MultiFab&     LAD_ucm,
    const amrex::MultiFab&     crown_area_frac_ucm,
    const amrex::MultiFab&     Cd_leaf_ucm,
    const amrex::iMultiFab&    is_tree_ucm,
    const amrex::Geometry&     geom_ucm,
    const amrex::Geometry&     geom_atm,
    int                        grid_ratio,
    int                        lev,
    bool                       ucm_debug = false);

/**
 * @brief Coarsen UCM fluxes to ATM grid (convention B: area-averaged, no divide-by-f_urb).
 *
 * When grid_ratio == 1, performs a direct copy. When grid_ratio > 1, uses
 * `amrex::average_down` with refinement vector (grid_ratio, grid_ratio, 1) to
 * spatially average the finer UCM flux to the coarser ATM grid (matching Dust pattern).
 *
 * **Phase 2.5 Convention B (conservation-preserving, area-averaged):**
 * Instead of a divide-by-urban-fraction (which would require injection-side reweighting),
 * this function multiplies UCM flux by is_urban mask before coarsening, then returns
 * the area-averaged result. This ensures conservation: the sum of ATM-grid fluxes
 * equals the sum of urban UCM-grid fluxes, with no injection-side reweighting needed.
 *
 * Formula:
 * ```
 * Q_atm(I,J) = (1 / N_total) * sum_{UCM cells in ATM cell} (is_urban * Q_ucm)
 * ```
 * where N_total = grid_ratio^2 (total UCM cells, urban + non-urban).
 *
 * The injection kernel `apply_ucm_tendency_to_cc_source` reads Q_atm AS-IS with no
 * further multiplication by f_urb. This mirrors the ERF-Fire convention on ERF-Hazard.
 *
 * **Debug trace format (IO rank only):**
 * ```
 * [UCM][2.5][coarsen_ucm_flux_to_atm]
 *   grid_ratio={ratio}
 *   before: Q_ucm min={min_ucm} max={max_ucm}
 *   after:  Q_atm min={min_atm} max={max_atm} (area-averaged)
 * ```
 *
 * @param[out] Q_atm_out      Output coarsened flux on ATM grid (ncomp=1) [W/m²]
 * @param[in]  Q_ucm          Input flux on UCM grid (ncomp=1): H_sensible or LE_latent [W/m²]
 * @param[in]  is_urban_ucm   Urban mask on UCM grid (0 or 1); used for masking before coarsening
 * @param[in]  geom_ucm       UCM geometry
 * @param[in]  geom_atm       ATM geometry
 * @param[in]  grid_ratio     Refinement factor (UCM vs ATM); typically 2 or 1
 * @param[in]  lev            AMR level (default 0)
 *
 * @note All ParallelFor kernels respect is_urban mask and GPU-safe (noexcept).
 * @note Ghost cells are properly handled by average_down.
 * @note Verified by check_conservation.py to satisfy H_atm_max <= H_ucm_max.
 */
void coarsen_ucm_flux_to_atm(
    amrex::MultiFab&           Q_atm_out,
    const amrex::MultiFab&     Q_ucm,
    const amrex::iMultiFab&    is_urban_ucm,
    const amrex::Geometry&     geom_ucm,
    const amrex::Geometry&     geom_atm,
    int                        grid_ratio,
    int                        lev);

/**
 * @brief Apply exponential-decay vertical tendency to cc_source (Phase 2.6: morphology-aware).
 *
 * Implements the WRF-SFIRE exponential injection pattern (Mandel 2011) with Phase 2.6 enhancements:
 * - Per-cell e-folding depth (alpha_ij) driven by mean building height (H_bldg_mean_atm)
 * - Surface term: road flux → k=klo only, no vertical decay
 * - Exponential term: wall+roof+AH flux → distributed vertically with alpha_ij decay
 *
 * **Physics (per column at each k):**
 * 
 * Per-cell alpha calculation:
 * \f[
 *   \alpha_{ij} = \max(\alpha_{\min}, \min(\alpha_{\max}, \alpha_{\text{scale}} \cdot H_{\text{bldg\_mean}}(I,J)))
 * \f]
 *
 * Surface term (roads only at k=klo):
 * \f[
 *   \theta_{\text{tend,road}}(I,J,k_{\text{lo}}) = f_{\text{urb}} \cdot \frac{H_{\text{road\_atm}}}{\\rho(k_{\text{lo}}) \cdot C_{p,d} \cdot dz(k_{\text{lo}})}
 * \f]
 *
 * Exponential term (walls+roofs distributed over column):
 * \f[
 *   z_k = z_{\text{phys\_cc}}(i,j,k) - z_{\text{phys\_cc}}(i,j,k_{\text{lo}})
 * \f]
 * \f[
 *   \theta_{\text{tend,exp}}(I,J,k) = f_{\text{urb}} \cdot \frac{H_{\text{wallroof\_atm}}}{\\rho(k) \cdot C_{p,d} \cdot \alpha_{ij}} 
 *   \cdot \exp\left(-\frac{z_k}{\alpha_{ij}}\right)
 * \f]
 *
 * Total tendency: `theta_tend = theta_tend_road + theta_tend_exp` (both added to cc_source).
 *
 * Non-urban cells (`f_urb < 0.01`): both fluxes are zero (safe).
 *
 * **Debug trace format (IO rank only, once per call):**
 * ```
 * [UCM][2.6][apply_ucm_tendency_to_cc_source]
 *   Morphology-aware per-cell alpha_ij: min={alpha_min_val} max={alpha_max_val} [m]
 *   Surface term contribution: sum={sum_road} over N_cells cells [K*kg/m^3/s]
 *   Exponential term contribution: sum={sum_exp} over N_cells cells [K*kg/m^3/s]
 *   [UCM][2.6] injection: surface term + morphology-aware exponential, alpha in [{alpha_min}, {alpha_max}]
 * ```
 *
 * If use_morphology_injection == false, falls back to uniform alpha_ucm (Phase 2.5 behavior).
 * If atm_feedback == 0.0, function prints a one-time warning and returns without modification.
 *
 * @param[in,out] cc_source              Cell-centered source term MultiFab (modified)
 * @param[in]     H_atm                  Total sensible heat flux on ATM grid [W/m²] (Phase 2.5 lumped, unused if use_morphology_injection)
 * @param[in]     H_road_atm             Road flux on ATM grid [W/m²] (Phase 2.6)
 * @param[in]     H_wallroof_atm         Wall+roof+AH flux on ATM grid [W/m²] (Phase 2.6)
 * @param[in]     H_bldg_mean_atm        Mean building height per ATM cell [m] (Phase 2.5 aggregates)
 * @param[in]     LE_atm                 Latent heat flux on ATM grid [W/m²]; may be nullptr
 * @param[in]     z_phys_cc              Physical z at cell centers on ATM grid [m]
 * @param[in]     S_old                  State (for density at each k); component 0 = RhoTheta
 * @param[in]     geom_atm               ATM geometry (to get dz)
 * @param[in]     is_urban_atm           is_urban mask on ATM grid (coarsened from UCM)
 * @param[in]     alpha_scale            Per-cell alpha multiplier: alpha_ij = alpha_scale * H_bldg_mean(I,J) (Phase 2.6)
 * @param[in]     alpha_min              Minimum e-folding depth [m] (Phase 2.6)
 * @param[in]     alpha_max              Maximum e-folding depth [m] (Phase 2.6)
 * @param[in]     use_morphology_injection  Enable Phase 2.6 split injection; false = fallback to uniform alpha_ucm (Phase 2.5)
 * @param[in]     alpha_ucm_fallback     Uniform e-folding depth [m] for fallback mode (Phase 2.5 backward compat)
 * @param[in]     feedback               Coupling strength [0.0, 1.0]; 0=off, 1=full
 * @param[in]     has_moisture           Whether moisture component exists
 * @param[in]     ucm_debug              Print per-step debug info (default false)
 * @param[in]     lev                    AMR level (default 0)
 *
 * @note Internally, uses `Cp_d = 1004.64 J/kg/K` for sensible heat; `L_V = 2.501e6 J/kg` for latent.
 * @note Handles terrain via z_phys_cc; flat terrain is special case where z_phys_cc(i,j,k) = k * dz.
 * @note All ParallelFor kernels respect is_urban mask and use correct GPU-safe signature.
 * @note LE_atm pointer may be nullptr; function must handle gracefully.
 * @note RK-stage safety: OWNS cc_source[RhoTheta_comp]. Zeros at entry, always uses `+=`.
 *
 * @param cc_source         Coarse-grid source term to be populated [output]
 * @param H_atm             Phase 2.5 lumped sensible flux (unused if use_facet3d_injection or use_morphology_injection)
 * @param H_road_atm        Phase 2.6+: road sensible flux [W/m²]
 * @param H_wall_atm        Phase 2.7+: wall sensible flux [W/m²]
 * @param H_roof_atm        Phase 2.7+: roof sensible flux including anthropogenic heat [W/m²]
 * @param H_bldg_mean_atm   Mean building height per ATM cell [m]
 * @param H_bldg_std_atm    Std dev of building height per ATM cell [m] (for continuous Gaussian mode)
 * @param lambda_p_atm      Plan-area density per ATM cell [-]
 * @param lambda_f_atm      Frontal-area density per ATM cell [-]
 * @param LE_atm            Latent heat flux (optional)
 * @param z_phys_nd         Physical z at nodes (terrain-following; nullptr for flat)
 * @param z_phys_cc         Physical z at cell centers (for reference)
 * @param S_old             Current state (for density extraction)
 * @param geom_atm          Coarse geometry
 * @param is_urban_atm      Urban mask on ATM grid
 * @param use_facet3d_injection       Phase 2.7: enable facet-3D geometric injection (default true)
 * @param use_gaussian_height_distribution  Phase 2.7: use continuous Gaussian mode (default false)
 * @param height_std_threshold_m      Phase 2.7: below this, Gaussian falls back to sharp [m]
 * @param feedback          Coupling strength [0, 1]
 * @param has_moisture      Whether moisture variables are present
 * @param ucm_debug         Enable debug output
 * @param lev               AMR level (default 0)
 *
 * @cite Martilli, Clappier & Rotach (2002), Boundary-Layer Meteorology 104:261–304
 * @cite Coceal & Belcher (2004), QJRMS 130:1349–1372
 */
void apply_ucm_tendency_to_cc_source(
    amrex::MultiFab&       cc_source,
    const amrex::MultiFab& H_atm,
    const amrex::MultiFab& H_road_atm,
    const amrex::MultiFab& H_wall_atm,
    const amrex::MultiFab& H_roof_atm,
    const amrex::MultiFab& H_bldg_mean_atm,
    const amrex::MultiFab& H_bldg_std_atm,
    const amrex::MultiFab& lambda_p_atm,
    const amrex::MultiFab& lambda_f_atm,
    const amrex::MultiFab* LE_atm,
    const amrex::MultiFab* z_phys_nd,
    const amrex::MultiFab& z_phys_cc,
    const amrex::MultiFab& S_old,
    const amrex::Geometry& geom_atm,
    const amrex::iMultiFab& is_urban_atm,
    const amrex::MultiFab* f_urb_atm,
    int                    interface_mode,
    bool                   use_facet3d_injection,
    bool                   use_gaussian_height_distribution,
    amrex::Real            height_std_threshold_m,
    amrex::Real            feedback_heat,           ///< Heat feedback strength [0,1] [Phase 2.11]
    amrex::Real            feedback_moisture,       ///< Moisture feedback strength [0,1] [Phase 2.11]
    bool                   has_moisture,
    int                    lev,
    bool                   ucm_debug = false);

/**
 * @brief Apply BEP momentum drag to xmom_src and ymom_src (Phase 2.8 compressible mode)
 *
 * Implements wall and roof momentum drag following Martilli et al. (2002) Section 4.
 * For each urban ATM cell k > klo, computes:
 *
 * Wall drag:
 * F_x_wall(k) = -f_urb * s_wall(k) * Cd_wall * |U_h| * u(k)
 * F_y_wall(k) = -f_urb * s_wall(k) * Cd_wall * |U_h| * v(k)
 * where s_wall(k) = 2 * lambda_f * wall_fraction(k) / H_bldg_mean
 *
 * Roof drag (at k_roof only):
 * F_x_roof(k_roof) = -f_urb * lambda_p * Cd_roof * |U_h| * u(k_roof) / dz(k_roof)
 * F_y_roof(k_roof) = -f_urb * lambda_p * Cd_roof * |U_h| * v(k_roof) / dz(k_roof)
 *
 * **MOST owns k=klo:** Drag is NOT applied at k=klo (surface layer); see early-exit guard.
 *
 * @param[in,out] xmom_src              x-momentum source [output, modified]
 * @param[in,out] ymom_src              y-momentum source [output, modified]
 * @param[in]     S_cons                Conserved variables (rho, rhotheta, etc) at current stage [kg/m^3, etc]
 * @param[in]     S_xmom                x-momentum (rho*u) at current stage
 * @param[in]     S_ymom                y-momentum (rho*v) at current stage
 * @param[in]     H_bldg_mean_atm       Mean building height per ATM cell [m]
 * @param[in]     H_bldg_std_atm        Std dev of building height per ATM cell [m]
 * @param[in]     lambda_p_atm          Plan-area density per ATM cell [-]
 * @param[in]     lambda_f_atm          Frontal-area density per ATM cell [-]
 * @param[in]     z_phys_nd             Physical z at nodes (terrain-following; nullptr for flat)
 * @param[in]     is_urban_atm          Urban mask on ATM grid (coarsened from UCM)
 * @param[in]     geom_atm              Coarse geometry
 * @param[in]     drag_mode             Resolved drag mode (Off/Explicit/Implicit)
 * @param[in]     Cd_wall               Wall drag coefficient (default 0.4 per Martilli 2002)
 * @param[in]     Cd_roof               Roof drag coefficient (default 0.15)
 * @param[in]     feedback              Coupling strength [0, 1]
 * @param[in]     use_gaussian_height_distribution  Phase 2.7: use continuous Gaussian mode
 * @param[in]     height_std_threshold_m Phase 2.7: below this, Gaussian falls back to sharp [m]
 * @param[in]     ucm_debug             Enable debug output
 * @param[in]     lev                   AMR level (default 0)
 *
 * @cite Martilli, Clappier & Rotach (2002), Boundary-Layer Meteorology 104:261–304, Section 4
 */
 
void apply_ucm_momentum_drag_to_source(
    amrex::MultiFab&       xmom_src,
    amrex::MultiFab&       ymom_src,
    const amrex::MultiFab& S_cons,
    const amrex::MultiFab& S_xmom,
    const amrex::MultiFab& S_ymom,
    const amrex::MultiFab& H_bldg_mean_atm,
    const amrex::MultiFab& H_bldg_std_atm,
    const amrex::MultiFab& lambda_p_atm,
    const amrex::MultiFab& lambda_f_atm,
    const amrex::MultiFab* z_phys_nd,
    const amrex::iMultiFab& is_urban_atm,
    const amrex::MultiFab* f_urb_atm,
    int                    interface_mode,
    const amrex::Geometry& geom_atm,
    WallDragMode           drag_mode,
    amrex::Real            Cd_wall,
    amrex::Real            Cd_roof,
    amrex::Real            feedback_momentum,     ///< Momentum drag feedback strength [0,1] [Phase 2.11]
    bool                   use_gaussian_height_distribution,
    amrex::Real            height_std_threshold_m,
    int                    lev,
    bool                   ucm_debug = false);

/**
 * @brief Apply post-projection multiplicative momentum drag correction (Phase 2.8 anelastic stub)
 *
 * **STUB FOR PHASE 2.8b VALIDATION:**
 * Implements unconditionally stable post-projection momentum decay as used in DALES.
 * After the anelastic projection enforces divergence-free flow, apply:
 *
 *   u^(n+1) ← u^(n+1) / (1 + dt * f_urb * s_wall * Cd_wall * |U^n|)
 *   v^(n+1) ← v^(n+1) / (1 + dt * f_urb * s_wall * Cd_wall * |U^n|)
 *
 * This path is code-complete but NOT extensively tested in this PR.
 * Validation deferred to Phase 2.8b (future PR with anelastic canonical test).
 *
 * @param[in,out] S_new                   Updated state after projection [modified]
 * @param[in]     H_bldg_mean_atm         Mean building height per ATM cell [m]
 * @param[in]     H_bldg_std_atm          Std dev of building height per ATM cell [m]
 * @param[in]     lambda_p_atm            Plan-area density per ATM cell [-]
 * @param[in]     lambda_f_atm            Frontal-area density per ATM cell [-]
 * @param[in]     z_phys_nd               Physical z at nodes (terrain-following; nullptr for flat)
 * @param[in]     is_urban_atm            Urban mask on ATM grid
 * @param[in]     geom_atm                Coarse geometry
 * @param[in]     Cd_wall                 Wall drag coefficient
 * @param[in]     Cd_roof                 Roof drag coefficient
 * @param[in]     dt                      Time step [s]
 * @param[in]     feedback                Coupling strength [0, 1]
 * @param[in]     use_gaussian_height_distribution  Phase 2.7: use continuous Gaussian mode
 * @param[in]     height_std_threshold_m  Phase 2.7: below this, Gaussian falls back to sharp
 * @param[in]     ucm_debug               Enable debug output
 * @param[in]     lev                     AMR level (default 0)
 *
 * @note Hook location: called after anelastic NodalProjectionSolve completes.
 * @note Slightly violates divergence-free (~O(dt·Cd·|U|·s_wall)) but next projection cleans up.
 * @cite Heus & Jonker (2008), DALES post-projection drag approach
 */
void apply_ucm_implicit_drag_correction(
    amrex::MultiFab&       S_new,
    const amrex::MultiFab& H_bldg_mean_atm,
    const amrex::MultiFab& H_bldg_std_atm,
    const amrex::MultiFab& lambda_p_atm,
    const amrex::MultiFab& lambda_f_atm,
    const amrex::MultiFab* z_phys_nd,
    const amrex::iMultiFab& is_urban_atm,
    const amrex::Geometry& geom_atm,
    amrex::Real            Cd_wall,
    amrex::Real            Cd_roof,
    amrex::Real            dt,
    amrex::Real            feedback,
    bool                   use_gaussian_height_distribution,
    amrex::Real            height_std_threshold_m,
    int                    lev,
    bool                   ucm_debug = false);

#endif  // ERF_UCM_ATM_COUPLING_H
