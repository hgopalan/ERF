#ifndef ERF_LEVELSET_ADVECTION_H
#define ERF_LEVELSET_ADVECTION_H

#include <AMReX_MultiFab.H>
#include <ERF_FireGrid.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <cmath>
#include <ERF_NumericalSchemes.H>

/**
 * @file ERF_LevelSetAdvection.H
 * @brief Level-set advection using Godunov upwinding and SSP-RK3
 *
 * Ported from wildfire_levelset/src/advection.H
 * Solves ∂φ/∂t = −R(x,y) * (|∇φ| − ε Δφ) via:
 *   - First-order Godunov upwind |grad phi| for spatial discretization
 *   - 3-stage Strong Stability Preserving RK3 for time integration
 */

namespace fire_levelset {

/**
 * @struct Scratch
 * @brief Stage fields of the SSP-RK3 advection and of the reinitialisation,
 *        allocated once on the fire grid instead of on every subcycle
 *
 * Every advection call used to allocate five MultiFabs (three with three
 * ghost cells) and ParallelCopy between them, and every reinitialisation
 * four more, up to a thousand times per atmospheric step. FireLayer owns one
 * Scratch per fire grid and hands it to the drivers; a caller that passes
 * none (the unit tests) gets a local one.
 */
struct Scratch
{
    amrex::MultiFab phi_n, phi_star, phi_hat, rhs, R_dir;   ///< advection stages
    amrex::MultiFab phi_in, S, phi0, stage, stage2;          ///< reinitialisation stages

    void define (const amrex::BoxArray& ba, const amrex::DistributionMapping& dm)
    {
        const int ng = 3;
        phi_n.define(ba, dm, 1, ng);
        phi_star.define(ba, dm, 1, ng);
        phi_hat.define(ba, dm, 1, ng);
        rhs.define(ba, dm, 1, 0);
        R_dir.define(ba, dm, 1, 0);
        phi_in.define(ba, dm, 1, ng);
        S.define(ba, dm, 1, ng);
        phi0.define(ba, dm, 1, ng);
        stage.define(ba, dm, 1, ng);
        stage2.define(ba, dm, 1, ng);
    }
    bool defined () const { return phi_n.ok(); }

    /// The scratch to use for phi: the given one (defined on phi's layout), or local
    static Scratch& select (Scratch* given, Scratch& local, const amrex::MultiFab& phi)
    {
        Scratch& sc = given ? *given : local;
        if (!sc.defined()) { sc.define(phi.boxArray(), phi.DistributionMap()); }
        // the kernels index the stages by phi's MFIter (MultiFab::Copy's own assert)
        AMREX_ALWAYS_ASSERT(sc.phi_n.boxArray() == phi.boxArray()
                            && sc.phi_n.DistributionMap() == phi.DistributionMap());
        return sc;
    }
};

/**
 * @brief Arrival time of a cell the front crossed within a substep
 *
 * phi_old > 0 >= phi_new over a substep of length dt from t0: the front passed
 * the cell centre at t0 + dt phi_old / (phi_old - phi_new), the linear
 * interpolation of phi in time, clamped to the substep. Stamping the start of
 * the substep instead (the practice until 2026-10) dated every cell up to one
 * substep early, a mean bias of -dt/2 = -0.2 h/R at the default CFL, of the
 * size of the arrival-time errors the Verification cases measure.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real levelset_crossing_time (amrex::Real t0, amrex::Real dt,
                                    amrex::Real phi_old, amrex::Real phi_new) noexcept
{
    const amrex::Real drop = phi_old - phi_new;
    // divide by the floored drop before the select (a cell that was already
    // burned has drop ~ 0, and the unselected x/0 would be speculated)
    const amrex::Real frac = phi_old / amrex::max(drop, amrex::Real(1.0e-30));
    const amrex::Real f = (drop > amrex::Real(1.0e-30) && phi_old > amrex::Real(0.0))
                        ? amrex::min(amrex::max(frac, amrex::Real(0.0)), amrex::Real(1.0))
                        : amrex::Real(0.0);
    return t0 + f * dt;
}

/**
 * @brief Stamp the arrival time of the cells that burned during a substep
 *
 * @param[in,out] arrival  -1 until a cell burns
 * @param[in]     phi_old  Level set at the start of the substep
 * @param[in]     phi_new  Level set at its end
 * @param[in]     t0, dt   Start and length of the substep
 */
inline void stamp_arrival_time (amrex::MultiFab& arrival,
                                const amrex::MultiFab& phi_old,
                                const amrex::MultiFab& phi_new,
                                amrex::Real t0, amrex::Real dt)
{
    for (amrex::MFIter mfi(arrival, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto at = arrival.array(mfi);
        auto po = phi_old.const_array(mfi);
        auto pn = phi_new.const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (pn(i, j, k) < amrex::Real(0.0) && at(i, j, k) < amrex::Real(0.0)) {
                at(i, j, k) = levelset_crossing_time(t0, dt, po(i, j, k), pn(i, j, k));
            }
        });
    }
}

/**
 * @brief Compute level-set RHS: ∂φ/∂t = −R * (|∇φ| − ε Δφ)
 *
 * Uses the Godunov upwind norm for |∇φ| and central differences for the
 * Laplacian. The one-sided derivatives are first order, or HJ-WENO5-Z with
 * a WENO `grad_scheme`, which reads all three ghost cells of
 * the stage fields.
 *
 * @param[out] rhs         RHS field (modified in-place)
 * @param[in]  phi         Level-set field with ghosts
 * @param[in]  R_mf        Rate-of-spread field
 * @param[in]  dx, dy      Cell sizes
 * @param[in]  eps_visc    Artificial viscosity coefficient away from the front
 *                         (grad_scheme carries the near-front value, if any)
 */
inline void compute_levelset_rhs(amrex::MultiFab& rhs,
                                  const amrex::MultiFab& phi,
                                  const amrex::MultiFab& R_mf,
                                  amrex::Real dx, amrex::Real dy,
                                  amrex::Real eps_visc,
                                  const amrex::MultiFab* slopes = nullptr,
                                  const amrex::MultiFab* mask = nullptr,
                                  bool wall_extrap = false,
                                  const LevelSetGradient& grad_scheme = LevelSetGradient{})
{
    using namespace amrex;

    const Real dx2i = 1.0 / (dx * dx);
    const Real dy2i = 1.0 / (dy * dy);

    const bool use_terrain = (slopes != nullptr);

    for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto rhs_arr = rhs.array(mfi);
        auto phi_arr = phi.const_array(mfi);
        auto R_arr = R_mf.const_array(mfi);
        const WallStencil wall = make_wall_stencil(mask, wall_extrap, mfi);

        Array4<const Real> slope_arr;
        if (use_terrain) slope_arr = slopes->const_array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
            int i = iv[0], j = iv[1], k = iv[2];

            // Compute |∇φ|, projected onto the terrain surface when slopes are
            // supplied so that R is a spread rate along the ground rather than
            // in map view.
            const Real sx = use_terrain ? slope_arr(i, j, k, 0) : Real(0.0);
            const Real sy = use_terrain ? slope_arr(i, j, k, 1) : Real(0.0);
            Real norm_grad = fire_levelset::godunov_norm_grad_phi(phi_arr, dx, dy,
                                                                  i, j, k, sx, sy, wall,
                                                                  grad_scheme);

            // Compute Δφ = ∂²φ/∂x² + ∂²φ/∂y², through the same wall stencil
            Real laplacian = (wall.at(phi_arr, i, j, k, 1, 0) - 2.0*phi_arr(i, j, k)
                              + wall.at(phi_arr, i, j, k, -1, 0)) * dx2i
                           + (wall.at(phi_arr, i, j, k, 0, 1) - 2.0*phi_arr(i, j, k)
                              + wall.at(phi_arr, i, j, k, 0, -1)) * dy2i;

            // RHS: -R * (|∇φ| - ε Δφ), with ε the near-front value inside
            // the viscosity band when two values are set
            Real R_val = R_arr(i, j, k);
            const Real eps = grad_scheme.eps_at(phi_arr(i, j, k), eps_visc);
            rhs_arr(i, j, k) = -R_val * (norm_grad - eps * laplacian);
        });
    }
}

/**
 * @brief Compute level-set RHS with the wind/slope advective term upwinded
 * SEPARATELY from the isotropic term, instead of folding both into one
 * scalar R(n) multiplied by a single Godunov |grad phi| (compute_levelset_rhs
 * above).
 *
 * Only valid for erf.fire.directional_wind_coupling = "advective" with
 * directional_shape = "projection" and ros_model = "rothermel", where the
 * physical R(n)|grad phi| reduces exactly to a genuine two-term Hamiltonian:
 *
 *   phi_t = -R0*|grad phi| - max(Vw.grad phi, 0) - max(Vs.grad phi, 0)
 *
 * with Vw = R0*phi_w(|U|)/|U| * U (wind) and Vs = R0*phi_s(|slope|)/|slope| *
 * slope (terrain), each clamped to its own max(.,0) exactly as
 * rothermel_ros_cell's phi_w/phi_s cosine clamps do. R0 and Vw/Vs are known a
 * priori from the wind/slope fields (independent of phi), so each advective
 * term is upwinded by the SIGN OF ITS OWN V component-by-component -- ordinary
 * linear-advection upwind, not a front-normal-derived rule. See
 * ERF_DirectionalRos.H's advect_levelset_directional_rk3_split() and the
 * erf.fire.directional_split_hamiltonian doc comment in ERF_FireParams.H for
 * the rationale.
 *
 * @param[out] rhs      RHS field (modified in-place)
 * @param[in]  phi      Level-set field with ghosts
 * @param[in]  VwVs     5-component field: (Vwx, Vwy, Vsx, Vsy) [m/s] and the
 *                      cell's isotropic (no-wind, no-slope) rate R0 [m/s],
 *                      built once from wind/slope/fuel (independent of phi)
 * @param[in]  dx, dy   Cell sizes
 * @param[in]  eps_visc Artificial viscosity coefficient
 */
inline void compute_levelset_rhs_split_advective(amrex::MultiFab& rhs,
                                                  const amrex::MultiFab& phi,
                                                  const amrex::MultiFab& VwVs,
                                                  amrex::Real dx, amrex::Real dy,
                                                  amrex::Real eps_visc,
                                                  const amrex::MultiFab* slopes = nullptr,
                                                  const amrex::MultiFab* mask = nullptr,
                                                  bool wall_extrap = false,
                                                  const LevelSetGradient& grad_scheme = LevelSetGradient{})
{
    using namespace amrex;

    const Real dx2i = 1.0 / (dx * dx);
    const Real dy2i = 1.0 / (dy * dy);
    const bool use_terrain = (slopes != nullptr);

    for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto rhs_arr = rhs.array(mfi);
        auto phi_arr = phi.const_array(mfi);
        auto v_arr = VwVs.const_array(mfi);
        const WallStencil wall = make_wall_stencil(mask, wall_extrap, mfi);

        Array4<const Real> slope_arr;
        if (use_terrain) slope_arr = slopes->const_array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
            int i = iv[0], j = iv[1], k = iv[2];

            const Real sx = use_terrain ? slope_arr(i, j, k, 0) : Real(0.0);
            const Real sy = use_terrain ? slope_arr(i, j, k, 1) : Real(0.0);
            Real norm_grad = fire_levelset::godunov_norm_grad_phi(phi_arr, dx, dy,
                                                                  i, j, k, sx, sy, wall,
                                                                  grad_scheme);

            Real laplacian = (wall.at(phi_arr, i, j, k, 1, 0) - 2.0*phi_arr(i, j, k)
                              + wall.at(phi_arr, i, j, k, -1, 0)) * dx2i
                           + (wall.at(phi_arr, i, j, k, 0, 1) - 2.0*phi_arr(i, j, k)
                              + wall.at(phi_arr, i, j, k, 0, -1)) * dy2i;
            const Real eps = grad_scheme.eps_at(phi_arr(i, j, k), eps_visc);

            // Each advective term upwinded by the sign of its OWN velocity
            // component, independent of the front normal -- ordinary
            // linear-advection upwind (one_sided_derivatives returns D- in
            // dxm/dym, D+ in dxp/dyp; for phi_t + a*phi_x = 0 with a > 0 the
            // correct upwind derivative is the backward/left-biased one, D-).
            Real dxm, dxp, dym, dyp;
            fire_levelset::one_sided_derivatives(phi_arr, dx, i, j, k, 1, 0, grad_scheme, wall, dxm, dxp);
            fire_levelset::one_sided_derivatives(phi_arr, dy, i, j, k, 0, 1, grad_scheme, wall, dym, dyp);

            const Real Vwx = v_arr(i, j, k, 0), Vwy = v_arr(i, j, k, 1);
            const Real Vsx = v_arr(i, j, k, 2), Vsy = v_arr(i, j, k, 3);

            const Real phix_w = (Vwx > Real(0.0)) ? dxm : dxp;
            const Real phiy_w = (Vwy > Real(0.0)) ? dym : dyp;
            const Real adv_w = amrex::max(Vwx*phix_w + Vwy*phiy_w, Real(0.0));

            const Real phix_s = (Vsx > Real(0.0)) ? dxm : dxp;
            const Real phiy_s = (Vsy > Real(0.0)) ? dym : dyp;
            const Real adv_s = amrex::max(Vsx*phix_s + Vsy*phiy_s, Real(0.0));

            const Real R0 = v_arr(i, j, k, 4);
            rhs_arr(i, j, k) = -R0*(norm_grad - eps*laplacian) - adv_w - adv_s;
        });
    }
}

/**
 * @brief Zero a rate-of-spread field where a non-burnable mask is set
 *
 * @param[in,out] R     Rate of spread [m/s]
 * @param[in]     mask  Non-burnable mask (>0.5 = non-burnable), or nullptr for no-op
 */
inline void zero_ros_in_mask(amrex::MultiFab& R, const amrex::MultiFab* mask)
{
    if (mask == nullptr) { return; }
    for (amrex::MFIter mfi(R, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& r = R.array(mfi);
        auto const& m = mask->const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (m(i, j, k) > amrex::Real(0.5)) { r(i, j, k) = amrex::Real(0.0); }
        });
    }
}

/**
 * @brief Multiply a rate-of-spread field by a per-cell factor
 *
 * @param[in,out] R      Rate of spread [m/s]
 * @param[in]     scale  Factor on the same grid, or nullptr for no-op
 */
inline void scale_ros_field(amrex::MultiFab& R, const amrex::MultiFab* scale)
{
    if (scale == nullptr) { return; }
    for (amrex::MFIter mfi(R, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& r = R.array(mfi);
        auto const& s = scale->const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            r(i, j, k) *= s(i, j, k);
        });
    }
}

/**
 * @brief Hold the level set at or above `value` where a non-burnable mask is set
 *
 * @param[in,out] phi    Level-set field
 * @param[in]     mask   Non-burnable mask (>0.5 = non-burnable), or nullptr for no-op
 * @param[in]     value  Level-set value kept in mask cells (positive = unburned)
 */
inline void hold_phi_in_mask(amrex::MultiFab& phi, const amrex::MultiFab* mask, amrex::Real value)
{
    if (mask == nullptr) { return; }
    for (amrex::MFIter mfi(phi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.growntilebox(amrex::min(phi.nGrowVect(), mask->nGrowVect()));
        auto const& p = phi.array(mfi);
        auto const& m = mask->const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (m(i, j, k) > amrex::Real(0.5)) { p(i, j, k) = amrex::max(p(i, j, k), value); }
        });
    }
}

/**
 * @brief SSP-RK3 stage updates on the valid cells
 *
 * Kept out of advect_levelset_rk3_with_fill: CUDA rejects a device lambda
 * inside a function template whose template argument is a caller's local
 * lambda type, which is how every ROS model passes its fill_ros.
 */
/// Stage 1: phi* = phi* + dt L(phi*)
inline void rk3_stage1_update(amrex::MultiFab& phi_star, const amrex::MultiFab& rhs, amrex::Real dt)
{
    using namespace amrex;
    for (MFIter mfi(phi_star); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto ps = phi_star.array(mfi);
        auto r  = rhs.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
            ps(iv) = ps(iv) + dt * r(iv);
        });
    }
}

/// Stage 2: phi** = (3/4) phi^n + (1/4)(phi* + dt L(phi*)), with phi_hat holding phi^n on entry
inline void rk3_stage2_update(amrex::MultiFab& phi_hat, const amrex::MultiFab& phi_star,
                              const amrex::MultiFab& rhs, amrex::Real dt)
{
    using namespace amrex;
    for (MFIter mfi(phi_hat); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto ph = phi_hat.array(mfi);
        auto ps = phi_star.const_array(mfi);
        auto r  = rhs.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
            ph(iv) = 0.75_rt * ph(iv) + 0.25_rt * (ps(iv) + dt * r(iv));
        });
    }
}

/// Stage 3: phi^{n+1} = (1/3) phi^n + (2/3)(phi** + dt L(phi**))
inline void rk3_stage3_update(amrex::MultiFab& phi, const amrex::MultiFab& phi_hat,
                              const amrex::MultiFab& phi_n, const amrex::MultiFab& rhs, amrex::Real dt)
{
    using namespace amrex;
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto pnew = phi.array(mfi);
        auto ph   = phi_hat.const_array(mfi);
        auto r    = rhs.const_array(mfi);
        auto pn   = phi_n.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
            pnew(iv) = (1.0_rt/3.0_rt) * pn(iv) + (2.0_rt/3.0_rt) * (ph(iv) + dt * r(iv));
        });
    }
}

/**
 * @brief SSP-RK3 level-set step with a rate of spread rebuilt every stage
 *
 * Shared driver for every direction-dependent path: the caller supplies
 * fill_ros(R, phi_stage), which writes the ROS field to use for that stage
 * from the stage's level set (typically from its front normal). Same Godunov
 * spatial scheme and stage arithmetic as advect_levelset_weno5z_rk3().
 *
 * @param[in,out] phi       Level-set field
 * @param[in]     slopes    Terrain slopes [dz/dx, dz/dy]; |∇φ| is projected
 *                          onto the terrain surface
 * @param[in]     geom      Fire-grid geometry
 * @param[in]     dt        Timestep [s]
 * @param[in]     eps_visc  Artificial viscosity coefficient
 * @param[in]     fill_ros  Callable (MultiFab& R, const MultiFab& phi_stage)
 * @param[in]     mask      Optional non-burnable mask; the ROS is zeroed there
 *                          after every fill
 * @param[in]     wall_extrap  Extrapolate phi into mask cells inside every
 *                          stencil (see WallStencil); needs a mask
 * @param[in]     ros_scale Optional per-cell factor applied to the ROS after
 *                          every fill (the front acceleration clock)
 * @param[in]     scratch   Stage fields allocated once (FireLayer); null allocates locally
 */
template <class FillRos>
inline void advect_levelset_rk3_with_fill(amrex::MultiFab& phi,
                                          const amrex::MultiFab& slopes,
                                          const amrex::Geometry& geom,
                                          amrex::Real dt,
                                          amrex::Real eps_visc,
                                          FillRos&& fill_ros,
                                          const amrex::MultiFab* mask = nullptr,
                                          bool wall_extrap = false,
                                          const LevelSetGradient& grad_scheme = LevelSetGradient{},
                                          const amrex::MultiFab* ros_scale = nullptr,
                                          Scratch* scratch = nullptr)
{
    using namespace amrex;

    const auto dx = geom.CellSizeArray();

    Scratch local;
    Scratch& sc = Scratch::select(scratch, local, phi);
    MultiFab& phi_n    = sc.phi_n;
    MultiFab& phi_star = sc.phi_star;
    MultiFab& phi_hat  = sc.phi_hat;
    MultiFab& rhs      = sc.rhs;
    MultiFab& R_dir    = sc.R_dir;

    MultiFab::Copy(phi_n, phi, 0, 0, 1, 0);
    fire_fill_boundary(phi_n, geom);

    // Stage 1: phi* = phi^n + dt L(phi^n)
    MultiFab::Copy(phi_star, phi_n, 0, 0, 1, 0);
    fire_fill_boundary(phi_star, geom);
    fill_ros(R_dir, phi_star);
    scale_ros_field(R_dir, ros_scale);
    zero_ros_in_mask(R_dir, mask);
    compute_levelset_rhs(rhs, phi_star, R_dir, dx[0], dx[1], eps_visc, &slopes, mask, wall_extrap, grad_scheme);
    rk3_stage1_update(phi_star, rhs, dt);
    fire_fill_boundary(phi_star, geom);

    // Stage 2: phi** = (3/4)phi^n + (1/4)(phi* + dt L(phi*))
    fill_ros(R_dir, phi_star);
    scale_ros_field(R_dir, ros_scale);
    zero_ros_in_mask(R_dir, mask);
    compute_levelset_rhs(rhs, phi_star, R_dir, dx[0], dx[1], eps_visc, &slopes, mask, wall_extrap, grad_scheme);
    MultiFab::Copy(phi_hat, phi_n, 0, 0, 1, 0);
    rk3_stage2_update(phi_hat, phi_star, rhs, dt);
    fire_fill_boundary(phi_hat, geom);

    // Stage 3: phi^{n+1} = (1/3)phi^n + (2/3)(phi** + dt L(phi**))
    fill_ros(R_dir, phi_hat);
    scale_ros_field(R_dir, ros_scale);
    zero_ros_in_mask(R_dir, mask);
    compute_levelset_rhs(rhs, phi_hat, R_dir, dx[0], dx[1], eps_visc, &slopes, mask, wall_extrap, grad_scheme);
    rk3_stage3_update(phi, phi_hat, phi_n, rhs, dt);
}

/**
 * @brief Advect level-set using Godunov upwind spatial + SSP-RK3 temporal integration
 *
 * Solves ∂φ/∂t = −R(x,y) * (|∇φ| − ε Δφ) with a single RK3 step.
 * Multi-stage calls advance by dt; CFL-based subcycling is handled by caller.
 *
 * Note: vel_eff is accepted for interface consistency with other advection
 * routines but is not used here; ROS magnitude is taken directly from R_mf.
 *
 * @param[in,out] phi       Level-set field (modified in-place)
 * @param[in]     vel_eff   Wind velocity (unused; ROS taken from R_mf)
 * @param[in]     R_mf      Rate-of-spread field [m/s]
 * @param[in]     geom      Domain geometry
 * @param[in]     dt        Timestep
 * @param[in]     eps_visc  Artificial viscosity coefficient (erf.fire.levelset.eps_visc)
 * @param[in]     slopes    Terrain slopes [dz/dx, dz/dy]; when supplied, |∇φ| is
 *                          projected onto the terrain surface so that R is a
 *                          spread rate along the ground. nullptr keeps the
 *                          map-view interpretation.
 * @param[in]     mask      Optional non-burnable mask, only read with wall_extrap
 * @param[in]     wall_extrap  Extrapolate phi into mask cells inside every stencil
 * @param[in]     scratch   Stage fields allocated once (FireLayer); null allocates locally
 */
inline void advect_levelset_weno5z_rk3(amrex::MultiFab& phi,
                                        const amrex::MultiFab& /*vel_eff*/,
                                        const amrex::MultiFab& R_mf,
                                        const amrex::Geometry& geom,
                                        amrex::Real dt,
                                        amrex::Real eps_visc,
                                        const amrex::MultiFab* slopes = nullptr,
                                        const amrex::MultiFab* mask = nullptr,
                                        bool wall_extrap = false,
                                        const LevelSetGradient& grad_scheme = LevelSetGradient{},
                                        Scratch* scratch = nullptr)
{
    using namespace amrex;

    const auto dx = geom.CellSizeArray();

    // Stage MultiFabs with 3 ghost cells (the WENO5-Z stencil reads all three)
    Scratch local;
    Scratch& sc = Scratch::select(scratch, local, phi);
    MultiFab& phi_n    = sc.phi_n;
    MultiFab& phi_star = sc.phi_star;
    MultiFab& phi_hat  = sc.phi_hat;
    MultiFab& rhs      = sc.rhs;

    // Save initial phi (input)
    MultiFab::Copy(phi_n, phi, 0, 0, 1, 0);
    fire_fill_boundary(phi_n, geom);

    // Stage 1: φ* = φ^n + Δt L(φ^n)
    MultiFab::Copy(phi_star, phi_n, 0, 0, 1, 0);
    // The copy fills the valid region only, so the three ghost cells the
    // gradient-Z stencil in compute_levelset_rhs reads are refilled here.
    fire_fill_boundary(phi_star, geom);
    compute_levelset_rhs(rhs, phi_star, R_mf, dx[0], dx[1], eps_visc, slopes, mask, wall_extrap, grad_scheme);
    for (MFIter mfi(phi_star); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto ps = phi_star.array(mfi);
        auto r = rhs.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
            ps(iv) = ps(iv) + dt * r(iv);
        });
    }
    fire_fill_boundary(phi_star, geom);

    // Stage 2: φ** = (3/4)φ^n + (1/4)(φ* + Δt L(φ*))
    compute_levelset_rhs(rhs, phi_star, R_mf, dx[0], dx[1], eps_visc, slopes, mask, wall_extrap, grad_scheme);
    MultiFab::Copy(phi_hat, phi_n, 0, 0, 1, 0);
    for (MFIter mfi(phi_hat); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto ph = phi_hat.array(mfi);
        auto ps = phi_star.const_array(mfi);
        auto r = rhs.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
            ph(iv) = 0.75_rt * ph(iv) + 0.25_rt * (ps(iv) + dt * r(iv));
        });
    }
    fire_fill_boundary(phi_hat, geom);

    // Stage 3: φ^{n+1} = (1/3)φ^n + (2/3)(φ** + Δt L(φ**))
    compute_levelset_rhs(rhs, phi_hat, R_mf, dx[0], dx[1], eps_visc, slopes, mask, wall_extrap, grad_scheme);
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto pnew = phi.array(mfi);
        auto ph = phi_hat.const_array(mfi);
        auto r = rhs.const_array(mfi);
        auto pn = phi_n.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
            pnew(iv) = (1.0_rt/3.0_rt) * pn(iv) + (2.0_rt/3.0_rt) * (ph(iv) + dt * r(iv));
        });
    }
}

} // namespace fire_levelset

#endif