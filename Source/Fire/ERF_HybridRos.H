#ifndef ERF_HYBRID_ROS_H
#define ERF_HYBRID_ROS_H

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <ERF_BalbiModel.H>
#include <ERF_DirectionalRos.H>
#include <ERF_LevelSetAdvection.H>

/**
 * @file ERF_HybridRos.H
 * @brief Hybrid rate of spread: two models blended with a per-cell weight
 *
 *   R = (1 - w) R_primary + w R_secondary,   w in [0, 1]
 *
 * The isotropic path blends the two fire_ros fields once per fire step. The
 * direction-dependent level-set path rebuilds both models along the front
 * normal at every Runge-Kutta stage and blends them there, so the hybrid gets
 * head, flank and backing spread from each member exactly as the single-model
 * directional paths do. A weight of exactly 0 or 1 everywhere reproduces the
 * corresponding single-model result.
 */

/// Model code meaning "Balbi" in HybridDirectionalSpec, alongside DIRECTIONAL_ROS_*.
inline constexpr int HYBRID_MODEL_BALBI = -1;

/**
 * @struct HybridDirectionalSpec
 * @brief Everything the directional hybrid fill needs, by reference
 *
 * Generic models carry their coefficients in a DirectionalRosState and read
 * the effective wind; Balbi carries its own coefficients and parameter block
 * and reads whichever wind erf.fire.balbi.wind_source selects.
 */
struct HybridDirectionalSpec
{
    int                 primary_model   = DIRECTIONAL_ROS_ROTHERMEL;
    DirectionalRosState primary_state{};
    int                 secondary_model = HYBRID_MODEL_BALBI;
    DirectionalRosState secondary_state{};

    const amrex::MultiFab*         wind_eff   = nullptr;  ///< Effective wind [m/s]
    const amrex::MultiFab*         balbi_wind = nullptr;  ///< Wind the Balbi kernels consume
    const BalbiComputed*           bc         = nullptr;  ///< Balbi default coefficients
    const FireParams::BalbiParams* bp         = nullptr;  ///< Balbi parameters
    const BalbiFieldInputs*        balbi_in   = nullptr;  ///< Balbi per-cell fields
    const amrex::MultiFab*         weight     = nullptr;  ///< Secondary weight w in [0,1]
};

/**
 * @brief In place blend R = (1 - w) R + w R_sec
 */
inline void blend_ros_fields(amrex::MultiFab& R,
                             const amrex::MultiFab& R_sec,
                             const amrex::MultiFab& weight)
{
    for (amrex::MFIter mfi(R, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& ros = R.array(mfi);
        auto const& sec = R_sec.const_array(mfi);
        auto const& w   = weight.const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const amrex::Real wt = w(i, j, k);
            ros(i, j, k) = (amrex::Real(1.0) - wt) * ros(i, j, k) + wt * sec(i, j, k);
        });
    }
}

/**
 * @brief Directional ROS of one hybrid member from the front normal
 */
inline void fill_hybrid_member_directional(amrex::MultiFab& R,
                                           const amrex::MultiFab& phi,
                                           const amrex::MultiFab& slopes,
                                           const amrex::Geometry& geom,
                                           int model,
                                           const DirectionalRosState& state,
                                           const HybridDirectionalSpec& spec,
                                           const amrex::MultiFab* mask = nullptr,
                                           bool wall_extrap = false)
{
    if (model == HYBRID_MODEL_BALBI) {
        fill_balbi_ros_directional(R, phi, *spec.balbi_wind, slopes, geom,
                                   *spec.bc, *spec.bp, *spec.balbi_in, mask, wall_extrap);
    } else {
        fill_directional_ros(R, phi, *spec.wind_eff, slopes, geom, state, mask, wall_extrap);
    }
}

/**
 * @brief Directional hybrid ROS: both members along the front normal, blended
 */
inline void fill_hybrid_directional_ros(amrex::MultiFab& R,
                                        const amrex::MultiFab& phi,
                                        const amrex::MultiFab& slopes,
                                        const amrex::Geometry& geom,
                                        const HybridDirectionalSpec& spec,
                                        const amrex::MultiFab* mask = nullptr,
                                        bool wall_extrap = false)
{
    fill_hybrid_member_directional(R, phi, slopes, geom,
                                   spec.primary_model, spec.primary_state, spec,
                                   mask, wall_extrap);
    amrex::MultiFab R_sec(R.boxArray(), R.DistributionMap(), 1, 0);
    fill_hybrid_member_directional(R_sec, phi, slopes, geom,
                                   spec.secondary_model, spec.secondary_state, spec,
                                   mask, wall_extrap);
    blend_ros_fields(R, R_sec, *spec.weight);
}

/**
 * @brief Advect the level set with the direction-dependent hybrid ROS
 */
inline void advect_levelset_hybrid_rk3(amrex::MultiFab& phi,
                                       const amrex::MultiFab& slopes,
                                       const amrex::Geometry& geom,
                                       amrex::Real dt,
                                       amrex::Real eps_visc,
                                       const HybridDirectionalSpec& spec,
                                       const amrex::MultiFab* nonburnable = nullptr,
                                       bool wall_extrap = false)
{
    fire_levelset::advect_levelset_rk3_with_fill(phi, slopes, geom, dt, eps_visc,
        [&](amrex::MultiFab& R, const amrex::MultiFab& phi_stage) {
            fill_hybrid_directional_ros(R, phi_stage, slopes, geom, spec, nonburnable, wall_extrap);
        }, nonburnable, wall_extrap);
}

#endif  // ERF_HYBRID_ROS_H
