/**
 * @file ERF_UCMRadiosityLW.H
 * @brief Phase 5.1c: LW multi-bounce radiosity (2×2 Cramer, lagged T_skin)
 *
 * Consumes Phase 5.1a view factors + previous-timestep T_skin to compute
 * effective LW incident at wall and road facets after infinite-bounce
 * reflection. Uses lagged approach (approach b): T_skin values are from
 * end of previous SEB call, so radiosity is completely decoupled from
 * the current Newton iteration.
 *
 * See problem-statement physics section for derivation.
 */

#ifndef ERF_UCM_RADIOSITY_LW_H
#define ERF_UCM_RADIOSITY_LW_H

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>

// Stefan-Boltzmann constant [W/m²/K⁴]
constexpr amrex::Real UCM_SIGMA_SB = 5.670374419e-8;

/**
 * @brief Compute multi-bounce LW effective incident at wall and road facets
 *
 * @param[in]  T_wall_prev   Previous-timestep wall skin temperature [K]
 * @param[in]  T_road_prev   Previous-timestep road skin temperature [K]
 * @param[in]  T_sky         Effective sky temperature = (LW_down/σ)^(1/4) [K]
 * @param[in]  eps_w         Wall LW emissivity [-]
 * @param[in]  eps_r         Road LW emissivity [-]
 * @param[in]  F_ww          F_wall_wall view factor [-]
 * @param[in]  F_wr          F_wall_road view factor [-]
 * @param[in]  F_rw          F_road_wall view factor (per wall) [-]
 * @param[in]  F_ws          F_wall_sky view factor [-]
 * @param[in]  F_rs          F_road_sky view factor [-]
 * @param[out] LW_wall_eff   Effective LW incident on wall (multi-bounce) [W/m²]
 * @param[out] LW_road_eff   Effective LW incident on road (multi-bounce) [W/m²]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void lw_radiosity_multi_bounce(amrex::Real T_wall_prev, amrex::Real T_road_prev,
                                amrex::Real T_sky,
                                amrex::Real eps_w, amrex::Real eps_r,
                                amrex::Real F_ww, amrex::Real F_wr, amrex::Real F_rw,
                                amrex::Real F_ws, amrex::Real F_rs,
                                amrex::Real& LW_wall_eff, amrex::Real& LW_road_eff)
{
    const amrex::Real sig = UCM_SIGMA_SB;
    const amrex::Real T_w4 = T_wall_prev * T_wall_prev * T_wall_prev * T_wall_prev;
    const amrex::Real T_r4 = T_road_prev * T_road_prev * T_road_prev * T_road_prev;
    const amrex::Real T_s4 = T_sky * T_sky * T_sky * T_sky;

    const amrex::Real one_m_eps_w = 1.0 - eps_w;
    const amrex::Real one_m_eps_r = 1.0 - eps_r;

    // RHS: emission from facet + sky contribution reflected off facet
    const amrex::Real rhs_w = eps_w * sig * T_w4 + one_m_eps_w * F_ws * sig * T_s4;
    const amrex::Real rhs_r = eps_r * sig * T_r4 + one_m_eps_r * F_rs * sig * T_s4;

    // Matrix determinant
    const amrex::Real det = (1.0 - one_m_eps_w * F_ww)
                            + 2.0 * one_m_eps_w * one_m_eps_r * F_wr * F_rw;
    const amrex::Real det_safe = amrex::max(det, amrex::Real(1.0e-6));

    // Cramer solve for radiosities B_w, B_r (LW leaving each facet)
    const amrex::Real B_w = (rhs_w + one_m_eps_w * F_wr * rhs_r) / det_safe;
    const amrex::Real B_r = (2.0 * one_m_eps_r * F_rw * rhs_w
                             + (1.0 - one_m_eps_w * F_ww) * rhs_r) / det_safe;

    // Effective LW incident (preserves SEB solver's LW contract)
    LW_wall_eff = F_ww * B_w + F_wr * B_r + F_ws * sig * T_s4;
    LW_road_eff = 2.0 * F_rw * B_w + F_rs * sig * T_s4;
}

#endif  // ERF_UCM_RADIOSITY_LW_H
