#ifndef ERF_FUEL_MOISTURE_STICK_H
#define ERF_FUEL_MOISTURE_STICK_H

#include <AMReX_REAL.H>
#include <AMReX_GpuDevice.H>
#include <cmath>
#include <ERF_FuelMoisture.H>

/**
 * @file ERF_FuelMoistureStick.H
 * @brief Dead-fuel moisture as radial diffusion in a cylindrical stick
 *
 * The time-lag model relaxes each class's moisture toward the equilibrium
 * with one time constant, so a class responds at the same rate whether the
 * change comes from a brief humidity swing or a long one. A fuel particle is
 * a cylinder: moisture enters and leaves through its surface and diffuses
 * inward, so the surface layers follow the air within minutes while the core
 * lags by hours, and the volume-average moisture the fire sees has the
 * diurnal shape Nelson (2000) measured on 10-hour sticks.
 *
 * This option (erf.fire.moisture_model = stick) solves that diffusion for
 * each dead class in every fire cell,
 *
 *   dM/dt = (1/r) d/dr ( r D dM/dr ),   0 < r < R_c,
 *
 * on N shells of equal thickness with an implicit (backward Euler)
 * tridiagonal step, the surface held at the equilibrium moisture of the air
 * with the same adsorption/desorption hysteresis the time-lag model uses,
 * or at the rain-wet value while it rains. The diffusivity is set so that
 * the slowest radial mode has the class's nominal time lag,
 *
 *   D_c = R_c^2 / (lambda_1^2 tau_c),  lambda_1 = 2.4048 (first zero of J_0),
 *
 * scaled by the same temperature factor as the time lag, so the long-time
 * response equals the time-lag model's and the short-time response is the
 * diffusive one. The class moisture handed to the rate-of-spread models is
 * the volume average over the shells.
 *
 * What this is and is not: the framework, the boundary treatment and the
 * lag-calibrated diffusivity follow Nelson (2000); his species constants for
 * the diffusivity, the bound-water isotherm and the surface exchange are not
 * reproduced, and there is no solar heating of the stick (the surface sees
 * the air temperature). Radii default to representative particle sizes,
 * 0.15 cm (1 h), 0.635 cm (the standard 10-h stick) and 2.5 cm (100 h).
 */

namespace FuelStickConst {
    constexpr int         MAX_SHELLS = 16;
    constexpr amrex::Real LAMBDA1    = 2.404825557695773;   ///< first zero of J_0
}

/// Diffusivity [cm^2/h] that gives a cylinder of radius R [cm] the time lag tau [h]
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real stick_diffusivity(amrex::Real R_cm, amrex::Real tau_h) noexcept
{
    return R_cm * R_cm / (FuelStickConst::LAMBDA1 * FuelStickConst::LAMBDA1 * tau_h);
}

/// Volume average of the shell moistures (shell i spans [i, i+1] dR, volume ~ 2i+1)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real stick_volume_average(const amrex::Real* M, int N) noexcept
{
    amrex::Real s = 0.0;
    for (int i = 0; i < N; ++i) { s += (2 * i + 1) * M[i]; }
    return s / (amrex::Real(N) * amrex::Real(N));
}

/**
 * @brief Advance one stick by dt_h [h] with the surface held at M_surf.
 *
 * Finite volumes on N shells of thickness dR = R/N, shell i between faces
 * i dR and (i+1) dR, volume (per unit length and angle) V_i = (2i+1) dR^2 / 2,
 * face conductance D r_f / dR, and the outer face at half a shell from the
 * surface value. Backward Euler, Thomas algorithm.
 *
 * @param[in,out] M      shell moistures [fraction], M[0] the core, M[N-1] the surface shell
 * @param[in]     N      number of shells (<= MAX_SHELLS)
 * @param[in]     R_cm   stick radius [cm]
 * @param[in]     D      diffusivity [cm^2/h]
 * @param[in]     M_surf surface moisture [fraction]
 * @param[in]     dt_h   step [h]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void stick_advance(amrex::Real* M, int N, amrex::Real R_cm, amrex::Real D,
                   amrex::Real M_surf, amrex::Real dt_h) noexcept
{
    if (N < 1 || dt_h <= 0.0) return;
    const amrex::Real dR = R_cm / N;
    amrex::Real a[FuelStickConst::MAX_SHELLS], b[FuelStickConst::MAX_SHELLS],
                c[FuelStickConst::MAX_SHELLS], d[FuelStickConst::MAX_SHELLS];
    for (int i = 0; i < N; ++i) {
        const amrex::Real V    = 0.5 * (2 * i + 1) * dR * dR;
        const amrex::Real g_lo = (i > 0) ? D * (i * dR) / dR : 0.0;                 // inner face
        const amrex::Real g_hi = (i < N - 1) ? D * ((i + 1) * dR) / dR              // outer face
                                             : D * R_cm / (0.5 * dR);              // surface, half a shell away
        a[i] = -g_lo;
        c[i] = -g_hi;
        b[i] = V / dt_h + g_lo + g_hi;
        d[i] = V / dt_h * M[i];
        if (i == N - 1) { d[i] += g_hi * M_surf; c[i] = 0.0; }
    }
    // Thomas
    for (int i = 1; i < N; ++i) {
        const amrex::Real w = a[i] / b[i - 1];
        b[i] -= w * c[i - 1];
        d[i] -= w * d[i - 1];
    }
    M[N - 1] = d[N - 1] / b[N - 1];
    for (int i = N - 2; i >= 0; --i) { M[i] = (d[i] - c[i] * M[i + 1]) / b[i]; }
    for (int i = 0; i < N; ++i) {
        M[i] = amrex::min(amrex::max(M[i], amrex::Real(FuelMoistureConst::M_MIN)), amrex::Real(FuelMoistureConst::M_MAX));
    }
}

/**
 * @brief One class of one cell for one step: surface condition, diffusivity, advance.
 * @return the volume-average moisture after the step
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real stick_advance_class(amrex::Real* M, int N, amrex::Real R_cm, amrex::Real tau_h,
                                amrex::Real RH, amrex::Real T_C, amrex::Real rain_mm_hr,
                                amrex::Real rain_surface_moisture, amrex::Real diffusivity_scale,
                                amrex::Real dt_h) noexcept
{
    // Surface: the air's equilibrium with hysteresis on the surface shell, or rain-wet.
    amrex::Real M_surf = compute_emc_with_hysteresis(RH, M[N - 1]);
    if (rain_mm_hr >= FuelMoistureConst::MIN_RAIN_RATE) { M_surf = amrex::max(M_surf, rain_surface_moisture); }
    // Diffusivity: lag-calibrated, faster when warm (same factor as the time lag).
    const amrex::Real D = diffusivity_scale * stick_diffusivity(R_cm, tau_h * compute_temp_correction(T_C));
    stick_advance(M, N, R_cm, D, M_surf, dt_h);
    return stick_volume_average(M, N);
}

#endif
