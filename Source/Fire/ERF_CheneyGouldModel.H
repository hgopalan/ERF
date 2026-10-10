#ifndef ERF_CHENEY_GOULD_MODEL_H
#define ERF_CHENEY_GOULD_MODEL_H

#include <AMReX_MultiFab.H>
#include <AMReX_Gpu.H>
#include <cmath>
#include <ERF_Rothermel.H>

/**
 * @file ERF_CheneyGouldModel.H
 * @brief Cheney, Gould and Catchpole (1998) grassland fire spread model, and
 *        the simple grass fit this code carried under that name until October 2026
 *
 * ros_model = "cheney_gould" is the CSIRO grassland fire spread model of
 * Cheney, Gould and Catchpole (1998), the model behind the CSIRO Grassland
 * Fire Spread Meter and the Australian Fire Danger Rating System's grassland
 * module. The head rate of spread in km/h is
 *
 *   R = (a_1 + a_2 U10) phi_M phi_C                      U10 < 5 km/h
 *   R = (b_1 + b_2 (U10 - 5)^0.844) phi_M phi_C           U10 >= 5 km/h
 *
 * with U10 the 10 m open wind in km/h, the coefficients of the pasture type
 * (natural: 0.054, 0.269, 1.4, 0.838; grazed: 0.054, 0.209, 1.1, 0.715), the
 * dead fine fuel moisture function
 *
 *   phi_M = exp(-0.108 M)                                M < 12 %
 *   phi_M = 0.684 - 0.0342 M   (U10 < 10 km/h)           12 <= M < 20 %
 *   phi_M = 0.547 - 0.0228 M   (U10 >= 10 km/h)          12 <= M < 24 %
 *
 * (zero beyond), and the curing coefficient of the degree of curing C [%],
 * by default the Cruz et al. (2015) curve
 *
 *   phi_C = 1.036 / (1 + 103.989 exp(-0.0996 (C - 20)))
 *
 * (erf.fire.cheney_gould.curing_curve = "cruz2015"), or the 1998 curve
 * phi_C = 1.12 / (1 + 59.2 exp(-0.124 (C - 50))) ("cheney1998"). The wind is
 * the reference-height wind (erf.fire.cheney_gould.wind_source = "reference",
 * so set erf.fire.wind_ref_ht = 10 without erf.fire.use_per_fuel_wind_ht)
 * or the midflame wind after the Wind
 * Adjustment Factor ("midflame"). Not appropriate for forest or shrub fuels.
 *
 * ros_model = "grass_simple" is the fit that this code presented as
 * "Cheney-Gould (1998)" before October 2026 and that no publication describes:
 *
 *   R = R_b (1 + 0.15 U (c + 0.2)) 20 / (M + 1),   R_b = max(0.08 c e^{-0.01 M}, 0.001) m/s
 *
 * with U the midflame wind in m/s, M in % and c the curing fraction. It is
 * about 26x the 1998 model's rate in calm, cured grass at 10 % moisture and
 * under a third of it at 20 km/h; it is kept so earlier results can be reproduced.
 *
 * References:
 *   Cheney, N.P., Gould, J.S. & Catchpole, W.R. (1998). Prediction of fire
 *     spread in grasslands. Int. J. Wildland Fire, 8(1), 1-13.
 *   Cruz, M.G., Gould, J.S., Kidnie, S., Bessell, R., Nichols, D. &
 *     Slijepcevic, A. (2015). Effects of curing on grassfires: II. Effect of
 *     grass senescence on the rate of fire spread. Int. J. Wildland Fire, 24,
 *     838-848.
 */

/// Selectors of the grass model state, from erf.fire.ros_model and erf.fire.cheney_gould.*
namespace cheney_gould {
    constexpr int model_cg98         = 0;   ///< ros_model = "cheney_gould": Cheney, Gould and Catchpole (1998)
    constexpr int model_grass_simple = 1;   ///< ros_model = "grass_simple": the pre-2026-10 fit
    constexpr int pasture_natural    = 0;   ///< erf.fire.cheney_gould.pasture = "natural"
    constexpr int pasture_grazed     = 1;   ///< erf.fire.cheney_gould.pasture = "grazed"
    constexpr int curing_cruz2015    = 0;   ///< erf.fire.cheney_gould.curing_curve = "cruz2015"
    constexpr int curing_cheney1998  = 1;   ///< erf.fire.cheney_gould.curing_curve = "cheney1998"
}

/**
 * @struct CheneyGouldComputed
 * @brief Precomputed grass-model state (depends on moisture and curing), trivially
 *        copyable so the directional kernels can capture it
 */
struct CheneyGouldComputed
{
    int         model     = cheney_gould::model_cg98;       ///< cheney_gould::model_*
    int         pasture   = cheney_gould::pasture_natural;  ///< cheney_gould::pasture_*
    amrex::Real moisture  = 10.0;   ///< Dead fine fuel moisture [%] the state was built with
    amrex::Real curing    = 1.0;    ///< Degree of curing [0-1] the state was built with
    amrex::Real phi_C     = 1.0;    ///< Curing coefficient of the 1998 model
    amrex::Real ros_backing = 0.0;  ///< Backing rate [m/s] of the grass_simple fit
};

/// Curing coefficient phi_C of the degree of curing C [%], by curve (cheney_gould::curing_*)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cheney_gould_phi_C(amrex::Real C_pct, int curve) noexcept
{
    const amrex::Real C = amrex::max(amrex::Real(0.0), amrex::min(C_pct, amrex::Real(100.0)));
    if (curve == cheney_gould::curing_cheney1998) {
        return amrex::Real(1.12) / (amrex::Real(1.0) + amrex::Real(59.2) * std::exp(amrex::Real(-0.124) * (C - amrex::Real(50.0))));
    }
    return amrex::Real(1.036) / (amrex::Real(1.0) + amrex::Real(103.989) * std::exp(amrex::Real(-0.0996) * (C - amrex::Real(20.0))));
}

/// Moisture coefficient phi_M of the dead fine fuel moisture M [%] and the 10 m wind [km/h]
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cheney_gould_phi_M(amrex::Real M_pct, amrex::Real U10_kmh) noexcept
{
    const amrex::Real M = amrex::max(M_pct, amrex::Real(0.0));
    if (M < amrex::Real(12.0)) { return std::exp(amrex::Real(-0.108) * M); }
    const amrex::Real phi = (U10_kmh < amrex::Real(10.0))
                          ? amrex::Real(0.684) - amrex::Real(0.0342) * M
                          : amrex::Real(0.547) - amrex::Real(0.0228) * M;
    return amrex::max(phi, amrex::Real(0.0));
}

/**
 * @brief Cheney, Gould and Catchpole (1998) head rate of spread [m/s]
 *
 * @param[in] U10_mps   10 m open wind in the spread direction [m/s] (>= 0)
 * @param[in] M_pct     Dead fine fuel moisture [%]
 * @param[in] phi_C     Curing coefficient (cheney_gould_phi_C)
 * @param[in] pasture   cheney_gould::pasture_natural or ::pasture_grazed
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cheney_gould_1998_ros(amrex::Real U10_mps, amrex::Real M_pct,
                                  amrex::Real phi_C, int pasture) noexcept
{
    const amrex::Real U = amrex::Real(3.6) * amrex::max(U10_mps, amrex::Real(0.0));   // km/h
    const bool grazed = (pasture == cheney_gould::pasture_grazed);
    const amrex::Real a1 = amrex::Real(0.054);
    const amrex::Real a2 = grazed ? amrex::Real(0.209) : amrex::Real(0.269);
    const amrex::Real b1 = grazed ? amrex::Real(1.1)   : amrex::Real(1.4);
    const amrex::Real b2 = grazed ? amrex::Real(0.715) : amrex::Real(0.838);
    // the high-wind branch's pow is evaluated from a floored argument so that
    // nothing speculated under the fpe traps raises a flag below 5 km/h
    const amrex::Real excess = amrex::max(U - amrex::Real(5.0), amrex::Real(0.0));
    const amrex::Real R_lo = a1 + a2 * U;
    const amrex::Real R_hi = b1 + b2 * std::pow(excess, amrex::Real(0.844));
    const amrex::Real R_kmh = ((U < amrex::Real(5.0)) ? R_lo : R_hi) * cheney_gould_phi_M(M_pct, U) * phi_C;
    return amrex::max(R_kmh / amrex::Real(3.6), amrex::Real(0.0));
}

/// The pre-2026-10 grass fit, ros_model = "grass_simple" (see the file comment)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real grass_simple_ros(amrex::Real u_wind, amrex::Real ros_backing,
                             amrex::Real moisture, amrex::Real curing) noexcept
{
    const amrex::Real wind_factor = amrex::Real(0.15) * u_wind * (curing + amrex::Real(0.2));
    constexpr amrex::Real moisture_scale = 20.0;   // reference moisture [%]
    const amrex::Real moisture_factor = moisture_scale / (moisture + amrex::Real(1.0));
    const amrex::Real ros_forward = ros_backing * (amrex::Real(1.0) + wind_factor) * moisture_factor;
    return amrex::max(ros_forward, amrex::Real(0.0));
}

/**
 * @brief Head rate of spread [m/s] of the selected grass model
 *
 * @param[in] u_wind  Wind in the spread direction [m/s]: the 10 m (reference) or
 *                    midflame wind for the 1998 model as erf.fire.cheney_gould.wind_source
 *                    selects, the midflame wind for grass_simple
 * @param[in] cg      Precomputed state (compute_cheney_gould_params)
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cheney_gould_ros(amrex::Real u_wind, const CheneyGouldComputed& cg) noexcept
{
    if (cg.model == cheney_gould::model_grass_simple) {
        return grass_simple_ros(amrex::max(u_wind, amrex::Real(0.0)), cg.ros_backing, cg.moisture, cg.curing);
    }
    return cheney_gould_1998_ros(u_wind, cg.moisture, cg.phi_C, cg.pasture);
}

/**
 * @brief Build the grass-model state for the deck's moisture and curing
 *
 * Carries both inputs so that every path (the isotropic fill and the
 * directional kernel) evaluates cheney_gould_ros with the same moisture and
 * curing, including the dynamic-moisture update that rebuilds this struct each
 * step. The inputs are range-checked in FireParams (moisture 1-40 %, curing
 * 0-1); the clamps below only guard a struct built by hand.
 *
 * @param[in] cgp          moisture [%] and curing [0-1]
 * @param[in] model        cheney_gould::model_cg98 or ::model_grass_simple
 * @param[in] pasture      cheney_gould::pasture_*
 * @param[in] curing_curve cheney_gould::curing_*
 */
inline CheneyGouldComputed compute_cheney_gould_params(
    const FireParams::CheneyGouldParams& cgp,
    int model = cheney_gould::model_cg98,
    int pasture = cheney_gould::pasture_natural,
    int curing_curve = cheney_gould::curing_cruz2015) noexcept
{
    CheneyGouldComputed cg;
    cg.model    = model;
    cg.pasture  = pasture;
    cg.moisture = amrex::max(amrex::Real(1.0), amrex::min(cgp.moisture, amrex::Real(40.0)));   // [%]
    cg.curing   = amrex::max(amrex::Real(0.0), amrex::min(cgp.curing,   amrex::Real(1.0)));    // [0-1]
    cg.phi_C    = cheney_gould_phi_C(amrex::Real(100.0) * cg.curing, curing_curve);
    // grass_simple: backing rate 0.08 c e^{-0.01 M}, floored at 1 mm/s
    cg.ros_backing = amrex::max(amrex::Real(0.08) * cg.curing * std::exp(amrex::Real(-0.01) * cg.moisture),
                                amrex::Real(0.001));
    return cg;
}

/**
 * @brief Fill fire_ros [m/s] with the head rate of the selected grass model
 *
 * @param[out] fire_ros   Rate of spread [m/s]
 * @param[in]  fire_wind  The wind the model consumes (reference or midflame, see
 *                        cheney_gould_ros) [m/s]
 * @param[in]  cg         Precomputed state
 */
inline void fill_cheney_gould_ros(
    amrex::MultiFab& fire_ros,
    const amrex::MultiFab& fire_wind,
    const CheneyGouldComputed& cg)
{
    for (amrex::MFIter mfi(fire_ros, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto ros_arr  = fire_ros.array(mfi);
        auto wind_arr = fire_wind.const_array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (const amrex::IntVect& iv) {
            const int i = iv[0];
            const int j = iv[1];
            const int k = 0;
            const amrex::Real u = wind_arr(i, j, k, 0);
            const amrex::Real v = wind_arr(i, j, k, 1);
            const amrex::Real wind_mag = std::sqrt(u*u + v*v);
            ros_arr(i, j, k, 0) = cheney_gould_ros(wind_mag, cg);
        });
    }
}

#endif
