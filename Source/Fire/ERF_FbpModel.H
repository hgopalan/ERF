#ifndef ERF_FBP_MODEL_H
#define ERF_FBP_MODEL_H

#include <AMReX_MultiFab.H>
#include <AMReX_Gpu.H>
#include <cmath>
#include <string>

/**
 * @file ERF_FbpModel.H
 * @brief Canadian Forest Fire Behavior Prediction (FBP) System rate of spread
 *
 * Forestry Canada Fire Danger Group (1992), with the equations of Wotton,
 * Alexander and Taylor (2009). Sixteen fuel types: C1-C7 conifer, D1
 * deciduous, M1-M4 mixedwood, S1-S3 slash, O1a/O1b grass. The surface rate of
 * spread is
 *
 *   RSI = a (1 - exp(-b ISI))^c                       [m/min]
 *   ISI = 0.208 f(F) exp(0.05039 W)                   (W the 10 m wind, km/h)
 *   f(F) = 91.9 exp(-0.1386 m) (1 + m^5.31 / 4.93e7),  m = 147.2 (101 - FFMC) / (59.5 + FFMC)
 *   ROS = RSI BE,  BE = exp(50 ln(q) (1/BUI - 1/BUI0))  (the buildup effect; 1 for grass)
 *
 * with the mixedwood types weighted between C2 and D1 by the percent
 * conifer (M1, M2 with the D1 share at 0.2) and between the dead-fir
 * coefficients and D1 by the percent dead fir (M3, M4), and the grass types
 * scaled by the curing factor of the degree of curing C,
 * CF = 0.005 (exp(0.061 C) - 1) below 58.8% and 0.176 + 0.02 (C - 58.8) above.
 *
 * Slope enters the way the system prescribes: the zero-wind rate on the
 * slope, RSF = RSZ SF with SF = exp(3.533 (GS/100)^1.2) (GS the slope in
 * percent, capped at 70), is inverted to the ISI it implies, ISF, and that to
 * an equivalent wind WSE = ln(ISF / (0.208 f(F))) / 0.05039 (with the 2009
 * form above 40 km/h), which is added to the wind before the ISI is formed.
 * On the level-set path the wind and slope are the components along the front
 * normal, so the sum is the normal component of the system's vector sum.
 *
 * The system's inputs are daily indices, so FFMC, BUI, curing and the
 * percentages are deck inputs (erf.fire.fbp.*), uniform over the domain, and
 * the wind is the 10 m open wind: erf.fire.fbp.wind_source = reference
 * (default) reads the reference-height wind, so erf.fire.wind_ref_ht should
 * be 10 m; midflame reads the wind after the Wind Adjustment Factor. C6 is
 * given its surface rate only (no crown component). Rates are converted to
 * m/s for the propagation.
 */

/// FBP fuel type ids
enum FbpFuelType : int {
    FBP_C1 = 0, FBP_C2, FBP_C3, FBP_C4, FBP_C5, FBP_C6, FBP_C7, FBP_D1,
    FBP_M1, FBP_M2, FBP_M3, FBP_M4, FBP_S1, FBP_S2, FBP_S3, FBP_O1A, FBP_O1B,
    FBP_NTYPES
};

/// Coefficients of one fuel type: a, b, c of RSI and q, BUI0 of the buildup effect
struct FbpCoeffs { amrex::Real a, b, c, q, bui0; };

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
FbpCoeffs fbp_coeffs(int type) noexcept
{
    switch (type) {
        case FBP_C1:  return {90.0,  0.0649, 4.5, 0.90, 72.0};
        case FBP_C2:  return {110.0, 0.0282, 1.5, 0.70, 64.0};
        case FBP_C3:  return {110.0, 0.0444, 3.0, 0.75, 62.0};
        case FBP_C4:  return {110.0, 0.0293, 1.5, 0.80, 66.0};
        case FBP_C5:  return {30.0,  0.0697, 4.0, 0.80, 56.0};
        case FBP_C6:  return {30.0,  0.0800, 3.0, 0.80, 62.0};
        case FBP_C7:  return {45.0,  0.0305, 2.0, 0.85, 106.0};
        case FBP_D1:  return {30.0,  0.0232, 1.6, 0.90, 32.0};
        case FBP_M1:  return {0.0,   0.0,    0.0, 0.80, 50.0};   // weighted C2 / D1
        case FBP_M2:  return {0.0,   0.0,    0.0, 0.80, 50.0};
        case FBP_M3:  return {120.0, 0.0572, 1.4, 0.80, 50.0};   // 100 % dead fir part
        case FBP_M4:  return {100.0, 0.0404, 1.48, 0.80, 50.0};
        case FBP_S1:  return {75.0,  0.0297, 1.3, 0.75, 38.0};
        case FBP_S2:  return {40.0,  0.0438, 1.7, 0.75, 63.0};
        case FBP_S3:  return {55.0,  0.0829, 3.2, 0.75, 31.0};
        case FBP_O1A: return {190.0, 0.0310, 1.4, 1.00, 1.0};    // grass: no buildup effect
        case FBP_O1B: return {250.0, 0.0350, 1.7, 1.00, 1.0};
        default:      return {110.0, 0.0282, 1.5, 0.70, 64.0};
    }
}

/// Fuel type id from its name (C1..C7, D1, M1..M4, S1..S3, O1A/O1a, O1B/O1b); -1 if unknown
inline int fbp_type_from_name(const std::string& s)
{
    static const char* names[FBP_NTYPES] = {"C1","C2","C3","C4","C5","C6","C7","D1","M1","M2","M3","M4","S1","S2","S3","O1A","O1B"};
    std::string u; for (char ch : s) { u += static_cast<char>(std::toupper(static_cast<unsigned char>(ch))); }
    for (int t = 0; t < FBP_NTYPES; ++t) { if (u == names[t]) return t; }
    return -1;
}

/// Precomputed, trivially copyable state for the kernels
struct FbpComputed
{
    int         type      = FBP_C2;
    amrex::Real fF        = 0.0;    ///< FFMC function f(F)
    amrex::Real be        = 1.0;    ///< buildup effect
    amrex::Real cf        = 1.0;    ///< curing factor (grass) or 1
    amrex::Real pc        = 50.0;   ///< percent conifer (M1, M2)
    amrex::Real pdf       = 50.0;   ///< percent dead fir (M3, M4)
    bool        use_slope = true;
};

/// FFMC function f(F)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real fbp_ffmc_function(amrex::Real ffmc) noexcept
{
    const amrex::Real m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc);
    return 91.9 * std::exp(-0.1386 * m) * (1.0 + std::pow(m, 5.31) / 4.93e7);
}

/// Initial spread index from f(F) and the 10 m wind [km/h]
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real fbp_isi(amrex::Real fF, amrex::Real W_kmh) noexcept
{
    return 0.208 * fF * std::exp(0.05039 * W_kmh);
}

/// Grass curing factor of the degree of curing C [%] (Wotton et al. 2009)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real fbp_curing_factor(amrex::Real C) noexcept
{
    C = amrex::max(amrex::Real(0.0), amrex::min(C, amrex::Real(100.0)));
    return (C < 58.8) ? 0.005 * (std::exp(0.061 * C) - 1.0) : 0.176 + 0.02 * (C - 58.8);
}

/// Buildup effect for BUI and the type's q, BUI0
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real fbp_buildup_effect(int type, amrex::Real bui) noexcept
{
    const FbpCoeffs k = fbp_coeffs(type);
    if (bui <= 0.0 || k.q >= 1.0) return 1.0;
    return std::exp(50.0 * std::log(k.q) * (1.0 / bui - 1.0 / k.bui0));
}

/// a (1 - exp(-b ISI))^c
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real fbp_rsi_basic(const FbpCoeffs& k, amrex::Real isi) noexcept
{
    return k.a * std::pow(amrex::max(amrex::Real(0.0), 1.0 - std::exp(-k.b * isi)), k.c);
}

/// Surface rate of spread [m/min] of a type at an ISI, before the buildup effect
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real fbp_rsi(const FbpComputed& s, amrex::Real isi) noexcept
{
    switch (s.type) {
        case FBP_M1: return (s.pc / 100.0) * fbp_rsi_basic(fbp_coeffs(FBP_C2), isi)
                          + (1.0 - s.pc / 100.0) * fbp_rsi_basic(fbp_coeffs(FBP_D1), isi);
        case FBP_M2: return (s.pc / 100.0) * fbp_rsi_basic(fbp_coeffs(FBP_C2), isi)
                          + 0.2 * (1.0 - s.pc / 100.0) * fbp_rsi_basic(fbp_coeffs(FBP_D1), isi);
        case FBP_M3: return (s.pdf / 100.0) * fbp_rsi_basic(fbp_coeffs(FBP_M3), isi)
                          + (1.0 - s.pdf / 100.0) * fbp_rsi_basic(fbp_coeffs(FBP_D1), isi);
        case FBP_M4: return (s.pdf / 100.0) * fbp_rsi_basic(fbp_coeffs(FBP_M4), isi)
                          + 0.2 * (1.0 - s.pdf / 100.0) * fbp_rsi_basic(fbp_coeffs(FBP_D1), isi);
        case FBP_O1A:
        case FBP_O1B: return s.cf * fbp_rsi_basic(fbp_coeffs(s.type), isi);
        default:      return fbp_rsi_basic(fbp_coeffs(s.type), isi);
    }
}

/// ISI at which a basic type spreads at rsf [m/min] (inverse of fbp_rsi_basic)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real fbp_isf_basic(const FbpCoeffs& k, amrex::Real rsf) noexcept
{
    if (k.a <= 0.0 || rsf <= 0.0) return 0.0;
    const amrex::Real r = amrex::min(std::pow(rsf / k.a, 1.0 / k.c), amrex::Real(0.999));
    return std::log(1.0 - r) / (-k.b);
}

/**
 * @brief Equivalent wind of a slope [km/h]: the wind that alone would give the
 *        zero-wind rate on the slope (Forestry Canada 1992; Wotton et al. 2009).
 * @param s        precomputed state
 * @param slope    rise over run along the spread direction (dz/ds), only the uphill part counts
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real fbp_slope_equivalent_wind(const FbpComputed& s, amrex::Real slope) noexcept
{
    if (!s.use_slope || slope <= 0.0) return 0.0;
    const amrex::Real gs = amrex::min(100.0 * slope, amrex::Real(70.0));     // percent, capped as the system does
    const amrex::Real sf = std::exp(3.533 * std::pow(gs / 100.0, 1.2));
    const amrex::Real isz = fbp_isi(s.fF, 0.0);                              // zero-wind ISI
    amrex::Real isf;
    switch (s.type) {
        case FBP_M1:
        case FBP_M2: {
            const amrex::Real w = s.pc / 100.0;
            const amrex::Real rsf_c2 = fbp_rsi_basic(fbp_coeffs(FBP_C2), isz) * sf;
            const amrex::Real rsf_d1 = fbp_rsi_basic(fbp_coeffs(FBP_D1), isz) * sf;
            isf = w * fbp_isf_basic(fbp_coeffs(FBP_C2), rsf_c2) + (1.0 - w) * fbp_isf_basic(fbp_coeffs(FBP_D1), rsf_d1);
            break;
        }
        case FBP_M3:
        case FBP_M4: {
            const amrex::Real w = s.pdf / 100.0;
            const amrex::Real rsf_m  = fbp_rsi_basic(fbp_coeffs(s.type), isz) * sf;
            const amrex::Real rsf_d1 = fbp_rsi_basic(fbp_coeffs(FBP_D1), isz) * sf;
            isf = w * fbp_isf_basic(fbp_coeffs(s.type), rsf_m) + (1.0 - w) * fbp_isf_basic(fbp_coeffs(FBP_D1), rsf_d1);
            break;
        }
        case FBP_O1A:
        case FBP_O1B: {
            const FbpCoeffs k = fbp_coeffs(s.type);
            const amrex::Real rsf = s.cf * fbp_rsi_basic(k, isz) * sf;
            isf = fbp_isf_basic(k, rsf / amrex::max(s.cf, amrex::Real(1.0e-6)));
            break;
        }
        default: {
            const FbpCoeffs k = fbp_coeffs(s.type);
            isf = fbp_isf_basic(k, fbp_rsi_basic(k, isz) * sf);
        }
    }
    if (isf <= 0.0) return 0.0;
    const amrex::Real ratio = isf / (0.208 * s.fF);
    amrex::Real wse = std::log(amrex::max(ratio, amrex::Real(1.0e-12))) / 0.05039;
    if (wse > 40.0) {
        // Wotton et al. (2009) eq. 43b for equivalent winds above 40 km/h
        const amrex::Real arg = 1.0 - isf / (2.496 * s.fF);
        wse = (arg > 0.0) ? 28.0 - std::log(arg) / 0.0818 : 112.0;
    }
    return amrex::max(wse, amrex::Real(0.0));
}

/**
 * @brief FBP rate of spread [m/s] for a wind [m/s] and a slope along the spread direction
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real fbp_ros(const FbpComputed& s, amrex::Real U_mps, amrex::Real slope) noexcept
{
    const amrex::Real W = 3.6 * amrex::max(U_mps, amrex::Real(0.0)) + fbp_slope_equivalent_wind(s, slope);
    const amrex::Real isi = fbp_isi(s.fF, W);
    return fbp_rsi(s, isi) * s.be / 60.0;
}

/**
 * @brief Fill fire_ros [m/s] with the FBP rate from the wind magnitude and the
 *        slope along the wind (head rate everywhere; the level-set path's
 *        directional option evaluates the rate along the front normal instead).
 */
inline void fill_fbp_ros(amrex::MultiFab& fire_ros,
                         const amrex::MultiFab& wind,
                         const amrex::MultiFab& slopes,
                         const FbpComputed& st)
{
    for (amrex::MFIter mfi(fire_ros, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto ros_arr   = fire_ros.array(mfi);
        auto wind_arr  = wind.const_array(mfi);
        auto slope_arr = slopes.const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (const amrex::IntVect& iv) {
            const int i = iv[0], j = iv[1], k = 0;
            const amrex::Real u = wind_arr(i, j, k, 0), v = wind_arr(i, j, k, 1);
            const amrex::Real U = std::sqrt(u * u + v * v);
            // slope along the wind direction (uphill part counts)
            amrex::Real s_along = 0.0;
            if (U > 1.0e-6) { s_along = (slope_arr(i, j, k, 0) * u + slope_arr(i, j, k, 1) * v) / U; }
            ros_arr(i, j, k, 0) = fbp_ros(st, U, s_along);
        });
    }
}

#endif
