#ifndef ERF_BEHAVE_MODEL_H
#define ERF_BEHAVE_MODEL_H

/**
 * @file ERF_BehaveModel.H
 * @brief BEHAVE multi-class Rothermel rate-of-spread model (Phase 15)
 *
 * Implements the full multi-class Rothermel fire spread model following
 * Andrews (2018). This extends the single-class model in ERF_Rothermel.H
 * to account for six fuel size classes:
 *   1. Dead 1-hour
 *   2. Dead 10-hour
 *   3. Dead 100-hour
 *   4. Live herbaceous
 *   5. Live woody
 *   6. Dead herbaceous (transferred from live across the moisture window
 *      erf.fire.behave.dynamic_transfer_lo/hi)
 *
 * Key features:
 * - Dynamic load transfer from live herbaceous to dead category based on moisture
 * - Per-class reaction intensity with separate dead and live damping coefficients
 * - Live moisture of extinction computed from dead/live fuel ratio
 * - Surface-area-weighted net loads and heat sink of Rothermel (1972) eqs. 53-57
 *   (erf.fire.behave.net_load = "weighted", default; "sum" is the plain class sum
 *   this code used until October 2026)
 *
 * This model is selected by ros_model = "behave" in ParmParse configuration.
 * Selected references:
 * - Andrews, P.L. (2018). The Rothermel surface fire spread model and associated
 *   developments: A comprehensive explanation. USDA Forest Service GTR-RMRS-371.
 * - WRF-SFIRE openwfm/WRF-SFIRE/phys/module_fr_sfire_phys.F, subroutine fire_ros_behave().
 * - Anderson, H.E. (1982). Aids to determining fuel models for estimating fire
 *   behavior. USDA Forest Service INT-122.
 */

#include <AMReX_MultiFab.H>
#include <ERF_FuelModels.H>
#include <ERF_Rothermel.H>
#include <cmath>
#include <limits>

/**
 * @struct BehaveState
 * @brief Pre-computed BEHAVE multi-class Rothermel state for one cell.
 *
 * Contains all pre-computed quantities that depend on fuel model and
 * per-cell moisture. Computed once per timestep for the uniform-moisture path,
 * or per cell inside the ROS kernel when moisture varies.
 */
struct BehaveState
{
    // Net fuel loads [lb/ft²] after mineral content correction
    amrex::Real wn_dead;         ///< Net dead fuel load
    amrex::Real wn_live;         ///< Net live fuel load

    // Heat content factors [BTU/lb × dimensionless]
    amrex::Real fuelheat_dead;   ///< Weighted dead heat content factor
    amrex::Real fuelheat_live;   ///< Weighted live heat content factor

    // Damping coefficients (Rothermel equations)
    amrex::Real etam_dead;       ///< Dead moisture damping (Rothermel Eq. 29)
    amrex::Real etam_live;       ///< Live moisture damping (Rothermel Eq. 29 with live M_x)
    amrex::Real etas_dead;       ///< Dead mineral damping (Rothermel Eq. 30)
    amrex::Real etas_live;       ///< Live mineral damping (Rothermel Eq. 30)

    // Heat sink [BTU/lb × lb/ft³]
    amrex::Real heat_sink;       ///< Combined dead+live heat sink

    // Reaction velocity and propagating flux ratio
    amrex::Real gamma;           ///< Optimum reaction velocity Γ [1/min]
    amrex::Real xifr;            ///< Propagating flux ratio ξ

    // No-wind ROS [m/s]
    amrex::Real r_0;

    // Wind/slope coefficients (identical to single-class path)
    amrex::Real C;               ///< Wind factor coefficient
    amrex::Real B;               ///< Wind factor exponent
    amrex::Real beta_ratio_E;    ///< (β/β_opt)^E
    amrex::Real phi_s_const;     ///< Slope factor constant
    amrex::Real U_max_ftmin;     ///< MEWS cap [ft/min]
    amrex::Real wind_conv;       ///< m/s → ft/min conversion factor
};

/**
 * @brief Fraction of the live herbaceous load moved to the dead herbaceous class.
 *
 * Linear in the live herbaceous moisture M across the transfer window: 1 at and
 * below transfer_lo, 0 at and above transfer_hi, (hi - M) / (hi - lo) between.
 * FireParams aborts unless transfer_hi > transfer_lo. The ramp was hard-coded
 * before as 1.333 - 1.11 M, the default window rounded to three digits, which
 * transferred up to 1e-3 more of the load just below M = 1.20.
 *
 * @param[in] mc_herba    Live herbaceous fuel moisture [fraction]
 * @param[in] transfer_lo Moisture at and below which the whole load transfers [fraction]
 * @param[in] transfer_hi Moisture at and above which none transfers [fraction]
 * @return Transferred fraction in [0,1]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real behave_herb_transfer_fraction(amrex::Real mc_herba,
                                          amrex::Real transfer_lo,
                                          amrex::Real transfer_hi) noexcept
{
    const amrex::Real T = (transfer_hi - mc_herba) / (transfer_hi - transfer_lo);
    return amrex::max(amrex::Real(0.0), amrex::min(amrex::Real(1.0), T));
}

/**
 * @brief Fuel parameters BEHAVE starts from: the fuel table without its own
 *        curing transfer.
 *
 * get_fuel_params moves a Scott-Burgan dynamic model's cured herbaceous load
 * to the 1-h class at the live moisture it is given, which the single-class
 * Rothermel and Balbi paths need. BEHAVE transfers the load itself, into its
 * dead herbaceous class (compute_behave_state, Step 1), so it takes the
 * untransferred table. Handing it the cured table transferred the load twice:
 * 5/9 of GR2's herbaceous load dead at M = 0.90 instead of 1/3. Anderson codes
 * ignore the live moisture, so they are unaffected.
 *
 * @param[in] code     Fuel model code
 * @param[in] fuel_set FUEL_SET_ANDERSON13 or FUEL_SET_SCOTT_BURGAN40
 * @return Fuel parameters with the herbaceous load still live
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
FuelModelParams behave_fuel_params(int code, int fuel_set)
{
    return get_fuel_params(code, fuel_set, amrex::Real(-1.0));
}

/**
 * @brief Compute BEHAVE multi-class Rothermel state for a single cell.
 *
 * Ports fire_ros_behave() from WRF-SFIRE module_fr_sfire_phys.F, implementing:
 *   - Dynamic load transfer from live herbaceous to dead at low moisture
 *   - Area-weighted SAV for dead and live fuel classes
 *   - Live moisture of extinction from fuel load ratio
 *   - Per-class moisture and mineral damping coefficients
 *   - Multi-class reaction intensity
 *   - Multi-class heat sink
 *
 * @param[in] fp         FuelModelParams — all fuel properties, with the
 *                       herbaceous load untransferred (behave_fuel_params)
 * @param[in] mc_1hr     1-hour dead fuel moisture [fraction]
 * @param[in] mc_10hr    10-hour dead fuel moisture [fraction]
 * @param[in] mc_100hr   100-hour dead fuel moisture [fraction]
 * @param[in] mc_herba   Live herbaceous fuel moisture [fraction]
 * @param[in] mc_woody   Live woody fuel moisture [fraction]
 * @param[in] transfer_lo Live herbaceous moisture at and below which the whole
 *                        herbaceous load transfers to dead [fraction]
 *                        (erf.fire.behave.dynamic_transfer_lo)
 * @param[in] transfer_hi Live herbaceous moisture at and above which none
 *                        transfers [fraction] (erf.fire.behave.dynamic_transfer_hi)
 * @param[in] use_wind_limit  Apply the wind limit (erf.fire.use_wind_limit)
 * @param[in] wind_limit_mode Which limit (erf.fire.wind_limit), fire_wind_limit::*
 * @param[in] net_load_mode   behave_net_load::weighted (default): the net load of
 *                        each category is the surface-area-weighted mean of its
 *                        classes, (w_n)_i = sum_j f_ij (w_n)_ij, and the heat sink
 *                        rho_b sum_i f_i sum_j f_ij eps_ij Q_ig,ij, Rothermel (1972)
 *                        eqs. 53-57 (Albini 1976, Andrews 2018). behave_net_load::sum:
 *                        the plain sum of the class loads with one eps_h(sigma_dead)
 *                        and a load-weighted Q_ig, this code's form until October
 *                        2026, which carried 3.3x FM10's and 5.2x FM13's net load
 *                        (3.0x and 4.3x their zero-wind rate, the sink changing too) at
 *                        zero wind (erf.fire.behave.net_load = "sum")
 *
 * @return BehaveState with all precomputed coefficients
 *
 * SAV (Surface Area to Volume) values:
 *   Dead 1-hr:      fp.sigma_d1       [1/ft]  (from FuelModelParams)
 *   Dead 10-hr:     109.0             [1/ft]  (fixed standard value)
 *   Dead 100-hr:    30.0              [1/ft]  (fixed standard value)
 *   Live herba:     fp.sigma_lh       [1/ft]  (from FuelModelParams)
 *   Live woody:     fp.sigma_lw       [1/ft]  (from FuelModelParams)
 *   Dead herba (transferred): fp.sigma_lh [1/ft] (same as live herba)
 *
 * The dead herbaceous class takes the 1-hour moisture mc_1hr (Scott and
 * Burgan 2005, p. 10). It carried a fixed 0.15 before, which put fuels with a
 * dead moisture of extinction of 0.15 near extinction once most of the
 * herbaceous load had cured.
 *
 * Reference: Andrews (2018) GTR-RMRS-371 pp. 16–38.
 */
/// How BEHAVE forms the net fuel load of a category, erf.fire.behave.net_load
namespace behave_net_load {
    constexpr int weighted = 0;   ///< "weighted": Rothermel's f_ij-weighted loads and heat sink (default)
    constexpr int sum      = 1;   ///< "sum": the plain sum of the classes (before 2026-10)
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
BehaveState compute_behave_state(
    const FuelModelParams& fp,
    amrex::Real mc_1hr,
    amrex::Real mc_10hr,
    amrex::Real mc_100hr,
    amrex::Real mc_herba,
    amrex::Real mc_woody,
    amrex::Real transfer_lo,
    amrex::Real transfer_hi,
    bool use_wind_limit = true,
    int wind_limit_mode = fire_wind_limit::rothermel,
    int net_load_mode = behave_net_load::weighted)
{
    BehaveState bs;

    using amrex::Real;

    // Standard constants (Rothermel 1972, Andrews 2018)
    const Real S_T = 0.0555;   // Total mineral content
    const Real S_e = 0.010;    // Effective mineral content
    const Real FT_MIN_TO_M_S = 0.00508;

    // Fixed SAV values for standard fuel classes [1/ft]
    const Real sigma_d10 = 109.0;
    const Real sigma_d100 = 30.0;

    // =====================================================================
    // Step 1: Dynamic load transfer (live herbaceous to dead)
    // =====================================================================
    // The whole live herbaceous load transfers at and below transfer_lo, none
    // at and above transfer_hi, and a linear share between.
    // Reference: WRF-SFIRE fire_ros_behave, lines ~384–396

    const Real T_herba = behave_herb_transfer_fraction(mc_herba, transfer_lo, transfer_hi);
    Real fuelload_herba_dead = T_herba * fp.w_lh;                 // [lb/ft²]
    Real fuelload_herba_live = fp.w_lh - fuelload_herba_dead;

    // =====================================================================
    // Step 2: Area-weighted SAV for dead and live
    // =====================================================================
    // A_k = (sigma_k / rho_p) * fuelload_k  (surface area per cell per class)
    // f_k = A_k / A_total                    (weighting factor within category)
    // savr = f_dead * savr_dead + f_live * savr_live

    Real A_d1 = (fp.sigma_d1 / fp.rho_p) * fp.w_d1;
    Real A_d10 = (sigma_d10 / fp.rho_p) * fp.w_d10;
    Real A_d100 = (sigma_d100 / fp.rho_p) * fp.w_d100;
    Real A_dh = (fp.sigma_lh / fp.rho_p) * fuelload_herba_dead;
    Real A_d_total = A_d1 + A_d10 + A_d100 + A_dh;

    Real A_lh = (fp.sigma_lh / fp.rho_p) * fuelload_herba_live;
    Real A_lw = (fp.sigma_lw / fp.rho_p) * fp.w_lw;
    Real A_l_total = A_lh + A_lw;

    // Weighting factors. The totals are clamped BEFORE the select: at -O3
    // clang evaluates the division of the unselected arm anyway, and 0/0 for
    // a fuel with no live (or no dead) load trips amrex.fpe_trap_invalid.
    // The clamp is the identity for every normal total.
    constexpr Real tiny = std::numeric_limits<Real>::min();
    const Real A_d_safe = amrex::max(A_d_total, tiny);
    const Real A_l_safe = amrex::max(A_l_total, tiny);
    Real f_d1 = (A_d_total > 1.0e-12) ? A_d1 / A_d_safe : 0.0;
    Real f_d10 = (A_d_total > 1.0e-12) ? A_d10 / A_d_safe : 0.0;
    Real f_d100 = (A_d_total > 1.0e-12) ? A_d100 / A_d_safe : 0.0;
    Real f_dh = (A_d_total > 1.0e-12) ? A_dh / A_d_safe : 0.0;
    Real f_lh = (A_l_total > 1.0e-12) ? A_lh / A_l_safe : 0.0;
    Real f_lw = (A_l_total > 1.0e-12) ? A_lw / A_l_safe : 0.0;

    // Area-weighted SAV
    Real sigma_dead = f_d1 * fp.sigma_d1 + f_d10 * sigma_d10 + f_d100 * sigma_d100 + f_dh * fp.sigma_lh;
    Real sigma_live = f_lh * fp.sigma_lh + f_lw * fp.sigma_lw;

    // Total fuels
    Real w_dead = fp.w_d1 + fp.w_d10 + fp.w_d100 + fuelload_herba_dead;
    Real w_live = fuelload_herba_live + fp.w_lw;
    Real w_0 = w_dead + w_live;

    const Real w_0_safe = amrex::max(w_0, tiny);
    Real f_dead = (w_0 > 1.0e-12) ? w_dead / w_0_safe : 0.0;
    Real f_live = (w_0 > 1.0e-12) ? w_live / w_0_safe : 0.0;

    // =====================================================================
    // Step 3: Live moisture of extinction
    // =====================================================================
    // Dead/live loading ratio W' of Rothermel (1972) Eq. 88 (Albini 1976,
    // Andrews 2018 Eq. 59), evaluated with the category-mean SAVs rather
    // than per class. Reference: WRF-SFIRE fire_ros_behave, lines ~421-430

    Real fuelmce = fp.Mx;  // Dead fuel moisture of extinction (from FuelModelParams)

    // Effective heating numbers: exp(-138/sigma) for the dead category and
    // exp(-500/sigma) for the live one (Rothermel Eq. 88; the two exponents
    // differ because live fuel needs a thicker heated layer to ignite)
    Real exp_dead = w_dead * std::exp(-138.0 / amrex::max(static_cast<amrex::Real>(sigma_dead), static_cast<amrex::Real>(1.0)));
    Real exp_live = w_live * std::exp(-500.0 / amrex::max(static_cast<amrex::Real>(sigma_live), static_cast<amrex::Real>(1.0)));

    Real W_prima = (exp_live > 1.0e-12) ? (exp_dead / amrex::max(exp_live, tiny)) : 0.0;

    // Weighted dead fuel moisture; the dead herbaceous class takes the 1-h moisture
    Real fmc_dead = f_d1 * mc_1hr + f_d10 * mc_10hr + f_d100 * mc_100hr + f_dh * mc_1hr;
    Real M_prima_f = fmc_dead;

    // Live moisture of extinction from Rothermel live fuel path
    Real fuelmce_live = 2.9 * W_prima * (1.0 - M_prima_f / fuelmce) - 0.226;
    fuelmce_live = amrex::max(fuelmce_live, fuelmce);  // Ensure ≥ dead fuel Mx

    // =====================================================================
    // Step 4: Per-class moisture damping
    // =====================================================================
    // Rothermel Eq. 29: η_M = max(0, 1 - 2.59*r + 5.11*r² - 3.52*r³)
    //   where r = min(1, M / M_x)
    // Reference: WRF-SFIRE lines ~431–445

    fmc_dead = f_d1 * mc_1hr + f_d10 * mc_10hr + f_d100 * mc_100hr + f_dh * mc_1hr;
    Real fmc_live = f_lh * mc_herba + f_lw * mc_woody;

    Real r_dead = amrex::min(static_cast<amrex::Real>(1.0), static_cast<amrex::Real>(fmc_dead / fuelmce));
    Real r_live = amrex::min(static_cast<amrex::Real>(1.0), static_cast<amrex::Real>(fmc_live / fuelmce_live));

    // Moisture damping coefficients
    auto etam_poly = [](Real r) -> Real {
        return amrex::max(0.0, 1.0 - 2.59*r + 5.11*r*r - 3.52*r*r*r);
    };
    bs.etam_dead = etam_poly(r_dead);
    bs.etam_live = etam_poly(r_live);

    // Mineral damping (Rothermel Eq. 30)
    bs.etas_dead = 0.174 * std::pow(S_e, -0.19);
    bs.etas_live = 0.174 * std::pow(S_e, -0.19);

    // =====================================================================
    // Step 5: Net fuel loads and heat content
    // =====================================================================
    // Rothermel (1972) eq. 53 with Albini's (1976) revision (Andrews 2018,
    // eq. 61): the net load of a category is (w_n)_i = sum_j g_ij (w_n)_ij,
    // where g_ij is the surface-area fraction f_ij summed over the classes
    // of the same size-class group (boundaries 1200, 192, 96, 48 and 16
    // ft^-1), so classes that share a size class are added, not averaged,
    // and g_ij = f_ij where every class is alone in its group (the Anderson
    // 1-h, 10-h and 100-h classes). The plain sum of the classes, kept as
    // erf.fire.behave.net_load = "sum", overstates the load that drives the
    // reaction intensity by the inverse of the 1-h class's area share: 1.8x
    // for FM2, 3.3x for FM10 and 5.2x for FM13 (the zero-wind rate 1.45x,
    // 3.0x and 4.3x, since the heat sink changes with it).
    const bool weighted = (net_load_mode == behave_net_load::weighted);
    if (weighted) {
        auto size_class = [](Real s) -> int {
            return (s >= 1200.0) ? 0 : (s >= 192.0) ? 1 : (s >= 96.0) ? 2 : (s >= 48.0) ? 3 : (s >= 16.0) ? 4 : 5;
        };
        const int c_d1 = size_class(fp.sigma_d1), c_d10 = size_class(sigma_d10),
                  c_d100 = size_class(sigma_d100), c_dh = size_class(fp.sigma_lh);
        auto g_dead = [&](int c) -> Real {
            return (c_d1 == c ? f_d1 : 0.0) + (c_d10 == c ? f_d10 : 0.0)
                 + (c_d100 == c ? f_d100 : 0.0) + (c_dh == c ? f_dh : 0.0);
        };
        const int c_lh = size_class(fp.sigma_lh), c_lw = size_class(fp.sigma_lw);
        auto g_live = [&](int c) -> Real { return (c_lh == c ? f_lh : 0.0) + (c_lw == c ? f_lw : 0.0); };
        bs.wn_dead = (g_dead(c_d1) * fp.w_d1 + g_dead(c_d10) * fp.w_d10 + g_dead(c_d100) * fp.w_d100
                      + g_dead(c_dh) * fuelload_herba_dead) * (1.0 - S_T);
        bs.wn_live = (g_live(c_lh) * fuelload_herba_live + g_live(c_lw) * fp.w_lw) * (1.0 - S_T);
    } else {
        bs.wn_dead = w_dead * (1.0 - S_T);
        bs.wn_live = w_live * (1.0 - S_T);
    }

    bs.fuelheat_dead = fp.heat_content;  // Dead heat content per Anderson
    bs.fuelheat_live = fp.heat_content;  // Live heat content assumed same

    // =====================================================================
    // Step 6: Bulk density and packing ratio
    // =====================================================================

    Real rho_b = w_0 / fp.delta;  // Bulk density [lb/ft³]
    Real beta = rho_b / fp.rho_p;  // Packing ratio

    // =====================================================================
    // Step 7: Reaction velocity parameters
    // =====================================================================
    // Use dead fuel SAV as characteristic value (weighted across dead classes)
    Real sigma = sigma_dead;
    sigma = amrex::max(static_cast<amrex::Real>(sigma), static_cast<amrex::Real>(100.0));  // Minimum SAV guard

    Real beta_op = 3.348 * std::pow(sigma, -0.8189);
    Real sigma_1p5 = std::pow(sigma, 1.5);
    Real Gamma_max = sigma_1p5 / (495.0 + 0.0594 * sigma_1p5);
    Real A = 133.0 * std::pow(sigma, -0.7913);
    Real beta_ratio = beta / beta_op;
    Real Gamma_prime = Gamma_max * std::pow(beta_ratio, A) * std::exp(A * (1.0 - beta_ratio));

    bs.gamma = Gamma_prime;

    // =====================================================================
    // Step 8: Propagating flux ratio and heat of preignition
    // =====================================================================
    // Rothermel Eq. 42 and Eq. 12

    bs.xifr = std::exp((0.792 + 0.681 * std::sqrt(sigma)) * (beta + 0.1))
              / (192.0 + 0.2595 * sigma);

    // Weighted moisture for heat of preignition
    Real M_weighted = f_dead * fmc_dead + f_live * fmc_live;
    Real Q_ig = 250.0 + 1116.0 * M_weighted;

    // =====================================================================
    // Step 9: Heat sink
    // =====================================================================
    // Rothermel (1972) eqs. 55-57: rho_b sum_i f_i sum_j f_ij eps_ij Q_ig,ij
    // with eps_ij = exp(-138/sigma_ij), Q_ig,ij = 250 + 1116 M_ij and the
    // category weights f_i the surface-area shares. The "sum" form keeps the
    // one eps_h(sigma_dead) and the load-weighted Q_ig of the earlier code.
    if (weighted) {
        const Real tiny_sav = 1.0;
        const Real eps_d1   = std::exp(-138.0 / amrex::max(fp.sigma_d1, tiny_sav));
        const Real eps_d10  = std::exp(-138.0 / sigma_d10);
        const Real eps_d100 = std::exp(-138.0 / sigma_d100);
        const Real eps_lh   = std::exp(-138.0 / amrex::max(fp.sigma_lh, tiny_sav));
        const Real eps_lw   = std::exp(-138.0 / amrex::max(fp.sigma_lw, tiny_sav));
        const Real sink_dead = f_d1   * eps_d1   * (250.0 + 1116.0 * mc_1hr)
                             + f_d10  * eps_d10  * (250.0 + 1116.0 * mc_10hr)
                             + f_d100 * eps_d100 * (250.0 + 1116.0 * mc_100hr)
                             + f_dh   * eps_lh   * (250.0 + 1116.0 * mc_1hr);
        const Real sink_live = f_lh * eps_lh * (250.0 + 1116.0 * mc_herba)
                             + f_lw * eps_lw * (250.0 + 1116.0 * mc_woody);
        const Real A_T = amrex::max(A_d_total + A_l_total, tiny);
        const Real f_dead_A = (A_d_total + A_l_total > 1.0e-12) ? A_d_total / A_T : 1.0;
        bs.heat_sink = rho_b * (f_dead_A * sink_dead + (1.0 - f_dead_A) * sink_live);
    } else {
        Real eps_h = std::exp(-138.0 / sigma);
        bs.heat_sink = rho_b * eps_h * Q_ig;
    }

    // =====================================================================
    // Step 10: Reaction intensity and no-wind ROS
    // =====================================================================
    // I_R = Γ * (w_n,dead * heat_dead * η_M,dead * η_s,dead +
    //            w_n,live * heat_live * η_M,live * η_s,live)
    // R_0 = (I_R * ξ) / heat_sink

    Real I_R = bs.gamma * (bs.wn_dead * bs.fuelheat_dead * bs.etam_dead * bs.etas_dead
                          + bs.wn_live * bs.fuelheat_live * bs.etam_live * bs.etas_live);
    I_R = amrex::max(static_cast<amrex::Real>(I_R), static_cast<amrex::Real>(0.01));

    Real R0_ft_min = (I_R * bs.xifr) / amrex::max(static_cast<amrex::Real>(bs.heat_sink), static_cast<amrex::Real>(1.0));

    // =====================================================================
    // Step 11: Wind and slope factor coefficients
    // =====================================================================
    // Identical to single-class Rothermel path (Eqs. 47–51)

    bs.r_0 = R0_ft_min * FT_MIN_TO_M_S;  // Convert ft/min → m/s

    bs.C = 7.47 * std::exp(-0.133 * std::pow(sigma, 0.55));
    bs.B = 0.02526 * std::pow(sigma, 0.54);
    Real E = 0.715 * std::exp(-3.59e-4 * sigma);

    bs.beta_ratio_E = std::pow(beta_ratio, -E);
    bs.phi_s_const = 5.275 * std::pow(beta, -0.3);

    // Wind limit (Rothermel's 0.9 I_R, or the fuel-class rule), none when
    // erf.fire.use_wind_limit = false
    bs.U_max_ftmin = rothermel_wind_cap_ftmin(sigma, I_R, use_wind_limit, wind_limit_mode);
    bs.wind_conv = 196.85;  // m/s → ft/min

    return bs;
}

/**
 * @brief Compute BEHAVE ROS for a single cell (GPU device function).
 *
 * Applies wind and slope factors to the precomputed no-wind ROS.
 * The wind and slope factor formulae are identical to the single-class
 * Rothermel path — only the precomputed coefficients and R_0 differ.
 *
 * @param[in] ux_eff  Effective wind u-component [m/s]
 * @param[in] uy_eff  Effective wind v-component [m/s]
 * @param[in] sx      Slope dz/dx [dimensionless]
 * @param[in] sy      Slope dz/dy [dimensionless]
 * @param[in] bs      BehaveState with precomputed coefficients
 *
 * @return Rate of spread [m/s]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real behave_ros_cell(
    amrex::Real ux_eff, amrex::Real uy_eff,
    amrex::Real sx, amrex::Real sy,
    const BehaveState& bs) noexcept
{
    using amrex::Real;

    // Effective wind speed [m/s]
    Real U_eff = std::sqrt(ux_eff*ux_eff + uy_eff*uy_eff);

    // Wind factor: φ_w = C * U_eff^B * (β/β_opt)^E
    // Convert to ft/min for Rothermel calculation
    Real U_eff_ftmin = U_eff * bs.wind_conv;

    // Apply MEWS cap
    U_eff_ftmin = amrex::min(U_eff_ftmin, bs.U_max_ftmin);

    // Slope magnitude and squared tangent
    Real tan_slope_sq = sx*sx + sy*sy;

    // Cap at tan²(80°) ≈ 32.16
    tan_slope_sq = amrex::min(static_cast<amrex::Real>(tan_slope_sq), static_cast<amrex::Real>(32.0));

    // Slope factor (Rothermel Eq. 51)
    Real phi_s = bs.phi_s_const * tan_slope_sq;

    // Wind factor
    Real phi_w = bs.C * std::pow(U_eff_ftmin, bs.B) * bs.beta_ratio_E;

    // Rate of spread: R = R0 * (1 + φ_w + φ_s)
    Real ROS = bs.r_0 * (1.0 + phi_w + phi_s);

    return amrex::max(static_cast<amrex::Real>(ROS), static_cast<amrex::Real>(0.0));
}

/**
 * @brief Fill fire_ros MultiFab using BEHAVE multi-class Rothermel model.
 *
 * @param[out] fire_ros        ROS field [m/s] on fire grid
 * @param[in]  fire_wind_eff   Effective wind (u,v) [m/s]
 * @param[in]  fire_slopes     Terrain slopes (dz/dx, dz/dy)
 * @param[in]  bs_default      Default BehaveState for uniform moisture
 * @param[in]  transfer_lo     Live herbaceous transfer window, lower edge [fraction]
 * @param[in]  transfer_hi     Live herbaceous transfer window, upper edge [fraction]
 * @param[in]  fire_fuel_mc    Per-cell fuel moisture (5 components, optional)
 * @param[in]  moisture_dynamic Whether per-cell moisture should be used
 * @param[in]  use_wind_limit  Wind limit of the per-cell states (erf.fire.use_wind_limit)
 * @param[in]  wind_limit_mode Which limit (erf.fire.wind_limit), fire_wind_limit::*
 * @param[in]  net_load_mode   behave_net_load::weighted or ::sum (erf.fire.behave.net_load)
 */
inline void fill_behave_ros(
    amrex::MultiFab& fire_ros,
    const amrex::MultiFab& fire_wind_eff,
    const amrex::MultiFab& fire_slopes,
    const FuelModelParams& fp,
    const BehaveState& bs_default,
    amrex::Real transfer_lo,
    amrex::Real transfer_hi,
    const amrex::MultiFab* fire_fuel_mc = nullptr,
    bool moisture_dynamic = false,
    bool use_wind_limit = true,
    int wind_limit_mode = fire_wind_limit::rothermel,
    int net_load_mode = behave_net_load::weighted)
{
    // When moisture_dynamic and fire_fuel_mc is provided, compute per-cell
    // BehaveState using per-cell moisture. Otherwise use bs_default for all cells.

    if (!moisture_dynamic || fire_fuel_mc == nullptr) {
        // Uniform moisture path: use bs_default for all cells
        for (amrex::MFIter mfi(fire_ros, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            auto ros = fire_ros.array(mfi);
            auto wind = fire_wind_eff.const_array(mfi);
            auto slopes = fire_slopes.const_array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (const amrex::IntVect& iv) {
                int i = iv[0], j = iv[1];
                ros(i, j, 0) = behave_ros_cell(wind(i,j,0,0), wind(i,j,0,1),
                                               slopes(i,j,0,0), slopes(i,j,0,1),
                                               bs_default);
            });
        }
    } else {
        // Per-cell moisture path: build the BehaveState inside the kernel from
        // that cell's moisture.
        //
        // This previously ran compute_behave_state in a LoopOnCpu and stashed the
        // result in a scratch MultiFab. That was wrong twice over. The host loop
        // dereferenced device memory, which faults under CUDA. And the packing
        // step assumed a BehaveState could be written contiguously through
        // &bs_arr(i,j,k,0), but Array4 stores components plane-by-plane rather
        // than interleaved per cell: component c sits numPts() Reals away, not
        // one. Writing sizeof(BehaveState) bytes at that address therefore ran
        // off the end of component 0 and overwrote other cells' data, and the
        // matching read returned a struct assembled from unrelated cells.
        //
        // compute_behave_state is pure arithmetic (exp/pow/sqrt), so evaluating
        // it per cell on the device removes the scratch fab, the host pass and
        // the aliasing bug together.
        for (amrex::MFIter mfi(fire_ros, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            auto ros = fire_ros.array(mfi);
            auto wind = fire_wind_eff.const_array(mfi);
            auto slopes = fire_slopes.const_array(mfi);
            auto mc = fire_fuel_mc->const_array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (const amrex::IntVect& iv) {
                int i = iv[0], j = iv[1];

                const BehaveState bs = compute_behave_state(fp,
                    mc(i,j,0,0), mc(i,j,0,1), mc(i,j,0,2),
                    mc(i,j,0,3), mc(i,j,0,4),
                    transfer_lo, transfer_hi, use_wind_limit, wind_limit_mode, net_load_mode);

                ros(i, j, 0) = behave_ros_cell(wind(i,j,0,0), wind(i,j,0,1),
                                               slopes(i,j,0,0), slopes(i,j,0,1),
                                               bs);
            });
        }
    }
}

#endif  // ERF_BEHAVE_MODEL_H
