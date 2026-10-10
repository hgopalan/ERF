#ifndef ERF_CROWN_FIRE_H
#define ERF_CROWN_FIRE_H

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>
#include <ERF_FireGrid.H>
#include <cmath>
#include <utility>

/**
 * @file ERF_CrownFire.H
 * @brief Crown fire initiation and active crown fire rate of spread
 *
 * Implements multiple crown fire models from published literature:
 *   1. Van Wagner (1977) criterion for surface-to-crown fire transition
 *   2. Cruz et al. (2005) model for active crown fire rate of spread
 *   3. Rothermel (1991) crown ROS as multiple of surface ROS
 *   4. Van Wagner proxy model for crown ROS
 *   5. Van Wagner passive-to-active blending
 *   6. Crown Fraction Burned (CFB) diagnostic
 *   7. The extension of the crown rate into the band the front can reach
 *      (extend_crown_ros_to_front), without which the level-set paths never
 *      moved the front at the crown rate
 *
 * The per-cell crown state (activation, rate, canopy load) lives in
 * FireLayer::apply_crown_fire_ros() and compute_heat_flux_and_diagnostics();
 * the two MultiFab kernels this header carried until 2026-10
 * (update_crown_fire_status, fill_crown_heat_flux) had no caller and a
 * different physics from the live path, and were removed.
 *
 * Assumptions:
 *   - Canopy parameters are uniform across the domain (erf.fire.crown.*);
 *     there is no canopy raster
 *   - Cruz et al. (2005) was calibrated for North American conifer stands and is
 *     not appropriate for grass or shrub fuel models FM1–FM3, FM5–FM6
 *   - Crown fire cells do not return to surface fire state once activated
 *
 * References:
 *   Van Wagner, C.E. (1977). Conditions for the start and spread of crown fire.
 *     Canadian Journal of Forest Research, 7(1), 23–34.
 *   Cruz, M.G., Alexander, M.E. & Wakimoto, R.H. (2005). Development and testing
 *     of models for predicting crown fire rate of spread in conifer forest stands.
 *     Canadian Journal of Forest Research, 35(7), 1626–1639.
 *   Rothermel, R.C. (1991). Predicting behavior and size of crown fires in the
 *     northern Rocky Mountain conifers. USDA Forest Service INT-RP-438.
 *   Scott, J.H. & Reinhardt, E.D. (2001). Assessing crown fire potential by linking
 *     models of surface and crown fire behavior. USDA Forest Service RMRS-RP-29.
 */

/**
 * @brief Van Wagner (1977) critical fireline intensity for crown fire initiation
 *
 *   I_0 = ( 0.010 C_BH (460 + 25.9 m) )^(3/2)   [kW/m]
 *
 * with C_BH the canopy base height [m] and m the foliar moisture content in
 * percent of dry mass. This is Van Wagner's eq. 4 with his h = 460 + 25.9 m
 * (kJ/kg) and C = 0.010. Until 2026-09 the 3/2 power was missing and the
 * moisture entered as the excess over a "critical" value, which made I_0 16x
 * too low at the defaults (114 instead of 1883 kW/m for C_BH = 5 m, m = 100 %)
 * and treated dry foliage as unable to crown.
 *
 * @param[in] C_BH              Canopy base height [m], floored at 0.1 m
 * @param[in] foliar_moisture   Foliar moisture [fraction of dry mass]
 * @return Critical surface fireline intensity [kW/m]
 *
 * Reference: Van Wagner, C.E. (1977). Conditions for the start and spread of
 *   crown fire. Canadian Journal of Forest Research, 7(1), 23–34.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real van_wagner_critical_intensity(amrex::Real C_BH, amrex::Real foliar_moisture) noexcept
{
    C_BH = amrex::max(static_cast<amrex::Real>(C_BH), static_cast<amrex::Real>(0.1));
    const amrex::Real m_pct = amrex::max(static_cast<amrex::Real>(foliar_moisture), static_cast<amrex::Real>(0.0)) * 100.0;
    const amrex::Real base  = 0.010 * C_BH * (460.0 + 25.9 * m_pct);
    return base * std::sqrt(base);
}

/**
 * @brief Cruz et al. (2005) active crown fire rate of spread
 *
 * Formula: R [m/min] = 11.02 * U_10^0.9 * C_BD^0.19 * exp(-0.17 * MC10_pct)
 * Returns value in m/s (input formula is m/min, so divide by 60).
 *
 * Note: moisture_10hr is converted from fraction to percent for the formula.
 *
 * @param[in] U_10_ms       10-meter wind speed [m/s] (converted to km/h internally)
 * @param[in] C_BD          Canopy bulk density [kg/m³]
 * @param[in] moisture_10hr 10-hr dead fuel moisture [fraction, 0-1]
 * @return Crown fire rate of spread [m/s], or 0 when U_10_ms <= 0
 *
 * Reference: Cruz, M.G., Alexander, M.E. & Wakimoto, R.H. (2005). Development
 *   and testing of models for predicting crown fire rate of spread in conifer
 *   forest stands. Canadian Journal of Forest Research, 35(7), 1626–1639.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cruz_crown_ros(amrex::Real U_10_ms, amrex::Real C_BD, amrex::Real moisture_10hr) noexcept
{
    // Zero or negative wind produces zero crown ROS
    if (U_10_ms <= 0.0) {
        return 0.0;
    }

    // Convert wind from m/s to km/h
    amrex::Real U_kmh = U_10_ms * 3.6;
    U_kmh = amrex::max(static_cast<amrex::Real>(U_kmh), static_cast<amrex::Real>(1.0));  // Minimum 1 km/h (model not calibrated at calm)

    // Convert moisture from fraction to percent
    amrex::Real MC10_pct = moisture_10hr * 100.0;

    // Clamp inputs to valid ranges
    C_BD = amrex::max(static_cast<amrex::Real>(C_BD), static_cast<amrex::Real>(0.01));
    MC10_pct = amrex::max(static_cast<amrex::Real>(MC10_pct), static_cast<amrex::Real>(0.0));

    // Cruz formula: R [m/min] = 11.02 * U^0.90 * CBD^0.19 * exp(-0.17 * MC10)
    constexpr amrex::Real A_COEFF  = 11.02;   // leading coefficient
    constexpr amrex::Real W_EXP    = 0.90;    // wind exponent
    constexpr amrex::Real CBD_EXP  = 0.19;    // CBD exponent
    constexpr amrex::Real MC_COEFF = 0.17;    // moisture coefficient

    amrex::Real R_m_min = A_COEFF
                        * std::pow(U_kmh, W_EXP)
                        * std::pow(C_BD, CBD_EXP)
                        * std::exp(-MC_COEFF * MC10_pct);

    // Convert m/min to m/s
    amrex::Real R_m_s = R_m_min / 60.0;
    return amrex::max(static_cast<amrex::Real>(0.0), static_cast<amrex::Real>(R_m_s));
}

/**
 * @brief Rothermel (1991) active crown fire rate of spread
 *
 * Simple empirical formula: R_crown = 3.34 * R_surface
 *
 * @param[in] R_surface_ms  Surface fire rate of spread [m/s]
 * @return Crown fire rate of spread [m/s]
 *
 * Reference: Rothermel, R.C. (1991). Predicting behavior and size of crown fires
 *   in the northern Rocky Mountain conifers. USDA Forest Service INT-RP-438.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_rothermel_1991_crown_ros(amrex::Real R_surface_ms) noexcept
{
    constexpr amrex::Real ROTHERMEL_MULT = 3.34;
    return amrex::max(static_cast<amrex::Real>(ROTHERMEL_MULT * R_surface_ms), static_cast<amrex::Real>(0.0));
}

/**
 * @brief Van Wagner (1977) proxy active crown fire rate of spread
 *
 * Formula: R_crown = (3.0 / CBD) * moisture_factor / 60.0
 * where moisture_factor = max(0.3, min(1.0, 1.0 - (FMC - 100) / 200))
 *
 * @param[in] C_BD              Canopy bulk density [kg/m³]
 * @param[in] foliar_moisture   Foliar moisture [fraction, 0-1]
 * @return Crown fire rate of spread [m/s]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_van_wagner_proxy_ros(amrex::Real C_BD, amrex::Real foliar_moisture) noexcept
{
    // Convert moisture to percent
    amrex::Real FMC_pct = foliar_moisture * 100.0;

    // Clamp CBD
    C_BD = amrex::max(static_cast<amrex::Real>(C_BD), static_cast<amrex::Real>(0.01));

    // Base crown ROS [m/min] = 3.0 / CBD
    amrex::Real R_m_min = 3.0 / C_BD;

    // Moisture factor (reduction for higher moisture)
    amrex::Real mf = 1.0 - (FMC_pct - 100.0) / 200.0;
    mf = amrex::max(static_cast<amrex::Real>(0.3), static_cast<amrex::Real>(amrex::min(static_cast<amrex::Real>(1.0), static_cast<amrex::Real>(mf))));

    R_m_min *= mf;

    // Convert m/min to m/s
    return amrex::max(0.0, R_m_min / 60.0);
}

/**
 * @brief Van Wagner (1977) passive-to-active crown fire blending
 *
 * A continuous crowning fraction CF blends between surface and active crown ROS:
 *   CF = min((I_B / I_o)^(2/3), 1.0)
 *   R_total = R_surface * (1 - CF) + R_crown_active * CF
 *
 * @param[in] R_surface_ms       Surface fire ROS [m/s]
 * @param[in] R_crown_active_ms  Active crown fire ROS [m/s]
 * @param[in] I_B_kwm            Byram fireline intensity [kW/m]
 * @param[in] I_o_kwm            Van Wagner critical surface intensity [kW/m]
 * @return Blended ROS [m/s]
 *
 * Reference: Van Wagner, C.E. (1977). Conditions for the start and spread of
 *   crown fire. Canadian Journal of Forest Research, 7(1), 23–34.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_van_wagner_passive_blend(
    amrex::Real R_surface_ms,
    amrex::Real R_crown_active_ms,
    amrex::Real I_B_kwm,
    amrex::Real I_o_kwm) noexcept
{
    // floored threshold inside the division: a no-canopy cell has I_o = 0
    // and the early return does not stop the optimiser evaluating I_B/0
    amrex::Real ratio = amrex::max(static_cast<amrex::Real>(I_B_kwm / amrex::max(I_o_kwm, amrex::Real(1.0e-6))), static_cast<amrex::Real>(0.0));
    if (I_o_kwm < 1.0e-6 || I_B_kwm <= 0.0) return R_surface_ms;

    amrex::Real CF = amrex::min(std::pow(ratio, 2.0 / 3.0), 1.0);
    return R_surface_ms * (1.0 - CF) + amrex::max(R_crown_active_ms, R_surface_ms) * CF;
}

/**
 * @brief Crown Fraction Burned (CFB) diagnostic
 *
 * Computes the crown fraction burned, which distinguishes between passive
 * (torching) and active crown fire. Used for emissions and fire effects modeling.
 *
 * Formula: CFB = (R_total - R_surface) / (R_active_crown - R_surface)
 * Clipped to range [0, 1].
 *
 * @param[in] R_total            Total (observed or modeled) ROS [m/s]
 * @param[in] R_surface          Surface fire ROS [m/s]
 * @param[in] R_active_crown     Active crown fire ROS [m/s]
 * @return CFB value [0-1]
 *
 * Reference: Scott, J.H. & Reinhardt, E.D. (2001). Assessing crown fire potential
 *   by linking models of surface and crown fire behavior. USDA Forest Service
 *   Research Paper RMRS-RP-29.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_crown_fraction_burned(
    amrex::Real R_total,
    amrex::Real R_surface,
    amrex::Real R_active_crown) noexcept
{
    // Ensure positive rates
    R_total       = amrex::max(static_cast<amrex::Real>(R_total), static_cast<amrex::Real>(0.0));
    R_surface     = amrex::max(static_cast<amrex::Real>(R_surface), static_cast<amrex::Real>(0.0));
    R_active_crown = amrex::max(static_cast<amrex::Real>(R_active_crown), static_cast<amrex::Real>(0.0));

    // Scott & Reinhardt (2001) CFB formula, with the denominator floored
    // before the early return (the unselected x/0 is speculated)
    amrex::Real CFB = (R_total - R_surface) / amrex::max(R_active_crown - R_surface, amrex::Real(1.0e-8));

    // If active crown ROS not greater than surface, no crown fire
    if (R_active_crown <= R_surface + 1.0e-8) {
        return 0.0;
    }

    // Clamp to valid range [0, 1]
    CFB = amrex::max(static_cast<amrex::Real>(0.0), static_cast<amrex::Real>(amrex::min(static_cast<amrex::Real>(1.0), static_cast<amrex::Real>(CFB))));

    return CFB;
}

/**
 * @brief Carry the crown rate of burned cells into the unburned band the front
 *        can reach, and return the per-cell factor the directional level-set
 *        paths apply.
 *
 * apply_crown_fire_ros() writes the crown rate into fire_ros in burned cells
 * only. The isotropic level set advances each cell at its own rate, so the
 * unburned cells next to the front kept the surface rate and a crown fire
 * never moved the front on that path; the directional, hybrid and Balbi
 * paths rebuild the rate from the model inside every Runge-Kutta stage and
 * never read fire_ros at all. Only the FARSITE front_cell update, which takes
 * the larger of a front cell's own rate and its burned neighbours', saw it.
 *
 * As apply_fire_acceleration_front() does for its clock, every unburned cell
 * within n_ext = 2 + ceil(R_max dt / dx) cells of a crowned cell (the cells
 * the front can reach within the step plus the stencil's neighbours) takes the
 * largest crown rate of its burned or already reached 4-neighbours, if that
 * is above its own surface rate; the factor R_crown / R_surface (1 elsewhere)
 * is written to `factor`, which the per-stage paths multiply into the rate
 * they rebuild, the way the acceleration and suppression factors are.
 *
 * @param[in,out] ros       Rate of spread [m/s]: the crown rate in crowned
 *                          cells on entry, extended into the band on return
 * @param[in]     ros_surf  Surface rate of spread [m/s], the factor's denominator
 * @param[in]     phi       Level set (< 0 burned)
 * @param[in]     crown_ros Crown rate of spread [m/s] in crowned cells
 * @param[in]     crown_act Crown activation flag (>= 0.5 crowned)
 * @param[in]     geom      Fire-grid geometry
 * @param[in]     dt        Atmospheric step [s]
 * @param[out]    factor    ros / ros_surf per cell, 1 where nothing changed
 * @param[in]     nonburnable  Mask (> 0.5 non-burnable); a mask cell neither
 *                          takes nor passes on a crown rate, so a firebreak or
 *                          a structure stops the extension (nullptr: no mask)
 * @return the number of cells that took a crown rate from a neighbour (over ranks)
 */
inline amrex::Long extend_crown_ros_to_front(amrex::MultiFab& ros,
                                             const amrex::MultiFab& ros_surf,
                                             const amrex::MultiFab& phi,
                                             const amrex::MultiFab& crown_ros,
                                             const amrex::MultiFab& crown_act,
                                             const amrex::Geometry& geom,
                                             amrex::Real dt,
                                             amrex::MultiFab& factor,
                                             const amrex::MultiFab* nonburnable = nullptr)
{
    using namespace amrex;
    const bool use_mask = (nonburnable != nullptr);
    using namespace amrex::literals;

    const auto dx    = geom.CellSizeArray();
    const Real h     = amrex::min(dx[0], dx[1]);
    const Real R_max = amrex::max(ros.max(0), 0.0_rt);
    const int  n_ext = 2 + static_cast<int>(std::ceil(R_max * dt / h));

    // Crown rate of the crowned cells, -1 elsewhere; one ghost cell for the
    // extension passes, which read the four neighbours across box boundaries.
    const IntVect ng(1, 1, 0);
    MultiFab c_ext(ros.boxArray(), ros.DistributionMap(), 1, ng);
    MultiFab c_tmp(ros.boxArray(), ros.DistributionMap(), 1, ng);
    ReduceOps<ReduceOpMax> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;
    for (MFIter mfi(c_ext); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto ce = c_ext.array(mfi);
        auto p  = phi.const_array(mfi);
        auto cr = crown_ros.const_array(mfi);
        auto ca = crown_act.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            ce(i, j, k) = (p(i, j, k) < 0.0_rt && ca(i, j, k) >= 0.5_rt) ? cr(i, j, k) : -1.0_rt;
        });
        reduce_op.eval(bx, reduce_data, [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
            return { (p(i, j, k) < 0.0_rt && ca(i, j, k) >= 0.5_rt) ? cr(i, j, k) : -1.0_rt };
        });
    }
    Real c_any = amrex::get<0>(reduce_data.value(reduce_op));
    ParallelDescriptor::ReduceRealMax(c_any);
    factor.setVal(1.0_rt);
    if (!(c_any > 0.0_rt)) { Gpu::streamSynchronize(); return 0; }

    // Carry the crowned cells' rate into the unburned band, one cell a pass
    for (int pass = 0; pass < n_ext; ++pass) {
        fire_fill_boundary(c_ext, geom);
        for (MFIter mfi(c_tmp); mfi.isValid(); ++mfi) {
            auto src = c_ext.const_array(mfi);
            auto dst = c_tmp.array(mfi);
            auto p   = phi.const_array(mfi);
            Array4<const Real> msk;
            if (use_mask) { msk = nonburnable->const_array(mfi); }
            ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                Real c = src(i, j, k);
                const bool blocked = use_mask && msk(i, j, k) > 0.5_rt;
                if (c < 0.0_rt && p(i, j, k) >= 0.0_rt && !blocked) {
                    c = amrex::max(amrex::max(src(i-1, j, k), src(i+1, j, k)),
                                   amrex::max(src(i, j-1, k), src(i, j+1, k)));
                }
                dst(i, j, k) = c;
            });
        }
        std::swap(c_ext, c_tmp);
    }

    ReduceOps<ReduceOpSum> count_op;
    ReduceData<unsigned long long> count_data(count_op);
    using CountTuple = typename decltype(count_data)::Type;
    for (MFIter mfi(ros); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto R  = ros.array(mfi);
        auto Rs = ros_surf.const_array(mfi);
        auto ce = c_ext.const_array(mfi);
        auto p  = phi.const_array(mfi);
        auto fc = factor.array(mfi);
        count_op.eval(bx, count_data, [=] AMREX_GPU_DEVICE (int i, int j, int k) -> CountTuple {
            const Real surf = amrex::max(Rs(i, j, k), 0.0_rt);
            unsigned long long took = 0ull;
            if (p(i, j, k) >= 0.0_rt && ce(i, j, k) > surf) {
                R(i, j, k) = ce(i, j, k);
                took = 1ull;
            }
            // the factor the per-stage paths apply to the rate they rebuild;
            // divide by the floored rate before the select (a non-burnable
            // cell has surf = 0 and the unselected 0/0 would be speculated)
            const Real ratio = R(i, j, k) / amrex::max(surf, 1.0e-30_rt);
            fc(i, j, k) = (surf > 1.0e-30_rt) ? amrex::max(ratio, 1.0_rt) : 1.0_rt;
            return { took };
        });
    }
    amrex::Long n_took = static_cast<amrex::Long>(amrex::get<0>(count_data.value(count_op)));
    ParallelDescriptor::ReduceLongSum(n_took);
    // The kernels above read c_ext, which is freed on return.
    Gpu::streamSynchronize();
    return n_took;
}

#endif  // ERF_CROWN_FIRE_H