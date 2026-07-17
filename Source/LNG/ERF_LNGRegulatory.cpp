/**
 * @file ERF_LNGRegulatory.cpp
 * @brief NFPA 59A regulatory compliance diagnostics implementation
 * @note Phase 7
 */

#include "ERF_LNGRegulatory.H"

#ifdef ERF_USE_LNG

/// Update 1-hour running exponential moving average of conc_sfc.
/// C_avg(t) = C_avg(t-dt) * (T-dt)/T + C_now * dt/T, T=3600 s
void update_lng_1h_average(amrex::MultiFab& lng_conc_1h_avg,
                            const amrex::MultiFab& lng_conc_sfc,
                            amrex::Real dt,
                            amrex::Real averaging_period)
{
    const amrex::Real weight_new = dt / averaging_period;
    const amrex::Real weight_old = 1.0 - weight_new;

    for (amrex::MFIter mfi(lng_conc_1h_avg, true); mfi.isValid(); ++mfi) {
        const auto& bx = mfi.tilebox();
        auto avg_arr = lng_conc_1h_avg.array(mfi);
        auto conc_arr = lng_conc_sfc.const_array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            avg_arr(i, j, k) = weight_old * avg_arr(i, j, k) + weight_new * conc_arr(i, j, k);
        });
    }

    // FillBoundary to exchange ghost cells across ranks
    lng_conc_1h_avg.FillBoundary(lng_conc_1h_avg.nGrow() > 0
        ? amrex::IntVect::TheUnitVector()
        : amrex::IntVect::TheZeroVector());
}

/// Compute exceedance flag: 1 where 1h-average >= threshold, else 0.
/// threshold in vol/vol; converted internally from conc [kg/m^3].
void compute_lng_exceedance(amrex::MultiFab& lng_exceed_flag,
                             const amrex::MultiFab& lng_conc_1h_avg,
                             amrex::Real rho_vapor,
                             amrex::Real mol_weight_LNG,
                             amrex::Real threshold_vol_frac)
{
    // Convert threshold from vol/vol to kg/m^3
    // threshold_vol_frac * rho_air = mass fraction
    // Molar volume of LNG vapor vs air at same T,P gives concentration ratio
    // Simplified: conc_threshold [kg/m^3] = threshold_vol_frac * rho_vapor
    const amrex::Real conc_threshold = threshold_vol_frac * rho_vapor;

    for (amrex::MFIter mfi(lng_exceed_flag, true); mfi.isValid(); ++mfi) {
        const auto& bx = mfi.tilebox();
        auto flag_arr = lng_exceed_flag.array(mfi);
        auto conc_arr = lng_conc_1h_avg.const_array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            flag_arr(i, j, k) = (conc_arr(i, j, k) >= conc_threshold) ? 1.0 : 0.0;
        });
    }

    // FillBoundary to exchange ghost cells across ranks
    lng_exceed_flag.FillBoundary(lng_exceed_flag.nGrow() > 0
        ? amrex::IntVect::TheUnitVector()
        : amrex::IntVect::TheZeroVector());
}

/// Estimate NFPA 59A exclusion zone radius [m]:
/// furthest distance from (pool_cx, pool_cy) where exceed_flag > 0.5.
/// MPI-safe: LoopOnCpu local result followed by ReduceRealMax across all ranks.
amrex::Real compute_exclusion_zone_radius(
    const amrex::MultiFab& lng_exceed_flag,
    const amrex::Geometry& geom_lng,
    amrex::Real pool_cx,
    amrex::Real pool_cy)
{
    // Find max radius across all MPI ranks where exceed_flag > 0.5
    amrex::Real local_max_r = 0.0;

    for (amrex::MFIter mfi(lng_exceed_flag); mfi.isValid(); ++mfi) {
        const auto& bx = mfi.validbox();
        auto flag_arr = lng_exceed_flag.const_array(mfi);

        // CPU loop to find max radius (rank-local)
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
            if (flag_arr(i, j, k) > 0.5) {
                amrex::Real x = geom_lng.ProbLo(0) + (i + 0.5) * geom_lng.CellSize(0);
                amrex::Real y = geom_lng.ProbLo(1) + (j + 0.5) * geom_lng.CellSize(1);
                amrex::Real r = std::sqrt((x - pool_cx) * (x - pool_cx) +
                                          (y - pool_cy) * (y - pool_cy));
                local_max_r = std::max(local_max_r, r);
            }
        });
    }

    // MPI reduce — all ranks participate before any IOProcessor guard (Rule B1)
    amrex::ParallelDescriptor::ReduceRealMax(local_max_r);
    return local_max_r;
}

#endif /* ERF_USE_LNG */
