/**
 * @file ERF_LNGRegulatory.H
 * @brief NFPA 59A regulatory compliance diagnostics for LNG dispersion.
 * @note Phase 7
 *
 * Implements:
 *   1. Running 1-hour time-average of lng_conc_sfc per cell
 *   2. Exceedance flag where 1h-average >= nfpa59a_exclusion_conc
 *   3. Exclusion zone radius estimate (distance from pool centre to
 *      furthest LFL/2 exceedance cell)
 *   4. CSV output: step, time_s, exclusion_zone_radius_m,
 *      conc_1h_max_kg_m3, n_cells_exceed, fence_line_conc_kg_m3
 *
 * References:
 *   NFPA 59A (2023). Standard for the Production, Storage, and
 *     Handling of LNG.
 *   49 CFR Part 193 — LNG facilities, exclusion zones.
 *   ERF_DustNAAQSOutput.H (pattern source).
 */
#ifndef ERF_LNG_REGULATORY_H
#define ERF_LNG_REGULATORY_H
#ifdef ERF_USE_LNG

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParallelDescriptor.H>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>

namespace LNGRegulatoryConst {
    constexpr double NFPA59A_HALF_LFL = 0.025; ///< 1/2 LFL exclusion zone [vol/vol]
    constexpr double AVERAGING_PERIOD_S = 3600.0; ///< 1-hour running average [s]
    constexpr double RHO_AIR_STD = 1.225;   ///< kg/m^3
    constexpr double MOL_WEIGHT_AIR = 28.97; ///< g/mol
}

/// Update 1-hour running exponential moving average of conc_sfc.
/// Pattern: Dust update_running_average (ERF_DustPM.H).
/// C_avg(t) = C_avg(t-dt) * (T-dt)/T + C_now * dt/T, T=3600 s
/// MPI-safe: ParallelFor kernel, no collective ops.
void update_lng_1h_average(amrex::MultiFab& lng_conc_1h_avg,
                            const amrex::MultiFab& lng_conc_sfc,
                            amrex::Real dt,
                            amrex::Real averaging_period = LNGRegulatoryConst::AVERAGING_PERIOD_S);

/// Compute exceedance flag: 1 where 1h-average >= threshold, else 0.
/// threshold in vol/vol; converted internally from conc [kg/m^3].
void compute_lng_exceedance(amrex::MultiFab& lng_exceed_flag,
                             const amrex::MultiFab& lng_conc_1h_avg,
                             amrex::Real rho_vapor,
                             amrex::Real mol_weight_LNG,
                             amrex::Real threshold_vol_frac);

/// Estimate NFPA 59A exclusion zone radius [m]:
/// furthest distance from (pool_cx, pool_cy) where exceed_flag > 0.5.
/// MPI-safe: ReduceReal pattern (all ranks participate before IOProcessor).
amrex::Real compute_exclusion_zone_radius(
    const amrex::MultiFab& lng_exceed_flag,
    const amrex::Geometry& geom_lng,
    amrex::Real pool_cx,
    amrex::Real pool_cy);

/// Write CSV header for regulatory output. IOProcessor-only (no collectives).
inline void write_lng_regulatory_header(const std::string& filename)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    std::ofstream out(filename, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        amrex::Warning("[LNG WARNING] Could not create " + filename);
        return;
    }
    out << "# NFPA 59A LNG Regulatory Output\n"
        << "# exclusion_threshold = 1/2 LFL = 2.5 vol%\n"
        << "# References: NFPA 59A (2023), 49 CFR Part 193\n"
        << "step,time_s,exclusion_zone_radius_m,"
        << "conc_1h_max_kg_m3,n_cells_exceed\n";
}

/// Append one row. MPI rule B1: all reductions BEFORE IOProcessor guard.
inline void append_lng_regulatory_row(
    int step, double time_s,
    const std::string& filename,
    amrex::Real exclusion_radius_m,
    const amrex::MultiFab& lng_conc_1h_avg,
    const amrex::MultiFab& lng_exceed_flag)
{
    // ── ALL MPI-collective reductions first (every rank participates) ───
    amrex::Real conc_1h_max = lng_conc_1h_avg.max(0);  // MPI_Allreduce
    amrex::Real n_exceed     = lng_exceed_flag.sum(0);  // MPI_Allreduce

    // ── File write: IOProcessor only ─────────────────────────────────────
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    std::ofstream out(filename, std::ios::app);
    if (!out.is_open()) {
        amrex::Warning("[LNG WARNING] Could not open " + filename);
        return;
    }
    out << step << ","
        << std::scientific << std::setprecision(6) << time_s << ","
        << std::scientific << std::setprecision(6) << exclusion_radius_m << ","
        << std::scientific << std::setprecision(6) << conc_1h_max << ","
        << static_cast<long>(n_exceed) << "\n";
}

#endif /* ERF_USE_LNG */
#endif /* ERF_LNG_REGULATORY_H */
