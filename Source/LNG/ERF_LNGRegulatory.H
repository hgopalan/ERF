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
 *      conc_1h_max_kg_m3, n_cells_exceed
 *
 * References:
 *   NFPA 59A (2023). Standard for the Production, Storage, and
 *     Handling of LNG.
 *   49 CFR Part 193 — LNG facilities, exclusion zones.
 *   ERF_DustNAAQSOutput.H (pattern source).
 *
 * MPI Rules applied (see LNG_MPI_SKILLS.md):
 *   Rule B1: all MultiFab::sum/max BEFORE IOProcessor() guard
 *   Rule B4: FillBoundary passes geom_lng.periodicity()
 *   Rule D1: geom_lng passed explicitly to functions needing FillBoundary
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
    constexpr double NFPA59A_HALF_LFL    = 0.025;    ///< 1/2 LFL exclusion zone [vol/vol]
    constexpr double AVERAGING_PERIOD_S  = 3600.0;   ///< 1-hour running average [s]
    constexpr double RHO_AIR_STD         = 1.225;    ///< kg/m^3
    constexpr double MOL_WEIGHT_AIR      = 28.97;    ///< g/mol
}

/**
 * @brief Update 1-hour running exponential moving average of conc_sfc.
 * @param[in,out] lng_conc_1h_avg  Running average [kg/m^3]
 * @param[in]     lng_conc_sfc     Current surface concentration [kg/m^3]
 * @param[in]     geom_lng         LNG grid geometry (needed for FillBoundary)
 * @param[in]     dt               Timestep [s]
 * @param[in]     averaging_period Averaging window [s] (default 3600 s)
 *
 * C_avg(t) = C_avg(t-dt) * (T-dt)/T + C_now * dt/T, T=3600 s
 * MPI-safe: ParallelFor kernel + FillBoundary with periodicity (Rule B4).
 * geom_lng required — MultiFab does not store its own geometry (Rule D1).
 */
void update_lng_1h_average(amrex::MultiFab& lng_conc_1h_avg,
                            const amrex::MultiFab& lng_conc_sfc,
                            const amrex::Geometry& geom_lng,
                            amrex::Real dt,
                            amrex::Real averaging_period = LNGRegulatoryConst::AVERAGING_PERIOD_S);

/**
 * @brief Compute exceedance flag: 1 where 1h-average >= threshold, else 0.
 * @param[out] lng_exceed_flag    Exceedance mask [0/1]
 * @param[in]  lng_conc_1h_avg   1-hour averaged concentration [kg/m^3]
 * @param[in]  geom_lng          LNG grid geometry (needed for FillBoundary)
 * @param[in]  rho_vapor         LNG vapor density [kg/m^3]
 * @param[in]  mol_weight_LNG    LNG molar mass [g/mol]
 * @param[in]  threshold_vol_frac Exceedance threshold [vol/vol]
 *
 * Converts vol/vol threshold to kg/m^3: conc_threshold = threshold * rho_vapor.
 * MPI-safe: FillBoundary with periodicity (Rule B4).
 * geom_lng required — MultiFab does not store its own geometry (Rule D1).
 */
void compute_lng_exceedance(amrex::MultiFab& lng_exceed_flag,
                             const amrex::MultiFab& lng_conc_1h_avg,
                             const amrex::Geometry& geom_lng,
                             amrex::Real rho_vapor,
                             amrex::Real mol_weight_LNG,
                             amrex::Real threshold_vol_frac);

/**
 * @brief Estimate NFPA 59A exclusion zone radius [m].
 * @param[in] lng_exceed_flag  Exceedance mask [0/1]
 * @param[in] geom_lng         LNG grid geometry
 * @param[in] pool_cx          Pool centre x [m]
 * @param[in] pool_cy          Pool centre y [m]
 * @return Maximum radius from pool centre where exceed_flag > 0.5 [m]
 *
 * MPI-safe: LoopOnCpu local max followed by ReduceRealMax across all ranks
 * (Rule B1 — reduction before any IOProcessor guard).
 */
amrex::Real compute_exclusion_zone_radius(
    const amrex::MultiFab& lng_exceed_flag,
    const amrex::Geometry& geom_lng,
    amrex::Real pool_cx,
    amrex::Real pool_cy);

/**
 * @brief Write CSV header for regulatory output.
 * IOProcessor-only — no MultiFab collectives inside, guard is safe here.
 */
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

/**
 * @brief Append one row to regulatory CSV.
 *
 * Rule B1: all MPI-collective reductions (max, sum) BEFORE IOProcessor guard.
 * IOProcessor guard wraps only the file write.
 */
inline void append_lng_regulatory_row(
    int step, double time_s,
    const std::string& filename,
    amrex::Real exclusion_radius_m,
    const amrex::MultiFab& lng_conc_1h_avg,
    const amrex::MultiFab& lng_exceed_flag)
{
    // ── ALL MPI-collective reductions first (every rank participates) ─────
    amrex::Real conc_1h_max = lng_conc_1h_avg.max(0);  // MPI_Allreduce
    amrex::Real n_exceed    = lng_exceed_flag.sum(0);   // MPI_Allreduce

    // ── File write: IOProcessor only ─────────────────────────────────────
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    std::ofstream out(filename, std::ios::app);
    if (!out.is_open()) {
        amrex::Warning("[LNG WARNING] Could not open " + filename);
        return;
    }
    out << step << ","
        << std::scientific << std::setprecision(6) << time_s             << ","
        << std::scientific << std::setprecision(6) << exclusion_radius_m << ","
        << std::scientific << std::setprecision(6) << conc_1h_max        << ","
        << static_cast<long>(n_exceed) << "\n";
}

#endif /* ERF_USE_LNG */
#endif /* ERF_LNG_REGULATORY_H */