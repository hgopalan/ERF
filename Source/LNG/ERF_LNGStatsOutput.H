/**
 * @file ERF_LNGStatsOutput.H
 * @brief LNG CSV diagnostics output
 * @details
 * Header-only functions for writing LNG time-series diagnostics to CSV.
 *
 * CRITICAL MPI RULE: MultiFab::sum(), max(), min() are MPI-collective — ALL
 * ranks must call them. The IOProcessor guard must come AFTER all reductions,
 * not before. Placing the guard before reductions causes a deadlock on >1 rank
 * because rank 0 enters MPI_Allreduce while other ranks have already returned.
 *
 * @see Source/Dust/ERF_DustStatsOutput.H — Dust module stats pattern (reference)
 */

#ifndef ERF_LNG_STATS_OUTPUT_H
#define ERF_LNG_STATS_OUTPUT_H

#ifdef ERF_USE_LNG

#include <fstream>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <iomanip>

/**
 * @brief Write CSV header for LNG diagnostics
 * @param[in] filename    CSV file path (e.g., "lng_diag.csv")
 */
inline void write_lng_stats_header(const std::string& filename)
{
    // Header write is not collective — IOProcessor guard is safe here
    if (!amrex::ParallelDescriptor::IOProcessor()) return;

    std::ofstream out(filename, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        amrex::Warning("[LNG WARNING] Could not open " + filename + " for writing");
        return;
    }

    out << "step,time_s,pool_cells,pool_area_m2,pool_mass_kg,total_vapor_mass_kg,"
        << "evap_flux_max_kg_m2_s,vapor_conc_max_kg_m3,lfl_area_m2,ufl_area_m2\n";
    out.close();
}

/**
 * @brief Append one row to LNG diagnostics CSV (Phase 2+)
 *
 * All MultiFab reductions (sum/max) are called on ALL ranks before the
 * IOProcessor guard. Only the file write is guarded.
 */
inline void append_lng_stats_phase2(int                           step,
                                    double                        time_s,
                                    const std::string&            filename,
                                    const amrex::MultiFab*        pool_depth_mf,
                                    const amrex::MultiFab*        pool_mask_mf,
                                    const amrex::MultiFab*        evap_flux_mf,
                                    const amrex::MultiFab*        conc_mf,
                                    const amrex::Geometry&        geom_lng,
                                    amrex::Real                   rho_LNG,
                                    amrex::Real                   lfl_area = 0.0,
                                    amrex::Real                   ufl_area = 0.0)
{
    // ── ALL MPI-collective reductions first (every rank participates) ─────
    double pool_mass_kg          = 0.0;
    double pool_area_m2          = 0.0;
    int    pool_cells            = 0;
    double evap_flux_max_kg_m2_s = 0.0;
    double vapor_conc_max_kg_m3  = 0.0;

    if (pool_depth_mf) {
        const auto& dx      = geom_lng.CellSize();
        amrex::Real cell_area = dx[0] * dx[1];
        amrex::Real depth_sum = pool_depth_mf->sum(0);  // MPI-collective
        pool_mass_kg = static_cast<double>(depth_sum * rho_LNG * cell_area);
    }

    if (pool_mask_mf) {
        const auto& dx       = geom_lng.CellSize();
        amrex::Real cell_area = dx[0] * dx[1];
        amrex::Real mask_sum  = pool_mask_mf->sum(0);   // MPI-collective
        pool_cells   = static_cast<int>(mask_sum);
        pool_area_m2 = static_cast<double>(mask_sum * cell_area);
    }

    if (evap_flux_mf)
        evap_flux_max_kg_m2_s = static_cast<double>(evap_flux_mf->max(0));  // MPI-collective

    if (conc_mf)
        vapor_conc_max_kg_m3 = static_cast<double>(conc_mf->max(0));        // MPI-collective

    // ── File write: IOProcessor only ─────────────────────────────────────
    if (!amrex::ParallelDescriptor::IOProcessor()) return;

    std::ofstream out(filename, std::ios::app);
    if (!out.is_open()) {
        amrex::Warning("[LNG WARNING] Could not open " + filename + " for appending");
        return;
    }

    out << step << ","
        << std::scientific << std::setprecision(6) << time_s         << ","
        << pool_cells                                                  << ","
        << std::scientific << std::setprecision(6) << pool_area_m2    << ","
        << std::scientific << std::setprecision(6) << pool_mass_kg    << ","
        << std::scientific << std::setprecision(6) << 0.0             << ","
        << std::scientific << std::setprecision(6) << evap_flux_max_kg_m2_s << ","
        << std::scientific << std::setprecision(6) << vapor_conc_max_kg_m3  << ","
        << std::scientific << std::setprecision(6) << lfl_area        << ","
        << std::scientific << std::setprecision(6) << ufl_area        << "\n";
    out.close();
}

/**
 * @brief Append one row to LNG diagnostics CSV (Phase 1 stub)
 *
 * No MultiFab reductions — all values are zero. IOProcessor guard is safe here.
 */
inline void append_lng_stats(int                       step,
                              double                    time_s,
                              const std::string&        filename,
                              const amrex::MultiFab*    evap_flux_mf,
                              const amrex::MultiFab*    temp_mf,
                              const amrex::MultiFab*    conc_mf)
{
    // Phase 1: all zeros, no reductions needed — IOProcessor guard is safe
    if (!amrex::ParallelDescriptor::IOProcessor()) return;

    std::ofstream out(filename, std::ios::app);
    if (!out.is_open()) {
        amrex::Warning("[LNG WARNING] Could not open " + filename + " for appending");
        return;
    }

    out << step << ","
        << std::scientific << std::setprecision(6) << time_s << ","
        << 0   << ","
        << std::scientific << std::setprecision(6) << 0.0 << ","
        << std::scientific << std::setprecision(6) << 0.0 << ","
        << std::scientific << std::setprecision(6) << 0.0 << ","
        << std::scientific << std::setprecision(6) << 0.0 << ","
        << std::scientific << std::setprecision(6) << 0.0 << ","
        << std::scientific << std::setprecision(6) << 0.0 << ","
        << std::scientific << std::setprecision(6) << 0.0 << "\n";
    out.close();
}

#endif /* ERF_USE_LNG */

#endif /* ERF_LNG_STATS_OUTPUT_H */