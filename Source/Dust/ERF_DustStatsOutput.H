/**
 * @file ERF_DustStatsOutput.H
 * @brief CSV time series output for dust diagnostics.
 *
 * Pattern: ERF_FireStatsOutput.H in Source/Fire/:
 *   https://github.com/hgopalan/ERF/blob/ERF-Fire/Source/Fire/ERF_FireStatsOutput.H
 *
 * Writes per-step scalar diagnostics to dust_diag.dat:
 *   step, time[s], emission_total[kg/s], deposition_total[kg],
 *   ustar_max[m/s], flux_max[kg/m^2/s], conc_sfc_max[kg/m^3]
 * The totals are integrals over the dust grid: the flux summed over every bin
 * and every cell times the cell area, the deposited mass per area summed over
 * the cells times the cell area. (Until October 2026 the "total" was the
 * bin-0 flux summed over the cells with no area, 4e5x off on the 375 m
 * canonical grid, and the deposition column was a cell sum of kg/m2.)
 *
 * Header written on first call (file does not exist).
 * Subsequent calls append one row.
 * Only IOProcessor writes; all ranks compute reductions.
 */
#ifndef ERF_DUST_STATS_OUTPUT_H
#define ERF_DUST_STATS_OUTPUT_H
#ifdef ERF_USE_DUST
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <fstream>
#include <string>

inline void write_dust_stats_header(const std::string& filename)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    {   // a restart keeps the earlier leg; the header is written once per file
        std::ifstream probe(filename);
        if (probe.good() && probe.peek() != std::ifstream::traits_type::eof()) return;
    }
    std::ofstream out(filename, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        amrex::Print() << "[DUST] WARNING: Cannot open dust_diag file: "
                       << filename << "\n";
        return;
    }
    out << "# ERF-Dust diagnostic time series\n"
        << "step,time_s,emission_total_kg_s,deposition_total_kg,"
        << "ustar_max_m_s,flux_max_kg_m2_s,conc_sfc_max_kg_m3\n";
}

inline void append_dust_stats(int step, amrex::Real time_s,
                               const std::string& filename,
                               const amrex::MultiFab* emission_flux,
                               const amrex::MultiFab* deposition_rate,
                               const amrex::MultiFab* ustar_in,
                               const amrex::MultiFab* conc_sfc,
                               amrex::Real cell_area)
{
    // Compute domain scalars on all ranks then reduce.
    amrex::Real em_sum = 0.0;
    if (emission_flux) {
        for (int b = 0; b < emission_flux->nComp(); ++b) em_sum += emission_flux->sum(b);
        em_sum *= cell_area;                                   // [kg/s]
    }
    amrex::Real dep_sum = deposition_rate ? deposition_rate->sum(0) * cell_area : 0.0;   // [kg]
    amrex::Real us_max  = ustar_in        ? ustar_in->max(0)        : 0.0;
    amrex::Real fl_max  = emission_flux   ? emission_flux->max(0)   : 0.0;
    amrex::Real cs_max  = conc_sfc        ? conc_sfc->max(0)        : 0.0;

    if (!amrex::ParallelDescriptor::IOProcessor()) return;

    bool file_exists = std::ifstream(filename).good();
    std::ofstream out(filename, std::ios::app);
    if (!out.is_open()) {
        amrex::Print() << "[DUST] WARNING: Cannot append to dust_diag file: "
                       << filename << "\n";
        return;
    }
    if (!file_exists) {
        out << "# ERF-Dust diagnostic time series\n"
            << "step,time_s,emission_total_kg_s,deposition_total_kg,"
            << "ustar_max_m_s,flux_max_kg_m2_s,conc_sfc_max_kg_m3\n";
    }
    out << step    << "," << time_s  << ","
        << em_sum  << "," << dep_sum << ","
        << us_max  << "," << fl_max  << ","
        << cs_max  << "\n";
}
#endif
#endif // ERF_DUST_STATS_OUTPUT_H
