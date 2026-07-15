/**
 * @file ERF_LNGStatsOutput.H
 * @brief LNG CSV diagnostics output
 * @details
 * Header-only functions for writing LNG time-series diagnostics to CSV.
 * In Phase 1, the CSV header is written; rows are added in Phase 2+ as physics is enabled.
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
 * @details
 * Creates/truncates file and writes header line listing all diagnostic columns.
 * Only IOProcessor writes to avoid file conflicts.
 * 
 * Header format (Phase 1 columns, expanded in future phases):
 * ```
 * step,time_s,pool_cells,pool_area_m2,pool_mass_kg,total_vapor_mass_kg,evap_flux_max_kg_m2_s,vapor_conc_max_kg_m3,lfl_area_m2,ufl_area_m2
 * ```
 */
inline void write_lng_stats_header(const std::string& filename)
{
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
 * @brief Append one row to LNG diagnostics CSV
 * @param[in] step               Timestep number
 * @param[in] time_s             Simulation time [s]
 * @param[in] filename           CSV file path
 * @param[in] pool_depth_mf      Pool depth [m]
 * @param[in] pool_mask_mf       Pool mask [0/1]
 * @param[in] evap_flux_mf       Evaporation flux [kg/m^2/s]
 * @param[in] conc_mf            Concentration [kg/m^3]
 * @param[in] geom_lng           LNG geometry for cell area
 * @param[in] rho_LNG            LNG density [kg/m^3]
 * @details
 * Phase 2: computes pool mass from depth, pool cells from mask, evap flux max.
 * Rows beyond Phase 2 will expand with additional diagnostics.
 * Only IOProcessor writes; all ranks compute reductions.
 */
inline void append_lng_stats_phase2(int                           step,
                                    double                        time_s,
                                    const std::string&            filename,
                                    const amrex::MultiFab*        pool_depth_mf,
                                    const amrex::MultiFab*        pool_mask_mf,
                                    const amrex::MultiFab*        evap_flux_mf,
                                    const amrex::MultiFab*        conc_mf,
                                    const amrex::Geometry&        geom_lng,
                                    amrex::Real                   rho_LNG)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    
    // Compute Phase 2 diagnostics
    int    pool_cells = 0;
    double pool_mass_kg = 0.0;
    double pool_area_m2 = 0.0;
    double evap_flux_max_kg_m2_s = 0.0;
    double vapor_conc_max_kg_m3 = 0.0;
    
    if (pool_depth_mf) {
        const auto& dx = geom_lng.CellSize();
        amrex::Real cell_area = dx[0] * dx[1];
        
        // Compute pool mass from depth
        amrex::Real depth_sum = pool_depth_mf->sum(0);
        pool_mass_kg = depth_sum * rho_LNG * cell_area;
    }
    
    if (pool_mask_mf) {
        const auto& dx = geom_lng.CellSize();
        amrex::Real cell_area = dx[0] * dx[1];
        
        // Count pool cells and compute area
        pool_cells = (int)pool_mask_mf->sum(0);
        pool_area_m2 = pool_cells * cell_area;
    }
    
    if (evap_flux_mf) {
        evap_flux_max_kg_m2_s = evap_flux_mf->max(0);
    }
    
    if (conc_mf) {
        vapor_conc_max_kg_m3 = conc_mf->max(0);
    }
    
    std::ofstream out(filename, std::ios::app);
    if (!out.is_open()) {
        amrex::Warning("[LNG WARNING] Could not open " + filename + " for appending");
        return;
    }
    
    out << step << ","
        << std::scientific << std::setprecision(6) << time_s << ","
        << pool_cells << ","
        << std::scientific << std::setprecision(6) << pool_area_m2 << ","
        << std::scientific << std::setprecision(6) << pool_mass_kg << ","
        << std::scientific << std::setprecision(6) << 0.0 << ","  // total_vapor_mass (Phase 3+)
        << std::scientific << std::setprecision(6) << evap_flux_max_kg_m2_s << ","
        << std::scientific << std::setprecision(6) << vapor_conc_max_kg_m3 << ","
        << std::scientific << std::setprecision(6) << 0.0 << ","  // lfl_area (Phase 5+)
        << std::scientific << std::setprecision(6) << 0.0 << "\n"; // ufl_area (Phase 5+)
    out.close();
}

/**
 * @brief Append one row to LNG diagnostics CSV (Phase 1 stub version)
 * @param[in] step               Timestep number
 * @param[in] time_s             Simulation time [s]
 * @param[in] filename           CSV file path
 * @param[in] evap_flux_mf       Evaporation flux [kg/m^2/s]
 * @param[in] temp_mf            Temperature field [K]
 * @param[in] conc_mf            Concentration [kg/m^3]
 * @details
 * Phase 1: all diagnostics are zero (stubs); row is written to establish format.
 * Only IOProcessor writes; all ranks compute reductions (for future phases).
 */
inline void append_lng_stats(int                       step,
                              double                    time_s,
                              const std::string&        filename,
                              const amrex::MultiFab*    evap_flux_mf,
                              const amrex::MultiFab*    temp_mf,
                              const amrex::MultiFab*    conc_mf)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    
    // Phase 1: all zeros
    int    pool_cells             = 0;
    double pool_area_m2           = 0.0;
    double pool_mass_kg           = 0.0;
    double total_vapor_mass_kg    = 0.0;
    double evap_flux_max_kg_m2_s  = 0.0;
    double vapor_conc_max_kg_m3   = 0.0;
    double lfl_area_m2            = 0.0;
    double ufl_area_m2            = 0.0;
    
    std::ofstream out(filename, std::ios::app);
    if (!out.is_open()) {
        amrex::Warning("[LNG WARNING] Could not open " + filename + " for appending");
        return;
    }
    
    out << step << ","
        << std::scientific << std::setprecision(6) << time_s << ","
        << pool_cells << ","
        << std::scientific << std::setprecision(6) << pool_area_m2 << ","
        << std::scientific << std::setprecision(6) << pool_mass_kg << ","
        << std::scientific << std::setprecision(6) << total_vapor_mass_kg << ","
        << std::scientific << std::setprecision(6) << evap_flux_max_kg_m2_s << ","
        << std::scientific << std::setprecision(6) << vapor_conc_max_kg_m3 << ","
        << std::scientific << std::setprecision(6) << lfl_area_m2 << ","
        << std::scientific << std::setprecision(6) << ufl_area_m2 << "\n";
    out.close();
}

#endif /* ERF_USE_LNG */

#endif /* ERF_LNG_STATS_OUTPUT_H */
