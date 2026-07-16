/**
 * @file ERF_LNGReceptorOutput.H
 * @brief LNG receptor point sampling (Phase 6)
 * @details
 * Header-only functions for sampling LNG vapor concentration at user-defined
 * receptor points (x,y locations) and writing CSV time series.
 * Implements the MPI-safe pattern documented in LNG_MPI_SKILLS.md Rule B1:
 * all MPI-collective reductions BEFORE the IOProcessor guard.
 *
 * Pattern: Source/Dust/ERF_DustMSHAOutput.H
 */

#ifndef ERF_LNG_RECEPTOR_OUTPUT_H
#define ERF_LNG_RECEPTOR_OUTPUT_H

#ifdef ERF_USE_LNG

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParallelReduce.H>
#include <fstream>
#include <iomanip>

/**
 * @brief Write CSV header for a receptor point file
 * @param[in] filename       CSV file path (e.g., "lng_receptor_center.csv")
 * @param[in] receptor_name  Human-readable receptor name (e.g., "center")
 * @param[in] rx            Receptor x-coordinate [m]
 * @param[in] ry            Receptor y-coordinate [m]
 * @details
 * IOProcessor-only (no MPI collectives). Creates a new CSV file with header line
 * containing receptor metadata and column names. Called once at initialization.
 */
inline void write_lng_receptor_header(const std::string& filename,
                                       const std::string& receptor_name,
                                       double rx, double ry)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) return;

    std::ofstream out(filename, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        amrex::Warning("[LNG WARNING] Could not create " + filename);
        return;
    }

    out << "# Receptor: " << receptor_name
        << "  x=" << rx << "  y=" << ry << "\n";
    out << "step,time_s,conc_sfc_kg_m3,vol_fraction,lfl_flag\n";
}

/**
 * @brief Append one row per timestep for a receptor point
 * @param[in] step                Timestep number
 * @param[in] time_s              Simulation time [s]
 * @param[in] filename            CSV file path to append to
 * @param[in] receptor_name       Human-readable receptor name
 * @param[in] rx                  Receptor x-coordinate [m]
 * @param[in] ry                  Receptor y-coordinate [m]
 * @param[in] conc_sfc_mf         LNG vapor concentration on LNG grid [kg/m^3]
 * @param[in] geom_lng            LNG grid geometry
 * @param[in] rho_vapor           Vapor density reference [kg/m^3]
 * @param[in] mol_weight_LNG      LNG molecular weight [g/mol]
 * @param[in] lfl_threshold       Lower flammability limit [vol/vol]
 * @details
 * MPI rule (LNG_MPI_SKILLS.md B1): all reductions BEFORE IOProcessor guard.
 * Samples concentration at nearest grid point containing (rx, ry).
 * Converts to volume fraction using vapor density and molecular weight.
 * Sets lfl_flag=1 if volume fraction exceeds LFL threshold.
 * Appends one CSV row with: step, time_s, conc_sfc, vol_fraction, lfl_flag.
 */
inline void append_receptor_sample(int step, double time_s,
                                    const std::string& filename,
                                    const std::string& receptor_name,
                                    double rx, double ry,
                                    const amrex::MultiFab& conc_sfc_mf,
                                    const amrex::Geometry& geom_lng,
                                    double rho_vapor,
                                    double mol_weight_LNG,
                                    double lfl_threshold)
{
    // Find nearest grid cell index — MPI-safe: each rank checks its owned boxes
    const auto& dx  = geom_lng.CellSize();
    const auto& plo = geom_lng.ProbLo();
    int i_rec = static_cast<int>((rx - plo[0]) / dx[0]);
    int j_rec = static_cast<int>((ry - plo[1]) / dx[1]);
    i_rec = std::max(0, std::min(i_rec, geom_lng.Domain().bigEnd(0)));
    j_rec = std::max(0, std::min(j_rec, geom_lng.Domain().bigEnd(1)));

    // Extract value at receptor — ReduceSum over all ranks to get correct value
    // ── MPI-collective reductions BEFORE IOProcessor guard ────────────────────
    amrex::Real conc_val = 0.0;
    for (amrex::MFIter mfi(conc_sfc_mf); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        if (bx.contains(amrex::IntVect(i_rec, j_rec, 0))) {
            auto arr = conc_sfc_mf.array(mfi);
            conc_val = arr(i_rec, j_rec, 0);
        }
    }
    // MPI-collective: sum across ranks (only the owning rank has non-zero)
    amrex::ParallelDescriptor::ReduceRealSum(conc_val);

    // All reductions done — IOProcessor guard safe here (Rule B1)
    if (!amrex::ParallelDescriptor::IOProcessor()) return;

    double vol_frac = (rho_vapor > 1e-10)
        ? (conc_val / rho_vapor) * (28.97 / mol_weight_LNG) : 0.0;
    int lfl_flag = (vol_frac >= lfl_threshold) ? 1 : 0;

    std::ofstream out(filename, std::ios::app);
    if (!out.is_open()) {
        amrex::Warning("[LNG WARNING] Could not open " + filename);
        return;
    }

    out << step << ","
        << std::scientific << std::setprecision(6) << time_s << ","
        << std::scientific << std::setprecision(6) << conc_val << ","
        << std::scientific << std::setprecision(6) << vol_frac << ","
        << lfl_flag << "\n";
}

#endif /* ERF_USE_LNG */

#endif /* ERF_LNG_RECEPTOR_OUTPUT_H */
