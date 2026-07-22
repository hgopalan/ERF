/**
 * @file ERF_UCMDiagnostics.cpp
 * @brief Implementation of diagnostics CSV writer for UCM statistics
 *
 * References:
 *  - Source/Dust/ERF_DustDiagnostics.cpp
 */

#include <UrbanCanopy/ERF_UCMDiagnostics.H>
#include <AMReX_Print.H>
#include <AMReX_ParallelDescriptor.H>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cmath>

UCMDiagnostics::UCMDiagnostics(const UCMParams& params, int lev)
    : m_params(params), m_lev(lev)
{
    // Build full path to diagnostics file
    m_diag_file = params.ucm_diag_file;
    if (m_diag_file.empty()) {
        m_diag_file = "ucm_diag.dat";
    }
    // TODO: In full ERF integration, prepend erf.plot_file_base if available

    // Initialize file on IO rank
    if (amrex::ParallelDescriptor::IOProcessor()) {
        initialize_file();
    }
}

UCMDiagnostics::~UCMDiagnostics() = default;

void UCMDiagnostics::initialize_file()
{
    // Check if file exists and is non-empty
    std::ifstream ifs(m_diag_file.c_str());
    bool file_exists = ifs.good();
    ifs.close();

    // Open file for append; if new, write header
    std::ofstream ofs(m_diag_file.c_str(), std::ios::app);

    if (!file_exists || ofs.tellp() == 0) {
        // Write header
        ofs << "step,time_s,T_skin_roof_max,T_skin_wall_max,T_skin_road_max,";
        ofs << "T_canyon_max,H_sensible_max,H_sensible_sum,LE_latent_max,";
        ofs << "H_road_max,H_wall_max,H_roof_max,AH_max\n";
    }

    ofs.close();
}

void UCMDiagnostics::write_row(int nstep, amrex::Real time,
                               amrex::Real T_roof_max, amrex::Real T_wall_max, amrex::Real T_road_max,
                               amrex::Real T_canyon_max, amrex::Real H_max, amrex::Real H_sum,
                               amrex::Real LE_max,
                               amrex::Real H_road_max, amrex::Real H_wall_max, 
                               amrex::Real H_roof_max, amrex::Real AH_max)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) return;

    std::ofstream ofs(m_diag_file.c_str(), std::ios::app);
    ofs << std::setprecision(10);
    ofs << nstep << ","
        << time << ","
        << T_roof_max << ","
        << T_wall_max << ","
        << T_road_max << ","
        << T_canyon_max << ","
        << H_max << ","
        << H_sum << ","
        << LE_max << ","
        << H_road_max << ","
        << H_wall_max << ","
        << H_roof_max << ","
        << AH_max << "\n";
    ofs.close();
}

void UCMDiagnostics::append(const UCMFields& fields, int nstep, amrex::Real time, int /*lev*/)
{
    // Duplicate-write guard
    if (nstep == m_last_write_step) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.4][UCMDiagnostics::append] (skipped, already written at step "
                           << nstep << ")\n";
        }
        return;
    }
    m_last_write_step = nstep;

    // Check that required fields exist
    if (!fields.T_skin_roof || !fields.T_skin_wall || !fields.T_skin_road ||
        !fields.T_canyon_air || !fields.H_sensible || !fields.LE_latent ||
        !fields.is_urban || !fields.H_road || !fields.H_wall || !fields.H_roof || !fields.AH)
    {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.3][UCMDiagnostics::append] ERROR: One or more required fields is nullptr\n";
        }
        return;
    }

    // NOTE: These reductions ignore the is_urban mask (Phase 1.4). Non-urban cells
    // are expected to hold neutral defaults (fluxes = 0, temps = initial values),
    // so max/sum are still meaningful. A masked reduction can be added later.
    const int  comp  = 0;
    const bool local = false; // perform MPI reduction inside MultiFab::max/sum

    // Phase 2.2 diagnostics
    amrex::Real T_roof_max   = fields.T_skin_roof->max(comp, 0, local);
    amrex::Real T_wall_max   = fields.T_skin_wall->max(comp, 0, local);
    amrex::Real T_road_max   = fields.T_skin_road->max(comp, 0, local);
    amrex::Real T_canyon_max = fields.T_canyon_air->max(comp, 0, local);
    amrex::Real H_max        = fields.H_sensible->max(comp, 0, local);
    amrex::Real H_sum        = fields.H_sensible->sum(comp, local);
    amrex::Real LE_max       = fields.LE_latent->max(comp, 0, local);

    // Phase 2.3: Facet-split fluxes and AH (computed OUTSIDE IOProcessor guard)
    amrex::Real H_road_max   = fields.H_road->max(comp, 0, local);
    amrex::Real H_wall_max   = fields.H_wall->max(comp, 0, local);
    amrex::Real H_roof_max   = fields.H_roof->max(comp, 0, local);
    amrex::Real AH_max       = fields.AH->max(comp, 0, local);

    // Write to file on IO rank
    if (amrex::ParallelDescriptor::IOProcessor()) {
        write_row(nstep, time,
                  T_roof_max, T_wall_max, T_road_max,
                  T_canyon_max, H_max, H_sum, LE_max,
                  H_road_max, H_wall_max, H_roof_max, AH_max);

        // Debug trace
        amrex::Print() << "[UCM][2.3][UCMDiagnostics::append]\n";
        amrex::Print() << "  step=" << nstep << " time=" << time << "\n";
        amrex::Print() << "  T_skin_roof_max=" << T_roof_max
                       << " T_canyon_max="     << T_canyon_max << "\n";
        amrex::Print() << "  H_sensible_max="  << H_max
                       << " H_sensible_sum="   << H_sum << "\n";
        amrex::Print() << "  H_road_max=" << H_road_max
                       << " H_wall_max=" << H_wall_max
                       << " H_roof_max=" << H_roof_max
                       << " AH_max=" << AH_max << "\n";

        // Phase 2.3 sum-invariant check: is H_sensible == H_road + H_wall + H_roof ?
        const amrex::Real facet_raw_sum = H_road_max + H_wall_max + H_roof_max;
        const amrex::Real residual_raw  = H_max - facet_raw_sum;
        amrex::Print() << "  [sum_check] H_sensible=" << H_max
                       << "  raw_sum(H_road+H_wall+H_roof)=" << facet_raw_sum
                       << "  residual=" << residual_raw
                       << (std::abs(residual_raw) < 1.0e-6 * std::max(1.0, std::abs(H_max))
                               ? "  [OK]" : "  [FAIL]")
                       << "\n";
    }
}