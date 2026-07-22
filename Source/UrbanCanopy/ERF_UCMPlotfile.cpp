/**
 * @file ERF_UCMPlotfile.cpp
 * @brief Implementation of plotfile writer for UCM diagnostics
 *
 * References:
 *  - Source/Dust/ERF_DustPlotfile.cpp
 *  - AMReX_VisMF.H
 */

#include <UrbanCanopy/ERF_UCMPlotfile.H>
#include <UrbanCanopy/ERF_UCMPlotfileCatalog.H>
#include <AMReX_VisMF.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_ParallelDescriptor.H>
#include <sstream>
#include <iomanip>

UCMPlotfile::UCMPlotfile(const UCMParams& params, int lev)
    : m_params(params), m_lev(lev)
{
}

UCMPlotfile::~UCMPlotfile() = default;

std::string UCMPlotfile::get_plotfile_name(int nstep) const
{
    // Use plot_file_base if available, otherwise current directory
    std::string base_dir = ".";  // Default to current directory
    // TODO: In full ERF integration, query erf.plot_file_base via ParmParse
    
    // Format: plt_ucm_000000
    std::ostringstream oss;
    oss << base_dir << "/plt_ucm_" << std::setfill('0') << std::setw(6) << nstep;
    return oss.str();
}

void UCMPlotfile::write(const UCMFields& fields, const UCMGrid& grid,
                       int nstep, amrex::Real time, bool is_final,
                       int lev)
{
    // Duplicate-write guard
    if (nstep == m_last_write_step) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.4][UCMPlotfile::write] (skipped, already written at step " << nstep << ")\n";
        }
        return;
    }
    m_last_write_step = nstep;

    // Check that all required fields are present
    if (!fields.H_bldg || !fields.W_road || !fields.W_roof ||
        !fields.albedo_roof || !fields.albedo_wall || !fields.albedo_road ||
        !fields.emissivity_roof || !fields.emissivity_wall || !fields.emissivity_road ||
        !fields.T_skin_roof || !fields.T_skin_wall || !fields.T_skin_road ||
        !fields.T_canyon_air || !fields.H_sensible || !fields.LE_latent ||
        !fields.is_urban || !fields.SVF_wall || !fields.SVF_road || !fields.SVF_roof) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.4][UCMPlotfile::write] ERROR: One or more required fields is nullptr\n";
        }
        return;
    }

    std::string plotfile_name = get_plotfile_name(nstep);

    // Create a temporary MultiFab to hold all components
    amrex::MultiFab ucm_plot(fields.H_bldg->boxArray(), fields.H_bldg->DistributionMap(),
                             UCMPlot_ncomp, 0);

    // Copy fields into components (by reference, since MultiFabs are already on UCM grid)
    amrex::MultiFab::Copy(ucm_plot, *fields.H_bldg,          0, UCMPlot_H_bldg,          1, 0);
    amrex::MultiFab::Copy(ucm_plot, *fields.W_road,          0, UCMPlot_W_road,          1, 0);
    amrex::MultiFab::Copy(ucm_plot, *fields.W_roof,          0, UCMPlot_W_roof,          1, 0);
    amrex::MultiFab::Copy(ucm_plot, *fields.albedo_roof,     0, UCMPlot_albedo_roof,     1, 0);
    amrex::MultiFab::Copy(ucm_plot, *fields.albedo_wall,     0, UCMPlot_albedo_wall,     1, 0);
    amrex::MultiFab::Copy(ucm_plot, *fields.albedo_road,     0, UCMPlot_albedo_road,     1, 0);
    amrex::MultiFab::Copy(ucm_plot, *fields.emissivity_roof, 0, UCMPlot_emissivity_roof, 1, 0);
    amrex::MultiFab::Copy(ucm_plot, *fields.emissivity_wall, 0, UCMPlot_emissivity_wall, 1, 0);
    amrex::MultiFab::Copy(ucm_plot, *fields.emissivity_road, 0, UCMPlot_emissivity_road, 1, 0);
    amrex::MultiFab::Copy(ucm_plot, *fields.T_skin_roof,     0, UCMPlot_T_skin_roof,     1, 0);
    amrex::MultiFab::Copy(ucm_plot, *fields.T_skin_wall,     0, UCMPlot_T_skin_wall,     1, 0);
    amrex::MultiFab::Copy(ucm_plot, *fields.T_skin_road,     0, UCMPlot_T_skin_road,     1, 0);
    amrex::MultiFab::Copy(ucm_plot, *fields.T_canyon_air,    0, UCMPlot_T_canyon_air,    1, 0);
    amrex::MultiFab::Copy(ucm_plot, *fields.H_sensible,      0, UCMPlot_H_sensible,      1, 0);
    amrex::MultiFab::Copy(ucm_plot, *fields.LE_latent,       0, UCMPlot_LE_latent,       1, 0);
    // is_urban is iMultiFab, need to cast to Real
    for (amrex::MFIter mfi(ucm_plot, false); mfi.isValid(); ++mfi) {
        auto dst = ucm_plot.array(mfi);
        auto src = fields.is_urban->const_array(mfi);
        amrex::ParallelFor(mfi.fabbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                dst(i, j, k, UCMPlot_is_urban) = static_cast<amrex::Real>(src(i, j, k, 0));
            });
    }

    // Phase 2.4: SVF (sky view factors) from shadowing model
    amrex::MultiFab::Copy(ucm_plot, *fields.SVF_wall,        0, UCMPlot_SVF_wall,        1, 0);
    amrex::MultiFab::Copy(ucm_plot, *fields.SVF_road,        0, UCMPlot_SVF_road,        1, 0);
    amrex::MultiFab::Copy(ucm_plot, *fields.SVF_roof,        0, UCMPlot_SVF_roof,        1, 0);

    // Build component names vector
    amrex::Vector<std::string> varnames(UCMPlot_ncomp);
    for (int i = 0; i < UCMPlot_ncomp; ++i) {
        varnames[i] = UCMPlotfileComponentName(i);
    }

    // Write plotfile using WriteSingleLevelPlotfile (handles directory, Header, Level_0/)
    amrex::WriteSingleLevelPlotfile(plotfile_name,
                                    ucm_plot,
                                    varnames,
                                    grid.geom,
                                    time,
                                    nstep);

    // Debug trace
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "[UCM][1.4][UCMPlotfile::write]\n";
        amrex::Print() << "  step=" << nstep << " time=" << time << " s\n";
        amrex::Print() << "  plotfile: " << plotfile_name << "/  (directory)\n";
        amrex::Print() << "  ncomp=" << UCMPlot_ncomp << " (";
        for (int i = 0; i < UCMPlot_ncomp; ++i) {
            amrex::Print() << varnames[i];
            if (i < UCMPlot_ncomp - 1) amrex::Print() << ", ";
        }
        amrex::Print() << ")\n";
        amrex::Print() << "  grid: " << grid.ba.size() << " boxes\n";
    }
}
