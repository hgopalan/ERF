/**
 * @file ERF_UCMAtmPlotfile.cpp
 * @brief Implementation of plotfile writer for UCM ATM-grid aggregates
 *
 * References:
 *  - Source/Fire/ERF_FirePlotfile.cpp on ERF-Hazard
 *  - Source/UrbanCanopy/ERF_UCMPlotfile.cpp
 *  - AMReX_VisMF.H
 */

#include <UrbanCanopy/ERF_UCMAtmPlotfile.H>
#include <AMReX_VisMF.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_ParallelDescriptor.H>
#include <sstream>
#include <iomanip>
#include <fstream>

std::string UCMAtmPlotfile::get_plotfile_name(int step) const
{
    // Use plot_file_base if available, otherwise current directory
    std::string base_dir = ".";  // Default to current directory
    
    // Format: plt_ucm_atm_000000
    std::ostringstream oss;
    oss << base_dir << "/plt_ucm_atm_" << std::setfill('0') << std::setw(6) << step;
    return oss.str();
}

void UCMAtmPlotfile::write(int                       step,
                           amrex::Real               time,
                           const amrex::MultiFab&    f_urb_atm,
                           const amrex::MultiFab&    H_bldg_mean_atm,
                           const amrex::MultiFab&    H_bldg_std_atm,
                           const amrex::MultiFab&    lambda_p_atm,
                           const amrex::MultiFab&    lambda_f_atm,
                           const amrex::MultiFab&    H_atm,
                           const amrex::Geometry&    geom,
                           bool                      ucm_debug,
                           int                       lev)
{
    using namespace amrex;

    // Duplicate-write guard
    if (step == m_last_write_step) {
        if (ParallelDescriptor::IOProcessor()) {
            Print() << "[UCM][2.5-followup][UCMAtmPlotfile::write] (skipped, already written at step " << step << ")\n";
        }
        return;
    }
    m_last_write_step = step;

    // Check that all required fields are present
    if (!f_urb_atm.ok() || !H_bldg_mean_atm.ok() || !H_bldg_std_atm.ok() ||
        !lambda_p_atm.ok() || !lambda_f_atm.ok() || !H_atm.ok()) {
        if (ParallelDescriptor::IOProcessor()) {
            Print() << "[UCM][2.5-followup][UCMAtmPlotfile::write] ERROR: One or more input MultiFabs is invalid\n";
        }
        return;
    }

    std::string plotfile_name = get_plotfile_name(step);

    // Extract klo_atm from geometry (k-index of first ATM level)
    const int klo_atm = geom.Domain().smallEnd(2);

    // Build a 2D slab BoxArray at k=klo_atm
    BoxList bl;
    for (int i = 0; i < f_urb_atm.boxArray().size(); ++i) {
        Box b = f_urb_atm.boxArray()[i];
        b.setSmall(2, klo_atm);
        b.setBig(2,   klo_atm);
        bl.push_back(b);
    }
    BoxArray ba_slab(std::move(bl));

    // Create a 2D slab MultiFab to hold all 6 components
    amrex::MultiFab atm_plot_slab(ba_slab, f_urb_atm.DistributionMap(), 6, 0);
    atm_plot_slab.setVal(0.0);

    // Component numbering
    static const int comp_f_urb = 0;
    static const int comp_H_bldg_mean = 1;
    static const int comp_H_bldg_std = 2;
    static const int comp_lambda_p = 3;
    static const int comp_lambda_f = 4;
    static const int comp_H_atm = 5;

    // Copy fields into slab components using ParallelCopy
    atm_plot_slab.ParallelCopy(f_urb_atm,        0, comp_f_urb,        1, 0, 0);
    atm_plot_slab.ParallelCopy(H_bldg_mean_atm,  0, comp_H_bldg_mean,  1, 0, 0);
    atm_plot_slab.ParallelCopy(H_bldg_std_atm,   0, comp_H_bldg_std,   1, 0, 0);
    atm_plot_slab.ParallelCopy(lambda_p_atm,     0, comp_lambda_p,     1, 0, 0);
    atm_plot_slab.ParallelCopy(lambda_f_atm,     0, comp_lambda_f,     1, 0, 0);
    atm_plot_slab.ParallelCopy(H_atm,            0, comp_H_atm,        1, 0, 0);

    // Build component names vector
    Vector<std::string> varnames(6);
    varnames[comp_f_urb]        = "f_urb";
    varnames[comp_H_bldg_mean]  = "H_bldg_mean";
    varnames[comp_H_bldg_std]   = "H_bldg_std";
    varnames[comp_lambda_p]     = "lambda_p";
    varnames[comp_lambda_f]     = "lambda_f";
    varnames[comp_H_atm]        = "H_atm";

    // Build a 2D geometry matching the slab (same x,y domain, z thickness = 1 cell)
    // Extract the domain box from geometry, modify Z dimension
    IntVect domain_lo = geom.Domain().smallEnd();
    IntVect domain_hi = geom.Domain().bigEnd();
    domain_lo[2] = klo_atm;
    domain_hi[2] = klo_atm;  // 1-cell thickness in Z
    Box domain_slab(domain_lo, domain_hi);

    // Get coordinate arrays from original geometry (for x and y)
    const Real* prob_lo = geom.ProbLo();
    const Real* prob_hi = geom.ProbHi();
    const Real* dx      = geom.CellSize();

    // Compute new prob_hi[2] and dx[2] for 1-cell thickness
    Real prob_lo_slab[3] = {prob_lo[0], prob_lo[1], prob_lo[2]};
    Real prob_hi_slab[3] = {prob_hi[0], prob_hi[1], prob_lo[2] + dx[2]};  // 1 cell thickness

    // Create the 2D slab geometry
    Geometry geom_slab(domain_slab, RealBox(prob_lo_slab, prob_hi_slab));

    // Write plotfile using WriteSingleLevelPlotfile (handles directory, Header, Level_0/)
    amrex::WriteSingleLevelPlotfile(plotfile_name,
                                    atm_plot_slab,
                                    varnames,
                                    geom_slab,
                                    time,
                                    step);

    // Debug trace
    if (ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.5-followup][UCMAtmPlotfile::write]\n";
        Print() << "  step=" << step << " time=" << time << " s\n";
        Print() << "  plotfile: " << plotfile_name << "/  (directory)\n";
        Print() << "  ncomp=6 (f_urb, H_bldg_mean, H_bldg_std, lambda_p, lambda_f, H_atm)\n";
        Print() << "  grid (2D slab): " << domain_slab.smallEnd()[0] << "-" << domain_slab.bigEnd()[0] 
                << " x " << domain_slab.smallEnd()[1] << "-" << domain_slab.bigEnd()[1] << " x " << klo_atm << "\n";
    }
}
