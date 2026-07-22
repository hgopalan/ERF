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

    // Create a temporary MultiFab to hold all 6 components on the ATM grid
    amrex::MultiFab atm_plot(f_urb_atm.boxArray(), f_urb_atm.DistributionMap(), 6, 0);

    // Component numbering
    static const int comp_f_urb = 0;
    static const int comp_H_bldg_mean = 1;
    static const int comp_H_bldg_std = 2;
    static const int comp_lambda_p = 3;
    static const int comp_lambda_f = 4;
    static const int comp_H_atm = 5;

    // Copy fields into components
    MultiFab::Copy(atm_plot, f_urb_atm,        0, comp_f_urb,        1, 0);
    MultiFab::Copy(atm_plot, H_bldg_mean_atm,  0, comp_H_bldg_mean,  1, 0);
    MultiFab::Copy(atm_plot, H_bldg_std_atm,   0, comp_H_bldg_std,   1, 0);
    MultiFab::Copy(atm_plot, lambda_p_atm,     0, comp_lambda_p,     1, 0);
    MultiFab::Copy(atm_plot, lambda_f_atm,     0, comp_lambda_f,     1, 0);
    MultiFab::Copy(atm_plot, H_atm,            0, comp_H_atm,        1, 0);

    // Build component names vector
    Vector<std::string> varnames(6);
    varnames[comp_f_urb]        = "f_urb";
    varnames[comp_H_bldg_mean]  = "H_bldg_mean";
    varnames[comp_H_bldg_std]   = "H_bldg_std";
    varnames[comp_lambda_p]     = "lambda_p";
    varnames[comp_lambda_f]     = "lambda_f";
    varnames[comp_H_atm]        = "H_atm";

    // Write plotfile using WriteSingleLevelPlotfile (handles directory, Header, Level_0/)
    amrex::WriteSingleLevelPlotfile(plotfile_name,
                                    atm_plot,
                                    varnames,
                                    geom,
                                    time,
                                    step);

    // Debug trace
    if (ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.5-followup][UCMAtmPlotfile::write]\n";
        Print() << "  step=" << step << " time=" << time << " s\n";
        Print() << "  plotfile: " << plotfile_name << "/  (directory)\n";
        Print() << "  ncomp=6 (f_urb, H_bldg_mean, H_bldg_std, lambda_p, lambda_f, H_atm)\n";
    }
}
