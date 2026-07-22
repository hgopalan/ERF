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
                           const amrex::MultiFab&    H_road_atm,
                           const amrex::MultiFab&    H_wallroof_atm,
                           const amrex::Geometry&    geom,
                           bool                      ucm_debug,
                           int                       lev)
{
    using namespace amrex;

    // Duplicate-write guard
    if (step == m_last_write_step) {
        if (ParallelDescriptor::IOProcessor()) {
            Print() << "[UCM][2.6-followup][UCMAtmPlotfile::write] (skipped, already written at step " << step << ")\n";
        }
        return;
    }
    m_last_write_step = step;

    // Check that all required fields are present
    if (!f_urb_atm.ok() || !H_bldg_mean_atm.ok() || !H_bldg_std_atm.ok() ||
        !lambda_p_atm.ok() || !lambda_f_atm.ok() || !H_atm.ok() ||
        !H_road_atm.ok() || !H_wallroof_atm.ok()) {
        if (ParallelDescriptor::IOProcessor()) {
            Print() << "[UCM][2.6-followup][UCMAtmPlotfile::write] ERROR: One or more input MultiFabs is invalid\n";
        }
        return;
    }

    std::string plotfile_name = get_plotfile_name(step);

    // Extract 2D slab (k=klo only) from the 3D ATM geometry/MFs
    const int klo = geom.Domain().smallEnd(2);
    
    // Build slab domain: same x,y extent, z limited to klo only
    Box slab_domain = geom.Domain();
    slab_domain.setSmall(2, klo);
    slab_domain.setBig(2, klo);

    // Build slab BoxArray by restricting each input box to k=klo
    BoxArray slab_ba = f_urb_atm.boxArray();
    BoxList bl;
    for (int i = 0; i < slab_ba.size(); ++i) {
        Box b = slab_ba[i];
        b.setSmall(2, klo);
        b.setBig(2, klo);
        bl.push_back(b);
    }
    slab_ba = BoxArray(std::move(bl));

    // Build slab Geometry: same x,y extent as ATM, z limited to one cell
    Real dz = geom.CellSize(2);
    RealBox slab_rb({geom.ProbLo(0), geom.ProbLo(1), geom.ProbLo(2) + klo*dz},
                    {geom.ProbHi(0), geom.ProbHi(1), geom.ProbLo(2) + (klo+1)*dz});
    Geometry slab_geom(slab_domain, slab_rb, geom.Coord(), geom.isPeriodic());

    // Build 8-component slab MultiFab (Phase 2.6: increased from 6 to 8) and copy each input
    MultiFab slab_mf(slab_ba, f_urb_atm.DistributionMap(), 8, 0);

    // Component numbering
    static const int comp_f_urb = 0;
    static const int comp_H_bldg_mean = 1;
    static const int comp_H_bldg_std = 2;
    static const int comp_lambda_p = 3;
    static const int comp_lambda_f = 4;
    static const int comp_H_atm = 5;
    static const int comp_H_road_atm = 6;      // Phase 2.6
    static const int comp_H_wallroof_atm = 7;  // Phase 2.6

    // Copy fields into components
    MultiFab::Copy(slab_mf, f_urb_atm,        0, comp_f_urb,        1, 0);
    MultiFab::Copy(slab_mf, H_bldg_mean_atm,  0, comp_H_bldg_mean,  1, 0);
    MultiFab::Copy(slab_mf, H_bldg_std_atm,   0, comp_H_bldg_std,   1, 0);
    MultiFab::Copy(slab_mf, lambda_p_atm,     0, comp_lambda_p,     1, 0);
    MultiFab::Copy(slab_mf, lambda_f_atm,     0, comp_lambda_f,     1, 0);
    MultiFab::Copy(slab_mf, H_atm,            0, comp_H_atm,        1, 0);
    MultiFab::Copy(slab_mf, H_road_atm,       0, comp_H_road_atm,   1, 0);  // Phase 2.6
    MultiFab::Copy(slab_mf, H_wallroof_atm,   0, comp_H_wallroof_atm, 1, 0);  // Phase 2.6

    // Build component names vector (Phase 2.6: expanded to 8)
    Vector<std::string> varnames(8);
    varnames[comp_f_urb]        = "f_urb";
    varnames[comp_H_bldg_mean]  = "H_bldg_mean";
    varnames[comp_H_bldg_std]   = "H_bldg_std";
    varnames[comp_lambda_p]     = "lambda_p";
    varnames[comp_lambda_f]     = "lambda_f";
    varnames[comp_H_atm]        = "H_atm";
    varnames[comp_H_road_atm]   = "H_road_atm";       // Phase 2.6
    varnames[comp_H_wallroof_atm] = "H_wallroof_atm"; // Phase 2.6

    // Write plotfile using WriteSingleLevelPlotfile with slab geometry
    WriteSingleLevelPlotfile(plotfile_name,
                             slab_mf,
                             varnames,
                             slab_geom,
                             time,
                             step);

    // Debug trace
    if (ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.6-followup][UCMAtmPlotfile::write]\n";
        Print() << "  step=" << step << " time=" << time << " s\n";
        Print() << "  plotfile: " << plotfile_name << "/  (directory, 2D slab nz=1)\n";
        Print() << "  ncomp=8 (f_urb, H_bldg_mean, H_bldg_std, lambda_p, lambda_f, H_atm, H_road_atm, H_wallroof_atm)\n";
    }
}
