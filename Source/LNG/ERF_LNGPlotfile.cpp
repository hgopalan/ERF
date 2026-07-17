/**
 * @file ERF_LNGPlotfile.cpp
 * @brief LNG plotfile output implementation (Phase 6)
 * @details
 * Implements WriteLNGPlotfile() following the MPI-safe pattern from
 * Source/Dust/ERF_DustPlotfile.cpp. All operations follow the strict 5-step
 * MPI pattern documented in LNG_MPI_SKILLS.md Rule B1:
 * 1. IOProcessor creates directories; Barrier
 * 2. ALL ranks call VisMF::Write (MPI-collective)
 * 3. IOProcessor writes Header and JSON metadata
 * 4. Barrier
 *
 * Pattern: Source/Dust/ERF_DustPlotfile.cpp
 */

#ifdef ERF_USE_LNG

#include "ERF_LNGPlotfile.H"
#include "ERF_LNGPlotfileCatalog.H"
#include "ERF_LNGLayer.H"
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_Print.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Utility.H>
#include <fstream>
#include <iomanip>

/**
 * @brief Write LNG metadata JSON sidecar
 * @param[in] plotfilename    Full path to plotfile directory
 * @param[in] time            Simulation time [s]
 * @param[in] step            Timestep number
 * @param[in] grid_ratio      LNG grid refinement ratio vs ATM
 * @param[in] n_vars          Number of output variables (17)
 * @details
 * IOProcessor-only — no MPI collectives inside, guard is safe here.
 * Creates LNGMetadata.json with format version, time, step, grid_ratio, and n_variables.
 */
static void write_lng_metadata_json(const std::string& plotfilename,
                                     double time, int step,
                                     int grid_ratio, int n_vars)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) return;

    const std::string filename = plotfilename + "/LNGMetadata.json";
    std::ofstream out(filename, std::ios::out | std::ios::trunc);
    if (!out.good()) amrex::FileOpenFailed(filename);

    out << "{\n"
        << "  \"format_version\": 1,\n"
        << "  \"time\": " << std::fixed << std::setprecision(15) << time << ",\n"
        << "  \"step\": " << step << ",\n"
        << "  \"grid_ratio\": " << grid_ratio << ",\n"
        << "  \"n_variables\": " << n_vars << "\n"
        << "}\n";

    if (!out.good()) amrex::FileOpenFailed(filename);
}

void WriteLNGPlotfile(const std::string& plotfile_prefix,
                       const LNGLayer& lng_layer,
                       double time, int step)
{
    const LNGGrid& lg     = lng_layer.get_lng_grid();
    int ncomp             = lng_plotfile_ncomp();
    auto varnames         = lng_plotfile_var_names();

    // Assemble all fields into one MultiFab. Null fields are zeroed.
    amrex::MultiFab mf(lg.ba, lg.dm, ncomp, 0);
    mf.setVal(0.0);

    auto copy_if = [&](const amrex::MultiFab* src, int dst_comp) {
        if (src) amrex::MultiFab::Copy(mf, *src, 0, dst_comp, 1, 0);
    };

    // Copy all 17 components in order (must match ERF_LNGPlotfileCatalog.H)
    copy_if(lng_layer.get_pool_depth(),  0);
    copy_if(lng_layer.get_pool_mask(),   1);
    copy_if(lng_layer.get_evap_flux(),   2);
    copy_if(lng_layer.get_latent_flux(), 3);
    copy_if(lng_layer.get_vapor_conc(),  4);
    copy_if(lng_layer.get_ustar(),       5);
    copy_if(lng_layer.get_tsfc(),        6);
    copy_if(lng_layer.get_pblh(),        7);
    copy_if(lng_layer.get_conc_sfc(),    8);
    copy_if(lng_layer.get_lfl_mask(),    9);
    copy_if(lng_layer.get_ufl_mask(),    10);
    
    // Wind field: extract u and v components from 2-component MultiFab
    if (lng_layer.get_wind_ref()) {
        amrex::MultiFab::Copy(mf, *lng_layer.get_wind_ref(), 0, 11, 1, 0);
        amrex::MultiFab::Copy(mf, *lng_layer.get_wind_ref(), 1, 12, 1, 0);
    }
    
    copy_if(lng_layer.get_gc_h(),        13);
    copy_if(lng_layer.get_gc_u(),        14);
    copy_if(lng_layer.get_gc_v(),        15);
    copy_if(lng_layer.get_gc_ri_flag(),  16);
    copy_if(lng_layer.get_conc_1h_avg(), 17);
    copy_if(lng_layer.get_exceed_flag(), 18);

    std::string plotfilename = amrex::Concatenate(plotfile_prefix, step, 5);

    // ── Step 1: IOProcessor creates directories; Barrier before collective write ─────
    if (amrex::ParallelDescriptor::IOProcessor()) {
        if (!amrex::UtilCreateDirectory(plotfilename, 0755))
            amrex::CreateDirectoryFailed(plotfilename);
        const std::string level_dir = plotfilename + "/Level_0";
        if (!amrex::UtilCreateDirectory(level_dir, 0755))
            amrex::CreateDirectoryFailed(level_dir);
    }
    amrex::ParallelDescriptor::Barrier();

    // ── Step 2: ALL ranks write owned fabs (MPI-collective) ───────────────────────
    amrex::VisMF::Write(mf, plotfilename + "/Level_0/Cell");

    // ── Step 3: IOProcessor writes AMReX Header ──────────────────────────────────
    if (amrex::ParallelDescriptor::IOProcessor()) {
        const std::string header_path = plotfilename + "/Header";
        std::ofstream hfile(header_path.c_str(),
                             std::ofstream::out | std::ofstream::trunc |
                             std::ofstream::binary);
        if (!hfile.good()) amrex::FileOpenFailed(header_path);

        amrex::Vector<amrex::BoxArray>  ba_vec   = {lg.ba};
        //amrex::Vector<std::string>      var_vec  = varnames;
        auto varnames_std = lng_plotfile_var_names();
        amrex::Vector<std::string> var_vec(varnames_std.begin(), varnames_std.end());
        amrex::Vector<amrex::Geometry>  geom_vec = {lg.geom};
        amrex::Vector<int>              steps    = {step};
        amrex::Vector<amrex::IntVect>   rr       = {};

        amrex::WriteGenericPlotfileHeader(hfile, 1, ba_vec, var_vec,
                                          geom_vec, time, steps, rr);
        hfile << "Level_0/Cell\n";

        if (!hfile.good()) amrex::FileOpenFailed(header_path);
        amrex::Print() << "[LNG] Writing LNG plotfile " << plotfilename << "\n";
    }
    amrex::ParallelDescriptor::Barrier();

    // ── Step 4: IOProcessor writes JSON metadata sidecar ──────────────────────────
    write_lng_metadata_json(plotfilename, time, step, lg.grid_ratio, ncomp);
}

#endif /* ERF_USE_LNG */
