/**
 * @file ERF_UCMAllocate.cpp
 * @brief Implementation of SLUCM MultiFab allocation and initialization
 *
 * Allocates all UCMFields MultiFabs on the UCM grid with appropriate
 * ghost cells and initializes them to zero (homogeneous values set later).
 *
 * References:
 *  - Source/Dust/ERF_DustLayer.cpp
 *  - Source/LNG/ERF_LNGLayer.cpp
 */

#include <ERF_UCMAllocate.H>
#include <AMReX_Print.H>
#include <unordered_map>
#include <cstdint>

using namespace amrex;

void allocate_ucm_fields(UCMFields& fields,
                         const UCMGrid& ucm_grid,
                         const UCMParams& params,
                         int lev)
{
    const BoxArray& ba = ucm_grid.ba;
    const DistributionMapping& dm = ucm_grid.dm;
    const IntVect ngrow(1, 1, 0);  // 1 ghost in x,y; 0 in z (2D slab contract)
    const int ncomp = 1;
    const int ncomp_slab = params.slab_N_layers;  // Phase 3.5A: multi-layer slab

    // Allocate each field with debug output
    fields.H_bldg = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][allocate_ucm_fields] H_bldg: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.W_road = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][allocate_ucm_fields] W_road: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.W_roof = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][allocate_ucm_fields] W_roof: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.albedo_roof = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][allocate_ucm_fields] albedo_roof: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.albedo_wall = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][allocate_ucm_fields] albedo_wall: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.albedo_road = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][allocate_ucm_fields] albedo_road: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.emissivity_roof = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][allocate_ucm_fields] emissivity_roof: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.emissivity_wall = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][allocate_ucm_fields] emissivity_wall: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.emissivity_road = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][allocate_ucm_fields] emissivity_road: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.T_skin_roof = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][allocate_ucm_fields] T_skin_roof: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.T_skin_wall = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][allocate_ucm_fields] T_skin_wall: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.T_skin_road = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][allocate_ucm_fields] T_skin_road: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.T_canyon_air = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][allocate_ucm_fields] T_canyon_air: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    // Phase 3.5A: Allocate multi-layer slab temperature fields
    fields.T_slab_roof = std::make_unique<MultiFab>(ba, dm, ncomp_slab, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][3.5A][allocate_ucm_fields] T_slab_roof: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp_slab << "\n";
    }

    fields.T_slab_wall = std::make_unique<MultiFab>(ba, dm, ncomp_slab, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][3.5A][allocate_ucm_fields] T_slab_wall: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp_slab << "\n";
    }

    fields.T_slab_road = std::make_unique<MultiFab>(ba, dm, ncomp_slab, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][3.5A][allocate_ucm_fields] T_slab_road: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp_slab << "\n";
    }

    fields.H_sensible = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][allocate_ucm_fields] H_sensible: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.LE_latent = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][allocate_ucm_fields] LE_latent: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.is_urban = std::make_unique<iMultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][allocate_ucm_fields] is_urban (iMultiFab): "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.mat_id_roof = std::make_unique<iMultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.1][allocate_ucm_fields] mat_id_roof (iMultiFab): "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.mat_id_wall = std::make_unique<iMultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.1][allocate_ucm_fields] mat_id_wall (iMultiFab): "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.mat_id_road = std::make_unique<iMultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.1][allocate_ucm_fields] mat_id_road (iMultiFab): "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    // Phase 2.2: thermal and aerodynamic properties
    fields.k_therm_roof = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.2][allocate_ucm_fields] k_therm_roof: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.k_therm_wall = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.2][allocate_ucm_fields] k_therm_wall: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.k_therm_road = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.2][allocate_ucm_fields] k_therm_road: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.rho_cp_roof = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.2][allocate_ucm_fields] rho_cp_roof: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.rho_cp_wall = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.2][allocate_ucm_fields] rho_cp_wall: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.rho_cp_road = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.2][allocate_ucm_fields] rho_cp_road: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.slab_L_roof = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.2][allocate_ucm_fields] slab_L_roof: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.slab_L_wall = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.2][allocate_ucm_fields] slab_L_wall: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.slab_L_road = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.2][allocate_ucm_fields] slab_L_road: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.z0_ucm = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.2][allocate_ucm_fields] z0_ucm: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.d_disp_ucm = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.2][allocate_ucm_fields] d_disp_ucm: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    // Phase 2.3: Facet-split fluxes and anthropogenic heat
    fields.H_road = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.3][allocate_ucm_fields] H_road: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.H_wall = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.3][allocate_ucm_fields] H_wall: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.H_roof = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.3][allocate_ucm_fields] H_roof: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    // Phase 3.5A-hotfix2: ATM injection heat fluxes (MOST-derived)
    fields.H_roof_atm = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][3.5A-hotfix2][allocate_ucm_fields] H_roof_atm: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.H_wall_atm = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][3.5A-hotfix2][allocate_ucm_fields] H_wall_atm: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.H_road_atm = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][3.5A-hotfix2][allocate_ucm_fields] H_road_atm: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.AH = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.3][allocate_ucm_fields] AH: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    // Phase 2.9: Per-cell anthropogenic heat override
    fields.AH_Wm2_ucm = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.9][allocate_ucm_fields] AH_Wm2_ucm: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.plan_area_frac = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.3][allocate_ucm_fields] plan_area_frac: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.ah_profile_id = std::make_unique<iMultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.3][allocate_ucm_fields] ah_profile_id (iMultiFab): "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    // Phase 2.4: Sky view factors (shadowing model)
    fields.SVF_wall = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.4][allocate_ucm_fields] SVF_wall: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.SVF_road = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.4][allocate_ucm_fields] SVF_road: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.SVF_roof = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.4][allocate_ucm_fields] SVF_roof: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    // Summary message
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][allocate_ucm_fields] allocated 31 MultiFabs on UCM grid "
                << "at lev=" << lev << "\n";
    }

    // Verify all are allocated
    AMREX_ALWAYS_ASSERT(fields.all_allocated());
}

void fill_ucm_fields_from_csv(UCMFields& fields,
                              const UCMGrid& ucm_grid,
                              const UCMBuildingLayoutReader& building_reader,
                              const UCMMaterialRegistry& material_registry,
                              const UCMParams& params,
                              int grid_ratio,
                              int lev,
                              bool ucm_debug)
{
    // Precondition checks
    AMREX_ALWAYS_ASSERT(fields.all_allocated());
    AMREX_ALWAYS_ASSERT(building_reader.size() > 0);
    AMREX_ALWAYS_ASSERT(material_registry.size() > 0);

    // grid_ratio no longer used here — CSV rows are UCM-indexed as of Phase 2.5-fix3.
    // Kept in the signature for ABI stability; will be removed in Phase 2.6.
    (void)grid_ratio;

    // Zero out all fields initially
    fields.H_bldg->setVal(0.0);
    fields.W_road->setVal(0.0);
    fields.W_roof->setVal(0.0);
    fields.albedo_roof->setVal(0.0);
    fields.albedo_wall->setVal(0.0);
    fields.albedo_road->setVal(0.0);
    fields.emissivity_roof->setVal(0.0);
    fields.emissivity_wall->setVal(0.0);
    fields.emissivity_road->setVal(0.0);
    fields.T_skin_roof->setVal(params.T_skin_init_K);
    fields.T_skin_wall->setVal(params.T_skin_init_K);
    fields.T_skin_road->setVal(params.T_skin_init_K);
    fields.T_canyon_air->setVal(params.T_canyon_init_K);
    fields.T_slab_roof->setVal(params.T_skin_init_K);
    fields.T_slab_wall->setVal(params.T_skin_init_K);
    fields.T_slab_road->setVal(params.T_skin_init_K);
    fields.H_sensible->setVal(0.0);
    fields.LE_latent->setVal(0.0);
    fields.is_urban->setVal(0);
    fields.mat_id_roof->setVal(0);
    fields.mat_id_wall->setVal(0);
    fields.mat_id_road->setVal(0);

    // Phase 2.2: zero out thermal properties (will be populated per-cell from CSV)
    fields.k_therm_roof->setVal(0.0);
    fields.k_therm_wall->setVal(0.0);
    fields.k_therm_road->setVal(0.0);
    fields.rho_cp_roof->setVal(0.0);
    fields.rho_cp_wall->setVal(0.0);
    fields.rho_cp_road->setVal(0.0);
    fields.slab_L_roof->setVal(0.0);
    fields.slab_L_wall->setVal(0.0);
    fields.slab_L_road->setVal(0.0);
    fields.z0_ucm->setVal(0.0);
    fields.d_disp_ucm->setVal(0.0);

    // Phase 2.3: zero out facet-split fluxes and anthropogenic heat
    fields.H_road->setVal(0.0);
    fields.H_wall->setVal(0.0);
    fields.H_roof->setVal(0.0);
    // Phase 3.5A-hotfix2: zero out ATM injection fluxes
    fields.H_road_atm->setVal(0.0);
    fields.H_wall_atm->setVal(0.0);
    fields.H_roof_atm->setVal(0.0);
    fields.AH->setVal(0.0);
    fields.AH_Wm2_ucm->setVal(0.0);  // Phase 2.9: zero out per-cell AH override
    fields.plan_area_frac->setVal(0.0);
    fields.ah_profile_id->setVal(0);

    // Get const references to the broadcast data
    const auto& rows = building_reader.rows();

    // Phase 2.5-fix3: CSV rows are UCM-indexed (one row per UCM cell).
    // Build a (i_ucm, j_ucm) -> row_index lookup so each UCM cell can be filled directly.
    std::unordered_map<std::int64_t, int> row_by_ucm_ij;
    row_by_ucm_ij.reserve(rows.size());
    int n_urban = 0, n_non_urban = 0;
    for (int r = 0; r < static_cast<int>(rows.size()); ++r) {
        const auto& row = rows[r];
        const std::int64_t key = (static_cast<std::int64_t>(row.i) << 32) |
                                 static_cast<std::uint32_t>(row.j);
        row_by_ucm_ij[key] = r;
        if (row.is_urban == 1) ++n_urban; else ++n_non_urban;
    }

    // Iterate the UCM grid and populate each cell from its matching CSV row.
    for (MFIter mfi(*(fields.H_bldg)); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto H_bldg_arr        = fields.H_bldg->array(mfi);
        auto W_road_arr        = fields.W_road->array(mfi);
        auto W_roof_arr        = fields.W_roof->array(mfi);
        auto albedo_roof_arr   = fields.albedo_roof->array(mfi);
        auto albedo_wall_arr   = fields.albedo_wall->array(mfi);
        auto albedo_road_arr   = fields.albedo_road->array(mfi);
        auto emissivity_roof_arr = fields.emissivity_roof->array(mfi);
        auto emissivity_wall_arr = fields.emissivity_wall->array(mfi);
        auto emissivity_road_arr = fields.emissivity_road->array(mfi);
        auto T_skin_roof_arr   = fields.T_skin_roof->array(mfi);
        auto T_skin_wall_arr   = fields.T_skin_wall->array(mfi);
        auto T_skin_road_arr   = fields.T_skin_road->array(mfi);
        auto T_canyon_air_arr  = fields.T_canyon_air->array(mfi);
        auto T_slab_roof_arr   = fields.T_slab_roof->array(mfi);
        auto T_slab_wall_arr   = fields.T_slab_wall->array(mfi);
        auto T_slab_road_arr   = fields.T_slab_road->array(mfi);
        auto is_urban_arr      = fields.is_urban->array(mfi);
        auto mat_id_roof_arr   = fields.mat_id_roof->array(mfi);
        auto mat_id_wall_arr   = fields.mat_id_wall->array(mfi);
        auto mat_id_road_arr   = fields.mat_id_road->array(mfi);
        auto k_therm_roof_arr  = fields.k_therm_roof->array(mfi);
        auto k_therm_wall_arr  = fields.k_therm_wall->array(mfi);
        auto k_therm_road_arr  = fields.k_therm_road->array(mfi);
        auto rho_cp_roof_arr   = fields.rho_cp_roof->array(mfi);
        auto rho_cp_wall_arr   = fields.rho_cp_wall->array(mfi);
        auto rho_cp_road_arr   = fields.rho_cp_road->array(mfi);
        auto slab_L_roof_arr   = fields.slab_L_roof->array(mfi);
        auto slab_L_wall_arr   = fields.slab_L_wall->array(mfi);
        auto slab_L_road_arr   = fields.slab_L_road->array(mfi);
        auto plan_area_frac_arr = fields.plan_area_frac->array(mfi);
        auto ah_profile_id_arr  = fields.ah_profile_id->array(mfi);
        auto AH_Wm2_ucm_arr     = fields.AH_Wm2_ucm->array(mfi);  // Phase 2.9

        for (int j_ucm = bx.smallEnd(1); j_ucm <= bx.bigEnd(1); ++j_ucm) {
            for (int i_ucm = bx.smallEnd(0); i_ucm <= bx.bigEnd(0); ++i_ucm) {
                const std::int64_t key = (static_cast<std::int64_t>(i_ucm) << 32) |
                                         static_cast<std::uint32_t>(j_ucm);
                auto it = row_by_ucm_ij.find(key);
                if (it == row_by_ucm_ij.end()) {
                    // No CSV row for this UCM cell — leave zero-initialized.
                    continue;
                }
                const auto& row = rows[it->second];

                IntVect iv(i_ucm, j_ucm, 0);

                // Always populate morphology + is_urban + raw mat_id.
                H_bldg_arr(iv, 0) = row.height_m;
                W_road_arr(iv, 0) = row.W_road_m;
                W_roof_arr(iv, 0) = row.W_roof_m;
                is_urban_arr(iv, 0) = row.is_urban;
                mat_id_roof_arr(iv, 0) = row.roof_mat_id;
                mat_id_wall_arr(iv, 0) = row.wall_mat_id;
                mat_id_road_arr(iv, 0) = row.road_mat_id;

                if (row.is_urban == 1) {
                    const auto& roof_mat = material_registry.lookup(row.roof_mat_id);
                    const auto& wall_mat = material_registry.lookup(row.wall_mat_id);
                    const auto& road_mat = material_registry.lookup(row.road_mat_id);

                    albedo_roof_arr(iv, 0) = roof_mat.albedo;
                    albedo_wall_arr(iv, 0) = wall_mat.albedo;
                    albedo_road_arr(iv, 0) = road_mat.albedo;

                    emissivity_roof_arr(iv, 0) = roof_mat.emissivity;
                    emissivity_wall_arr(iv, 0) = wall_mat.emissivity;
                    emissivity_road_arr(iv, 0) = road_mat.emissivity;

                    // Phase 2.2: thermal properties from material registry
                    k_therm_roof_arr(iv, 0) = roof_mat.k_therm_W_per_mK;
                    k_therm_wall_arr(iv, 0) = wall_mat.k_therm_W_per_mK;
                    k_therm_road_arr(iv, 0) = road_mat.k_therm_W_per_mK;
                    rho_cp_roof_arr(iv, 0) = roof_mat.rho_cp_J_per_m3K;
                    rho_cp_wall_arr(iv, 0) = wall_mat.rho_cp_J_per_m3K;
                    rho_cp_road_arr(iv, 0) = road_mat.rho_cp_J_per_m3K;
                    slab_L_roof_arr(iv, 0) = roof_mat.thickness_m;
                    slab_L_wall_arr(iv, 0) = wall_mat.thickness_m;
                    slab_L_road_arr(iv, 0) = road_mat.thickness_m;

                    // Phase 2.3: morphology-derived + AH profile id
                    plan_area_frac_arr(iv, 0) = static_cast<amrex::Real>(row.plan_area_frac);
                    ah_profile_id_arr(iv, 0)  = row.ah_profile_id;
                    
                    // Phase 2.9: per-cell AH override from CSV
                    AH_Wm2_ucm_arr(iv, 0) = row.AH_Wm2;
                } else {
                    // Non-urban cell: physically inert defaults so downstream kernels
                    // that don't check is_urban still produce sensible numbers.
                    albedo_roof_arr(iv, 0) = 0.0;
                    albedo_wall_arr(iv, 0) = 0.0;
                    albedo_road_arr(iv, 0) = 0.0;
                    emissivity_roof_arr(iv, 0) = 0.0;
                    emissivity_wall_arr(iv, 0) = 0.0;
                    emissivity_road_arr(iv, 0) = 0.0;
                    k_therm_roof_arr(iv, 0) = 0.1;
                    k_therm_wall_arr(iv, 0) = 0.1;
                    k_therm_road_arr(iv, 0) = 0.1;
                    rho_cp_roof_arr(iv, 0) = 1.0e5;
                    rho_cp_wall_arr(iv, 0) = 1.0e5;
                    rho_cp_road_arr(iv, 0) = 1.0e5;
                    slab_L_roof_arr(iv, 0) = 0.3;
                    slab_L_wall_arr(iv, 0) = 0.3;
                    slab_L_road_arr(iv, 0) = 0.3;
                    plan_area_frac_arr(iv, 0) = 0.0;
                    ah_profile_id_arr(iv, 0)  = 0;
                }

                // Initial temperatures (same for urban and non-urban).
                T_skin_roof_arr(iv, 0) = 293.15;
                T_skin_wall_arr(iv, 0) = 293.15;
                T_skin_road_arr(iv, 0) = 293.15;
                T_canyon_air_arr(iv, 0) = 293.15;
                
                // Phase 3.5A: Initialize all slab layers to uniform temperature
                for (int k = 0; k < T_slab_roof_arr.nComp(); ++k) {
                    T_slab_roof_arr(iv, k) = 293.15;
                    T_slab_wall_arr(iv, k) = 293.15;
                    T_slab_road_arr(iv, k) = 293.15;
                }
            }
        }
    }

    if (ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.1][fill_ucm_fields_from_csv] "
                << "populated fields from CSV at lev=" << lev
                << ": urban_cells=" << n_urban
                << ", non_urban_cells=" << n_non_urban << "\n";
    }

    // Phase 2.5-fix2: Task 1 — Debug instrumentation for is_urban iMultiFab propagation
    if (ucm_debug) {
        // Collective OUTSIDE IOProcessor (PR #209 rule).
        const int urb_min = fields.is_urban->min(0, 0);
        const int urb_max = fields.is_urban->max(0, 0);
        long n1 = 0, n0 = 0;
        for (amrex::MFIter mfi(*(fields.is_urban)); mfi.isValid(); ++mfi) {
            auto const a = fields.is_urban->const_array(mfi);
            const auto& bx = mfi.validbox();
            amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
                if (a(i,j,k) == 1) ++n1; else ++n0;
            });
        }
        amrex::ParallelDescriptor::ReduceLongSum(n1);
        amrex::ParallelDescriptor::ReduceLongSum(n0);
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.1][DEBUG][fill_ucm_fields_from_csv] "
                           << "is_urban iMultiFab populated: n_urban=" << n1
                           << " n_nonurban=" << n0
                           << " (min=" << urb_min << " max=" << urb_max << ")\n";
        }
    }
}

void fill_ucm_fields_homogeneous(UCMFields& fields,
                                  const UCMParams& params,
                                  int lev)
{
    // Precondition check
    AMREX_ALWAYS_ASSERT(fields.all_allocated());

    // Fill building morphology fields
    fields.H_bldg->setVal(params.H_bldg_uniform);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][fill_ucm_fields_homogeneous] H_bldg = "
                << params.H_bldg_uniform << " m\n";
    }

    fields.W_road->setVal(params.W_road_uniform);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][fill_ucm_fields_homogeneous] W_road = "
                << params.W_road_uniform << " m\n";
    }

    fields.W_roof->setVal(params.W_roof_uniform);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][fill_ucm_fields_homogeneous] W_roof = "
                << params.W_roof_uniform << " m\n";
    }

    // Fill shortwave albedo fields
    fields.albedo_roof->setVal(params.albedo_roof);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][fill_ucm_fields_homogeneous] albedo_roof = "
                << params.albedo_roof << "\n";
    }

    fields.albedo_wall->setVal(params.albedo_wall);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][fill_ucm_fields_homogeneous] albedo_wall = "
                << params.albedo_wall << "\n";
    }

    fields.albedo_road->setVal(params.albedo_road);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][fill_ucm_fields_homogeneous] albedo_road = "
                << params.albedo_road << "\n";
    }

    // Fill longwave emissivity fields
    fields.emissivity_roof->setVal(params.emissivity_roof);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][fill_ucm_fields_homogeneous] emissivity_roof = "
                << params.emissivity_roof << "\n";
    }

    fields.emissivity_wall->setVal(params.emissivity_wall);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][fill_ucm_fields_homogeneous] emissivity_wall = "
                << params.emissivity_wall << "\n";
    }

    fields.emissivity_road->setVal(params.emissivity_road);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][fill_ucm_fields_homogeneous] emissivity_road = "
                << params.emissivity_road << "\n";
    }

    // Fill temperature fields (Phase 3.5A: use T_skin_init_K and T_canyon_init_K)
    fields.T_skin_roof->setVal(params.T_skin_init_K);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][fill_ucm_fields_homogeneous] T_skin_roof = "
                << params.T_skin_init_K << " K\n";
    }

    fields.T_skin_wall->setVal(params.T_skin_init_K);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][fill_ucm_fields_homogeneous] T_skin_wall = "
                << params.T_skin_init_K << " K\n";
    }

    fields.T_skin_road->setVal(params.T_skin_init_K);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][fill_ucm_fields_homogeneous] T_skin_road = "
                << params.T_skin_init_K << " K\n";
    }

    fields.T_canyon_air->setVal(params.T_canyon_init_K);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][fill_ucm_fields_homogeneous] T_canyon_air = "
                << params.T_canyon_init_K << " K\n";
    }

    // Phase 3.5A: Fill multi-layer slab temperatures
    fields.T_slab_roof->setVal(params.T_skin_init_K);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][3.5A][fill_ucm_fields_homogeneous] T_slab_roof = "
                << params.T_skin_init_K << " K (all layers)\n";
    }

    fields.T_slab_wall->setVal(params.T_skin_init_K);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][3.5A][fill_ucm_fields_homogeneous] T_slab_wall = "
                << params.T_skin_init_K << " K (all layers)\n";
    }

    fields.T_slab_road->setVal(params.T_skin_init_K);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][3.5A][fill_ucm_fields_homogeneous] T_slab_road = "
                << params.T_skin_init_K << " K (all layers)\n";
    }

    // Fill flux fields (Phase 1.2: zero; Phase 1.3+ computed by SEB)
    fields.H_sensible->setVal(0.0);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][fill_ucm_fields_homogeneous] H_sensible = 0.0 W/m^2\n";
    }

    fields.LE_latent->setVal(0.0);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][fill_ucm_fields_homogeneous] LE_latent = 0.0 W/m^2\n";
    }

    // Fill urban mask (Phase 1.2: all 1; Phase 4.1: heterogeneous LSM/MOST bypass)
    fields.is_urban->setVal(1);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][fill_ucm_fields_homogeneous] is_urban = 1 everywhere "
                << "on UCM grid at lev=" << lev << "\n";
    }

    // Fill material ID fields (Phase 2.1: default to 0; CSV fill Phase 2.1+ will override)
    fields.mat_id_roof->setVal(0);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.1][fill_ucm_fields_homogeneous] mat_id_roof = 0 (default)\n";
    }

    fields.mat_id_wall->setVal(0);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.1][fill_ucm_fields_homogeneous] mat_id_wall = 0 (default)\n";
    }

    fields.mat_id_road->setVal(0);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.1][fill_ucm_fields_homogeneous] mat_id_road = 0 (default)\n";
    }

    // Fill thermal properties (Phase 2.2: defaults; CSV fill Phase 2.2+ will override)
    fields.k_therm_roof->setVal(params.k_therm_uniform);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.2][fill_ucm_fields_homogeneous] k_therm_roof = "
                << params.k_therm_uniform << " W/m/K\n";
    }

    fields.k_therm_wall->setVal(params.k_therm_uniform);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.2][fill_ucm_fields_homogeneous] k_therm_wall = "
                << params.k_therm_uniform << " W/m/K\n";
    }

    fields.k_therm_road->setVal(params.k_therm_uniform);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.2][fill_ucm_fields_homogeneous] k_therm_road = "
                << params.k_therm_uniform << " W/m/K\n";
    }

    fields.rho_cp_roof->setVal(params.rho_cp_uniform);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.2][fill_ucm_fields_homogeneous] rho_cp_roof = "
                << params.rho_cp_uniform << " J/m^3/K\n";
    }

    fields.rho_cp_wall->setVal(params.rho_cp_uniform);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.2][fill_ucm_fields_homogeneous] rho_cp_wall = "
                << params.rho_cp_uniform << " J/m^3/K\n";
    }

    fields.rho_cp_road->setVal(params.rho_cp_uniform);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.2][fill_ucm_fields_homogeneous] rho_cp_road = "
                << params.rho_cp_uniform << " J/m^3/K\n";
    }

    fields.slab_L_roof->setVal(params.slab_L);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.2][fill_ucm_fields_homogeneous] slab_L_roof = "
                << params.slab_L << " m\n";
    }

    fields.slab_L_wall->setVal(params.slab_L);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.2][fill_ucm_fields_homogeneous] slab_L_wall = "
                << params.slab_L << " m\n";
    }

    fields.slab_L_road->setVal(params.slab_L);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.2][fill_ucm_fields_homogeneous] slab_L_road = "
                << params.slab_L << " m\n";
    }

    // Phase 2.3: Facet-split sensible heat and anthropogenic heat
    fields.H_road->setVal(0.0);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.3][fill_ucm_fields_homogeneous] H_road = 0.0 W/m^2\n";
    }

    fields.H_wall->setVal(0.0);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.3][fill_ucm_fields_homogeneous] H_wall = 0.0 W/m^2\n";
    }

    fields.H_roof->setVal(0.0);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.3][fill_ucm_fields_homogeneous] H_roof = 0.0 W/m^2\n";
    }

    // Phase 3.5A-hotfix2: ATM injection heat fluxes
    fields.H_road_atm->setVal(0.0);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][3.5A-hotfix2][fill_ucm_fields_homogeneous] H_road_atm = 0.0 W/m^2\n";
    }

    fields.H_wall_atm->setVal(0.0);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][3.5A-hotfix2][fill_ucm_fields_homogeneous] H_wall_atm = 0.0 W/m^2\n";
    }

    fields.H_roof_atm->setVal(0.0);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][3.5A-hotfix2][fill_ucm_fields_homogeneous] H_roof_atm = 0.0 W/m^2\n";
    }

    fields.AH->setVal(0.0);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.3][fill_ucm_fields_homogeneous] AH = 0.0 W/m^2\n";
    }

    fields.plan_area_frac->setVal(params.plan_area_frac_uniform);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.3][fill_ucm_fields_homogeneous] plan_area_frac = "
                << params.plan_area_frac_uniform << "\n";
    }

    fields.ah_profile_id->setVal(0);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.3][fill_ucm_fields_homogeneous] ah_profile_id = 0\n";
    }

    // Note: z0 and d_disp are filled by fill_ucm_z0_and_disp, not here
}

bool UCMFields::all_allocated() const
{
    bool result = true;

    if (!H_bldg) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.2][all_allocated] MISSING: H_bldg\n";
        }
        result = false;
    }
    if (!W_road) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.2][all_allocated] MISSING: W_road\n";
        }
        result = false;
    }
    if (!W_roof) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.2][all_allocated] MISSING: W_roof\n";
        }
        result = false;
    }
    if (!albedo_roof) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.2][all_allocated] MISSING: albedo_roof\n";
        }
        result = false;
    }
    if (!albedo_wall) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.2][all_allocated] MISSING: albedo_wall\n";
        }
        result = false;
    }
    if (!albedo_road) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.2][all_allocated] MISSING: albedo_road\n";
        }
        result = false;
    }
    if (!emissivity_roof) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.2][all_allocated] MISSING: emissivity_roof\n";
        }
        result = false;
    }
    if (!emissivity_wall) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.2][all_allocated] MISSING: emissivity_wall\n";
        }
        result = false;
    }
    if (!emissivity_road) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.2][all_allocated] MISSING: emissivity_road\n";
        }
        result = false;
    }
    if (!T_skin_roof) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.2][all_allocated] MISSING: T_skin_roof\n";
        }
        result = false;
    }
    if (!T_skin_wall) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.2][all_allocated] MISSING: T_skin_wall\n";
        }
        result = false;
    }
    if (!T_skin_road) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.2][all_allocated] MISSING: T_skin_road\n";
        }
        result = false;
    }
    if (!T_canyon_air) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.2][all_allocated] MISSING: T_canyon_air\n";
        }
        result = false;
    }
    if (!T_slab_roof) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][3.5A][all_allocated] MISSING: T_slab_roof\n";
        }
        result = false;
    }
    if (!T_slab_wall) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][3.5A][all_allocated] MISSING: T_slab_wall\n";
        }
        result = false;
    }
    if (!T_slab_road) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][3.5A][all_allocated] MISSING: T_slab_road\n";
        }
        result = false;
    }
    if (!H_sensible) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.2][all_allocated] MISSING: H_sensible\n";
        }
        result = false;
    }
    if (!LE_latent) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.2][all_allocated] MISSING: LE_latent\n";
        }
        result = false;
    }
    if (!is_urban) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.2][all_allocated] MISSING: is_urban\n";
        }
        result = false;
    }
    if (!mat_id_roof) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.1][all_allocated] MISSING: mat_id_roof\n";
        }
        result = false;
    }
    if (!mat_id_wall) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.1][all_allocated] MISSING: mat_id_wall\n";
        }
        result = false;
    }
    if (!mat_id_road) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.1][all_allocated] MISSING: mat_id_road\n";
        }
        result = false;
    }
    if (!k_therm_roof) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.2][all_allocated] MISSING: k_therm_roof\n";
        }
        result = false;
    }
    if (!k_therm_wall) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.2][all_allocated] MISSING: k_therm_wall\n";
        }
        result = false;
    }
    if (!k_therm_road) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.2][all_allocated] MISSING: k_therm_road\n";
        }
        result = false;
    }
    if (!rho_cp_roof) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.2][all_allocated] MISSING: rho_cp_roof\n";
        }
        result = false;
    }
    if (!rho_cp_wall) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.2][all_allocated] MISSING: rho_cp_wall\n";
        }
        result = false;
    }
    if (!rho_cp_road) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.2][all_allocated] MISSING: rho_cp_road\n";
        }
        result = false;
    }
    if (!slab_L_roof) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.2][all_allocated] MISSING: slab_L_roof\n";
        }
        result = false;
    }
    if (!slab_L_wall) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.2][all_allocated] MISSING: slab_L_wall\n";
        }
        result = false;
    }
    if (!slab_L_road) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.2][all_allocated] MISSING: slab_L_road\n";
        }
        result = false;
    }
    if (!z0_ucm) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.2][all_allocated] MISSING: z0_ucm\n";
        }
        result = false;
    }
    if (!d_disp_ucm) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.2][all_allocated] MISSING: d_disp_ucm\n";
        }
        result = false;
    }
    if (!H_road) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.3][all_allocated] MISSING: H_road\n";
        }
        result = false;
    }
    if (!H_wall) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.3][all_allocated] MISSING: H_wall\n";
        }
        result = false;
    }
    if (!H_roof) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.3][all_allocated] MISSING: H_roof\n";
        }
        result = false;
    }
    if (!H_roof_atm) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][3.5A-hotfix2][all_allocated] MISSING: H_roof_atm\n";
        }
        result = false;
    }
    if (!H_wall_atm) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][3.5A-hotfix2][all_allocated] MISSING: H_wall_atm\n";
        }
        result = false;
    }
    if (!H_road_atm) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][3.5A-hotfix2][all_allocated] MISSING: H_road_atm\n";
        }
        result = false;
    }
    if (!AH) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.3][all_allocated] MISSING: AH\n";
        }
        result = false;
    }
    if (!plan_area_frac) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.3][all_allocated] MISSING: plan_area_frac\n";
        }
        result = false;
    }
    if (!ah_profile_id) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.3][all_allocated] MISSING: ah_profile_id\n";
        }
        result = false;
    }

    // Phase 2.4: Sky view factors (shadowing)
    if (!SVF_wall) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.4][all_allocated] MISSING: SVF_wall\n";
        }
        result = false;
    }

    if (!SVF_road) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.4][all_allocated] MISSING: SVF_road\n";
        }
        result = false;
    }

    if (!SVF_roof) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.4][all_allocated] MISSING: SVF_roof\n";
        }
        result = false;
    }

    return result;
}

void fill_ucm_z0_and_disp(UCMFields& f,
                         const UCMParams& params,
                         int lev)
{
    // Precondition check
    AMREX_ALWAYS_ASSERT(f.all_allocated());

    // CPU loop for one-time initialization (tiny cost)
    for (amrex::MFIter mfi(*f.H_bldg, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
       const amrex::Box& bx = mfi.tilebox();
       auto const h_a  = f.H_bldg->const_array(mfi);
       auto const u_a  = f.is_urban->const_array(mfi);
       auto z0_a       = f.z0_ucm->array(mfi);
       auto dd_a       = f.d_disp_ucm->array(mfi);
       const amrex::Real z0oH = params.z0_over_H;
       const amrex::Real  doH = params.d_over_H;

       amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
           if (u_a(i,j,0) == 1) {
               z0_a(i,j,0) = z0oH * h_a(i,j,0);
               dd_a(i,j,0) =  doH * h_a(i,j,0);
           } else {
               z0_a(i,j,0) = 0.1;   // MOST default
               dd_a(i,j,0) = 0.0;
           }
       });
    }

    // Collectives OUTSIDE IOProcessor guard (Phase 1.4 rule Bug #9, PR #201)
    amrex::Real z0_min = f.z0_ucm->min(0);
    amrex::Real z0_max = f.z0_ucm->max(0);
    amrex::Real dd_min = f.d_disp_ucm->min(0);
    amrex::Real dd_max = f.d_disp_ucm->max(0);

    if (params.ucm_debug && amrex::ParallelDescriptor::IOProcessor()) {
       amrex::Print() << "[UCM][2.2][fill_ucm_z0_and_disp] z0 min=" << z0_min
                      << " max=" << z0_max
                      << " d_disp min=" << dd_min
                      << " max=" << dd_max << "\n";
    }
}

void compute_anthropogenic_heat(amrex::MultiFab&        AH_out,
                               const amrex::iMultiFab& ah_profile_id,
                               const amrex::iMultiFab& is_urban,
                               const amrex::MultiFab&  AH_Wm2_ucm,
                               const UCMParams&        params,
                               amrex::Real             time,
                               int                     lev)
{
    const amrex::Real AH_const = params.AH_uniform_Wm2;
    const amrex::Real AH_peak  = params.AH_daytime_peak;
    const amrex::Real day_len  = 86400.0;
    const amrex::Real phase    = 2.0 * M_PI * (time / day_len) - 0.5 * M_PI;
    const amrex::Real diurnal  = std::max(0.0, std::cos(phase));

    // Phase 2.9: Track per-cell override stats
    int n_overridden = 0;
    amrex::Real min_override = std::numeric_limits<amrex::Real>::infinity();
    amrex::Real max_override = -std::numeric_limits<amrex::Real>::infinity();

    for (amrex::MFIter mfi(AH_out, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
       const amrex::Box& bx = mfi.tilebox();
       auto       ah_a = AH_out.array(mfi);
       auto const id_a = ah_profile_id.const_array(mfi);
       auto const ur_a = is_urban.const_array(mfi);
       auto const ah_csv_a = AH_Wm2_ucm.const_array(mfi);
       amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
           // Phase 2.9: First-line guard for urban cells
           if (ur_a(i,j,0) == 0) { 
               ah_a(i,j,0) = 0.0; 
               return; 
           }
           
           // Phase 2.9: Use per-cell AH override if > 0, else use ParmParse fallback
           const amrex::Real AH_csv = ah_csv_a(i,j,0);
           if (AH_csv > 0.0) {
               // Use per-cell override (no diurnal scaling for CSV-provided values)
               ah_a(i,j,0) = AH_csv;
           } else {
               // Use ParmParse fallback (with diurnal factor as before)
               const int pid = id_a(i,j,0);
               if      (pid == 1) ah_a(i,j,0) = AH_peak * diurnal;
               else               ah_a(i,j,0) = AH_const;
           }
       });
    }

    // Phase 2.9: Compute stats for per-cell overrides (reduction)
    for (amrex::MFIter mfi(AH_out); mfi.isValid(); ++mfi) {
        auto const ah_csv_a = AH_Wm2_ucm.const_array(mfi);
        auto const ur_a = is_urban.const_array(mfi);
        const amrex::Box& bx = mfi.validbox();
        amrex::LoopConcurrentOnCpu(bx, [&](int i, int j, int k) {
            if (ur_a(i,j,0) > 0 && ah_csv_a(i,j,0) > 0.0) {
                n_overridden++;
                min_override = std::min(min_override, ah_csv_a(i,j,0));
                max_override = std::max(max_override, ah_csv_a(i,j,0));
            }
        });
    }

    if (params.ucm_debug) {
       amrex::Real ah_min = AH_out.min(0, 0);
       amrex::Real ah_max = AH_out.max(0, 0);
       if (amrex::ParallelDescriptor::IOProcessor()) {
           amrex::Print() << "[UCM][2.9][compute_anthropogenic_heat] time=" << time
                          << "s AH min=" << ah_min
                          << " max=" << ah_max << " W/m^2"
                          << " diurnal_factor=" << diurnal << "\n";
           // Phase 2.9: Log per-cell override stats
           if (n_overridden > 0 && min_override < std::numeric_limits<amrex::Real>::infinity()) {
               amrex::Print() << "[UCM][2.9][AH] per-cell override applied to " << n_overridden
                              << " cells, min=" << min_override << " max=" << max_override
                              << " W/m^2\n";
           }
       }
    }
}