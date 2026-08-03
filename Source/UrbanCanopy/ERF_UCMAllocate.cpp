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
#include <cmath>
#include <limits>


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

    // Phase 5.2: HVAC profile selector
    fields.hvac_profile_id_map = std::make_unique<iMultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][5.2][allocate_ucm_fields] hvac_profile_id_map (iMultiFab): "
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

    // Phase 5.1a: Multi-facet view factors (geometry only)
    fields.F_wall_sky = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][5.1a][allocate_ucm_fields] F_wall_sky: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.F_wall_wall = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][5.1a][allocate_ucm_fields] F_wall_wall: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.F_wall_road = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][5.1a][allocate_ucm_fields] F_wall_road: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.F_road_sky = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][5.1a][allocate_ucm_fields] F_road_sky: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.F_road_wall = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][5.1a][allocate_ucm_fields] F_road_wall: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    fields.F_roof_sky = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][5.1a][allocate_ucm_fields] F_roof_sky: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    // Phase 5.2: HVAC diagnostic field
    fields.Q_HVAC_diag = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][5.2][allocate_ucm_fields] Q_HVAC_diag: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    // Phase 5.5: HVAC facet-split diagnostic fields
    fields.Q_HVAC_roof_diag = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    fields.Q_HVAC_wall_diag = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    fields.Q_HVAC_road_diag = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][5.5][allocate_ucm_fields] Q_HVAC_roof_diag, Q_HVAC_wall_diag, Q_HVAC_road_diag: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    // Phase 6.2a: Tree radiation diagnostic field
    fields.Q_tree_SW_abs = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][6.2a][allocate_ucm_fields] Q_tree_SW_abs: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    // Phase 5.3: Green roof and permeable pavement state fields
    fields.soil_moisture_roof = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    fields.soil_moisture_road = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    fields.LE_green_roof_diag = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    fields.LE_permeable_road_diag = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][5.3][allocate_ucm_fields] soil_moisture_roof, soil_moisture_road, "
                << "LE_green_roof_diag, LE_permeable_road_diag: "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
    }

    // Phase 5.3: Green roof and permeable pavement masks (iMultiFab for integer flags)
    int ncomp_i = 1;  // integer components
    fields.is_green_roof = std::make_unique<iMultiFab>(ba, dm, ncomp_i, ngrow);
    fields.is_permeable_road = std::make_unique<iMultiFab>(ba, dm, ncomp_i, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][5.3][allocate_ucm_fields] is_green_roof (iMultiFab), is_permeable_road (iMultiFab): "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp_i << "\n";
    }

    // Phase 6.1: Tree canopy 2D UCM fields
    // Note: Tree drag ATM aggregates (m_ucm_is_tree_atm, etc.) are ERF class members,
    // NOT allocated here. See ERF_Advance.cpp for ATM aggregate allocation pattern.
    fields.H_tree = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    fields.H_crown_base = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    fields.LAD_bulk = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    fields.crown_area_frac = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    fields.Cd_leaf = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
    fields.is_tree = std::make_unique<iMultiFab>(ba, dm, ncomp_i, ngrow);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][6.1][allocate_ucm_fields] H_tree, H_crown_base, LAD_bulk, crown_area_frac, Cd_leaf (MultiFab), "
                << "is_tree (iMultiFab): "
                << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << " or " << ncomp_i << "\n";
    }

 // Phase 6.2b: Crown SEB facet (4-var Newton solver)
// T_crown allocated ONLY in 4-var mode; left as nullptr in 3-var mode (Contract #30)
if (params.seb_mode == SEBMode::FourVar) {
   fields.T_crown = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
   fields.H_crown_up = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
   fields.H_crown_down = std::make_unique<MultiFab>(ba, dm, ncomp, ngrow);
   if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
       Print() << "[UCM][6.2b][allocate_ucm_fields] T_crown + H_crown_up/down (4-var mode): "
               << ba.size() << " boxes, ngrow=" << ngrow << ", ncomp=" << ncomp << "\n";
   }
}
    // In 3-var mode, T_crown remains nullptr (Contract #30)

    // Summary message
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
       Print() << "[UCM][1.2][allocate_ucm_fields] allocated 51 MultiFabs on UCM grid "
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
    fields.H_tree->setVal(0.0);
    fields.H_crown_base->setVal(0.0);
    fields.LAD_bulk->setVal(0.0);
    fields.crown_area_frac->setVal(0.0);
    fields.Cd_leaf->setVal(0.0);
    fields.is_tree->setVal(0);

    // Phase 6.2b: Crown SEB facet (4-var mode only)
    if (params.seb_mode == SEBMode::FourVar) {
       fields.T_crown->setVal(params.T_canyon_init_K);
      if (fields.H_crown_up)   fields.H_crown_up->setVal(0.0);
       if (fields.H_crown_down) fields.H_crown_down->setVal(0.0);
    }

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
    fields.hvac_profile_id_map->setVal(0);  // Phase 5.2: zero out HVAC profile ID
    fields.Q_HVAC_diag->setVal(0.0);         // Phase 5.2: zero out HVAC diagnostic
    // Phase 5.5: zero out HVAC facet-split diagnostics
    fields.Q_HVAC_roof_diag->setVal(0.0);
    fields.Q_HVAC_wall_diag->setVal(0.0);
    fields.Q_HVAC_road_diag->setVal(0.0);

    // Phase 6.2a: Zero out tree radiation diagnostic
    fields.Q_tree_SW_abs->setVal(0.0);

    // Phase 5.3: Initialize green roof and permeable pavement state
    fields.soil_moisture_roof->setVal(params.green_roof_soil_capacity_m);  // Initialize to full capacity
    fields.soil_moisture_road->setVal(params.permeable_road_soil_capacity_m);
    fields.LE_green_roof_diag->setVal(0.0);
    fields.LE_permeable_road_diag->setVal(0.0);
    fields.is_green_roof->setVal(0);  // Default: no green roofs
    fields.is_permeable_road->setVal(0);  // Default: no permeable roads

    // Get const references to the broadcast data
    const auto& rows = building_reader.rows();

    // Phase 3.7: Detect CSV mode. In physical mode, x_m and y_m are
    // physical coordinates in meters (typically hundreds to thousands).
    // In legacy mode, x and y are grid indices (small integers < nx_ucm).
    // Heuristic: if any x or y is non-integer OR >= nx_ucm*10, treat as physical.
    bool is_physical_mode = false;
    const int nx_ucm = ucm_grid.ba.minimalBox().length(0);
    const int ny_ucm = ucm_grid.ba.minimalBox().length(1);
    for (const auto& row : rows) {
        if (row.x != std::floor(row.x) || row.y != std::floor(row.y) ||
            row.x >= 10.0 * nx_ucm || row.y >= 10.0 * ny_ucm) {
            is_physical_mode = true;
            break;
        }
    }

    // Get UCM physical geometry (needed for physical mode nearest-neighbor)
    const amrex::Geometry& geom_ucm = ucm_grid.geom;
    const amrex::Real dx_ucm   = geom_ucm.CellSize(0);
    const amrex::Real dy_ucm   = geom_ucm.CellSize(1);
    const amrex::Real prob_lo_x = geom_ucm.ProbLo(0);
    const amrex::Real prob_lo_y = geom_ucm.ProbLo(1);

    if (ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][3.7][fill_ucm_fields_from_csv] mode="
                << (is_physical_mode ? "physical (nearest-neighbor)" : "legacy (index)")
                << " nx_ucm=" << nx_ucm << " ny_ucm=" << ny_ucm
                << " dx_ucm=" << dx_ucm << " prob_lo=(" << prob_lo_x
                << "," << prob_lo_y << ")\n";
    }

    // Legacy mode: build (i,j) -> row lookup table.
    // Physical mode: skip lookup table; will do nearest-neighbor per cell.
    std::unordered_map<std::int64_t, int> row_by_ucm_ij;
    int n_urban = 0, n_non_urban = 0;
    if (!is_physical_mode) {
        row_by_ucm_ij.reserve(rows.size());
        for (int r = 0; r < static_cast<int>(rows.size()); ++r) {
            const auto& row = rows[r];
            const int i = static_cast<int>(row.x);
            const int j = static_cast<int>(row.y);
            const std::int64_t key = (static_cast<std::int64_t>(i) << 32) |
                                     static_cast<std::uint32_t>(j);
            row_by_ucm_ij[key] = r;
            if (row.is_urban == 1) ++n_urban; else ++n_non_urban;
        }
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
        auto hvac_profile_id_arr = fields.hvac_profile_id_map->array(mfi);  // Phase 5.2
        auto AH_Wm2_ucm_arr     = fields.AH_Wm2_ucm->array(mfi);  // Phase 2.9

        // Phase 5.3: Green roof and permeable pavement fields
        auto is_green_roof_arr = fields.is_green_roof->array(mfi);
        auto is_permeable_road_arr = fields.is_permeable_road->array(mfi);
        auto soil_moisture_roof_arr = fields.soil_moisture_roof->array(mfi);
        auto soil_moisture_road_arr = fields.soil_moisture_road->array(mfi);
        auto LE_green_roof_diag_arr = fields.LE_green_roof_diag->array(mfi);
        auto LE_permeable_road_diag_arr = fields.LE_permeable_road_diag->array(mfi);

        for (int j_ucm = bx.smallEnd(1); j_ucm <= bx.bigEnd(1); ++j_ucm) {
            for (int i_ucm = bx.smallEnd(0); i_ucm <= bx.bigEnd(0); ++i_ucm) {

                int row_idx = -1;

                if (!is_physical_mode) {
                    // Legacy mode: exact (i,j) lookup
                    const std::int64_t key = (static_cast<std::int64_t>(i_ucm) << 32) |
                                             static_cast<std::uint32_t>(j_ucm);
                    auto it = row_by_ucm_ij.find(key);
                    if (it == row_by_ucm_ij.end()) continue;  // no match, leave zero
                    row_idx = it->second;
                } else {
                    // Phase 3.7 physical mode: nearest-neighbor lookup
                    const amrex::Real x_c = prob_lo_x + (i_ucm + 0.5) * dx_ucm;
                    const amrex::Real y_c = prob_lo_y + (j_ucm + 0.5) * dy_ucm;
                    amrex::Real min_dist_sq = std::numeric_limits<amrex::Real>::infinity();
                    for (int r = 0; r < static_cast<int>(rows.size()); ++r) {
                        const amrex::Real ddx = rows[r].x - x_c;
                        const amrex::Real ddy = rows[r].y - y_c;
                        const amrex::Real d2 = ddx*ddx + ddy*ddy;
                        if (d2 < min_dist_sq) {
                            min_dist_sq = d2;
                            row_idx = r;
                        }
                    }
                    if (row_idx < 0) continue;
                    if (rows[row_idx].is_urban == 1) ++n_urban; else ++n_non_urban;
                }

                const auto& row = rows[row_idx];

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
                    hvac_profile_id_arr(iv, 0) = row.hvac_profile_id;  // Phase 5.2

                    // Phase 5.3: Green roof and permeable pavement masks + initial soil moisture
                    is_green_roof_arr(iv, 0) = row.is_green_roof;
                    is_permeable_road_arr(iv, 0) = row.is_permeable_road;
                    // Initialize soil moisture: use CSV value if provided, else use params defaults
                    soil_moisture_roof_arr(iv, 0) = (row.soil_moisture_init_m3_per_m3 > 0.0) 
                        ? row.soil_moisture_init_m3_per_m3 
                        : params.green_roof_soil_capacity_m;
                    soil_moisture_road_arr(iv, 0) = (row.soil_moisture_init_m3_per_m3 > 0.0) 
                        ? row.soil_moisture_init_m3_per_m3 
                        : params.permeable_road_soil_capacity_m;
                    LE_green_roof_diag_arr(iv, 0) = 0.0;
                    LE_permeable_road_diag_arr(iv, 0) = 0.0;

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
                    hvac_profile_id_arr(iv, 0) = 0;  // Phase 5.2
                    
                    // Phase 5.3: Non-urban cells get no green roofs or permeable roads
                    is_green_roof_arr(iv, 0) = 0;
                    is_permeable_road_arr(iv, 0) = 0;
                    soil_moisture_roof_arr(iv, 0) = 0.0;
                    soil_moisture_road_arr(iv, 0) = 0.0;
                    LE_green_roof_diag_arr(iv, 0) = 0.0;
                    LE_permeable_road_diag_arr(iv, 0) = 0.0;
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

void fill_ucm_tree_fields_from_csv(UCMFields& fields,
                                    const UCMGrid& ucm_grid,
                                    const UCMTreeLayoutReader& tree_reader,
                                    const UCMParams& params,
                                    int lev,
                                    bool ucm_debug)
{
    AMREX_ALWAYS_ASSERT(fields.all_allocated());

    fields.H_tree->setVal(0.0);
    fields.H_crown_base->setVal(0.0);
    fields.LAD_bulk->setVal(0.0);
    fields.crown_area_frac->setVal(0.0);
    fields.Cd_leaf->setVal(0.0);
    fields.is_tree->setVal(0);

    const auto& rows = tree_reader.rows();
    if (rows.empty()) {
        if (ucm_debug && ParallelDescriptor::IOProcessor()) {
            Print() << "[UCM][6.1][fill_ucm_tree_fields_from_csv] lev=" << lev
                    << " rows=0 (tree fields remain zero)\n";
        }
        return;
    }

    const amrex::Geometry& geom_ucm = ucm_grid.geom;
    const amrex::Real dx_ucm = geom_ucm.CellSize(0);
    const amrex::Real dy_ucm = geom_ucm.CellSize(1);
    const amrex::Real prob_lo_x = geom_ucm.ProbLo(0);
    const amrex::Real prob_lo_y = geom_ucm.ProbLo(1);
    const amrex::Box domain = geom_ucm.Domain();

    for (const auto& row : rows) {
        if (row.is_tree == 0) continue;

        const int i_ucm = static_cast<int>(std::floor((row.x_m - prob_lo_x) / dx_ucm));
        const int j_ucm = static_cast<int>(std::floor((row.y_m - prob_lo_y) / dy_ucm));
        if (!domain.contains(amrex::IntVect(i_ucm, j_ucm, 0))) {
            amrex::Abort(std::string("[UCM][6.1][fill_ucm_tree_fields_from_csv] Tree row maps outside UCM domain: x_m=") +
                         std::to_string(row.x_m) + ", y_m=" + std::to_string(row.y_m) +
                         ", lev=" + std::to_string(lev));
        }

        const amrex::IntVect iv(i_ucm, j_ucm, 0);
        for (MFIter mfi(*fields.H_tree); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            if (!bx.contains(iv)) continue;

            auto H_tree_arr = fields.H_tree->array(mfi);
            auto H_crown_base_arr = fields.H_crown_base->array(mfi);
            auto LAD_bulk_arr = fields.LAD_bulk->array(mfi);
            auto crown_area_frac_arr = fields.crown_area_frac->array(mfi);
            auto Cd_leaf_arr = fields.Cd_leaf->array(mfi);
            auto is_tree_arr = fields.is_tree->array(mfi);

            const amrex::Real old_area = crown_area_frac_arr(iv, 0);
            const amrex::Real add_area = row.crown_area_frac;
            const amrex::Real new_area = old_area + add_area;
            const amrex::Real row_cd = (row.Cd_leaf > 0.0) ? row.Cd_leaf : params.Cd_leaf_default;

            if (is_tree_arr(iv, 0) == 0) {
                H_tree_arr(iv, 0) = row.H_tree_m;
                H_crown_base_arr(iv, 0) = row.H_crown_base_m;
                LAD_bulk_arr(iv, 0) = row.LAD_bulk;
                crown_area_frac_arr(iv, 0) = add_area;
                Cd_leaf_arr(iv, 0) = row_cd;
                is_tree_arr(iv, 0) = 1;
            } else {
                H_tree_arr(iv, 0) = std::max(H_tree_arr(iv, 0), row.H_tree_m);
                H_crown_base_arr(iv, 0) = std::min(H_crown_base_arr(iv, 0), row.H_crown_base_m);
                if (new_area > 0.0) {
                    LAD_bulk_arr(iv, 0) = (LAD_bulk_arr(iv, 0) * old_area + row.LAD_bulk * add_area) / new_area;
                    Cd_leaf_arr(iv, 0) = (Cd_leaf_arr(iv, 0) * old_area + row_cd * add_area) / new_area;
                }
                crown_area_frac_arr(iv, 0) = new_area;
            }
            break;
        }
    }

    if (ucm_debug) {
        const amrex::Real H_tree_min = fields.H_tree->min(0, 0);
        const amrex::Real H_tree_max = fields.H_tree->max(0, 0);
        const amrex::Real H_crown_base_min = fields.H_crown_base->min(0, 0);
        const amrex::Real H_crown_base_max = fields.H_crown_base->max(0, 0);
        const amrex::Real LAD_min = fields.LAD_bulk->min(0, 0);
        const amrex::Real LAD_max = fields.LAD_bulk->max(0, 0);
        const amrex::Real crown_min = fields.crown_area_frac->min(0, 0);
        const amrex::Real crown_max = fields.crown_area_frac->max(0, 0);
        const amrex::Real Cd_min = fields.Cd_leaf->min(0, 0);
        const amrex::Real Cd_max = fields.Cd_leaf->max(0, 0);
        const long n_tree_cells = fields.is_tree->sum(0);
        if (ParallelDescriptor::IOProcessor()) {
            Print() << "[UCM][6.1][fill_ucm_tree_fields_from_csv]\n"
                    << "  lev=" << lev << " tree_cells=" << n_tree_cells << "\n"
                    << "  H_tree: min=" << H_tree_min << " max=" << H_tree_max << " [m]\n"
                    << "  H_crown_base: min=" << H_crown_base_min << " max=" << H_crown_base_max << " [m]\n"
                    << "  LAD_bulk: min=" << LAD_min << " max=" << LAD_max << " [m^2/m^3]\n"
                    << "  crown_area_frac: min=" << crown_min << " max=" << crown_max << "\n"
                    << "  Cd_leaf: min=" << Cd_min << " max=" << Cd_max << "\n";
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

    // Phase 6.2b: Crown SEB facet (4-var mode only)
    if (params.seb_mode == SEBMode::FourVar) {
       fields.T_crown->setVal(params.T_canyon_init_K);
      if (fields.H_crown_up)   fields.H_crown_up->setVal(0.0);
       if (fields.H_crown_down) fields.H_crown_down->setVal(0.0);
       if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
           Print() << "[UCM][6.2b][fill_ucm_fields_homogeneous] T_crown = "
                   << params.T_canyon_init_K << " K\n";
       }
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

    // Phase 5.2: HVAC profile ID
    fields.hvac_profile_id_map->setVal(0);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][5.2][fill_ucm_fields_homogeneous] hvac_profile_id_map = 0\n";
    }

    // Phase 5.2: HVAC diagnostic field
    fields.Q_HVAC_diag->setVal(0.0);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][5.2][fill_ucm_fields_homogeneous] Q_HVAC_diag = 0.0 W/m^2\n";
    }

    // Phase 5.5: HVAC facet-split diagnostic fields
    fields.Q_HVAC_roof_diag->setVal(0.0);
    fields.Q_HVAC_wall_diag->setVal(0.0);
    fields.Q_HVAC_road_diag->setVal(0.0);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][5.5][fill_ucm_fields_homogeneous] Q_HVAC_roof_diag, Q_HVAC_wall_diag, Q_HVAC_road_diag = 0.0 W/m^2\n";
    }

    // Phase 6.2a: Tree radiation diagnostic field
    fields.Q_tree_SW_abs->setVal(0.0);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][6.2a][fill_ucm_fields_homogeneous] Q_tree_SW_abs = 0.0 W/m^2\n";
    }

    // Phase 5.3: Green roof and permeable pavement state
    fields.soil_moisture_roof->setVal(params.green_roof_soil_capacity_m);
    fields.soil_moisture_road->setVal(params.permeable_road_soil_capacity_m);
    fields.LE_green_roof_diag->setVal(0.0);
    fields.LE_permeable_road_diag->setVal(0.0);
    fields.is_green_roof->setVal(0);
    fields.is_permeable_road->setVal(0);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][5.3][fill_ucm_fields_homogeneous] soil_moisture_roof = " 
                << params.green_roof_soil_capacity_m << " m³/m³, soil_moisture_road = "
                << params.permeable_road_soil_capacity_m << " m³/m³\n";
        Print() << "[UCM][5.3][fill_ucm_fields_homogeneous] is_green_roof = 0, is_permeable_road = 0\n";
    }

    // Phase 6.1: Tree canopy fields (homogeneous initialization to zero)
    // Tree layout is populated from CSV if tree_drag_mode != off.
    fields.H_tree->setVal(0.0);
    fields.H_crown_base->setVal(0.0);
    fields.LAD_bulk->setVal(0.0);
    fields.crown_area_frac->setVal(0.0);
    fields.Cd_leaf->setVal(0.0);  // 0 → use Cd_leaf_default
    fields.is_tree->setVal(0);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][6.1][fill_ucm_fields_homogeneous] H_tree, H_crown_base, LAD_bulk, crown_area_frac, Cd_leaf = 0.0, is_tree = 0 "
                << "(homogeneous; CSV fill follows if tree_drag_mode != off)\n";
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

    // Phase 5.2: HVAC profile and diagnostic fields
    if (!hvac_profile_id_map) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][5.2][all_allocated] MISSING: hvac_profile_id_map\n";
        }
        result = false;
    }

    if (!Q_HVAC_diag) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][5.2][all_allocated] MISSING: Q_HVAC_diag\n";
        }
        result = false;
    }

    // Phase 5.5: HVAC facet-split diagnostic fields
    if (!Q_HVAC_roof_diag) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][5.5][all_allocated] MISSING: Q_HVAC_roof_diag\n";
        }
        result = false;
    }
    if (!Q_HVAC_wall_diag) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][5.5][all_allocated] MISSING: Q_HVAC_wall_diag\n";
        }
        result = false;
    }
    if (!Q_HVAC_road_diag) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][5.5][all_allocated] MISSING: Q_HVAC_road_diag\n";
        }
        result = false;
    }

    // Phase 6.2a: Tree radiation diagnostic
    if (!Q_tree_SW_abs) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][6.2a][all_allocated] MISSING: Q_tree_SW_abs\n";
        }
        result = false;
    }

    // Phase 5.3: Green roof and permeable pavement state
    if (!soil_moisture_roof) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][5.3][all_allocated] MISSING: soil_moisture_roof\n";
        }
        result = false;
    }
    if (!soil_moisture_road) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][5.3][all_allocated] MISSING: soil_moisture_road\n";
        }
        result = false;
    }
    if (!LE_green_roof_diag) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][5.3][all_allocated] MISSING: LE_green_roof_diag\n";
        }
        result = false;
    }
    if (!LE_permeable_road_diag) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][5.3][all_allocated] MISSING: LE_permeable_road_diag\n";
        }
        result = false;
    }
    if (!is_green_roof) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][5.3][all_allocated] MISSING: is_green_roof\n";
        }
        result = false;
    }
    if (!is_permeable_road) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][5.3][all_allocated] MISSING: is_permeable_road\n";
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

    // Phase 5.1a: Multi-facet view factors (geometry only)
    if (!F_wall_sky) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][5.1a][all_allocated] MISSING: F_wall_sky\n";
        }
        result = false;
    }

    if (!F_wall_wall) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][5.1a][all_allocated] MISSING: F_wall_wall\n";
        }
        result = false;
    }

    if (!F_wall_road) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][5.1a][all_allocated] MISSING: F_wall_road\n";
        }
        result = false;
    }

    if (!F_road_sky) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][5.1a][all_allocated] MISSING: F_road_sky\n";
        }
        result = false;
    }

    if (!F_road_wall) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][5.1a][all_allocated] MISSING: F_road_wall\n";
        }
        result = false;
    }

    if (!F_roof_sky) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][5.1a][all_allocated] MISSING: F_roof_sky\n";
        }
        result = false;
    }

    // Phase 6.1: Tree canopy fields
    if (!H_tree) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][6.1][all_allocated] MISSING: H_tree\n";
        }
        result = false;
    }
    if (!H_crown_base) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][6.1][all_allocated] MISSING: H_crown_base\n";
        }
        result = false;
    }
    if (!LAD_bulk) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][6.1][all_allocated] MISSING: LAD_bulk\n";
        }
        result = false;
    }
    if (!crown_area_frac) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][6.1][all_allocated] MISSING: crown_area_frac\n";
        }
        result = false;
    }
    if (!Cd_leaf) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][6.1][all_allocated] MISSING: Cd_leaf\n";
        }
        result = false;
    }
    if (!is_tree) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][6.1][all_allocated] MISSING: is_tree\n";
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