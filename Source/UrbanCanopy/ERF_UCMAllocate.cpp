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

    // Summary message
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][allocate_ucm_fields] allocated 19 MultiFabs on UCM grid "
                << "at lev=" << lev << "\n";
    }

    // Verify all are allocated
    AMREX_ALWAYS_ASSERT(fields.all_allocated());
}

void fill_ucm_fields_from_csv(UCMFields& fields,
                              const UCMGrid& ucm_grid,
                              const UCMBuildingLayoutReader& building_reader,
                              const UCMMaterialRegistry& material_registry,
                              int grid_ratio,
                              int lev,
                              bool ucm_debug)
{
    // Precondition checks
    AMREX_ALWAYS_ASSERT(fields.all_allocated());
    AMREX_ALWAYS_ASSERT(building_reader.size() > 0);
    AMREX_ALWAYS_ASSERT(material_registry.size() > 0);

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
    fields.T_skin_roof->setVal(293.15);
    fields.T_skin_wall->setVal(293.15);
    fields.T_skin_road->setVal(293.15);
    fields.T_canyon_air->setVal(293.15);
    fields.H_sensible->setVal(0.0);
    fields.LE_latent->setVal(0.0);
    fields.is_urban->setVal(0);
    fields.mat_id_roof->setVal(0);
    fields.mat_id_wall->setVal(0);
    fields.mat_id_road->setVal(0);

    // Get const references to the broadcast data
    const auto& rows = building_reader.rows();

    // Properly set values using MFIter loop to respect domain decomposition
    for (int row_idx = 0; row_idx < building_reader.size(); ++row_idx) {
        const auto& row = rows[row_idx];
        int i_atm = row.i;
        int j_atm = row.j;
        
        // Look up and pre-compute material properties
        const UCMMaterial& roof_mat = material_registry.lookup(row.roof_mat_id);
        const UCMMaterial& wall_mat = material_registry.lookup(row.wall_mat_id);
        const UCMMaterial& road_mat = material_registry.lookup(row.road_mat_id);
        
        // For each UCM cell in the domain, if it maps to this ATM cell, fill it
        for (MFIter mfi(*(fields.H_bldg)); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto H_bldg_arr = fields.H_bldg->array(mfi);
            auto W_road_arr = fields.W_road->array(mfi);
            auto W_roof_arr = fields.W_roof->array(mfi);
            auto albedo_roof_arr = fields.albedo_roof->array(mfi);
            auto albedo_wall_arr = fields.albedo_wall->array(mfi);
            auto albedo_road_arr = fields.albedo_road->array(mfi);
            auto emissivity_roof_arr = fields.emissivity_roof->array(mfi);
            auto emissivity_wall_arr = fields.emissivity_wall->array(mfi);
            auto emissivity_road_arr = fields.emissivity_road->array(mfi);
            auto T_skin_roof_arr = fields.T_skin_roof->array(mfi);
            auto T_skin_wall_arr = fields.T_skin_wall->array(mfi);
            auto T_skin_road_arr = fields.T_skin_road->array(mfi);
            auto T_canyon_air_arr = fields.T_canyon_air->array(mfi);
            auto is_urban_arr = fields.is_urban->array(mfi);
            auto mat_id_roof_arr = fields.mat_id_roof->array(mfi);
            auto mat_id_wall_arr = fields.mat_id_wall->array(mfi);
            auto mat_id_road_arr = fields.mat_id_road->array(mfi);

            // Loop over cells in this box
            for (int i_ucm = bx.smallEnd(0); i_ucm <= bx.bigEnd(0); ++i_ucm) {
                for (int j_ucm = bx.smallEnd(1); j_ucm <= bx.bigEnd(1); ++j_ucm) {
                    // Map UCM grid indices to ATM grid indices
                    int i_atm_cell = i_ucm / grid_ratio;
                    int j_atm_cell = j_ucm / grid_ratio;
                    
                    // If this UCM cell corresponds to the current building row, fill it
                    if (i_atm_cell == i_atm && j_atm_cell == j_atm) {
                        IntVect iv(i_ucm, j_ucm, 0);
                        
                        // Fill building morphology
                        H_bldg_arr(iv, 0) = row.height_m;
                        W_road_arr(iv, 0) = row.W_road_m;
                        W_roof_arr(iv, 0) = row.W_roof_m;
                        is_urban_arr(iv, 0) = row.is_urban;
                        
                        // Store material IDs
                        mat_id_roof_arr(iv, 0) = row.roof_mat_id;
                        mat_id_wall_arr(iv, 0) = row.wall_mat_id;
                        mat_id_road_arr(iv, 0) = row.road_mat_id;
                        
                        // Fill material properties
                        albedo_roof_arr(iv, 0) = roof_mat.albedo;
                        albedo_wall_arr(iv, 0) = wall_mat.albedo;
                        albedo_road_arr(iv, 0) = road_mat.albedo;
                        
                        emissivity_roof_arr(iv, 0) = roof_mat.emissivity;
                        emissivity_wall_arr(iv, 0) = wall_mat.emissivity;
                        emissivity_road_arr(iv, 0) = road_mat.emissivity;
                        
                        // Set initial temperatures
                        T_skin_roof_arr(iv, 0) = 293.15;
                        T_skin_wall_arr(iv, 0) = 293.15;
                        T_skin_road_arr(iv, 0) = 293.15;
                        T_canyon_air_arr(iv, 0) = 293.15;
                    }
                }
            }
        }
    }

    if (ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.1][fill_ucm_fields_from_csv] "
                << "populated fields from CSV at lev=" << lev << "\n";
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

    // Fill temperature fields (Phase 1.2: placeholders from test_surf_temp_K)
    fields.T_skin_roof->setVal(params.test_surf_temp_K);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][fill_ucm_fields_homogeneous] T_skin_roof = "
                << params.test_surf_temp_K << " K\n";
    }

    fields.T_skin_wall->setVal(params.test_surf_temp_K);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][fill_ucm_fields_homogeneous] T_skin_wall = "
                << params.test_surf_temp_K << " K\n";
    }

    fields.T_skin_road->setVal(params.test_surf_temp_K);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][fill_ucm_fields_homogeneous] T_skin_road = "
                << params.test_surf_temp_K << " K\n";
    }

    fields.T_canyon_air->setVal(params.test_surf_temp_K);
    if (params.ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.2][fill_ucm_fields_homogeneous] T_canyon_air = "
                << params.test_surf_temp_K << " K\n";
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

    return result;
}
