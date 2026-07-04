/**
 * \file ERF_FirePrerequisites.cpp
 *
 * \brief Implementation of fire module prerequisite validation.
 */

#include "ERF_FirePrerequisites.H"
#include "ERF_IndexDefines.H"
#include <AMReX_Orientation.H>

void verify_fire_prerequisites(
    const amrex::Vector<amrex::BCRec>& phys_bc_type,
    const SurfaceLayer* surface_layer,
    const amrex::Vector<amrex::BoxArray>& grids,
    const amrex::Vector<amrex::DistributionMapping>& dmap,
    const amrex::Vector<amrex::Geometry>& geom,
    const FireParams& fire_params,
    int lev)
{
    using amrex::Orientation;

    // Check 1: Surface layer BC type must be "surface_layer"
    amrex::BCRec bc_rec_zlo = phys_bc_type[Orientation(amrex::Direction::z, Orientation::low)];
    ERF_BC bc_type_zlo = bc_rec_zlo.lo(2);  // Get the z-low boundary condition
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        bc_type_zlo == ERF_BC::surface_layer,
        "Fire module requires z-boundary type to be 'surface_layer'.\n"
        "  Fix: Add 'erf.phys_bc_type.zlo = \"surface_layer\"' to inputs file.");

    // Check 2: SurfaceLayer pointer is not null
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        surface_layer != nullptr,
        "Fire module internal error: SurfaceLayer pointer is null.");

    // Check 3: u_star field is allocated
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        surface_layer->get_u_star(lev) != nullptr,
        "Fire module requires friction velocity (u_star) to be computed.\n"
        "  Fix: Ensure make_SurfaceLayer_at_level(0,...) is called before fire initialization.");

    // Check 4: z0 (roughness) field is set
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        surface_layer->get_z0(lev) != nullptr,
        "Fire module requires surface roughness (z0) to be set.\n"
        "  Fix: Add 'erf.most.z0 = <value>' to inputs file (e.g., 0.1 for grass).");

    // Check 5: olen (Obukhov length) field is computed
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        surface_layer->get_olen(lev) != nullptr,
        "Fire module internal error: Obukhov length (olen) field is not computed.");

    // Check 6: MAC-averaged velocity components are available
    const amrex::MultiFab* uavg = surface_layer->get_mac_avg(lev, 0);
    const amrex::MultiFab* vavg = surface_layer->get_mac_avg(lev, 1);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        uavg != nullptr && vavg != nullptr,
        "Fire module requires updated MOST fluxes (MAC-averaged winds).\n"
        "  Fix: Ensure SurfaceLayer::update_fluxes() has been called.");

    // Check 7: No z-direction domain decomposition (MPI z-split check)
    //          Each box must span the full z-domain
    int nz_domain = geom[lev].Domain().length(2);
    for (const auto& box : grids[lev]) {
        int nz_box = box.length(2);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            nz_box == nz_domain,
            "Fire module does not support MPI z-decomposition.\n"
            "  All atmospheric boxes must span the full z-domain.\n"
            "  Fix: Set 'erf.max_grid_size_z = " << nz_domain << "' and\n"
            "       'erf.blocking_factor_z = " << nz_domain << "' in inputs file.");
    }

    // Check 8: grid_ratio must be >= 1
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        fire_params.grid_ratio >= 1,
        "Fire grid refinement ratio must be >= 1.\n"
        "  Fix: Set 'erf.fire.grid_ratio' to 1 or greater.");

    // Check 9: All box x,y lengths divisible by grid_ratio
    int C = fire_params.grid_ratio;
    for (const auto& box : grids[lev]) {
        int nx = box.length(0);
        int ny = box.length(1);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            (nx % C == 0) && (ny % C == 0),
            "Fire grid refinement requires box dimensions to be divisible by grid_ratio.\n"
            "  Box x-length: " << nx << ", y-length: " << ny << "\n"
            "  Grid ratio: " << C << "\n"
            "  Fix: Adjust 'erf.max_grid_size' so all x,y box lengths are divisible by\n"
            "       'erf.fire.grid_ratio'.");
    }

    // Check 10: DistributionMapping consistency
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        dmap[lev].size() == grids[lev].size(),
        "Fire module internal error: DistributionMapping size mismatch.");

    // Check 11: Domain z-index starts at 0 (ground level)
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        geom[lev].Domain().smallEnd(2) == 0,
        "Fire module requires AMR level 0 at ground (z-index = 0).");

    // Check 12: Domain height >= wind reference height
    amrex::Real domain_height = geom[lev].ProbHi(2) - geom[lev].ProbLo(2);
    amrex::Real z_ref = fire_params.wind_ref_ht;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        domain_height > z_ref,
        "Fire module requires domain height > wind reference height.\n"
        "  Domain height: " << domain_height << " m\n"
        "  Wind reference height: " << z_ref << " m\n"
        "  Fix: Increase 'geometry.prob_hi(2)' in inputs file.");
}
