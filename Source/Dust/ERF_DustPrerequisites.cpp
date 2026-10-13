/**
 * @file ERF_DustPrerequisites.cpp
 * @brief Implementation of prerequisite verification for the ERF-Dust module.
 */

#include <ERF_DustPrerequisites.H>
#include <ERF.H>
#include <ERF_SurfaceLayer.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>

#include <string>

namespace {

// What to change when a box's x or y length is not a multiple of the grid ratio C.
// AMReX accepts only power-of-2 blocking factors, so the advice depends on C. With
// blocking factor 1 AMReX cuts a length of n cells into the fewest near-equal
// pieces no longer than max_grid_size (20 cells with max_grid_size 8 become
// 7 + 7 + 6), and cuts further when there are more ranks than boxes. On level 0,
// when max_grid_size covers the domain in x and y on more than one rank, ERF
// replaces those boxes with one per rank (ERF::MakeNewLevelFromScratch), whatever
// the blocking factor. On a refined level n is the refinement region's extent.
std::string box_ratio_advice (int C, int lev)
{
    const std::string c = std::to_string(C);
    std::string advice = (lev > 0)
        ? "On a refined level n is the refinement region's x,y extent in that level's "
          "cells, and that level's amr.max_grid_size and amr.blocking_factor apply. "
        : "";
    if ((C & (C - 1)) == 0) {
        advice += "Set amr.blocking_factor_x and amr.blocking_factor_y to a multiple of " + c
                + ", with amr.max_grid_size_x, amr.max_grid_size_y and the x,y cell counts n "
                  "multiples of it too.";
    } else {
        advice += "AMReX blocking factors must be powers of 2, which " + c + " is not, so set "
                  "the blocking factor to 1, make the x,y cell counts n multiples of " + c
                + ", and set amr.max_grid_size_x and amr.max_grid_size_y to a multiple of " + c
                + " that divides n: AMReX cuts a length of n cells into the fewest near-equal "
                  "pieces no longer than max_grid_size (n = 40 and max_grid_size = 20 give "
                  "20 + 20). Use at least as many boxes as ranks, since AMReX cuts the boxes "
                  "further otherwise; one box works on one rank.";
    }
    advice += " With more than one rank keep max_grid_size below n in x or y: when it covers "
              "the whole domain in both, ERF cuts level 0 into one box per rank itself, "
              "whatever the blocking factor.";
    return advice;
}

} // namespace

void verify_dust_prerequisites(const ERF&          erf,
                               const SurfaceLayer* surface_layer,
                               const DustParams&   dust_params)
{
    // Check 1: SurfaceLayer pointer is not null
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        surface_layer != nullptr,
        "[DUST] SurfaceLayer is required. Set: zlo.type = \"surface_layer\"");

    // The dust layer lives on level 0 only: its sources go into the level-0 RHS and
    // a finer level's average-down would overwrite them.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(erf.maxLevel() == 0,
        "[DUST] The dust module runs on a single level. Set: amr.max_level = 0");

    // Get atmospheric grid information
    const amrex::BoxArray& ba_atm = erf.boxArray(0);
    const amrex::DistributionMapping& dm_atm = erf.DistributionMap(0);
    const amrex::Geometry& geom_atm = erf.Geom(0);

    // Check 2: the dust scalar must be transported. erf.transport_scalar = false
    // removes the whole scalar block (dust included) from the advection and
    // from the slow RHS update, so the emission would be injected and dropped.
    {
        amrex::ParmParse pp_erf("erf");
        bool transport_scalar = true;
        pp_erf.query("transport_scalar", transport_scalar);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(transport_scalar,
            "[DUST] erf.transport_scalar = false drops the dust scalar from the transport; "
            "set erf.transport_scalar = true (the default) with erf.dust.enable");
    }
    // (erf.most.z0 is validated where the surface layer reads it; the dust
    // module takes the surface layer's u* and its own erf.dust.z0_dust)

    // Get domain information
    const amrex::Box& domain = geom_atm.Domain();
    int domain_nz = domain.length(2);
    // the wind extraction places the surface from the first two cell centres
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(domain_nz >= 2,
        "[DUST] the dust module needs at least two cells in z (amr.n_cell)");

    // Check 3: No z-direction MPI decomposition
    for (int i = 0; i < ba_atm.size(); ++i) {
        amrex::Box box = ba_atm[i];
        int box_nz = box.length(2);
        std::string msg = std::string("[DUST] Cannot decompose in z direction. ")
                        + "Set: amr.max_grid_size_z = " + std::to_string(domain_nz)
                        + " (or larger)";
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(box_nz == domain_nz, msg.c_str());
    }

    if (dust_params.dust_debug) {
        amrex::Print() << "[DUST DEBUG] Prerequisite check 3 passed: No z-decomposition. "
                       << "domain_nz=" << domain_nz << ", all boxes have nz=" << domain_nz << "\n";
    }

    // Check 4: grid_ratio >= 1
    std::string msg4 = std::string("[DUST] Invalid grid_ratio ")
                     + std::to_string(dust_params.grid_ratio)
                     + ". Set: erf.dust.grid_ratio >= 1";
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dust_params.grid_ratio >= 1, msg4.c_str());

    if (dust_params.dust_debug) {
        amrex::Print() << "[DUST DEBUG] Prerequisite check 4 passed: "
                       << "grid_ratio=" << dust_params.grid_ratio << " >= 1\n";
    }

    // Check 5: All atmospheric boxes have x,y sizes divisible by grid_ratio
    for (int i = 0; i < ba_atm.size(); ++i) {
        amrex::Box box = ba_atm[i];
        int box_nx = box.length(0);
        int box_ny = box.length(1);
        if (box_nx % dust_params.grid_ratio != 0 || box_ny % dust_params.grid_ratio != 0) {
            amrex::Abort("[DUST] Box sizes not divisible by grid_ratio: box " + std::to_string(i)
                         + " of level 0 is " + std::to_string(box_nx) + " x " + std::to_string(box_ny)
                         + " cells, and every atmospheric x,y box length must be divisible by "
                         "erf.dust.grid_ratio = " + std::to_string(dust_params.grid_ratio) + ". "
                         + box_ratio_advice(dust_params.grid_ratio, 0)
                         + " Or change erf.dust.grid_ratio.");
        }
    }

    if (dust_params.dust_debug) {
        amrex::Print() << "[DUST DEBUG] Prerequisite check 5 passed: "
                       << "All " << ba_atm.size() << " box(es) have x,y sizes divisible by "
                       << "grid_ratio=" << dust_params.grid_ratio << "\n";
    }

    // Check 6: DistributionMapping size matches BoxArray size
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        static_cast<int>(dm_atm.size()) == ba_atm.size(),
        "[DUST] DistributionMapping size mismatch. Check AMReX configuration.");

    if (dust_params.dust_debug) {
        amrex::Print() << "[DUST DEBUG] Prerequisite check 6 passed: "
                       << "DistributionMapping size=" << dm_atm.size()
                       << " matches BoxArray size=" << ba_atm.size() << "\n";
    }

    // Check 7: Domain z-index starts at 0
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        domain.smallEnd(2) == 0,
        "[DUST] Domain z-index must start at 0. Check Geometry configuration.");

    if (dust_params.dust_debug) {
        amrex::Print() << "[DUST DEBUG] Prerequisite check 7 passed: "
                       << "Domain z-index starts at 0\n";
    }

    // Check 8: Domain physical height exceeds MRF reference height
    // Default z_ref = 10.0 m (from erf.most.zref, read by SurfaceLayer)
    // For now, just ensure domain height is positive
    amrex::Real dz = geom_atm.ProbHi(2) - geom_atm.ProbLo(2);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        dz > 0.0,
        "[DUST] Domain physical height must be positive. Check Geometry configuration.");
    {
        std::string msg = std::string("[DUST] erf.dust.zref (") + std::to_string(dust_params.zref)
                        + " m) must lie below the domain top (" + std::to_string(dz) + " m)";
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dust_params.zref < dz, msg.c_str());
    }

    {
        // The wind at zref is interpolated between cell centres and clamped to
        // the lowest one: below half the first cell thickness it is the first
        // centre's wind, not the wind at zref.
        const amrex::Real dz0 = geom_atm.CellSize(2);
        if (dust_params.zref < 0.5 * dz0) {
            amrex::Print() << "[DUST] WARNING: erf.dust.zref = " << dust_params.zref
                           << " m is below the first cell centre (" << 0.5 * dz0
                           << " m); the dust wind is the first cell's wind, not the wind at zref\n";
        }
    }
    if (dust_params.dust_debug) {
        amrex::Print() << "[DUST DEBUG] Prerequisite check 8 passed: "
                       << "Domain physical height=" << dz << " m > 0\n";
    }

    // Check 9: the dust wind-extraction height is the surface layer's reference
    // height. The dust u* is the surface layer's (a log law between z0_dust and erf.dust.zref only with terrain_ustar = loglaw)
    // using the wind the surface layer sampled at erf.most.zref, so the two must
    // agree; with erf.most.zref unset the surface layer picks its own height
    // and the deck has to set erf.dust.zref to the same value.
    {
        amrex::ParmParse pp_most("erf.most");
        amrex::Real most_zref = -1.0;
        // MOSTAverage queryAdds its sentinel (-1) when the deck sets nothing, so a
        // non-positive value means "not specified", not a height.
        if (pp_most.query("zref", most_zref) && most_zref > 0.0) {
            std::string msg = std::string("[DUST] erf.dust.zref (")
                            + std::to_string(dust_params.zref)
                            + ") must equal erf.most.zref ("
                            + std::to_string(most_zref) + ")";
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::abs(dust_params.zref - most_zref) <= 1.0e-6 * std::max(most_zref, amrex::Real(1.0)),
                msg.c_str());
        } else {
            amrex::Print() << "[DUST] WARNING: erf.most.zref is not set; erf.dust.zref = "
                           << dust_params.zref << " m must match the surface layer's "
                           << "reference height\n";
        }
        if (dust_params.dust_debug) {
            amrex::Print() << "[DUST DEBUG] Prerequisite check 9 passed: "
                           << "erf.dust.zref=" << dust_params.zref << " m\n";
        }
    }

    amrex::Print() << "[DUST] All prerequisites verified\n";
}
