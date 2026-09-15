#include <ERF_FirePrerequisites.H>
#include <ERF.H>
#include <ERF_SurfaceLayer.H>
#include <ERF_IndexDefines.H>
#include <AMReX_ParmParse.H>

#include <string>

using namespace amrex;

int fire_anchor_level(const FireParams& fire_params, int finest_level)
{
    const int lev = (fire_params.anchor_level < 0) ? finest_level : fire_params.anchor_level;
    if (lev > finest_level) {
        Abort("[FIRE] erf.fire.anchor_level = " + std::to_string(fire_params.anchor_level)
              + " is above the finest level of this run (" + std::to_string(finest_level)
              + "). Set it to a level from 0 to " + std::to_string(finest_level)
              + ", or leave it unset for the finest level.");
    }
    return lev;
}

void verify_fire_prerequisites(const ERF& erf,
                               int lev,
                               const SurfaceLayer* surface_layer,
                               const FireParams& fire_params)
{
    // Check 1: Surface layer BC type is "surface_layer"
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        surface_layer != nullptr,
        "fire module requires SurfaceLayer. "
        "Check: phys_bc_type[zlo] == surface_layer AND erf.most.z0 is set");

    // Check 2: the level the fire grid refines exists. fire_anchor_level()
    // resolves the input and aborts above the finest level; this is the backstop
    // for any other caller.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(lev >= 0 && lev <= erf.finestLevel(),
        "[FIRE] Internal error: the fire grid's level is outside 0..finest_level");

    // Check 3: static refinement. The fire grid is built once from the boxes of
    // its level, and its heat, moisture and smoke go into that level's source,
    // so the level must keep its boxes for the whole run. Level 0 never regrids.
    if (lev > 0) {
        ParmParse pp_erf("erf");
        int regrid_int = -1;
        pp_erf.query("regrid_int", regrid_int);
        if (regrid_int > 0) {
            Abort("[FIRE] The fire grid on level " + std::to_string(lev)
                  + " is built once from that level's boxes, so the refinement must not change, but "
                  "erf.regrid_int = " + std::to_string(regrid_int) + " regrids it. Use static refinement "
                  "boxes with erf.regrid_int = -1, or set erf.fire.anchor_level = 0.");
        }

        // Check 4: the dust layer and the fire-dust coupling live on level 0 and
        // share the fire grid's index space; they cannot follow a finer fire grid.
        ParmParse pp_dust("erf.dust");
        bool dust_enable = false;
        pp_dust.query("enable", dust_enable);
        if (dust_enable) {
            Abort("[FIRE] The dust layer and the fire-dust coupling run on level 0, so they cannot "
                  "run with the fire grid on level " + std::to_string(lev) + " (erf.dust.enable = true). "
                  "Set erf.fire.anchor_level = 0, or turn the dust layer off.");
        }
    }

    // Check 7: No z-decomposition (all z-levels on same rank)
    // Use the public boxArray() accessor instead of the protected grids[] member
    const BoxArray& ba = erf.boxArray(lev);
    const Box& domain  = erf.Geom(lev).Domain();
    int domain_nz      = domain.length(2);

    for (int i = 0; i < ba.size(); ++i) {
        const Box& b  = ba[i];
        int box_nz    = b.length(2);
        // Build the message string before passing to the macro
        std::string msg = std::string("[FIRE] Cannot decompose in z direction. ")
                        + "Every box on level " + std::to_string(lev)
                        + " must span the level's full height, since the fire model interpolates "
                          "through whole columns: set amr.max_grid_size_z = " + std::to_string(domain_nz)
                        + " (or larger), and on a refined level use a refinement box that reaches the "
                          "domain top (erf.<box>.in_box_hi without a z bound)";
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(box_nz == domain_nz, msg.c_str());
    }

    // Check 7b: the boxes cover one rectangle. The fire grid treats the edges of
    // their bounding rectangle as its domain edges and fills its ghost cells
    // there; a hole or a second patch would leave cells with no neighbour data.
    {
        BoxList bl_2d;
        for (int i = 0; i < ba.size(); ++i) {
            Box b = ba[i];
            b.setSmall(2, 0);
            b.setBig(2, 0);
            bl_2d.push_back(b);
        }
        const BoxArray ba_2d(std::move(bl_2d));
        const Box region = ba_2d.minimalBox();
        if (region.numPts() != ba_2d.numPts()) {
            Abort("[FIRE] The boxes of level " + std::to_string(lev) + " cover "
                  + std::to_string(ba_2d.numPts()) + " of the " + std::to_string(region.numPts())
                  + " columns of their bounding rectangle, but the fire grid needs one rectangle. "
                  "Refine one rectangle on that level (a single refinement box), or set "
                  "erf.fire.anchor_level to a level whose boxes form one.");
        }
    }

    // Check 8: grid_ratio >= 1
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        fire_params.grid_ratio >= 1,
        "[FIRE] Fire grid_ratio requires value >= 1. Set: erf.fire.grid_ratio >= 1");

    // Check 9: All boxes divisible by grid_ratio in x,y
    int C = fire_params.grid_ratio;
    for (int i = 0; i < ba.size(); ++i) {
        const Box& b = ba[i];
        int nx = b.length(0);
        int ny = b.length(1);
        std::string msg = std::string("[FIRE] Box sizes not divisible by grid_ratio. ")
                        + "Adjust amr.max_grid_size_x and amr.max_grid_size_y so all x,y lengths "
                          "of the boxes on level " + std::to_string(lev) + " are divisible by "
                        + std::to_string(C);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE((nx % C == 0 && ny % C == 0), msg.c_str());
    }

    // Check 10: DistributionMapping size matches grid size
    // Use the public boxArray() accessor here too
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        erf.DistributionMap(lev).size() == erf.boxArray(lev).size(),
        "[FIRE] Internal error: dmap size != grids size");

    // Check 11: Domain z-index starts at 0
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        erf.Geom(lev).Domain().smallEnd(2) == 0,
        "[FIRE] Domain z requires start at index 0");

    // Check 11b: the level's domain starts at index 0 in x and y.
    //
    // A fire cell maps back to its atmospheric cell by i_a = i_f / C + atm_lo,
    // with atm_lo the lower corner of the fire region in the level's index space
    // (FireGrid::atm_lo), in ERF_FireGrid.H, ERF_FireAtmCoupling.H,
    // ERF_FireWindExtract.cpp and ERF_TerrainSlope.cpp; create_fire_grid() takes
    // the region's physical position from ProbLo plus its index offset from the
    // domain's corner. Every AMReX level has a zero-based domain, so this
    // asserts the convention rather than relying on it silently.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        erf.Geom(lev).Domain().smallEnd(0) == 0 && erf.Geom(lev).Domain().smallEnd(1) == 0,
        "[FIRE] Domain x and y require start at index 0: the fire-to-atmosphere "
        "index mapping assumes a zero-based domain");

    // Check 12: Domain height > wind_ref_ht
    Real prob_hi_z = erf.Geom(lev).ProbHi(2);
    std::string msg12 = std::string("[FIRE] Domain height ")
                      + std::to_string(prob_hi_z)
                      + " should exceed wind_ref_ht "
                      + std::to_string(fire_params.wind_ref_ht)
                      + ". Increase geometry.prob_hi(2)";
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(prob_hi_z > fire_params.wind_ref_ht, msg12.c_str());

    // Check 13: divisors and step sizes read straight from the inputs file.
    // ParmParse applies no bounds, and a zero reaches the solver as an integer
    // division by zero (SIGFPE) or, for cfl, a zero-length subcycle that never
    // terminates. Each is only checked on the path that actually uses it.
    if (fire_params.propagation_method == "levelset") {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            fire_params.levelset_reinit_every >= 1,
            "[FIRE] erf.fire.levelset.reinit_every must be >= 1; it is used as a "
            "modulus divisor, so 0 is an integer division by zero");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            fire_params.levelset_cfl > 0.0,
            "[FIRE] erf.fire.levelset.cfl must be > 0; a non-positive value gives a "
            "zero-length subcycle and the level-set loop never advances");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            fire_params.levelset_reinit_iters >= 0,
            "[FIRE] erf.fire.levelset.reinit_iters must be >= 0");
    }

    if (fire_params.spotting.enable) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            fire_params.spotting.spotting_interval >= 1,
            "[FIRE] erf.fire.spotting.spotting_interval must be >= 1; it is used as "
            "a modulus divisor, so 0 is an integer division by zero");
    }

    // Check 14: the FARSITE subcycle divides by cfl_fire in the same way.
    if (fire_params.propagation_method != "levelset") {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            fire_params.farsite_cfl_fire > 0.0,
            "[FIRE] erf.fire.farsite.cfl_fire must be > 0; a non-positive value "
            "makes every fire substep zero-length");
    }

    amrex::Print() << "[FIRE] All prerequisites verified successfully" << std::endl;
}
