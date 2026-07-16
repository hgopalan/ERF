/**
 * @file ERF_LNGPrerequisites.cpp
 * @brief LNG prerequisite validation implementation
 * @details
 * Implements check_lng_prerequisites() which validates configuration
 * and prints initialization summary.
 */

#include "ERF_LNGPrerequisites.H"
#include "ERF.H"
#include <cmath>
#include <iomanip>
#include <sstream>

void check_lng_prerequisites(const LNGParams& params,
                              const amrex::Geometry& geom_atm,
                              const amrex::BoxArray& ba_atm)
{
    // Check 1: grid_ratio >= 1
    if (params.grid_ratio < 1) {
        amrex::Abort("[LNG] ERROR: grid_ratio must be >= 1. Got: " +
                     std::to_string(params.grid_ratio));
    }

    // Check 2: ATM domain size divisible by grid_ratio in x,y
    const auto& atm_domain = geom_atm.Domain();
    int nx = atm_domain.length(0);
    int ny = atm_domain.length(1);
    if ((nx % params.grid_ratio != 0) || (ny % params.grid_ratio != 0)) {
        amrex::Abort("[LNG] ERROR: ATM domain size not divisible by grid_ratio.\n" +
                     std::string("           nx=") + std::to_string(nx) +
                     " ny=" + std::to_string(ny) +
                     " grid_ratio=" + std::to_string(params.grid_ratio) +
                     "\n           Try: amr.n_cell must be divisible by grid_ratio");
    }

    // Check 3: No z-direction MPI decomposition — mirrors Dust Check 3.
    // The LNG gravity current and reduce operations operate on 2D k=0 slabs.
    // If boxes are split in z, each MPI rank sees only a partial column and
    // ReduceSum/ReduceMax over k=0 cells produces wrong results.
    // Fix: set amrex.max_grid_size_z = <domain_nz> in the inputs file.
    {
        int domain_nz = atm_domain.length(2);
        for (int i = 0; i < ba_atm.size(); ++i) {
            amrex::Box box = ba_atm[i];
            int box_nz = box.length(2);
            std::string msg = std::string("[LNG] Cannot decompose in z direction. ")
                            + "Set: amrex.max_grid_size_z = "
                            + std::to_string(domain_nz);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(box_nz == domain_nz, msg.c_str());
        }
        if (params.lng_debug) {
            amrex::Print() << "[LNG DEBUG] Prerequisite check 3 passed: No z-decomposition. "
                           << "domain_nz=" << domain_nz
                           << ", all boxes have nz=" << domain_nz << "\n";
        }
    }

    // Check 4: All ATM boxes have x,y sizes divisible by grid_ratio
    for (int i = 0; i < ba_atm.size(); ++i) {
        amrex::Box box = ba_atm[i];
        int box_nx = box.length(0);
        int box_ny = box.length(1);
        if (box_nx % params.grid_ratio != 0 || box_ny % params.grid_ratio != 0) {
            amrex::Abort("[LNG] ERROR: Box sizes not divisible by grid_ratio. "
                         "All atmospheric box x,y sizes must be divisible by grid_ratio=" +
                         std::to_string(params.grid_ratio) +
                         ". Adjust grid_ratio or amrex.max_grid_size.");
        }
    }

    // Check 5: LNG composition mole fractions sum to ~1.0 (±1% tolerance)
    amrex::Real mole_sum = params.ch4_mole_fraction + params.c2h6_mole_fraction +
                           params.n2_mole_fraction;
    if (mole_sum < 0.99 || mole_sum > 1.01) {
        amrex::Warning("[LNG WARNING] LNG mole fractions sum to " +
                       std::to_string(mole_sum) + " (expected ~1.0).\n" +
                       "             CH4=" + std::to_string(params.ch4_mole_fraction) +
                       " C2H6=" + std::to_string(params.c2h6_mole_fraction) +
                       " N2=" + std::to_string(params.n2_mole_fraction));
    }

    // Check 6: lfl_vol_fraction < ufl_vol_fraction
    if (params.lfl_vol_fraction >= params.ufl_vol_fraction) {
        amrex::Abort("[LNG] ERROR: LFL >= UFL. LFL=" +
                     std::to_string(params.lfl_vol_fraction) +
                     " UFL=" + std::to_string(params.ufl_vol_fraction));
    }

    // Check 7: spill_rate_kg_s >= 0
    if (params.spill_rate_kg_s < 0.0) {
        amrex::Abort("[LNG] ERROR: spill_rate_kg_s must be >= 0. Got: " +
                     std::to_string(params.spill_rate_kg_s));
    }

    // Check 8: Domain z-index starts at 0
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        atm_domain.smallEnd(2) == 0,
        "[LNG] Domain z-index must start at 0. Check Geometry configuration.");

    // Print initialization summary if verbose >= 1
    if (params.verbose >= 1) {
        amrex::Print() << "[LNG] ============================================================\n"
                       << "[LNG] ERF-LNG Phase 1 initialized\n"
                       << "[LNG]   Pool area       : "
                       << std::fixed << std::setprecision(2) << params.pool_area_m2 << " m^2\n"
                       << "[LNG]   Spill rate      : "
                       << std::fixed << std::setprecision(2) << params.spill_rate_kg_s << " kg/s\n"
                       << "[LNG]   LNG composition : CH4="
                       << std::fixed << std::setprecision(2) << params.ch4_mole_fraction
                       << "  C2H6=" << params.c2h6_mole_fraction
                       << "  N2=" << params.n2_mole_fraction << "\n"
                       << "[LNG]   Mol. weight     : "
                       << std::fixed << std::setprecision(2) << params.mol_weight_LNG << " g/mol\n"
                       << "[LNG]   Boiling point   : "
                       << std::fixed << std::setprecision(2) << params.lng_boil_temp_K << " K\n"
                       << "[LNG]   LFL/UFL         : "
                       << std::fixed << std::setprecision(3) << params.lfl_vol_fraction
                       << " / " << params.ufl_vol_fraction << " (vol/vol)\n"
                       << "[LNG]   Grid ratio      : " << params.grid_ratio << "\n"
                       << "[LNG]   ATM feedback    : "
                       << std::fixed << std::setprecision(2) << params.atm_feedback << "\n"
                       << "[LNG]   Debug mode      : "
                       << (params.lng_debug ? "on" : "off") << "\n"
                       << "[LNG]   Verbose level   : " << params.verbose << "\n"
                       << "[LNG] ============================================================\n";
    }
}