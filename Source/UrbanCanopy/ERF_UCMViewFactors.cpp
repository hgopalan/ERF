/**
 * @file ERF_UCMViewFactors.cpp
 * @brief Implementation of Phase 5.1a view-factor computation
 */

#include <UrbanCanopy/ERF_UCMViewFactors.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <cmath>

void compute_view_factors(amrex::MultiFab& F_wall_sky,
                          amrex::MultiFab& F_wall_wall,
                          amrex::MultiFab& F_wall_road,
                          amrex::MultiFab& F_road_sky,
                          amrex::MultiFab& F_road_wall,
                          amrex::MultiFab& F_roof_sky,
                          const amrex::MultiFab& H_bldg,
                          const amrex::MultiFab& W_road,
                          const amrex::iMultiFab& is_urban,
                          int lev,
                          bool ucm_debug)
{
    amrex::ignore_unused(lev);

    constexpr amrex::Real eps = 1.0e-6;

    // Zero-init all outputs
    F_wall_sky.setVal(0.0);
    F_wall_wall.setVal(0.0);
    F_wall_road.setVal(0.0);
    F_road_sky.setVal(0.0);
    F_road_wall.setVal(0.0);
    F_roof_sky.setVal(1.0);   // Roof always sees full sky

    for (amrex::MFIter mfi(F_wall_sky, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();

        auto const H_a       = H_bldg.const_array(mfi);
        auto const W_a       = W_road.const_array(mfi);
        auto const is_urb    = is_urban.const_array(mfi);

        auto Fws_a  = F_wall_sky.array(mfi);
        auto Fww_a  = F_wall_wall.array(mfi);
        auto Fwr_a  = F_wall_road.array(mfi);
        auto Frs_a  = F_road_sky.array(mfi);
        auto Frw_a  = F_road_wall.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) noexcept {
            if (is_urb(i, j, 0) == 0) {
                // Non-urban: no canyon, no walls; roof-sky handled by setVal above
                return;
            }

            const amrex::Real H = H_a(i, j, 0);
            const amrex::Real W = W_a(i, j, 0);

            // Degenerate canyon: treat as isolated building
            if (H < eps || W < eps) {
                Fws_a(i, j, 0) = 1.0;
                Fww_a(i, j, 0) = 0.0;
                Fwr_a(i, j, 0) = 0.0;
                Frs_a(i, j, 0) = 1.0;
                Frw_a(i, j, 0) = 0.0;
                return;
            }

            const amrex::Real r = H / W;
            const amrex::Real d = std::sqrt(1.0 + r * r);

            // Hottel crossed-string 2D canyon view factors
            const amrex::Real F_wall_sky_val  = 0.5 * (1.0 + r - d);
            const amrex::Real F_wall_road_val = 0.5 * (1.0 + r - d);   // = F_wall_sky
            const amrex::Real F_wall_wall_val = d - r;                  // = 1 - 2*F_wall_sky

            const amrex::Real F_road_sky_val  = d - r;                  // = F_wall_wall
            const amrex::Real F_road_wall_val = 0.5 * (1.0 - (d - r));  // = (1 - F_road_sky)/2

            // Clamp to [0, 1] against floating-point drift near limits
            Fws_a(i, j, 0) = amrex::max(0.0, amrex::min(1.0, F_wall_sky_val));
            Fww_a(i, j, 0) = amrex::max(0.0, amrex::min(1.0, F_wall_wall_val));
            Fwr_a(i, j, 0) = amrex::max(0.0, amrex::min(1.0, F_wall_road_val));
            Frs_a(i, j, 0) = amrex::max(0.0, amrex::min(1.0, F_road_sky_val));
            Frw_a(i, j, 0) = amrex::max(0.0, amrex::min(1.0, F_road_wall_val));
        });
    }

    if (ucm_debug) {
        // Collectives on ALL ranks
        amrex::Real Fws_min = F_wall_sky.min(0, 0);
        amrex::Real Fws_max = F_wall_sky.max(0, 0);
        amrex::Real Fww_min = F_wall_wall.min(0, 0);
        amrex::Real Fww_max = F_wall_wall.max(0, 0);
        amrex::Real Fwr_min = F_wall_road.min(0, 0);
        amrex::Real Fwr_max = F_wall_road.max(0, 0);
        amrex::Real Frs_min = F_road_sky.min(0, 0);
        amrex::Real Frs_max = F_road_sky.max(0, 0);
        amrex::Real Frw_min = F_road_wall.min(0, 0);
        amrex::Real Frw_max = F_road_wall.max(0, 0);

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print()
                << "\n[UCM][5.1a][BANNER] View-factor field ranges (per-cell):\n"
                << "  F_wall_sky   min=" << Fws_min << " max=" << Fws_max << "\n"
                << "  F_wall_wall  min=" << Fww_min << " max=" << Fww_max << "\n"
                << "  F_wall_road  min=" << Fwr_min << " max=" << Fwr_max << "\n"
                << "  F_road_sky   min=" << Frs_min << " max=" << Frs_max << "\n"
                << "  F_road_wall  min=" << Frw_min << " max=" << Frw_max << "\n"
                << "  (F_roof_sky is uniformly 1.0 for all urban cells)\n\n";
        }
    }
}
