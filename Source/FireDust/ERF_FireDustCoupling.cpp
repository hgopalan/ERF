#include <ERF_FireDustCoupling.H>

#if defined(ERF_ENABLE_FIRE) && defined(ERF_USE_DUST)

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_MFIter.H>

using namespace amrex;

void FireDustCoupling::apply_burned_area_to_crust(MultiFab& dust_crust_index) const
{
    if (!enabled || fire_phi_scratch == nullptr) {
        return;
    }
    // fire_phi_scratch lives on the dust BoxArray (ERF.cpp copies the level set
    // onto it by index overlap with the dust periodicity), so burned cells are
    // read cell for cell: no host gather, no all-reduce, no domain-sized copy.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(fire_phi_scratch->boxArray() == dust_crust_index.boxArray(),
        "[FIRE-DUST] apply_burned_area_to_crust: the fire level-set scratch must be on the dust BoxArray");

    const amrex::Real reduction = post_fire_crust_reduction;

    for (MFIter mfi(dust_crust_index, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto crust = dust_crust_index.array(mfi);
        auto phi   = fire_phi_scratch->const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (phi(i, j, 0) < 0.0_rt) {
                crust(i, j, k) *= (1.0_rt - reduction);
                crust(i, j, k)  = amrex::max(crust(i, j, k), 0.0_rt);
            }
        });
    }

    if (debug) {
        amrex::Real crust_min = dust_crust_index.min(0);
        amrex::Real crust_max = dust_crust_index.max(0);
        amrex::Print() << "[DUST DEBUG] Fire-dust coupling: modified crust values "
                       << "crust_min=" << crust_min << ", crust_max=" << crust_max
                       << ", reduction=" << reduction << "\n";
    }
}

void FireDustCoupling::apply_fire_wind_to_dust_ustar(
    amrex::MultiFab&       dust_ustar_in,
    const amrex::MultiFab& fire_wind_scratch,
    const amrex::MultiFab& fire_phi_scratch,
    const amrex::Geometry& /*geom_dust*/,
    amrex::Real            z0,
    amrex::Real            zref,
    int                    C) const
{
    if (!enabled || !fire_wind_to_dust) return;

    // fire_wind_scratch lives on the DUST BoxArray (ERF.cpp copies the fire
    // wind onto it by index overlap), so the lookup is cell for cell. The
    // fire and dust grid ratios are asserted equal at start-up (ERF.cpp); a
    // ratio other than 1 would have read outside the fab.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(C == 1,
        "[FIRE-DUST] apply_fire_wind_to_dust_ustar: the fire wind scratch is on the dust grid, "
        "the fire-to-dust grid ratio must be 1 (erf.fire.grid_ratio == erf.dust.grid_ratio)");

    // Log-law constant: u* = U * kappa / ln(zref/z0); zref > z0 > 0 is
    // enforced where erf.fire_dust_wind_z0/zref are read (ERF.cpp)
    constexpr amrex::Real kappa = 0.4_rt;
    const amrex::Real log_ratio = std::log(amrex::max(zref, 1.0e-30_rt) / amrex::max(z0, 1.0e-30_rt));
    const amrex::Real inv_log   = 1.0_rt / amrex::max(log_ratio, 1.0e-30_rt);

    for (amrex::MFIter mfi(dust_ustar_in, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto ustar = dust_ustar_in.array(mfi);
        auto wind  = fire_wind_scratch.const_array(mfi);
        auto phi   = fire_phi_scratch.const_array(mfi);

        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                // Only inside the fire perimeter (phi < 0). Outside it the
                // reference wind is the atmospheric wind the surface layer
                // already saw, and a neutral log law on fire_dust_wind_z0
                // differs from the surface layer's u* by the roughness and
                // stability it ignores (1.5x domain-wide for most.z0 = 0.01
                // against 0.1), which until October 2026 overrode the surface
                // layer everywhere.
                if (phi(i, j, 0) >= 0.0) return;
                const amrex::Real u_avg = wind(i, j, 0, 0);
                const amrex::Real v_avg = wind(i, j, 0, 1);

                // Derive u* from fire wind speed via log-law
                const amrex::Real spd        = std::sqrt(u_avg*u_avg + v_avg*v_avg);
                const amrex::Real ustar_fire = spd * kappa * inv_log;

                // Dust emission driven by the larger of MRF u* and fire u*
                ustar(i, j, k) = amrex::max(ustar(i, j, k), ustar_fire);
            });
    }

    if (debug) {
        amrex::Real ustar_max = dust_ustar_in.max(0);
        amrex::Real wind_max  = fire_wind_scratch.max(0);
        amrex::Print() << "[DUST DEBUG] Phase 2 fire-wind coupling:"
                       << " fire_wind_max=" << wind_max << " m/s"
                       << " ustar_fire_max=" << ustar_max << " m/s"
                       << " C=" << C << "\n";
    }
}

#endif
