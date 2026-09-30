/**
 * \file ERF_ImmersedProjection.cpp
 *
 * Implicit immersed-forcing drag in the anelastic projection (erf.if_implicit_projection).
 *
 * The terrain immersed forcing drags the momentum of the solid faces toward zero at the rate
 * rate = beta_face * if_Cd_momentum / (dx dy dz)^(1/3). Applied explicitly, the drag is undone by
 * the projection, which treats the solid as fluid; the post-projection freeze of the solid
 * faces then leaves the surface faces (solid on one side only) unbalanced, so the solid row
 * under the surface becomes a mass sink and source. With the drag inside the projection,
 *
 *     m^{n+1} = (m* - grad phi) / (1 + dt rate) = sigma (m* - grad phi),
 *     div(sigma grad phi) = div(sigma m*),
 *
 * the projected momenta are divergence free everywhere, the solid included, and no freeze is
 * needed. This is the ImmersedTerrain.implicit_projection of kynema-sgf. The faces where the
 * source term applies the wall law instead of the drag keep sigma = 1. For buildings the rate is
 * the linear drag of ImmersedForcingBuildings_*mom (partial-wall and interior faces, from the
 * masks of ERF_ImmersedBuildingMasks.H), and the faces between two solid cells take the full rate.
 */
#include "ERF.H"
#include "ERF_ImmersedWallCell.H"
#include "ERF_ImmersedBuildingMasks.H"

#ifdef ERF_USE_FFT
#include "ERF_ImmersedPoisson.H"
#include <AMReX_GMRES.H>
#endif

using namespace amrex;

void ERF::make_if_projection_sigma (int lev, Real dt, Array<MultiFab,AMREX_SPACEDIM>& sigma)
{
    BL_PROFILE("ERF::make_if_projection_sigma()");

    AMREX_ALWAYS_ASSERT(terrain_blanking[lev]);
    const MultiFab& tb = *terrain_blanking[lev];

    // 1 in the valid cells of this level (and their periodic images), 0 elsewhere: a face is
    // interior to the level's grids when both of its cells are 1. The other faces carry the
    // domain or c/f boundary data of the projection and keep sigma = 1.
    iMultiFab inside(grids[lev], dmap[lev], 1, 1);
    inside.setVal(0);
    inside.setVal(1, 0, 1, 0);
    inside.FillBoundary(geom[lev].periodicity());

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        sigma[dir].define(convert(grids[lev], IntVect::TheDimensionVector(dir)), dmap[lev], 1, 0);
    }

    const Real* dx = geom[lev].CellSize();
    // drag rate of the solid faces in the source terms (ImmersedForcingTerrain_*mom); the
    // windspeed cap there (min(1, |u|)) is dropped, as in kynema-sgf's drag_coefficient / dz
    const Real rate_solid = solverChoice.if_Cd_momentum / std::cbrt(dx[0]*dx[1]*dx[2]);
    const Real small      = Real(0.005);   // small_volfrac of the terrain kernels
    const bool fraction_stress = solverChoice.if_fraction_stress;
    const bool use_most        = solverChoice.if_use_most;
    const bool buildings       = (solverChoice.buildings_type == BuildingsType::ImmersedForcing);
    const bool snap            = solverChoice.if_snap_partial_cells;
    const int klo = geom[lev].Domain().smallEnd(2);
    const int khi = geom[lev].Domain().bigEnd(2);


    if (buildings) {
        // The building kernels' face blanking: the face-centred fraction when the build has it, else
        // the mean of the two cells; with the snap, from the snapped cells (solid when either is)
        MultiFab const* tbf[AMREX_SPACEDIM] = {terrain_blanking_xface[lev].get(), terrain_blanking_yface[lev].get(),
                                               terrain_blanking_zface[lev].get()};
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(tb, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Array4<const Real> t  = tb.const_array(mfi);
            const Array4<const int>  in = inside.const_array(mfi);
            Array4<const Real> tf[AMREX_SPACEDIM];
            for (int d = 0; d < AMREX_SPACEDIM; ++d) { tf[d] = (tbf[d]) ? tbf[d]->const_array(mfi) : Array4<const Real>{}; }
            const Array4<Real> sg[AMREX_SPACEDIM] = {sigma[0].array(mfi), sigma[1].array(mfi), sigma[2].array(mfi)};

            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                const int di = (d == 0), dj = (d == 1), dk = (d == 2);
                const Array4<const Real> tfd = tf[d];
                const Array4<Real> sd = sg[d];
                auto snapb = [=] AMREX_GPU_DEVICE (Real b) noexcept -> Real {
                    return snap ? ((b >= myhalf) ? one : zero) : b;
                };
                // raw face blanking (what the explicit path freezes on) and the kernels' thresholded one
                auto raw = [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Real {
                    return (tfd) ? tfd(i,j,k) : myhalf * (t(i,j,k) + t(i-di,j-dj,k-dk));
                };
                auto fb = [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Real {
                    const Real b = snap ? amrex::max(snapb(t(i,j,k)), snapb(t(i-di,j-dj,k-dk))) : raw(i,j,k);
                    return (b < small) ? zero : b;
                };
                ParallelFor(mfi.nodaltilebox(d), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (in(i,j,k) == 0 || in(i-di,j-dj,k-dk) == 0) { sd(i,j,k) = one; return; }
                    const bool normal_face = snap && (snapb(t(i,j,k)) != snapb(t(i-di,j-dj,k-dk)));
                    const Real b = fb(i,j,k);
                    Real w;
                    if (d < 2) {
                        const Real below = (k == klo) ? zero : fb(i,j,k-1);
                        const Real lo = (d == 0) ? fb(i,j-1,k) : fb(i-1,j,k);
                        const Real hi = (d == 0) ? fb(i,j+1,k) : fb(i+1,j,k);
                        w = if_bld::horiz_masks(b, below, fb(i,j,k+1), lo, hi, normal_face, use_most, snap).linear(b);
                    } else {
                        w = if_bld::vert_masks(b, fb(i,j,k+1), fb(i,j-1,k), fb(i,j+1,k), fb(i-1,j,k), fb(i+1,j,k),
                                               k >= 1, normal_face, use_most, snap).linear(b);
                    }
                    // a face the explicit path freezes (solid on both sides) is held by the drag here,
                    // and the kernels apply no wall law on it
                    if (raw(i,j,k) == one) { w = one; }
                    sd(i,j,k) = one / (one + dt * w * rate_solid);
                });
            }
        }
        return;
    }

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(tb, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Array4<const Real> t  = tb.const_array(mfi);
        const Array4<const int>  in = inside.const_array(mfi);
        const Array4<Real> sx = sigma[0].array(mfi);
        const Array4<Real> sy = sigma[1].array(mfi);
        const Array4<Real> sz = sigma[2].array(mfi);

        // Horizontal faces: the drag of ImmersedForcingTerrain_{X,Y}mom (legacy) or
        // ImmersedForcingTerrain_HorizMom_FractionStress, zero where those apply the wall law
        auto horiz = [=] AMREX_GPU_DEVICE (int i, int j, int k, int io, int jo) noexcept -> Real
        {
            if (in(i,j,k) == 0 || in(i-io,j-jo,k) == 0) { return one; }
            auto beta = [&] (int kk) -> Real {
                const int kc = amrex::min(amrex::max(kk, klo), khi);
                const Real b = myhalf * (t(i,j,kc) + t(i-io,j-jo,kc));
                return (b < small) ? zero : b;
            };
            const Real b_k = beta(k);
            bool wall;
            if (fraction_stress) {
                wall = ib_wall::is_wall_cell(beta(k-1), b_k, beta(k+1), small);
            } else {
                // the legacy wall law acts on partial faces only here: a fully solid face under
                // the surface is held by the drag (ImmersedForcingTerrain_{X,Y}mom skips it)
                wall = use_most && (b_k > zero) && (b_k < one) && (beta(k+1) == zero);
            }
            const Real rate = wall ? zero : b_k * rate_solid;
            return one / (one + dt * rate);
        };

        ParallelFor(mfi.nodaltilebox(0), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            sx(i,j,k) = horiz(i, j, k, 1, 0);
        });
        ParallelFor(mfi.nodaltilebox(1), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            sy(i,j,k) = horiz(i, j, k, 0, 1);
        });
        // Vertical faces: the drag of ImmersedForcingTerrain_Zmom (both wall laws)
        ParallelFor(mfi.nodaltilebox(2), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            if (in(i,j,k) == 0 || in(i,j,k-1) == 0) {
                sz(i,j,k) = one;
            } else {
                const Real b_raw = myhalf * (t(i,j,k) + t(i,j,k-1));
                const Real b = (b_raw < small) ? zero : b_raw;
                sz(i,j,k) = one / (one + dt * b * rate_solid);
            }
        });
    }
}

void ERF::solve_with_if_gmres (int lev, const Box& subdomain, MultiFab& rhs, MultiFab& phi,
                               Array<MultiFab,AMREX_SPACEDIM>& fluxes,
                               Array<MultiFab const*,AMREX_SPACEDIM> const& sigma)
{
#ifdef ERF_USE_FFT
    BL_PROFILE("ERF::solve_with_if_gmres()");

    auto const dom_lo = lbound(Geom(lev).Domain());
    auto const dom_hi = ubound(Geom(lev).Domain());
    auto const sub_lo = lbound(subdomain);
    auto const sub_hi = ubound(subdomain);
    auto dx = Geom(lev).CellSizeArray();

    // Same subdomain geometry as build_fft_solvers
    Array<int,AMREX_SPACEDIM> is_per; is_per[0] = 0; is_per[1] = 0; is_per[2] = 0;
    if (Geom(lev).isPeriodic(0) && sub_lo.x == dom_lo.x && sub_hi.x == dom_hi.x) { is_per[0] = 1;}
    if (Geom(lev).isPeriodic(1) && sub_lo.y == dom_lo.y && sub_hi.y == dom_hi.y) { is_per[1] = 1;}
    int coord_sys = 0;
    Geometry my_geom;
    if (subdomain == Geom(lev).Domain()) {
        my_geom.define(Geom(lev).Domain(), Geom(lev).ProbDomain(), coord_sys, is_per);
    } else {
        RealBox rb( sub_lo.x   *dx[0],  sub_lo.y   *dx[1],  sub_lo.z   *dx[2],
                   (sub_hi.x+1)*dx[0], (sub_hi.y+1)*dx[1], (sub_hi.z+1)*dx[2]);
        my_geom.define(subdomain, rb, coord_sys, is_per);
    }

    ImmersedPoisson ip(my_geom, Geom(lev), rhs.boxArray(), rhs.DistributionMap(), domain_bc_type,
                       sigma, solverChoice.use_real_bcs);
    ip.usePrecond(true);

    GMRES<MultiFab, ImmersedPoisson> gmsolver;
    gmsolver.define(ip);
    gmsolver.setVerbose(mg_verbose);
    gmsolver.setRestartLength(50);
    gmsolver.solve(phi, rhs, solverChoice.poisson_reltol, solverChoice.poisson_abstol);

    ip.getFluxes(phi, fluxes);
#else
    amrex::ignore_unused(lev, rhs, phi, fluxes, sigma);
    amrex::Abort("erf.if_implicit_projection needs a build with FFT");
#endif

    ImposeBCsOnPhi(lev, phi, subdomain);
}
