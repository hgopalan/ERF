#include "ERF_Utils.H"
#include "ERF_ImmersedWallCell.H"

using namespace amrex;

/**
 * Convert conservative variables to primitive variables.
 *
 * @param[in] cons_state MultiFab containing conservative state variables
 * @param[out] S_prim MultiFab to be filled with primitive state variables
 * @param[in] ng Number of ghost cells
 */
void
cons_to_prim(const MultiFab& cons_state, MultiFab& S_prim, int ng)
{
    BL_PROFILE("cons_to_prim()");

    int ncomp_prim = S_prim.nComp();

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(cons_state,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& gbx = mfi.growntilebox(ng);
        const Array4<const Real>& cons_arr     = cons_state.array(mfi);
        const Array4<      Real>& prim_arr     = S_prim.array(mfi);

        //
        // We may need > one ghost cells of prim in order to compute higher order advective terms
       //
       amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
       {
           Real rho       = cons_arr(i,j,k,Rho_comp);
           Real rho_theta = cons_arr(i,j,k,RhoTheta_comp);
           prim_arr(i,j,k,PrimTheta_comp) = rho_theta / rho;
           for (int n = 1; n < ncomp_prim; ++n) {
               prim_arr(i,j,k,PrimTheta_comp + n) = cons_arr(i,j,k,RhoTheta_comp + n) / rho;
           }
       });
    } // mfi
}

/**
 * Fill the ghost cells of the wall distance field.
 *
 * The wall distance is only computed on the valid region of each box (whether
 * from the analytic shortcuts, the Poisson solve, or the thin-body
 * correction), so without this the ghost cells still hold the large sentinel
 * value they were initialized with. Any operator that reads wall distance from
 * a grown box -- e.g. the RANS Dirichlet-k boundary condition in
 * SurfaceLayer::update_fluxes -- would then propagate that sentinel into the
 * state.
 *
 * Ghost cells shared with a neighboring box or across a periodic boundary are
 * filled by FillBoundary; ghost cells outside the domain in a non-periodic
 * direction are filled by zero-order extrapolation. Note that the analytic
 * wall distances vary only with k, so the extrapolation is exact in those
 * cases.
 *
 * @param[inout] wdist MultiFab holding the wall distance
 * @param[in] geom Geometry at this level
 */
void
fill_wall_dist_ghost_cells (MultiFab& wdist, const Geometry& geom)
{
    BL_PROFILE("fill_wall_dist_ghost_cells()");

    wdist.FillBoundary(geom.periodicity());

    const Box& domain = geom.Domain();
    const auto dom_lo = amrex::lbound(domain);
    const auto dom_hi = amrex::ubound(domain);

    const GpuArray<int,AMREX_SPACEDIM> is_per = {AMREX_D_DECL(geom.isPeriodic(0),
                                                              geom.isPeriodic(1),
                                                              geom.isPeriodic(2))};

    for (MFIter mfi(wdist); mfi.isValid(); ++mfi)
    {
        const Box& gbx = mfi.fabbox();
        const Array4<Real>& d_arr = wdist.array(mfi);

        //
        // Note that the cells we read from are always inside the domain in
        // every non-periodic direction, hence already filled, and are disjoint
        // from the cells we write to.
        //
        ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const int ii = (is_per[0]) ? i : amrex::min(amrex::max(i,dom_lo.x),dom_hi.x);
            const int jj = (is_per[1]) ? j : amrex::min(amrex::max(j,dom_lo.y),dom_hi.y);
            const int kk = (is_per[2]) ? k : amrex::min(amrex::max(k,dom_lo.z),dom_hi.z);
            if (ii != i || jj != j || kk != k) {
                d_arr(i,j,k) = d_arr(ii,jj,kk);
            }
        });
    } // mfi
}

/**
 * Wall distance over terrain represented by immersed forcing. The terrain kernels
 * (ERF_ImmersedForcing.cpp) put the wall law in the top cell of the column that holds solid
 * (solid fraction above small_volfrac with none above it) and place the wall at the BOTTOM face
 * of that cell: target velocity at dz/2, friction velocity from the cell above at 3 dz/2, both
 * vertical. Measuring the RANS wall distance from the same face gives the wall cell d = dz/2 and
 * the cell above 3 dz/2, as on flat ground, so the length scale and the wall law agree. Solid
 * cells get their distance to that face, which keeps the length scale finite inside the body.
 *
 * @param[out] wdist Wall distance (valid cells)
 * @param[in] terrain_blank Cell-centred solid fraction
 * @param[in] z_phys_nd Node heights
 * @param[in] z_phys_cc Cell-centre heights
 * @param[in] domain Index domain of the level
 * @param[in] small_volfrac Solid fraction below which a cell counts as fluid
 */
void
immersed_wall_dist (MultiFab& wdist,
                    const MultiFab& terrain_blank,
                    const MultiFab& z_phys_nd,
                    const MultiFab& z_phys_cc,
                    const Box& domain,
                    Real small_volfrac,
                    bool true_surface,
                    MultiFab* wall_height,
                    const MultiFab* wall_height_crse)
{
    BL_PROFILE("immersed_wall_dist()");

    const int  klo     = domain.smallEnd(2);
    const int  khi     = domain.bigEnd(2);
    const Real small   = small_volfrac;
    const Real d_floor = std::numeric_limits<Real>::epsilon();
    const bool have_crse = (wall_height_crse != nullptr);

    for (MFIter mfi(wdist); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(have_crse || (bx.smallEnd(2) == klo && bx.bigEnd(2) == khi),
            "immersed_wall_dist: the wall distance over immersed terrain needs whole columns in each box");

        const int kb_lo = bx.smallEnd(2);
        const int kb_hi = bx.bigEnd(2);
        Box bx2d = bx; bx2d.setRange(2, klo);
        auto const tb_arr   = terrain_blank.const_array(mfi);
        auto const znd_arr  = z_phys_nd.const_array(mfi);
        auto const zcc_arr  = z_phys_cc.const_array(mfi);
        auto       dist_arr = wdist.array(mfi);
        auto const hw_arr   = (wall_height)  ? wall_height->array(mfi) : Array4<Real>{};
        auto const hc_arr   = (have_crse) ? wall_height_crse->const_array(mfi) : Array4<const Real>{};

        ParallelFor(bx2d, [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
        {
            auto z_face = [&] (int kk) -> Real {
                return fourth * ( znd_arr(i,j,kk) + znd_arr(i+1,j,kk)
                                + znd_arr(i,j+1,kk) + znd_arr(i+1,j+1,kk) );
            };
            // topmost cell of the box holding solid: beta >= small for the legacy wall (whose wall
            // is the bottom face of that cell), not fluid for the surface itself
            int kt = kb_lo - 1;
            for (int kk = kb_hi; kk >= kb_lo; --kk) {
                const bool solid = (true_surface) ? !ib_wall::is_fluid(tb_arr(i,j,kk), small)
                                                  : (tb_arr(i,j,kk) >= small);
                if (solid) { kt = kk; break; }
            }
            // The box sees the surface of its column if it holds the whole column, if it has
            // fluid above its top solid cell, or if it starts at the bottom and holds no solid.
            // Otherwise (a fine box wholly inside the solid, or above the surface) the wall
            // height comes from the coarser level.
            const bool whole  = (kb_lo == klo) && (kb_hi == khi);
            const bool sees   = whole || (kt >= kb_lo && kt < kb_hi) || (kt < kb_lo && kb_lo == klo);
            const bool local  = sees || !have_crse;

            // h: the wall face (legacy) or the surface height; kw: the wall cell of the surface
            // form (none below kb_lo). A local box without solid starts at the bottom (klo).
            Real h;
            int  kw = kb_lo - 1;
            if (!local) {
                h = hc_arr(i,j,klo);
            } else if (!true_surface) {
                h = z_face(amrex::max(kt, kb_lo));
            } else {
                kw = kb_lo;
                h  = z_face(kb_lo);
                if (kt >= kb_lo) {
                    const Real b = amrex::min(tb_arr(i,j,kt), one);
                    h  = z_face(kt) + b * (z_face(kt+1) - z_face(kt));
                    kw = ib_wall::is_solid(tb_arr(i,j,kt), small) ? kt + 1 : kt;
                }
            }
            if (hw_arr) { hw_arr(i,j,klo) = h; }
            for (int kk = kb_lo; kk <= kb_hi; ++kk) {
                // the wall cell of the surface form is represented by its fluid centroid
                const Real d = (kk == kw) ? myhalf * (z_face(kk+1) - h)
                                          : std::abs(zcc_arr(i,j,kk) - h);
                dist_arr(i,j,kk) = amrex::max(d, d_floor);
            }
        });
    } // mfi
}

/**
 * Wall-face masks of the fraction-stress wall law: zero on the edge that is the bottom face
 * of the wall cell of a face column, one elsewhere (see ERF_ImmersedWallCell.H). The solid
 * fraction on a face column is the mean of the two cells, as the momentum kernels read it.
 *
 * @param[in] terrain_blank Cell-centred solid fraction
 * @param[out] mask13 Mask on the xz edges
 * @param[out] mask23 Mask on the yz edges
 * @param[in] domain Index domain of the level
 * @param[in] small_volfrac Solid fraction below which a cell counts as fluid
 */
void
make_ib_wall_face_masks (const MultiFab& terrain_blank,
                         MultiFab& mask13,
                         MultiFab& mask23,
                         MultiFab& mask33,
                         const Box& domain,
                         Real small_volfrac,
                         int spacing,
                         Real dz,
                         Real z0)
{
    BL_PROFILE("make_ib_wall_face_masks()");
    AMREX_ALWAYS_ASSERT(terrain_blank.nGrowVect().allGT(mask13.nGrowVect()) &&
                        terrain_blank.nGrowVect().allGT(mask23.nGrowVect()) &&
                        terrain_blank.nGrowVect().allGE(mask33.nGrowVect()));

    const int  klo   = domain.smallEnd(2);
    const int  khi   = domain.bigEnd(2);
    const Real small = small_volfrac;

    for (MFIter mfi(mask13); mfi.isValid(); ++mfi) {
        auto const tb = terrain_blank.const_array(mfi);
        auto const m  = mask13.array(mfi);
        ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            auto b = [&] (int kk) -> Real {
                const int kc = amrex::min(amrex::max(kk, klo), khi);
                const Real v = myhalf * (tb(i-1,j,kc) + tb(i,j,kc));
                return (v < small) ? zero : v;
            };
            // zero on the bottom face of the wall cell, the spacing factor on its top face
            const bool wall_below = (k >= klo) && (k <= khi) && ib_wall::is_wall_cell(b(k-1), b(k), b(k+1), small);
            const bool wall_above = (k-1 >= klo) && (k-1 <= khi) && ib_wall::is_wall_cell(b(k-2), b(k-1), b(k), small);
            const Real g = ib_wall::face_above_factor(ib_wall::wall_beta(b(k-1), small), dz, z0, spacing);
            m(i,j,k) = wall_below ? zero : (wall_above ? g : one);
        });
    }
    for (MFIter mfi(mask23); mfi.isValid(); ++mfi) {
        auto const tb = terrain_blank.const_array(mfi);
        auto const m  = mask23.array(mfi);
        ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            auto b = [&] (int kk) -> Real {
                const int kc = amrex::min(amrex::max(kk, klo), khi);
                const Real v = myhalf * (tb(i,j-1,kc) + tb(i,j,kc));
                return (v < small) ? zero : v;
            };
            // zero on the bottom face of the wall cell, the spacing factor on its top face
            const bool wall_below = (k >= klo) && (k <= khi) && ib_wall::is_wall_cell(b(k-1), b(k), b(k+1), small);
            const bool wall_above = (k-1 >= klo) && (k-1 <= khi) && ib_wall::is_wall_cell(b(k-2), b(k-1), b(k), small);
            const Real g = ib_wall::face_above_factor(ib_wall::wall_beta(b(k-1), small), dz, z0, spacing);
            m(i,j,k) = wall_below ? zero : (wall_above ? g : one);
        });
    }
    for (MFIter mfi(mask33); mfi.isValid(); ++mfi) {
        auto const tb = terrain_blank.const_array(mfi);
        auto const m  = mask33.array(mfi);
        ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            auto b = [&] (int kk) -> Real {
                const Real v = tb(i,j,amrex::min(amrex::max(kk, klo), khi));
                return (v < small) ? zero : v;
            };
            // zero on the bottom face of the wall cell, the spacing factor on its top face
            const bool wall_below = (k >= klo) && (k <= khi) && ib_wall::is_wall_cell(b(k-1), b(k), b(k+1), small);
            const bool wall_above = (k-1 >= klo) && (k-1 <= khi) && ib_wall::is_wall_cell(b(k-2), b(k-1), b(k), small);
            const Real g = ib_wall::face_above_factor(ib_wall::wall_beta(b(k-1), small), dz, z0, spacing);
            m(i,j,k) = wall_below ? zero : (wall_above ? g : one);
        });
    }
}

/**
 * Lateral wall distance for immersed buildings; see ERF_Utils.H. The solid fraction is copied to a
 * MultiFab with nsearch ghost cells so the search sees the neighbouring boxes and periodic images.
 */
void
immersed_lateral_wall_dist (MultiFab& wdist, const MultiFab& terrain_blank, const Geometry& geom, int nsearch)
{
    BL_PROFILE("immersed_lateral_wall_dist()");
    AMREX_ALWAYS_ASSERT(nsearch >= 1);
    MultiFab beta(terrain_blank.boxArray(), terrain_blank.DistributionMap(), 1, IntVect(nsearch, nsearch, 0));
    beta.setVal(zero);
    MultiFab::Copy(beta, terrain_blank, 0, 0, 1, 0);
    beta.FillBoundary(geom.periodicity());
    const Real dx = geom.CellSize(0), dy = geom.CellSize(1);
    const Real half = myhalf;

    for (MFIter mfi(wdist, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const d = wdist.array(mfi);
        auto const b = beta.const_array(mfi);
        ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            if (b(i,j,k) >= half) { return; }   // solid: its value is not used
            Real dmin = d(i,j,k);
            for (int jj = -nsearch; jj <= nsearch; ++jj) {
            for (int ii = -nsearch; ii <= nsearch; ++ii) {
                if (b(i+ii,j+jj,k) >= half) {
                    // centre of this cell to the nearest point of the solid cell
                    const Real ex = amrex::max(Real(amrex::Math::abs(ii)) - half, zero) * dx;
                    const Real ey = amrex::max(Real(amrex::Math::abs(jj)) - half, zero) * dy;
                    dmin = amrex::min(dmin, std::sqrt(ex*ex + ey*ey));
                }
            }}
            d(i,j,k) = dmin;
        });
    }
}

/**
 * Fluid-weighted average down for immersed forcing; see ERF_Utils.H. The weighted sums are formed
 * on the coarsened fine layout (components 0..ncomp-1, then the weight and the weighted density),
 * copied to the coarse layout, and applied where the coarse cell holds fluid and the weight is
 * positive.
 */
void
if_average_down (const MultiFab& S_fine, MultiFab& S_crse,
                 const MultiFab& beta_fine, const MultiFab& beta_crse,
                 int scomp, int ncomp, const IntVect& ratio, bool mass_weighted)
{
    BL_PROFILE("if_average_down()");
    AMREX_ALWAYS_ASSERT(S_fine.is_cell_centered() && S_crse.is_cell_centered());
    AMREX_ALWAYS_ASSERT(scomp >= 0 && ncomp > 0 && scomp + ncomp <= S_crse.nComp());

    BoxArray cba = S_fine.boxArray(); cba.coarsen(ratio);
    const int nsum = ncomp + 2;
    MultiFab csum(cba, S_fine.DistributionMap(), nsum, 0);
    const int rx = ratio[0], ry = ratio[1], rz = ratio[2];

    for (MFIter mfi(csum, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const c = csum.array(mfi);
        auto const f = S_fine.const_array(mfi);
        auto const b = beta_fine.const_array(mfi);
        ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real w = zero, wr = zero;
            for (int n = 0; n < ncomp; ++n) { c(i,j,k,n) = zero; }
            for (int kk = k*rz; kk < (k+1)*rz; ++kk) {
            for (int jj = j*ry; jj < (j+1)*ry; ++jj) {
            for (int ii = i*rx; ii < (i+1)*rx; ++ii) {
                const Real wf = one - amrex::min(b(ii,jj,kk), one);
                w  += wf;
                wr += (mass_weighted) ? wf * f(ii,jj,kk,Rho_comp) : wf;
                for (int n = 0; n < ncomp; ++n) { c(i,j,k,n) += wf * f(ii,jj,kk,scomp+n); }
            }}}
            c(i,j,k,ncomp)   = w;
            c(i,j,k,ncomp+1) = wr;
        });
    }

    MultiFab csum_c(S_crse.boxArray(), S_crse.DistributionMap(), nsum, 0);
    csum_c.setVal(zero);
    csum_c.ParallelCopy(csum, 0, 0, nsum);

    for (MFIter mfi(S_crse, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const s  = S_crse.array(mfi);
        auto const c  = csum_c.const_array(mfi);
        auto const bc = beta_crse.const_array(mfi);
        ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const Real w = c(i,j,k,ncomp);
            if (w > zero && bc(i,j,k) < one) {
                if (!mass_weighted) {
                    for (int n = 0; n < ncomp; ++n) { s(i,j,k,scomp+n) = c(i,j,k,n) / w; }
                } else {
                    // the coarse density: averaged here when it is in the range, else its own
                    const Real rho_c = (scomp == Rho_comp) ? c(i,j,k,0) / w : s(i,j,k,Rho_comp);
                    const Real fac   = rho_c / c(i,j,k,ncomp+1);
                    for (int n = 0; n < ncomp; ++n) {
                        s(i,j,k,scomp+n) = (scomp+n == Rho_comp) ? rho_c : fac * c(i,j,k,n);
                    }
                }
            }
        });
    }
}

/**
 * Compute the total water mixing ratio from conservative variables.
 *
 * @param[in] cons_state MultiFab containing conservative state variables
 * @param[out] qt MultiFab to store the total water mixing ratio
 * @param[in] n_qstate_into_total Number of moisture components to include in the total
 */
void
make_qt(const MultiFab& cons_state, MultiFab& qt, int n_qstate_into_total)
{
    BL_PROFILE("make_qt()");

    // All moisture models are guaranteed to have RhoQ1_comp.
    MultiFab::Copy(qt, cons_state, RhoQ1_comp, 0, 1, qt.nGrowVect());

    for (int n = 1; n < n_qstate_into_total; ++n) {
        MultiFab::Add(qt, cons_state, RhoQ1_comp+n, 0, 1, qt.nGrowVect());
    }

    MultiFab::Divide(qt, cons_state, Rho_comp, 0, 1, qt.nGrowVect());
}

/**
 * Spread of the fine solid fractions under the coarse cells along the lateral coarse-fine faces of
 * a fine level (see ERF_Utils.H).
 */
std::pair<int, Real>
if_cf_surface_mismatch (const MultiFab& fine_tb,
                        const BoxArray& fine_grids,
                        const BoxArray& crse_grids,
                        const DistributionMapping& crse_dmap,
                        const Geometry& crse_geom,
                        const IntVect& ratio,
                        Real warn)
{
    const IntVect rr = ratio;
    const MultiFab& fine = fine_tb;
    const iMultiFab covered = makeFineMask(crse_grids, crse_dmap, IntVect(1,1,0), fine_grids, rr,
                                           crse_geom.periodicity(), 0, 1);

    // spread of the fine fractions under each coarse cell, formed on the coarsened fine layout
    MultiFab ftmp(amrex::coarsen(fine_grids, rr), fine_tb.DistributionMap(), 1, 0);
    const int rx = rr[0], ry = rr[1], rz = rr[2];
    for (MFIter mfi(ftmp, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const s = ftmp.array(mfi);
        auto const f = fine.const_array(mfi);
        ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real bmin = one, bmax = zero;
            for (int kk = k*rz; kk < (k+1)*rz; ++kk) {
            for (int jj = j*ry; jj < (j+1)*ry; ++jj) {
            for (int ii = i*rx; ii < (i+1)*rx; ++ii) {
                const Real b = amrex::min(amrex::max(f(ii,jj,kk), zero), one);
                bmin = amrex::min(bmin, b);
                bmax = amrex::max(bmax, b);
            }}}
            s(i,j,k) = bmax - bmin;
        });
    }
    MultiFab spread(crse_grids, crse_dmap, 1, 0);
    spread.setVal(zero);
    spread.ParallelCopy(ftmp, 0, 0, 1);

    const Box& dom = crse_geom.Domain();
    const int per_x = crse_geom.isPeriodic(0), per_y = crse_geom.isPeriodic(1);
    const int dlo_x = dom.smallEnd(0), dhi_x = dom.bigEnd(0), dlo_y = dom.smallEnd(1), dhi_y = dom.bigEnd(1);
    ReduceOps<ReduceOpSum, ReduceOpMax> reduce_op;
    ReduceData<int, Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;
    for (MFIter mfi(spread, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const m  = covered.const_array(mfi);
        auto const sp = spread.const_array(mfi);
        reduce_op.eval(mfi.tilebox(), reduce_data, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple
        {
            if (m(i,j,k) != 1) { return {0, zero}; }
            const bool edge = ((per_x || i > dlo_x) && m(i-1,j,k) == 0) || ((per_x || i < dhi_x) && m(i+1,j,k) == 0)
                           || ((per_y || j > dlo_y) && m(i,j-1,k) == 0) || ((per_y || j < dhi_y) && m(i,j+1,k) == 0);
            if (!edge) { return {0, zero}; }
            return {(sp(i,j,k) >= warn) ? 1 : 0, sp(i,j,k)};
        });
    }
    ReduceTuple hv = reduce_data.value(reduce_op);
    int  nflagged = amrex::get<0>(hv);
    Real worst    = amrex::get<1>(hv);
    ParallelDescriptor::ReduceIntSum(nflagged);
    ParallelDescriptor::ReduceRealMax(worst);
    return {nflagged, worst};
}

/**
 * Remove the net mass flux through the coarse-fine faces of each fine subdomain (see ERF_Utils.H).
 */
void
if_balance_cf_fluxes (const Geometry& geom,
                      const Vector<BoxArray>& subdomains,
                      const BoxArray& grids,
                      const Array<MultiFab*, 3>& mom,
                      const Array<const iMultiFab*, 3>& mask,
                      const Array<int, 3>& set_val,
                      const MultiFab& tbmf)
{
    const int nsub = static_cast<int>(subdomains.size());
    const Box dom = geom.Domain();
    const auto dxl = geom.CellSizeArray();
    const Real area[3] = {dxl[1]*dxl[2], dxl[0]*dxl[2], dxl[0]*dxl[1]};

    // subdomain of each fine box
    std::vector<int> box_sub(grids.size(), -1);
    for (int b = 0; b < static_cast<int>(grids.size()); ++b) {
        for (int isub = 0; isub < nsub; ++isub) {
            if (subdomains[isub].contains(grids[b])) { box_sub[b] = isub; break; }
        }
    }

    // net inflow and open (fluid) c/f area per subdomain
    std::vector<Real> net(nsub, Real(0.0)), open(nsub, Real(0.0));
    for (int dir = 0; dir < 3; ++dir) {
        const int sval = set_val[dir];
        const iMultiFab& msk = *mask[dir];
        const int di = (dir == 0), dj = (dir == 1), dk = (dir == 2);
        const int dlo = dom.smallEnd(dir), dhi = dom.bigEnd(dir) + 1;
        for (MFIter mfi(*mom[dir]); mfi.isValid(); ++mfi) {
            const int isub = box_sub[mfi.index()];
            if (isub < 0) { continue; }
            const Box vb = mfi.validbox();
            const int flo = vb.smallEnd(dir), fhi = vb.bigEnd(dir);
            auto const m  = mom[dir]->const_array(mfi);
            auto const mk = msk.const_array(mfi);
            auto const tb = tbmf.const_array(mfi);
            const Real a  = area[dir];
            ReduceOps<ReduceOpSum,ReduceOpSum> reduce_op;
            ReduceData<Real,Real> reduce_data(reduce_op);
            reduce_op.eval(vb, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> GpuTuple<Real,Real>
            {
                const int f = (dir == 0) ? i : ((dir == 1) ? j : k);
                if (mk(i,j,k) != sval || f == dlo || f == dhi || (f != flo && f != fhi)) { return {Real(0.0), Real(0.0)}; }
                const Real sgn = (f == flo) ? Real(1.0) : Real(-1.0);
                const bool solid = (tb(i,j,k) >= Real(1.0)) || (tb(i-di,j-dj,k-dk) >= Real(1.0));
                return {sgn * m(i,j,k) * a, solid ? Real(0.0) : a};
            });
            auto hv = reduce_data.value(reduce_op);
            net[isub]  += amrex::get<0>(hv);
            open[isub] += amrex::get<1>(hv);
        }
    }
    ParallelDescriptor::ReduceRealSum(net.data(),  nsub);
    ParallelDescriptor::ReduceRealSum(open.data(), nsub);

    // take the net inflow off the open faces: -sgn * c on each, c = net / open area
    for (int dir = 0; dir < 3; ++dir) {
        const int sval = set_val[dir];
        const iMultiFab& msk = *mask[dir];
        const int di = (dir == 0), dj = (dir == 1), dk = (dir == 2);
        const int dlo = dom.smallEnd(dir), dhi = dom.bigEnd(dir) + 1;
        for (MFIter mfi(*mom[dir]); mfi.isValid(); ++mfi) {
            const int isub = box_sub[mfi.index()];
            if (isub < 0 || !(open[isub] > Real(0.0))) { continue; }
            const Real c = net[isub] / open[isub];
            const Box vb = mfi.validbox();
            const int flo = vb.smallEnd(dir), fhi = vb.bigEnd(dir);
            auto const m  = mom[dir]->array(mfi);
            auto const mk = msk.const_array(mfi);
            auto const tb = tbmf.const_array(mfi);
            ParallelFor(vb, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                const int f = (dir == 0) ? i : ((dir == 1) ? j : k);
                if (mk(i,j,k) != sval || f == dlo || f == dhi || (f != flo && f != fhi)) { return; }
                const bool solid = (tb(i,j,k) >= Real(1.0)) || (tb(i-di,j-dj,k-dk) >= Real(1.0));
                if (solid) { return; }
                const Real sgn = (f == flo) ? Real(1.0) : Real(-1.0);
                m(i,j,k) -= sgn * c;
            });
        }
    }
}
