/**
 * \file ERF_PoissonWallDist.cpp
 */
#include "ERF.H"
#include "ERF_Constants.H"
#include "ERF_Utils.H"
#include "ERF_TerrainPoisson_3D_K.H"
#include "ERF_TerrainMetrics.H"

#include "AMReX_MLMG.H"
#include "AMReX_MLABecLaplacian.H"

using namespace amrex;

/**
 * Calculate wall distances using the Poisson equation
 *
 * The zlo boundary is assumed to correspond to the land surface. If there are
 * no boundary walls, then the other use case is to calculate wall distances
 * for immersed boundaries (embedded or thin body).
 *
 * See Tucker, P. G. (2003). Differential equation-based wall distance
 * computation for DES and RANS. Journal of Computational Physics,
 * 190(1), 229–Real(248.) https://doi.org/Real(10.1016)/S0021-9991(03)00272-9
 *
 * @param lev Level index for the wall-distance solve
 */
void ERF::poisson_wall_dist (int lev)
{
    BL_PROFILE("ERF::poisson_wall_dist()");

    bool havewall{false};
    Orientation zlo(Direction::z, Orientation::low);
    if ( ( phys_bc_type[zlo] == ERF_BC::surface_layer                      ) ||
         ( phys_bc_type[zlo] == ERF_BC::no_slip_wall                       ) )/*||
         ((phys_bc_type[zlo] == ERF_BC::slip_wall) && (dom_hi.z > dom_lo.z)) )*/
    {
        havewall = true;
    }

    auto const& geomdata = geom[lev];
    auto const& dxinv    = geomdata.InvCellSizeArray();

    auto const& zphys_arr = z_phys_nd[lev]->const_arrays();

    const bool if_buildings = (solverChoice.buildings_type == BuildingsType::ImmersedForcing);
    if ((solverChoice.terrain_type == TerrainType::ImmersedForcing || if_buildings) &&
        solverChoice.wall_dist_type == "terrain_height") {
        // Immersed terrain: measure from the wall of the immersed wall law (the bottom face of the
        // top cell holding solid), so the length scale and the wall law see the same geometry.
        // Immersed buildings (height maps over the domain bottom): measure from the top of the
        // solid in each column, the roof inside a footprint and the ground elsewhere
        Print() << "Calculating wall distance from the immersed forcing wall cell" << std::endl;
        AMREX_ALWAYS_ASSERT(terrain_blanking[lev]);
        const int klo = geomdata.Domain().smallEnd(2);
        const DistributionMapping& dm = walldist[lev]->DistributionMap();
        BoxList bl2d = walldist[lev]->boxArray().boxList();
        for (Box& b : bl2d) { b.setRange(2, klo); }
        BoxArray ba2d(std::move(bl2d));
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ba2d.isDisjoint(),
            "poisson_wall_dist: the immersed wall distance needs one box per column at each level");
        ib_wall_height[lev] = std::make_unique<MultiFab>(ba2d, dm, 1, 0);

        // A fine level need not reach the surface or the top: bring the coarser level's wall
        // height to this level's columns (piecewise constant) for the boxes that do not see it
        std::unique_ptr<MultiFab> height_crse;
        if (lev > 0) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ib_wall_height[lev-1],
                "poisson_wall_dist: the immersed wall distance of a fine level needs the coarser level's");
            const IntVect rr = refRatio(lev-1);
            BoxArray cba2d = ba2d; cba2d.coarsen(rr);
            MultiFab tmp(cba2d, dm, 1, 0);
            tmp.ParallelCopy(*ib_wall_height[lev-1], 0, 0, 1, IntVect(0), IntVect(0), geom[lev-1].periodicity());
            height_crse = std::make_unique<MultiFab>(ba2d, dm, 1, 0);
            for (MFIter mfi(*height_crse); mfi.isValid(); ++mfi) {
                auto const hf = height_crse->array(mfi);
                auto const hc = tmp.const_array(mfi);
                ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    hf(i,j,k) = hc(amrex::coarsen(i,rr[0]), amrex::coarsen(j,rr[1]), amrex::coarsen(k,rr[2]));
                });
            }
        }
        immersed_wall_dist(*walldist[lev], *terrain_blanking[lev], *z_phys_nd[lev], *z_phys_cc[lev],
                           geomdata.Domain(), Real(0.005),    // small_volfrac of the terrain kernels
                           solverChoice.if_fraction_stress || if_buildings,   // from the surface itself
                           ib_wall_height[lev].get(), height_crse.get());
        // vertical building walls: the nearest solid cell in the same horizontal plane
        if (if_buildings) {
            immersed_lateral_wall_dist(*walldist[lev], *terrain_blanking[lev], geomdata,
                                       solverChoice.if_wall_dist_search);
        }
        fill_wall_dist_ghost_cells(*walldist[lev], geom[lev]);
        return;
    }

    if (havewall) {
#if 1
        // Bypass wall dist calc in the trivial cases

        if (solverChoice.mesh_type == MeshType::ConstantDz) {
            Print() << "Directly calculating direct wall distance for constant dz" << std::endl;
            const auto prob_lo = geomdata.ProbLoArray();
            const auto dx = geomdata.CellSizeArray();
            for (MFIter mfi(*walldist[lev]); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.validbox();
                auto dist_arr = walldist[lev]->array(mfi);
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    dist_arr(i, j, k) = prob_lo[2] + (k + myhalf) * dx[2];
                });
            }
            fill_wall_dist_ghost_cells(*walldist[lev], geom[lev]);
            return;
        }

        if (solverChoice.mesh_type == MeshType::StretchedDz) {
            Print() << "Directly calculating direct wall distance for stretched dz" << std::endl;
            for (MFIter mfi(*walldist[lev],TileNoZ()); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.validbox();
                auto dist_arr = walldist[lev]->array(mfi);
                const auto zcc_arr = z_phys_cc[lev]->const_array(mfi);
                const auto znd_arr = z_phys_nd[lev]->const_array(mfi);
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    dist_arr(i, j, k) = zcc_arr(i, j, k) - znd_arr(i, j, 0);
                });
            }
            fill_wall_dist_ghost_cells(*walldist[lev], geom[lev]);
            return;
        }
#endif
    }
    else
    {
        Error("No solid boundaries in the computational domain");
    }

    if (havewall && solverChoice.wall_dist_type == "terrain_height") {
        // Height above the local surface projected on the surface normal:
        // d = (z_cc - z_surf) / sqrt(1 + h_xi^2 + h_eta^2), exact for a
        // plane, within a few percent of the true distance for hills with
        // slopes below about 0.3, and free of any linear solve (cf. the
        // terrain height used by the amr-wind immersed terrain and Kynema).
        Print() << "Calculating wall distance from the terrain height (normal-projected)" << std::endl;
        const int klo = geomdata.Domain().smallEnd(2);

        // The surface nodes z_nd(:,:,klo) live only in the boxes that touch the surface;
        // when the BoxArray is split in z the boxes above hold nodes from their own k
        // range, so gather the surface slab onto every box (the same 2D footprint, at klo).
        BoxList bl_surf = z_phys_nd[lev]->boxArray().boxList();
        for (auto& b : bl_surf) { b.setRange(2,klo); }
        BoxArray ba_surf(std::move(bl_surf));
        IntVect ng_surf = z_phys_nd[lev]->nGrowVect(); ng_surf[2] = 0;
        MultiFab znd_surf(ba_surf, z_phys_nd[lev]->DistributionMap(), 1, ng_surf);
        znd_surf.setVal(bogus_large_value);
        znd_surf.ParallelCopy(*z_phys_nd[lev], 0, 0, 1, ng_surf, ng_surf, geom[lev].periodicity());
        // Every node the stencil below reads (valid plus the x/y ghosts) must
        // have been gathered; the slab has no z ghosts, so reduce over ng_surf
        // rather than a scalar ghost count.
        Real znd_max = ReduceMax(znd_surf, ng_surf,
            [=] AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& a) -> Real
            {
                Real m = std::numeric_limits<Real>::lowest();
                amrex::Loop(bx, [&] (int i, int j, int k) { m = amrex::max(m, a(i,j,k)); });
                return m;
            });
        ParallelAllReduce::Max(znd_max, ParallelContext::CommunicatorSub());
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(znd_max < bogus_large_value,
            "poisson_wall_dist: the surface nodes were not gathered onto every box");

        for (MFIter mfi(*walldist[lev]); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto dist_arr = walldist[lev]->array(mfi);
            const auto zcc_arr = z_phys_cc[lev]->const_array(mfi);
            const auto znd_arr = znd_surf.const_array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Real z_surf = fourth * ( znd_arr(i,j,klo) + znd_arr(i+1,j,klo)
                                       + znd_arr(i,j+1,klo) + znd_arr(i+1,j+1,klo) );
                Real h_xi  = Compute_h_xi_AtKface (i, j, klo, dxinv, znd_arr);
                Real h_eta = Compute_h_eta_AtKface(i, j, klo, dxinv, znd_arr);
                Real dz_loc = zcc_arr(i,j,k) - z_surf;
                dist_arr(i, j, k) = amrex::max(dz_loc / std::sqrt(one + h_xi*h_xi + h_eta*h_eta),
                                               std::numeric_limits<Real>::epsilon());
            });
        }
        fill_wall_dist_ghost_cells(*walldist[lev], geom[lev]);
        return;
    }

    Print() << "Calculating Poisson wall distance for general terrain" << std::endl;

    // Make sure the solver only sees the levels over which we are solving
    Vector<Geometry>          geom_tmp; geom_tmp.push_back(geom[lev]);
    Vector<BoxArray>            ba_tmp;   ba_tmp.push_back(walldist[lev]->boxArray());
    Vector<DistributionMapping> dm_tmp;   dm_tmp.push_back(walldist[lev]->DistributionMap());

    Vector<MultiFab> rhs;
    Vector<MultiFab> phi;

    if (solverChoice.terrain_type == TerrainType::EB) {
        amrex::Error("Wall dist calc not implemented for EB");
    } else {
        rhs.resize(1);   rhs[0].define(ba_tmp[0], dm_tmp[0], 1, 0);
        phi.resize(1);   phi[0].define(ba_tmp[0], dm_tmp[0], 1, 1);
    }

    rhs[0].setVal(1.0);

    auto const dom_lo = lbound(geom[lev].Domain());
    auto const dom_hi = ubound(geom[lev].Domain());

    // ****************************************************************************
    // Initialize phi
    // (It is essential that we do this in order to fill the corners; this is
    // used if we include blanking.)
    // ****************************************************************************
    phi[0].setVal(0.0);

    // ****************************************************************************
    // Interior boundaries are marked with phi=0
    // ****************************************************************************
#if 0
    // Define an overset mask (0 or 1) to set dirichlet nodes on walls
    // 1 means the node is an unknown. 0 means it's known.
    iMultiFab mask(ba_tmp[0], dm_tmp[0], 1, 0);
    Vector<const iMultiFab*> overset_mask = {&mask};

    mask.setVal(1);
    if (solverChoice.advChoice.have_zero_flux_faces) {
        Warning("Poisson distance is inaccurate for bodies in open domains that are small compared to the domain size, skipping");
        return;

        Gpu::DeviceVector<IntVect> xfacelist, yfacelist, zfacelist;

        xfacelist.resize(solverChoice.advChoice.zero_xflux.size());
        yfacelist.resize(solverChoice.advChoice.zero_yflux.size());
        zfacelist.resize(solverChoice.advChoice.zero_zflux.size());

        if (xfacelist.size() > 0) {
            Gpu::copy(amrex::Gpu::hostToDevice,
                      solverChoice.advChoice.zero_xflux.begin(),
                      solverChoice.advChoice.zero_xflux.end(),
                      xfacelist.begin());
            Print() << "  masking interior xfaces" << std::endl;
        }
        if (yfacelist.size() > 0) {
            Gpu::copy(amrex::Gpu::hostToDevice,
                      solverChoice.advChoice.zero_yflux.begin(),
                      solverChoice.advChoice.zero_yflux.end(),
                      yfacelist.begin());
            Print() << "  masking interior yfaces" << std::endl;
        }
        if (zfacelist.size() > 0) {
            Gpu::copy(amrex::Gpu::hostToDevice,
                      solverChoice.advChoice.zero_zflux.begin(),
                      solverChoice.advChoice.zero_zflux.end(),
                      zfacelist.begin());
            Print() << "  masking interior zfaces" << std::endl;
        }

        for (MFIter mfi(phi[0]); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();

            auto phi_arr  = phi[0].array(mfi);
            auto mask_arr = mask.array(mfi);

            if (xfacelist.size() > 0) {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    for (int iface=0; iface < xfacelist.size(); ++iface) {
                        if ((i == xfacelist[iface][0]) &&
                            (j == xfacelist[iface][1]) &&
                            (k == xfacelist[iface][2]))
                        {
                            mask_arr(i, j  , k  ) = 0;
                            mask_arr(i, j  , k+1) = 0;
                            mask_arr(i, j+1, k  ) = 0;
                            mask_arr(i, j+1, k+1) = 0;
                        }
                    }
                });
            }

            if (yfacelist.size() > 0) {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    for (int iface=0; iface < yfacelist.size(); ++iface) {
                        if ((i == yfacelist[iface][0]) &&
                            (j == yfacelist[iface][1]) &&
                            (k == yfacelist[iface][2]))
                        {
                            mask_arr(i  , j, k  ) = 0;
                            mask_arr(i  , j, k+1) = 0;
                            mask_arr(i+1, j, k  ) = 0;
                            mask_arr(i+1, j, k+1) = 0;
                        }
                    }
                });
            }

            if (zfacelist.size() > 0) {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    for (int iface=0; iface < zfacelist.size(); ++iface) {
                        if ((i == xfacelist[iface][0]) &&
                            (j == xfacelist[iface][1]) &&
                            (k == xfacelist[iface][2]))
                        {
                            mask_arr(i  , j  , k) = 0;
                            mask_arr(i  , j+1, k) = 0;
                            mask_arr(i+1, j  , k) = 0;
                            mask_arr(i+1, j+1, k) = 0;
                        }
                    }
                });
            }
        }
    }
#endif

    // ****************************************************************************
    // Setup BCs, with solid domain boundaries being dirichlet
    // ****************************************************************************
    amrex::Array<amrex::LinOpBCType,AMREX_SPACEDIM> bc3d_lo, bc3d_hi;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (geom[lev].isPeriodic(dir)) {
            bc3d_lo[dir] = LinOpBCType::Periodic;
            bc3d_hi[dir] = LinOpBCType::Periodic;
        } else {
            bc3d_lo[dir] = LinOpBCType::Neumann;
            bc3d_hi[dir] = LinOpBCType::Neumann;
        }
    }
    if (havewall) {
        Print() << "  Poisson zlo BC is dirichlet" << std::endl;
        bc3d_lo[2] = LinOpBCType::Dirichlet;
    }
    Print() << "  bc lo : " << bc3d_lo << std::endl;
    Print() << "  bc hi : " << bc3d_hi << std::endl;

    if (!solverChoice.advChoice.have_zero_flux_faces && !havewall) {
        Error("No solid boundaries in the computational domain");
    }

    LPInfo info; // defaults

/* Nodal solver cannot have hidden dimensions */
#if 0
    // Allow a hidden direction if the domain is one cell wide
    if (dom_lo.x == dom_hi.x) {
        info.setHiddenDirection(0);
        Print() << "  domain is 2D in yz" << std::endl;
    } else if (dom_lo.y == dom_hi.y) {
        info.setHiddenDirection(1);
        Print() << "  domain is 2D in xz" << std::endl;
    } else if (dom_lo.z == dom_hi.z) {
        info.setHiddenDirection(2);
        Print() << "  domain is 2D in xy" << std::endl;
    }
#endif

#if 0
    Vector<EBFArrayBoxFactory const*> factory_vec;
    factory_vec.push_back(static_cast<FabFactory<FArrayBox> const*>(&EBFactory(lev));
#endif

    // ****************************************************************************
    // Setup Poisson problem
    // (A \alpha - B \nabla \cdot \beta \nabla ) \phi = f
    //
    // In physical space:
    //   \nabla \cdot \nabla \phi = -1
    //
    // In computational space:
    //   grad(phi) = T^T \nabla \phi
    // and
    //   \nabla \cdot (h_zeta T (T^T \nabla \phi)) = -h_zeta
    // where T = inv(J), T^T is the transpose of inv(J)
    //
    // Posed as -div(beta grad phi) = +h_zeta, i.e. B = +1, the positive
    // definite form MLABecLaplacian documents (B = -1 with f = -h_zeta is
    // the same equation and gave the same iterates). Note: on a 3D fitted
    // mesh with dx != dz this multigrid diverges (residual 18x after the
    // first cycle, 1e10 by iteration 100) with or without semi-coarsening
    // and independent of the box layout. Use erf.wall_dist_type =
    // terrain_height there.
    // ****************************************************************************
    constexpr Real constA = zero;
    constexpr Real constB = one;

    MLABecLaplacian mlabec(geom_tmp, ba_tmp, dm_tmp, info);

    mlabec.setScalars(constA, constB);
    mlabec.setACoeffs(0, zero);
#if 1
    // Set beta coefficients at faces
    Array<MultiFab, AMREX_SPACEDIM> beta;

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        BoxArray ba_face = ba_tmp[0];
        ba_face.surroundingNodes(idim);  // Convert to face-centered in direction idim
        beta[idim].define(ba_face, dm_tmp[0], 1, 0);
    }

    auto beta0_arr = beta[0].arrays();
    auto beta1_arr = beta[1].arrays();
    auto beta2_arr = beta[2].arrays();

    // Note: This ignores the off-diagonal components of (h_zeta T T^T), which
    //       is equivalent to assuming that h_xi and h_eta are small.

    ParallelFor(beta[0], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k) {
        beta0_arr[b](i, j, k) = Compute_h_zeta_AtIface(i, j, k, dxinv, zphys_arr[b]);;
    });
    ParallelFor(beta[1], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k) {
        beta1_arr[b](i, j, k) = Compute_h_zeta_AtJface(i, j, k, dxinv, zphys_arr[b]);;
    });
    ParallelFor(beta[2], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k) {
        Real inv_h_zeta = one / Compute_h_zeta_AtKface(i, j, k, dxinv, zphys_arr[b]);
        Real h_xi = Compute_h_xi_AtKface(i, j, k, dxinv, zphys_arr[b]);
        Real h_eta = Compute_h_eta_AtKface(i, j, k, dxinv, zphys_arr[b]);
        beta2_arr[b](i, j, k) = inv_h_zeta * (1 + h_xi*h_xi + h_eta*h_eta);
    });

    mlabec.setBCoeffs(0, GetArrOfConstPtrs(beta));

    // Set RHS := +h_zeta (for -div(beta grad phi) = h_zeta)
    auto rhs_arr = rhs[0].arrays();
    ParallelFor(rhs[0], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k) {
        rhs_arr[b](i, j, k) = Compute_h_zeta_AtCellCenter(i, j, k, dxinv, zphys_arr[b]);
    });
#else
    mlabec.setBCoeffs(0, one);
#endif

    mlabec.setDomainBC(bc3d_lo, bc3d_hi);

    if (lev > 0) {
        mlabec.setCoarseFineBC(nullptr, ref_ratio[lev-1], LinOpBCType::Neumann);
    }

    // If we have inhomogeneous BCs -- do this after setCoarseFineBC
    mlabec.setLevelBC(0, nullptr);

    // ****************************************************************************
    // Solve Poisson problem with MLMG
    // ****************************************************************************
    const Real reltol = solverChoice.poisson_reltol;
    const Real abstol = solverChoice.poisson_abstol;
    const int n_corr = solverChoice.ncorr;
    constexpr int max_iter = 100;

    MLMG mlmg(mlabec);
    mlmg.setMaxIter(max_iter);
    mlmg.setVerbose(mg_verbose);
    mlmg.setBottomVerbose(0);

    for (int icorr=0; icorr <= n_corr; ++icorr) {
        Print()<< "Solving wall distance poisson, icorr=" << icorr << std::endl;

        mlmg.solve(GetVecOfPtrs(phi),
                   GetVecOfConstPtrs(rhs),
                   reltol, abstol);

        // ****************************************************************************
        // Apply BCs: dirichlet (odd) on zlo, neumann (even) / periodic elsewhere
        // ****************************************************************************

        // Overwrite with periodic fill outside domain and fine-fine fill inside
        phi[0].FillBoundary(geom[lev].periodicity());

        if (!geom[lev].isPeriodic(0)) {
            for (MFIter mfi(phi[0],true); mfi.isValid(); ++mfi)
            {
                Box bx = mfi.tilebox();
                const Array4<Real>& phi_arr = phi[0].array(mfi);
                if (bx.smallEnd(0) <= dom_lo.x) {
                    ParallelFor(makeSlab(bx,0,dom_lo.x),
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        phi_arr(i-1,j,k) =  phi_arr(i,j,k); // even BC
                    });
                } // lo x
                if (bx.bigEnd(0) >= dom_hi.x) {
                    ParallelFor(makeSlab(bx,0,dom_hi.x),
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        phi_arr(i+1,j,k) =  phi_arr(i,j,k); // even BC
                    });
                } // hi x
            } // mfi
        } // not periodic in x

        if (!geom[lev].isPeriodic(1)) {
            for (MFIter mfi(phi[0],true); mfi.isValid(); ++mfi)
            {
                Box bx = mfi.tilebox();
                Box bx2(bx); bx2.grow(0,1);
                const Array4<Real>& phi_arr = phi[0].array(mfi);
                if (bx.smallEnd(1) <= dom_lo.y) {
                    ParallelFor(makeSlab(bx2,1,dom_lo.y),
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        phi_arr(i,j-1,k) =  phi_arr(i,j,k); // even BC
                    });
                } // lo y
                if (bx.bigEnd(1) >= dom_hi.y) {
                    ParallelFor(makeSlab(bx2,1,dom_hi.y),
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        phi_arr(i,j+1,k) =  phi_arr(i,j,k); // even BC
                    });
                } // hi y

            } // mfi
        } // not periodic in y

        for (MFIter mfi(phi[0],true); mfi.isValid(); ++mfi)
        {
            Box bx = mfi.tilebox();
            Box bx3(bx); bx3.grow(0,1); bx3.grow(1,1);
            const Array4<Real>& phi_arr = phi[0].array(mfi);
            if (bx.smallEnd(2) <= dom_lo.z) {
                ParallelFor(makeSlab(bx3,2,dom_lo.z),
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    phi_arr(i,j,k-1) = -phi_arr(i,j,k); // ODD BC
                });
            } // lo z
            if (bx.bigEnd(2) >= dom_hi.z) {
                ParallelFor(makeSlab(bx3,2,dom_hi.z),
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    phi_arr(i,j,k+1) =  phi_arr(i,j,k); // even BC
                });
            } // hi z
        } // mfi

        // ****************************************************************************
        // Compute grad(phi) to get distances
        // ****************************************************************************
        auto const& phi_arr = phi[0].const_arrays();
        //auto rhs_arr = rhs[0].arrays();
        auto dist_arr = walldist[lev]->arrays();

        ParallelFor(*walldist[lev], [=] AMREX_GPU_DEVICE(int b, int i, int j, int k) {
            // Cell-centred gradient of phi in physical space: centred
            // differences in computational space with the cell-centre
            // terrain metrics (chain rule for a mesh deformed in z only).
            // The face fluxes used before sat half a cell below and beside
            // the centre, which overstated |grad phi| by dz/2 and shortened
            // every distance by z dz / (2H) on a flat mesh (0.8 % for 64
            // cells). The ghost cells of phi carry the Dirichlet (odd) value
            // below the wall and even values elsewhere, so the stencil needs
            // nothing beyond one ghost cell and the cell's own eight nodes.
            const auto& p  = phi_arr[b];
            const auto& zp = zphys_arr[b];
            Real dpdxi   = myhalf * (p(i+1,j,k) - p(i-1,j,k)) * dxinv[0];
            Real dpdeta  = myhalf * (p(i,j+1,k) - p(i,j-1,k)) * dxinv[1];
            Real dpdzeta = myhalf * (p(i,j,k+1) - p(i,j,k-1)) * dxinv[2];
            Real h_zeta  = Compute_h_zeta_AtCellCenter(i, j, k, dxinv, zp);
            Real h_xi    = Compute_h_xi_AtCellCenter  (i, j, k, dxinv, zp);
            Real h_eta   = Compute_h_eta_AtCellCenter (i, j, k, dxinv, zp);
            Real dpdz = dpdzeta / h_zeta;
            Real dpdx = dpdxi  - h_xi  * dpdz;
            Real dpdy = dpdeta - h_eta * dpdz;

            Real magsqr_dphi = dpdx*dpdx + dpdy*dpdy + dpdz*dpdz;
            Real mag_dphi = std::sqrt(magsqr_dphi);
#if 1
            // Tucker 2003 Eqn 2
            dist_arr[b](i, j, k) = -mag_dphi + std::sqrt(magsqr_dphi + 2*phi_arr[b](i, j, k));
#else
            // DEBUG: output phi instead
            if (i==0 && j==0) AllPrint() << "walldist"<<IntVect(i,j,k) << " = " << dist_arr[b](i,j,k) << std::endl;
            dist_arr[b](i, j, k) = phi_arr[b](i, j, k);
#endif
            // Update RHS source term to explicitly include cross-terms
            if (n_corr > 0) {
                // d/dxi ( h_xi * dphi/dzeta )
                Real phi_zeta_xlo = fourth * dxinv[2] * ( phi_arr[b](i  , j, k+1) - phi_arr[b](i  , j, k-1)
                                                      + phi_arr[b](i-1, j, k+1) - phi_arr[b](i-1, j, k-1) );
                Real phi_zeta_xhi = fourth * dxinv[2] * ( phi_arr[b](i  , j, k+1) - phi_arr[b](i  , j, k-1)
                                                      + phi_arr[b](i+1, j, k+1) - phi_arr[b](i+1, j, k-1) );
                Real h_xi_xlo = Compute_h_xi_AtIface(i  , j, k, dxinv, zphys_arr[b]);
                Real h_xi_xhi = Compute_h_xi_AtIface(i+1, j, k, dxinv, zphys_arr[b]);

                // d/deta ( h_eta * dphi/dzeta )
                Real phi_zeta_ylo = fourth * dxinv[2] * ( phi_arr[b](i, j  , k+1) - phi_arr[b](i, j  , k-1)
                                                      + phi_arr[b](i, j-1, k+1) - phi_arr[b](i, j-1, k-1) );
                Real phi_zeta_yhi = fourth * dxinv[2] * ( phi_arr[b](i, j  , k+1) - phi_arr[b](i, j  , k-1)
                                                      + phi_arr[b](i, j+1, k+1) - phi_arr[b](i, j+1, k-1) );
                Real h_eta_ylo = Compute_h_eta_AtJface(i, j  , k, dxinv, zphys_arr[b]);
                Real h_eta_yhi = Compute_h_eta_AtJface(i, j+1, k, dxinv, zphys_arr[b]);

                // d/dzeta ( h_xi * dphi/dxi )
                Real phi_xi_zlo = fourth * dxinv[0] * ( phi_arr[b](i+1, j, k  ) - phi_arr[b](i-1, j, k  )
                                                    + phi_arr[b](i+1, j, k-1) - phi_arr[b](i-1, j, k-1) );
                Real phi_xi_zhi = fourth * dxinv[0] * ( phi_arr[b](i+1, j, k  ) - phi_arr[b](i-1, j, k  )
                                                    + phi_arr[b](i+1, j, k+1) - phi_arr[b](i-1, j, k+1) );
                Real h_xi_zlo = Compute_h_xi_AtKface(i, j, k  , dxinv, zphys_arr[b]);
                Real h_xi_zhi = Compute_h_xi_AtKface(i, j, k+1, dxinv, zphys_arr[b]);

                // d/dzeta ( h_eta * dphi/deta )
                Real phi_eta_zlo = fourth * dxinv[1] * ( phi_arr[b](i, j+1, k  ) - phi_arr[b](i, j-1, k  )
                                                     + phi_arr[b](i, j+1, k-1) - phi_arr[b](i, j-1, k-1) );
                Real phi_eta_zhi = fourth * dxinv[1] * ( phi_arr[b](i, j+1, k  ) - phi_arr[b](i, j-1, k  )
                                                     + phi_arr[b](i, j+1, k+1) - phi_arr[b](i, j-1, k+1) );
                Real h_eta_zlo = Compute_h_eta_AtKface(i, j, k  , dxinv, zphys_arr[b]);
                Real h_eta_zhi = Compute_h_eta_AtKface(i, j, k+1, dxinv, zphys_arr[b]);

                Real detJ = Compute_h_zeta_AtCellCenter(i, j, k, dxinv, zphys_arr[b]);

                // same sign convention as the predictor: -div(beta grad phi) = detJ - cross terms
                rhs_arr[b](i, j, k) = detJ
                                    - dxinv[0] * ( h_xi_xhi * phi_zeta_xhi - h_xi_xlo * phi_zeta_xlo)
                                    - dxinv[1] * ( h_eta_yhi * phi_zeta_yhi - h_eta_ylo * phi_zeta_ylo)
                                    - dxinv[2] * ( h_xi_zhi * phi_xi_zhi - h_xi_zlo * phi_xi_zlo
                                                 + h_eta_zhi * phi_eta_zhi - h_eta_zlo * phi_eta_zlo);
            }
        });
    } // corrector loop

    // The solve only fills the valid region, so fill the ghost cells here
    fill_wall_dist_ghost_cells(*walldist[lev], geom[lev]);
}

/**
 * Build the wall-face masks of the fraction-stress immersed wall law (see ERF_ImmersedWallCell.H)
 * from the solid fraction of this level. The vertical momentum fluxes tau13 and tau23 and the
 * matching coefficients of the implicit vertical solve are multiplied by them.
 *
 * @param lev Level index
 */
void ERF::make_ib_wall_face_masks_lev (int lev)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(solverChoice.mesh_type == MeshType::ConstantDz,
        "erf.if_wall_form = fraction_stress needs a constant-dz mesh");
    AMREX_ALWAYS_ASSERT(terrain_blanking[lev]);
    const BoxArray& ba = grids[lev];
    const DistributionMapping& dm = dmap[lev];
    ib_wall_face13[lev] = std::make_unique<MultiFab>(convert(ba, IntVect(1,0,1)), dm, 1, 1);
    ib_wall_face23[lev] = std::make_unique<MultiFab>(convert(ba, IntVect(0,1,1)), dm, 1, 1);
    ib_wall_face33[lev] = std::make_unique<MultiFab>(convert(ba, IntVect(0,0,1)), dm, 1, 1);
    ib_wall_hfx[lev]    = std::make_unique<MultiFab>(ba, dm, 1, 0);
    ib_wall_hfx[lev]->setVal(zero);
    make_ib_wall_face_masks(*terrain_blanking[lev], *ib_wall_face13[lev], *ib_wall_face23[lev],
                            *ib_wall_face33[lev],
                            geom[lev].Domain(), Real(0.005),   // small_volfrac of the terrain kernels
                            solverChoice.if_wall_face_spacing, geom[lev].CellSize(2), solverChoice.if_z0);
}

/**
 * Wall data of one level: the fraction-stress wall-face masks (erf.if_wall_form = fraction_stress)
 * and the RANS wall distance. Called for every level at start-up and again whenever a level is
 * made or remade by a regrid, since both depend on the grids and the level's solid fraction.
 *
 * @param lev Level index
 */
void ERF::make_wall_data_lev (int lev)
{
    if (solverChoice.if_fraction_stress) {
        make_ib_wall_face_masks_lev(lev);
    }

    check_if_cf_mismatch(lev);

    if (solverChoice.turbChoice[lev].rans_type != RANSType::None) {
        // Handle bottom boundary
        poisson_wall_dist(lev);

        // Correct the wall distance for immersed bodies
        if (solverChoice.advChoice.have_zero_flux_faces) {
            thinbody_wall_dist(walldist[lev],
                               solverChoice.advChoice.zero_xflux,
                               solverChoice.advChoice.zero_yflux,
                               solverChoice.advChoice.zero_zflux,
                               geom[lev],
                               z_phys_cc[lev]);

            // The correction is only applied on the valid region
            fill_wall_dist_ghost_cells(*walldist[lev], geom[lev]);
        }
    }
}

/**
 * Warn where a lateral coarse-fine boundary of level lev crosses the immersed surface in a way the
 * two levels describe differently (erf.if_cf_mismatch_warning; the terrain_cf_mismatch check of
 * kynema-sgf). For each coarse cell covered by level lev that has an uncovered lateral neighbour,
 * the spread (largest minus smallest) of the solid fractions of its fine cells is taken; a coarse
 * cell half solid over one fully solid and one fully fluid fine cell has a spread of 1, one over two
 * partial fine cells about 0.5. Where the spread reaches the threshold the c/f face fill cannot give
 * both levels the same flux near the surface, and the flow near the refinement edge is wrong (tens
 * of percent in a two-level ABL); refinement that keeps its lateral edges off the surface avoids it.
 *
 * @param lev Fine level just made
 */
void ERF::check_if_cf_mismatch (int lev)
{
    const Real warn = solverChoice.if_cf_mismatch_warning;
    if (lev < 1 || warn <= zero || !terrain_blanking[lev] || !terrain_blanking[lev-1]) { return; }

    const auto [nflagged, worst] = if_cf_surface_mismatch(*terrain_blanking[lev], grids[lev],
                                                          grids[lev-1], dmap[lev-1], geom[lev-1],
                                                          refRatio(lev-1), warn);
    if (nflagged > 0) {
        Print() << "WARNING: immersed forcing: " << nflagged << " level-" << lev-1
                << " cells along lateral coarse-fine faces of level " << lev
                << " where the levels disagree on the immersed surface (spread of the fine solid fraction >= "
                << warn << ", worst " << worst << "). Keep refinement edges off the surface, e.g. with a"
                << " refined band along the surface, a full-width level, or edges where the surface lies on a"
                << " coarse cell face (erf.if_cf_mismatch_warning)." << std::endl;
    }
}
