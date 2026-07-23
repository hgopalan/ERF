#include <ERF.H>
#include <ERF_Utils.H>
#include <ERF_UCMAtmPlotfile.H>

#ifdef ERF_USE_WINDFARM
#include <ERF_WindFarm.H>
#endif

using namespace amrex;

/**
 * Function that advances the solution at one level for a single time step --
 * this does some preliminaries then calls erf_advance
 *
 * @param[in] lev level of refinement (coarsest level is 0)
 * @param[in] time start time for time advance
 * @param[in] dt_lev time step for this time advance
 */

void
ERF::Advance (int lev, double time, double dt_lev, int iteration, int /*ncycle*/)
{
    BL_PROFILE("ERF::Advance()");

    // We must swap the pointers so the previous step's "new" is now this step's "old"
    std::swap(vars_old[lev], vars_new[lev]);

    MultiFab& S_old = vars_old[lev][Vars::cons];
    MultiFab& S_new = vars_new[lev][Vars::cons];

    MultiFab& U_old = vars_old[lev][Vars::xvel];
    MultiFab& V_old = vars_old[lev][Vars::yvel];
    MultiFab& W_old = vars_old[lev][Vars::zvel];

    MultiFab& U_new = vars_new[lev][Vars::xvel];
    MultiFab& V_new = vars_new[lev][Vars::yvel];
    MultiFab& W_new = vars_new[lev][Vars::zvel];

    // We need to set these because otherwise in the first call to erf_advance we may
    //    read uninitialized data on ghost values in setting the bc's on the velocities
    U_new.setVal(bogus_large_value,U_new.nGrowVect());
    V_new.setVal(bogus_large_value,V_new.nGrowVect());
    W_new.setVal(bogus_large_value,W_new.nGrowVect());

    //
    // NOTE: the momenta here are not fillpatched (they are only used as scratch space)
    // If lev == 0 we have already FillPatched this in ERF::TimeStep
    //
    if (lev > 0) {
        // Set ghost cells to bogus values so they aren't uninitialized
        W_old.setBndry(bogus_large_value);
        FillPatchFineLevel(lev, time, {&S_old, &U_old, &V_old, &W_old},
                           {&S_old, &rU_old[lev], &rV_old[lev], &rW_old[lev]},
                           base_state[lev], base_state[lev]);
    }

    //
    // So we must convert the fillpatched to momenta, including the ghost values
    //
    const MultiFab* c_vfrac = nullptr;
    if (solverChoice.terrain_type == TerrainType::EB) {
        c_vfrac = &((get_eb(lev).get_const_factory())->getVolFrac());
    }

    VelocityToMomentum(U_old, rU_old[lev].nGrowVect(),
                       V_old, rV_old[lev].nGrowVect(),
                       W_old, rW_old[lev].nGrowVect(),
                       S_old, rU_old[lev], rV_old[lev], rW_old[lev],
                       Geom(lev).Domain(),
                       domain_bcs_type, c_vfrac);

    // Update the inflow perturbation update time and amplitude
    if (solverChoice.use_perturbation(lev))
    {
        turbPert.calc_tpi_update(lev, dt_lev, U_old, V_old, S_old);
    }

    // If PerturbationType::Direct or CPM is selected, directly add the computed perturbation
    // on the conserved field
    if (solverChoice.use_direct_perturbation(lev))
    {
        if (solverChoice.use_wvel_perturbation(lev)) { // CPM_W
            auto m_ixtype = W_old.boxArray().ixType();
            for (MFIter mfi(W_old,TileNoZ()); mfi.isValid(); ++mfi) {
                Box bx  = mfi.tilebox();
                const Array4<Real> &cell_data  = W_old.array(mfi);
                const Array4<const Real> &pert_cell = turbPert.pb_cell[lev].array(mfi);
                turbPert.apply_tpi(lev, bx, -1, m_ixtype, cell_data, pert_cell);
            }
        } else {
            auto m_ixtype = S_old.boxArray().ixType(); // Conserved term
            for (MFIter mfi(S_old,TileNoZ()); mfi.isValid(); ++mfi) {
                Box bx  = mfi.tilebox();
                const Array4<Real> &cell_data  = S_old.array(mfi);
                const Array4<const Real> &pert_cell = turbPert.pb_cell[lev].array(mfi);
                turbPert.apply_tpi(lev, bx, RhoTheta_comp, m_ixtype, cell_data, pert_cell);
            }
        }
    }

    // configure SurfaceLayer params if needed
    if (phys_bc_type[Orientation(Direction::z,Orientation::low)] == ERF_BC::surface_layer) {
        if (m_SurfaceLayer) {
            IntVect ng = Theta_prim[lev]->nGrowVect();
            MultiFab::Copy(  *Theta_prim[lev], S_old, RhoTheta_comp, 0, 1, ng);
            MultiFab::Divide(*Theta_prim[lev], S_old, Rho_comp     , 0, 1, ng);
            if (solverChoice.moisture_type != MoistureType::None) {
                ng = Qv_prim[lev]->nGrowVect();

                MultiFab::Copy(  *Qv_prim[lev], S_old, RhoQ1_comp, 0, 1, ng);
                MultiFab::Divide(*Qv_prim[lev], S_old, Rho_comp  , 0, 1, ng);

                if (solverChoice.moisture_indices.qr > -1) {
                    MultiFab::Copy(  *Qr_prim[lev], S_old, solverChoice.moisture_indices.qr, 0, 1, ng);
                    MultiFab::Divide(*Qr_prim[lev], S_old, Rho_comp  , 0, 1, ng);
                } else {
                    Qr_prim[lev]->setVal(0);
                }
            }
            // NOTE: std::swap above causes the field ptrs to be out of date.
            //       Reassign the field ptrs for MAC avg computation.
            m_SurfaceLayer->update_mac_ptrs(lev, vars_old, Theta_prim, Qv_prim, Qr_prim);
            m_SurfaceLayer->update_pblh(lev, vars_old, z_phys_cc[lev].get(),
                                        solverChoice.moisture_indices);

#ifdef ERF_USE_NETCDF
            double elapsed_time_since_start_low = time + (start_time - start_low_time);
#else
            double elapsed_time_since_start_low = time;
#endif
            m_SurfaceLayer->update_fluxes(lev, time, elapsed_time_since_start_low,
                                          S_old, z_phys_nd[lev], walldist[lev]);
        }
    }

    // **************************************************************************************
    // Phase 1.3: Advance SLUCM facet SEB and slab conduction
    // **************************************************************************************
    #ifdef ERF_USE_UCM
    if (m_ucm_params.enable && m_ucm_layer[lev] != nullptr && m_SurfaceLayer) {

        const int gr      = m_ucm_params.grid_ratio;
        const int klo_atm = Geom(lev).Domain().smallEnd(2);

        // --- Build ATM-grid T_atm and q_atm (θ = ρθ/ρ, q = ρq/ρ) with 1 ghost ---
        amrex::MultiFab T_atm_3d(S_old.boxArray(), S_old.DistributionMap(), 1, amrex::IntVect(1,1,0));
        T_atm_3d.setVal(0.0);
        amrex::MultiFab::Copy  (T_atm_3d, S_old, RhoTheta_comp, 0, 1, T_atm_3d.nGrowVect());
        amrex::MultiFab::Divide(T_atm_3d, S_old, Rho_comp,      0, 1, T_atm_3d.nGrowVect());

        amrex::MultiFab q_atm_3d(S_old.boxArray(), S_old.DistributionMap(), 1, amrex::IntVect(1,1,0));
        q_atm_3d.setVal(0.0);
        if (solverChoice.moisture_type != MoistureType::None) {
            amrex::MultiFab::Copy  (q_atm_3d, S_old, RhoQ1_comp, 0, 1, q_atm_3d.nGrowVect());
            amrex::MultiFab::Divide(q_atm_3d, S_old, Rho_comp,   0, 1, q_atm_3d.nGrowVect());
        }

        // --- Interpolate U,V to cell centers on ATM (2D horizontal slab at k=klo) ---
        amrex::MultiFab U_cc_atm(S_old.boxArray(), S_old.DistributionMap(), 1, 0);
        amrex::MultiFab V_cc_atm(S_old.boxArray(), S_old.DistributionMap(), 1, 0);
        U_cc_atm.setVal(0.0);
        V_cc_atm.setVal(0.0);
        for (amrex::MFIter mfi(U_cc_atm, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            auto ucc = U_cc_atm.array(mfi);
            auto vcc = V_cc_atm.array(mfi);
            auto const uf = U_old.const_array(mfi);
            auto const vf = V_old.const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                ucc(i,j,k) = amrex::Real(0.5) * (uf(i,j,k) + uf(i+1,j,k));
                vcc(i,j,k) = amrex::Real(0.5) * (vf(i,j,k) + vf(i,j+1,k));
            });
        }

        // --- Allocate UCM-grid scratch and refine each ATM input onto UCM ---
        const auto& ba_u = m_ucm_grid[lev]->ba;
        const auto& dm_u = m_ucm_grid[lev]->dm;

        amrex::MultiFab T_atm_ucm(ba_u, dm_u, 1, 0);
        amrex::MultiFab q_atm_ucm(ba_u, dm_u, 1, 0);
        amrex::MultiFab U_atm_ucm(ba_u, dm_u, 1, 0);
        amrex::MultiFab V_atm_ucm(ba_u, dm_u, 1, 0);
        amrex::MultiFab ustar_ucm(ba_u, dm_u, 1, 0);
        amrex::MultiFab tstar_ucm(ba_u, dm_u, 1, 0);
        amrex::MultiFab qstar_ucm(ba_u, dm_u, 1, 0);

        refine_atm_to_ucm(T_atm_ucm, T_atm_3d,                                     gr, klo_atm);
        refine_atm_to_ucm(q_atm_ucm, q_atm_3d,                                     gr, klo_atm);
        refine_atm_to_ucm(U_atm_ucm, U_cc_atm,                                     gr, klo_atm);
        refine_atm_to_ucm(V_atm_ucm, V_cc_atm,                                     gr, klo_atm);
        refine_atm_to_ucm(ustar_ucm, *m_SurfaceLayer->get_u_star(lev),             gr, klo_atm);
        refine_atm_to_ucm(tstar_ucm, *m_SurfaceLayer->get_t_star(lev),             gr, klo_atm);
        refine_atm_to_ucm(qstar_ucm, *m_SurfaceLayer->get_q_star(lev),             gr, klo_atm);

        // --- Call UCMLayer::advance with UCM-grid inputs and UCM geometry ---
        m_ucm_layer[lev]->advance(*m_ucm_fields[lev], *m_ucm_forcing[lev], *m_ucm_grid[lev],
                                  ustar_ucm, tstar_ucm, qstar_ucm,
                                  U_atm_ucm, V_atm_ucm,
                                  *z_phys_cc[lev].get(),
                                  T_atm_ucm, q_atm_ucm,
                                  m_ucm_grid[lev]->geom,
                                  time, dt_lev, 1, lev);

        // One-per-step confirmation that SEB ran (appears once between consecutive
        // "Making slow rhs" lines, confirming the once-per-coarse-step contract)
        if (m_ucm_params.ucm_debug && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][step] SEB advanced at time=" << time
                           << " dt=" << dt_lev << " lev=" << lev << "\n";
        }
    }
    #endif

#if defined(ERF_USE_WINDFARM)
    // **************************************************************************************
    // Update the windfarm sources
    // **************************************************************************************
    if (solverChoice.windfarm_type != WindFarmType::None) {
        advance_windfarm(Geom(lev), dt_lev, S_old,
                         U_old, V_old, W_old, vars_windfarm[lev],
                         Nturb[lev], SMark[lev], time);
    }

#endif

    // **************************************************************************************
    // Update the radiation sources with the "old" state
    // **************************************************************************************
    advance_radiation(lev, S_old, dt_lev);

    // **************************************************************************************
    // Update the "old" state using SHOC
    // **************************************************************************************
    if (solverChoice.turbChoice[lev].uses_shoc_family()) {
        // Get SFC fluxes from SurfaceLayer
        if (m_SurfaceLayer) {
            Vector<const MultiFab*> mfs = {&S_old, &U_old, &V_old, &W_old};
            m_SurfaceLayer->impose_SurfaceLayer_bcs(lev, mfs, Tau[lev],
                                                    SFS_hfx1_lev[lev].get() , SFS_hfx2_lev[lev].get() , SFS_hfx3_lev[lev].get(),
                                                    SFS_q1fx1_lev[lev].get(), SFS_q1fx2_lev[lev].get(), SFS_q1fx3_lev[lev].get(),
                                                    z_phys_nd[lev].get());
        }

        // Apply SHOC before the dycore so it sees a coherent state.
        Real* w_sub = (solverChoice.custom_w_subsidence) ? d_w_subsid[lev].data() : nullptr;
        if (solverChoice.turbChoice[lev].uses_eamxx_shoc()) {
#ifdef ERF_USE_EAMXX_SHOC
            compute_shoc_tendencies(lev, &S_old, &U_old, &V_old, &W_old, w_sub,
                                    Tau[lev][TauType::tau13].get(), Tau[lev][TauType::tau23].get(),
                                    SFS_hfx3_lev[lev].get()       , SFS_q1fx3_lev[lev].get()      ,
                                    eddyDiffs_lev[lev].get()      , z_phys_nd[lev].get()          ,
                                    dt_lev);
#endif
        } else if (solverChoice.turbChoice[lev].uses_native_shoc()) {
            compute_native_shoc_tendencies(lev, &S_old, &U_old, &V_old, &W_old, w_sub,
                                           Tau[lev][TauType::tau13].get(), Tau[lev][TauType::tau23].get(),
                                           SFS_hfx3_lev[lev].get()       , SFS_q1fx3_lev[lev].get()      ,
                                           eddyDiffs_lev[lev].get()      , z_phys_nd[lev].get()          ,
                                           dt_lev);

            if (native_shoc_driver[lev] && native_shoc_driver[lev]->uses_state_update()) {
                // Native SHOC updates the old-time state before the dycore reads it.
                // Re-fill the updated state, velocities, and momenta now so the
                // pre-dycore checks and strain calculation see coherent fields.
                Vector<MultiFab*> mfs_vel = {&S_old, &U_old, &V_old, &W_old};
                if (lev == 0) {
                    FillPatchCrseLevel(lev, time, mfs_vel, false);
                    VelocityToMomentum(U_old, rU_old[lev].nGrowVect(),
                                       V_old, rV_old[lev].nGrowVect(),
                                       W_old, rW_old[lev].nGrowVect(),
                                       S_old, rU_old[lev], rV_old[lev], rW_old[lev],
                                       Geom(lev).Domain(),
                                       domain_bcs_type, c_vfrac);
                } else {
                    Vector<MultiFab*> mfs_mom = {&S_old, &rU_old[lev], &rV_old[lev], &rW_old[lev]};
                    FillPatchFineLevel(lev, time, mfs_vel, mfs_mom,
                                       base_state[lev], base_state[lev],
                                       true, false);
                }
            }
        }
    }

    const BoxArray&            ba = S_old.boxArray();
    const DistributionMapping& dm = S_old.DistributionMap();

    int nvars = S_old.nComp();

    // Source array for conserved cell-centered quantities -- this will be filled
    //     in the call to make_sources in ERF_TI_slow_rhs_pre.H
    MultiFab cc_source(ba,dm,nvars,1); cc_source.setVal(0);

    // Source arrays for momenta -- these will be filled
    //     in the call to make_mom_sources in ERF_TI_slow_rhs_pre.H
    BoxArray ba_x(ba); ba_x.surroundingNodes(0);
    MultiFab xmom_source(ba_x,dm,1,1); xmom_source.setVal(0);

    BoxArray ba_y(ba); ba_y.surroundingNodes(1);
    MultiFab ymom_source(ba_y,dm,1,1); ymom_source.setVal(0);

    BoxArray ba_z(ba); ba_z.surroundingNodes(2);
    MultiFab zmom_source(ba_z,dm,1,1); zmom_source.setVal(0);
    MultiFab    buoyancy(ba_z,dm,1,1); buoyancy.setVal(0);

    // **************************************************************************************
    // Phase 1.4: Cache UCM ATM fluxes once per coarse step.
    // The actual per-RK-stage injection happens in ERF_TI_slow_rhs_pre.H via
    // apply_ucm_tendency_to_cc_source(), which overwrites cc_src[RhoTheta_comp]
    // after each make_sources() reset.  See ERF_UCMAtmCoupling.cpp for the
    // RK-stage safety contract.
    // **************************************************************************************
    #ifdef ERF_USE_UCM
    if (m_ucm_params.enable && m_ucm_layer[lev] != nullptr && m_ucm_params.atm_feedback > 0.0) {
        // Allocate cached ATM-grid flux MultiFabs on first call
        if (!m_ucm_H_atm[lev]) {
            m_ucm_H_atm[lev] = std::make_unique<amrex::MultiFab>(ba, dm, 1, 0);
            m_ucm_H_atm[lev]->setVal(0.0);
        }
        if (solverChoice.moisture_type != MoistureType::None && !m_ucm_LE_atm[lev]) {
            m_ucm_LE_atm[lev] = std::make_unique<amrex::MultiFab>(ba, dm, 1, 0);
            m_ucm_LE_atm[lev]->setVal(0.0);
        }

        // Phase 2.5: Allocate morphology aggregates on ATM grid (first call)
        if (!m_ucm_f_urb_atm[lev]) {
            m_ucm_f_urb_atm[lev] = std::make_unique<amrex::MultiFab>(ba, dm, 1, 0);
            m_ucm_f_urb_atm[lev]->setVal(0.0);
        }
        if (!m_ucm_H_bldg_mean_atm[lev]) {
            m_ucm_H_bldg_mean_atm[lev] = std::make_unique<amrex::MultiFab>(ba, dm, 1, 0);
            m_ucm_H_bldg_mean_atm[lev]->setVal(0.0);
        }
        if (!m_ucm_H_bldg_std_atm[lev]) {
            m_ucm_H_bldg_std_atm[lev] = std::make_unique<amrex::MultiFab>(ba, dm, 1, 0);
            m_ucm_H_bldg_std_atm[lev]->setVal(0.0);
        }
        if (!m_ucm_lambda_p_atm[lev]) {
            m_ucm_lambda_p_atm[lev] = std::make_unique<amrex::MultiFab>(ba, dm, 1, 0);
            m_ucm_lambda_p_atm[lev]->setVal(0.0);
        }
        if (!m_ucm_lambda_f_atm[lev]) {
            m_ucm_lambda_f_atm[lev] = std::make_unique<amrex::MultiFab>(ba, dm, 1, 0);
            m_ucm_lambda_f_atm[lev]->setVal(0.0);
        }

        // Phase 2.5: Compute morphology aggregates from UCM grid to ATM grid
        aggregate_ucm_morphology_to_atm(
            *m_ucm_f_urb_atm[lev],
            *m_ucm_H_bldg_mean_atm[lev],
            *m_ucm_H_bldg_std_atm[lev],
            *m_ucm_lambda_p_atm[lev],
            *m_ucm_lambda_f_atm[lev],
            *m_ucm_fields[lev]->H_bldg,
            *m_ucm_fields[lev]->W_road,
            *m_ucm_fields[lev]->plan_area_frac,
            *m_ucm_fields[lev]->is_urban,
            m_ucm_grid[lev]->geom, Geom(lev),
            m_ucm_params.grid_ratio,
            m_ucm_params.ucm_debug, lev);

        // Phase 2.5: One-time BANNER for aggregates (collective min/max outside IOProcessor guard)
        static bool aggregate_banner_printed = false;
        if (!aggregate_banner_printed && m_ucm_params.ucm_debug) {
            aggregate_banner_printed = true;
            // Collectives outside IOProcessor guard (PR #209 rule)
            const amrex::Real fu_min = m_ucm_f_urb_atm[lev]->min(0, 0);
            const amrex::Real fu_max = m_ucm_f_urb_atm[lev]->max(0, 0);
            const amrex::Real Hm_min = m_ucm_H_bldg_mean_atm[lev]->min(0, 0);
            const amrex::Real Hm_max = m_ucm_H_bldg_mean_atm[lev]->max(0, 0);
            const amrex::Real Hs_max = m_ucm_H_bldg_std_atm[lev]->max(0, 0);
            const amrex::Real lp_max = m_ucm_lambda_p_atm[lev]->max(0, 0);
            const amrex::Real lf_max = m_ucm_lambda_f_atm[lev]->max(0, 0);
            if (amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "\n[UCM][2.5-followup][BANNER] ATM-grid aggregates:\n"
                               << "  f_urb        min=" << fu_min << " max=" << fu_max << "\n"
                               << "  H_bldg_mean  min=" << Hm_min << " max=" << Hm_max << " m\n"
                               << "  H_bldg_std   max=" << Hs_max << " m\n"
                               << "  lambda_p     max=" << lp_max << "\n"
                               << "  lambda_f     max=" << lf_max << "\n\n";
            }
        }

        // Coarsen UCM fluxes from UCM grid to ATM grid (lagged; constant across RK stages)
        // Phase 2.5: Use area-averaged coarsening (convention B)
        coarsen_ucm_flux_to_atm(*m_ucm_H_atm[lev], *m_ucm_fields[lev]->H_sensible,
                                *m_ucm_fields[lev]->is_urban,
                                m_ucm_grid[lev]->geom, Geom(lev),
                                m_ucm_params.grid_ratio, lev);
        if (solverChoice.moisture_type != MoistureType::None && m_ucm_fields[lev]->LE_latent) {
            coarsen_ucm_flux_to_atm(*m_ucm_LE_atm[lev], *m_ucm_fields[lev]->LE_latent,
                                    *m_ucm_fields[lev]->is_urban,
                                    m_ucm_grid[lev]->geom, Geom(lev),
                                    m_ucm_params.grid_ratio, lev);
        }

        // Phase 2.6: Separate coarsening for road and wall+roof+AH channels.
        // Phase 2.7: Further split roof and wall into separate ATM-grid fields.
        // H_road_ucm is already populated by SEB; H_wall + H_roof already include AH per Phase 2.3.
        if (!m_ucm_H_road_atm[lev]) {
           m_ucm_H_road_atm[lev] = std::make_unique<amrex::MultiFab>(ba, dm, 1, 0);
           m_ucm_H_road_atm[lev]->setVal(0.0);
        }
        // Phase 2.7: Create separate wall and roof ATM-grid fields
        if (!m_ucm_H_wall_atm[lev]) {
           m_ucm_H_wall_atm[lev] = std::make_unique<amrex::MultiFab>(ba, dm, 1, 0);
           m_ucm_H_wall_atm[lev]->setVal(0.0);
        }
        if (!m_ucm_H_roof_atm[lev]) {
           m_ucm_H_roof_atm[lev] = std::make_unique<amrex::MultiFab>(ba, dm, 1, 0);
           m_ucm_H_roof_atm[lev]->setVal(0.0);
        }

        coarsen_ucm_flux_to_atm(*m_ucm_H_road_atm[lev], *m_ucm_fields[lev]->H_road,
                               *m_ucm_fields[lev]->is_urban,
                               m_ucm_grid[lev]->geom, Geom(lev),
                               m_ucm_params.grid_ratio, lev);

        // Phase 2.7: Coarsen wall and roof fluxes separately
        coarsen_ucm_flux_to_atm(*m_ucm_H_wall_atm[lev], *m_ucm_fields[lev]->H_wall,
                               *m_ucm_fields[lev]->is_urban,
                               m_ucm_grid[lev]->geom, Geom(lev),
                               m_ucm_params.grid_ratio, lev);

        coarsen_ucm_flux_to_atm(*m_ucm_H_roof_atm[lev], *m_ucm_fields[lev]->H_roof,
                               *m_ucm_fields[lev]->is_urban,
                               m_ucm_grid[lev]->geom, Geom(lev),
                               m_ucm_params.grid_ratio, lev);

        // Debug print for Phase 2.6/2.7 fields
        if (m_ucm_params.ucm_debug) {
           const amrex::Real h_road_min = m_ucm_H_road_atm[lev]->min(0, 0);
           const amrex::Real h_road_max = m_ucm_H_road_atm[lev]->max(0, 0);
           const amrex::Real h_wall_min = m_ucm_H_wall_atm[lev]->min(0, 0);
           const amrex::Real h_wall_max = m_ucm_H_wall_atm[lev]->max(0, 0);
           const amrex::Real h_roof_min = m_ucm_H_roof_atm[lev]->min(0, 0);
           const amrex::Real h_roof_max = m_ucm_H_roof_atm[lev]->max(0, 0);
           if (amrex::ParallelDescriptor::IOProcessor()) {
               amrex::Print() << "[UCM][2.7][coarsen_ucm_flux_to_atm] Facet-split fluxes:\n"
                              << "  H_road_atm  min=" << h_road_min << " max=" << h_road_max << " [W/m2]\n"
                              << "  H_wall_atm  min=" << h_wall_min << " max=" << h_wall_max << " [W/m2]\n"
                              << "  H_roof_atm  min=" << h_roof_min << " max=" << h_roof_max << " [W/m2]\n";
           }
        }

        // Build is_urban mask on ATM grid if not already allocated
        if (!m_ucm_is_urban_atm[lev]) {
            m_ucm_is_urban_atm[lev] = std::make_unique<amrex::iMultiFab>(ba, dm, 1, 0);
            m_ucm_is_urban_atm[lev]->setVal(1);
        }

        // Diagnostics output (once per coarse step)
        if (m_ucm_params.ucm_diag_file.size() > 0) {
            if (!m_ucm_diagnostics[lev]) {
                m_ucm_diagnostics[lev] = std::make_unique<UCMDiagnostics>(m_ucm_params, lev);
            }
            m_ucm_diagnostics[lev]->append(*m_ucm_fields[lev], iteration, time,
                                           m_ucm_f_urb_atm[lev].get(),
                                           m_ucm_H_bldg_mean_atm[lev].get(),
                                           m_ucm_H_bldg_std_atm[lev].get(),
                                           m_ucm_lambda_f_atm[lev].get(),
                                           m_ucm_H_atm[lev].get(),
                                           lev);
        }

        // Plotfile output (once per coarse step)
        if (m_ucm_params.ucm_plot_int > 0 && (iteration % m_ucm_params.ucm_plot_int == 0)) {
            if (!m_ucm_plotfile[lev]) {
                m_ucm_plotfile[lev] = std::make_unique<UCMPlotfile>(m_ucm_params, lev);
            }
            m_ucm_plotfile[lev]->write(*m_ucm_fields[lev], *m_ucm_grid[lev],
                                       iteration, time, false, lev);
        }
        
        // Phase 2.5: ATM-grid aggregate plotfile output (once per coarse step)
        // Phase 2.7: Now with separate wall and roof fluxes
        if (m_ucm_params.ucm_atm_plot_int > 0 &&
           (iteration % m_ucm_params.ucm_atm_plot_int == 0))
        {
           // Phase 2.7: Updated call with 9 components (split H_wallroof into H_wall and H_roof)
           m_ucm_atm_plotfile[lev]->write(
               iteration,
               time,
               *m_ucm_f_urb_atm[lev],
               *m_ucm_H_bldg_mean_atm[lev],
               *m_ucm_H_bldg_std_atm[lev],
               *m_ucm_lambda_p_atm[lev],
               *m_ucm_lambda_f_atm[lev],
               *m_ucm_H_atm[lev],
               *m_ucm_H_road_atm[lev],       // Phase 2.6: road flux
               *m_ucm_H_wall_atm[lev],       // Phase 2.7: wall flux
               *m_ucm_H_roof_atm[lev],       // Phase 2.7: roof flux (incl AH)
               Geom(lev),
               m_ucm_params.ucm_debug,
               lev);
        }
    }
    #endif

    amrex::Vector<MultiFab> state_old;
    amrex::Vector<MultiFab> state_new;

    // **************************************************************************************
    // Here we define state_old and state_new which are to be advanced
    // **************************************************************************************
    // Initial solution
    // Note that "old" and "new" here are relative to each RK stage.
    state_old.push_back(MultiFab(S_old      , amrex::make_alias, 0, nvars)); // cons
    state_old.push_back(MultiFab(rU_old[lev], amrex::make_alias, 0,     1)); // xmom
    state_old.push_back(MultiFab(rV_old[lev], amrex::make_alias, 0,     1)); // ymom
    state_old.push_back(MultiFab(rW_old[lev], amrex::make_alias, 0,     1)); // zmom

    // Final solution
    // state_new at the end of the last RK stage holds the t^{n+1} data
    state_new.push_back(MultiFab(S_new      , amrex::make_alias, 0, nvars)); // cons
    state_new.push_back(MultiFab(rU_new[lev], amrex::make_alias, 0,     1)); // xmom
    state_new.push_back(MultiFab(rV_new[lev], amrex::make_alias, 0,     1)); // ymom
    state_new.push_back(MultiFab(rW_new[lev], amrex::make_alias, 0,     1)); // zmom

    // **************************************************************************************
    // Tests on the reasonableness of the solution before the dycore
    // **************************************************************************************
    // Test for NaNs after dycore
    if (check_for_nans > 1) {
        if (verbose > 1) {
            amrex::Print() << "Testing old state and vels for NaNs before dycore" << std::endl;
        }
        check_state_for_nans(S_old);
        check_vels_for_nans(rU_old[lev],rV_old[lev],rW_old[lev]);
    }

    // We only test on low temp if we have a moisture model because we are protecting against
    //    the test on low temp inside the moisture models
    if (solverChoice.moisture_type != MoistureType::None) {
        if (verbose > 1) {
            amrex::Print() << "Testing on low temperature before dycore" << std::endl;
        }
        check_for_low_temp(S_old);
    } else {
        if (verbose > 1) {
            amrex::Print() << "Testing on negative temperature before dycore" << std::endl;
        }
        check_for_negative_theta(S_old);
    }
    // Before calling advance_dycore, enable the rhotheta_src path for UCM
    // NOTE: UCM injection now happens per-RK-stage in ERF_TI_slow_rhs_pre.H via
    //       apply_ucm_tendency_to_cc_source() with overwrite semantics.
    //       The custom_rhotheta_forcing path is NOT used for UCM — do NOT set it here.
    #ifdef ERF_USE_UCM
    // (no custom_rhotheta_forcing for UCM — see RK-stage safety contract in
    //  Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp)
    #endif

    // **************************************************************************************
    // Update the dycore
    // **************************************************************************************
    advance_dycore(lev, state_old, state_new,
                   U_old, V_old, W_old,
                   U_new, V_new, W_new,
                   cc_source, xmom_source, ymom_source, zmom_source, buoyancy,
                   Geom(lev), dt_lev, time);

    // **************************************************************************************
    // Tests on the reasonableness of the solution after the dycore
    // **************************************************************************************
    // Test for NaNs after dycore
    if (check_for_nans > 0) {
        if (verbose > 1) {
            amrex::Print() << "Testing new state and vels for NaNs after dycore" << std::endl;
        }
        check_state_for_nans(S_new);
        check_vels_for_nans(rU_new[lev],rV_new[lev],rW_new[lev]);
    }

    // We only test on low temp if we have a moisture model because we are protecting against
    //    the test on low temp inside the moisture models
    if (solverChoice.moisture_type != MoistureType::None) {
        if (verbose > 1) {
            amrex::Print() << "Testing on low temperature after dycore" << std::endl;
        }
        check_for_low_temp(S_new);
    } else {
        // Otherwise we will test on negative (rhotheta) coming out of the dycore
        if (verbose > 1) {
            amrex::Print() << "Testing on negative temperature after dycore" << std::endl;
        }
        check_for_negative_theta(S_new);
    }

    // **************************************************************************************
    // Update the microphysics (moisture)
    // **************************************************************************************
    if (!solverChoice.moisture_tight_coupling)
    {
        advance_microphysics(lev, S_new, dt_lev, iteration, time);

        // Test for NaNs after microphysics
        if (check_for_nans > 0) {
            amrex::Print() << "Testing new state for NaNs after advance_microphysics" << std::endl;
            check_state_for_nans(S_new);
        }
    }

    // **************************************************************************************
    // Update the land surface model
    // **************************************************************************************
    double time_at_end_of_step = time+dt_lev;
    advance_lsm(lev, S_new, U_new, V_new, time_at_end_of_step, dt_lev);

#ifdef ERF_USE_PARTICLES
    // **************************************************************************************
    // Update the particle positions
    // **************************************************************************************
   evolveTracers(lev, dt_lev, vars_new, z_phys_nd);
#endif

    // ***********************************************************************************************
    // Impose domain boundary conditions here so that in FillPatching the fine data we won't
    // need to re-fill these
    // ***********************************************************************************************
    if (lev < finest_level) {
         IntVect ngvect_vels = vars_new[lev][Vars::xvel].nGrowVect();
         (*physbcs_cons[lev])(vars_new[lev][Vars::cons], vars_new[lev][Vars::xvel], vars_new[lev][Vars::yvel],
                              0,vars_new[lev][Vars::cons].nComp(),
                              vars_new[lev][Vars::cons].nGrowVect(),time,BCVars::cons_bc,true);
            (*physbcs_u[lev])(vars_new[lev][Vars::xvel], vars_new[lev][Vars::xvel], vars_new[lev][Vars::yvel],
                              ngvect_vels,time,BCVars::xvel_bc,true);
            (*physbcs_v[lev])(vars_new[lev][Vars::yvel], vars_new[lev][Vars::xvel], vars_new[lev][Vars::yvel],
                              ngvect_vels,time,BCVars::yvel_bc,true);
            (*physbcs_w[lev])(vars_new[lev][Vars::zvel], vars_new[lev][Vars::xvel], vars_new[lev][Vars::yvel],
                              ngvect_vels,time,BCVars::zvel_bc,true);
    }

    // **************************************************************************************
    // Register old and new coarse data if we are at a level less than the finest level
    // **************************************************************************************
    if (lev < finest_level) {
        if (cf_width > 0) {
            // We must fill the ghost cells of these so that the parallel copy works correctly
            state_old[IntVars::cons].FillBoundary(geom[lev].periodicity());
            state_new[IntVars::cons].FillBoundary(geom[lev].periodicity());
            FPr_c[lev].RegisterCoarseData({&state_old[IntVars::cons], &state_new[IntVars::cons]},
                                          {time, time+dt_lev});
        }

        if (cf_width >= 0) {
            // We must fill the ghost cells of these so that the parallel copy works correctly
            state_old[IntVars::xmom].FillBoundary(geom[lev].periodicity());
            state_new[IntVars::xmom].FillBoundary(geom[lev].periodicity());
            FPr_u[lev].RegisterCoarseData({&state_old[IntVars::xmom], &state_new[IntVars::xmom]},
                                          {time, time+dt_lev});

            state_old[IntVars::ymom].FillBoundary(geom[lev].periodicity());
            state_new[IntVars::ymom].FillBoundary(geom[lev].periodicity());
            FPr_v[lev].RegisterCoarseData({&state_old[IntVars::ymom], &state_new[IntVars::ymom]},
                                          {time, time+dt_lev});

            state_old[IntVars::zmom].FillBoundary(geom[lev].periodicity());
            state_new[IntVars::zmom].FillBoundary(geom[lev].periodicity());
            FPr_w[lev].RegisterCoarseData({&state_old[IntVars::zmom], &state_new[IntVars::zmom]},
                                          {time, time+dt_lev});
        }

            //
            // Now create a MultiFab that holds (S_new - S_old) / dt from the coarse level interpolated
            //     on to the coarse/fine boundary at the fine resolution
            //
            Interpolater* mapper_f = &face_cons_linear_interp;

            MultiFab temp_state(zmom_crse_rhs[lev+1].boxArray(),zmom_crse_rhs[lev+1].DistributionMap(),1,0);
            InterpFromCoarseLevel(temp_state,            IntVect{0}, IntVect{0}, state_old[IntVars::zmom], 0, 0, 1,
                                  geom[lev], geom[lev+1], refRatio(lev), mapper_f, domain_bcs_type, BCVars::zvel_bc);
            InterpFromCoarseLevel(zmom_crse_rhs[lev+1],  IntVect{0}, IntVect{0}, state_new[IntVars::zmom], 0, 0, 1,
                                  geom[lev], geom[lev+1], refRatio(lev), mapper_f, domain_bcs_type, BCVars::zvel_bc);
            MultiFab::Subtract(zmom_crse_rhs[lev+1],temp_state,0,0,1,IntVect{0});

            Real inv_dt = static_cast<Real>(one/dt_lev);
            zmom_crse_rhs[lev+1].mult(inv_dt,0,1,0);
    }

    // ***********************************************************************************************
    // Update the time averaged velocities if they are requested
    // ***********************************************************************************************
    if (solverChoice.time_avg_vel) {
        Time_Avg_Vel_atCC(dt[lev], t_avg_cnt[lev], vel_t_avg[lev].get(), U_new, V_new, W_new);
    }
}