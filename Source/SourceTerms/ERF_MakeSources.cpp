#include <AMReX_MultiFab.H>
#include <AMReX_ArrayLim.H>
#include <AMReX_BCRec.H>
#include <AMReX_TableData.H>
#include <AMReX_GpuContainers.H>

#include <ERF_NumericalDiffusion.H>
#include <ERF_SrcHeaders.H>
#include <ERF_TI_slow_headers.H>

using namespace amrex;

/**
 * Function for computing the slow RHS for the evolution equations for the density, potential temperature and momentum.
 *
 * @param[in]  level level of resolution
 * @param[in]  nrk   which RK stage
 * @param[in]  dt    slow time step
 * @param[in]  S_data current solution
 * @param[in]  S_prim primitive variables (i.e. conserved variables divided by density)
 * @param[in] source source terms for conserved variables
 * @param[in]  geom   Container for geometric information
 * @param[in]  solverChoice  Container for solver parameters
 * @param[in] mapfac_u map factor at x-faces
 * @param[in] mapfac_v map factor at y-faces
 * @param[in] dptr_rhotheta_src  custom temperature source term
 * @param[in] dptr_rhoqt_src  custom moisture source term
 * @param[in] dptr_wbar_sub  subsidence source term
 * @param[in] d_rayleigh_ptrs_at_lev  Vector of {strength of Rayleigh damping, reference value of theta} used to define Rayleigh damping
 */

void make_sources (int level,
                   int /*nrk*/, Real dt, Real time,
                   Vector<MultiFab>& S_data,
                   const  MultiFab & S_prim,
                          MultiFab & source,
                   std::unique_ptr<MultiFab>& z_phys_cc,
#ifdef ERF_USE_RRTMGP
                   const MultiFab* qheating_rates,
#endif
                   const Geometry geom,
                   const SolverChoice& solverChoice,
                   std::unique_ptr<MultiFab>& /*mapfac_u*/,
                   std::unique_ptr<MultiFab>& /*mapfac_v*/,
                   std::unique_ptr<MultiFab>& mapfac_m,
                   const Real* dptr_rhotheta_src,
                   const Real* dptr_rhoqt_src,
                   const Real* dptr_wbar_sub,
                   const Vector<Real*> d_rayleigh_ptrs_at_lev,
                   InputSoundingData& input_sounding_data,
                   TurbulentPerturbation& turbPert)
{
    BL_PROFILE_REGION("erf_make_sources()");

    // *****************************************************************************
    // Initialize source to zero since we re-compute it every RK stage
    // *****************************************************************************
    source.setVal(0.0);

    const bool l_use_ndiff      = solverChoice.use_NumDiff;
    const bool use_terrain      = solverChoice.terrain_type != TerrainType::None;

    TurbChoice tc = solverChoice.turbChoice[level];
    const bool l_use_KE  =  ( (tc.les_type == LESType::Deardorff) ||
                              (tc.pbl_type == PBLType::MYNN25) );
    const bool l_diff_KE = tc.diffuse_KE_3D;

    const Box& domain = geom.Domain();

    const GpuArray<Real, AMREX_SPACEDIM> dxInv = geom.InvCellSizeArray();

    Real* thetabar = d_rayleigh_ptrs_at_lev[Rayleigh::thetabar];

    // *****************************************************************************
    // Planar averages for subsidence terms
    // *****************************************************************************
    Table1D<Real>      dptr_r_plane, dptr_t_plane, dptr_qv_plane, dptr_qc_plane;
    TableData<Real, 1>  r_plane_tab,  t_plane_tab,  qv_plane_tab,  qc_plane_tab;
    if (dptr_wbar_sub || solverChoice.nudging_from_input_sounding)
    {
        // Rho
        PlaneAverage r_ave(&(S_data[IntVars::cons]), geom, solverChoice.ave_plane, true);
        r_ave.compute_averages(ZDir(), r_ave.field());

        int ncell = r_ave.ncell_line();
        Gpu::HostVector<    Real> r_plane_h(ncell);
        Gpu::DeviceVector<  Real> r_plane_d(ncell);

        r_ave.line_average(Rho_comp, r_plane_h);

        Gpu::copyAsync(Gpu::hostToDevice, r_plane_h.begin(), r_plane_h.end(), r_plane_d.begin());

        Real* dptr_r = r_plane_d.data();

        IntVect ng_c = S_data[IntVars::cons].nGrowVect();
        Box tdomain  = domain; tdomain.grow(2,ng_c[2]);
        r_plane_tab.resize({tdomain.smallEnd(2)}, {tdomain.bigEnd(2)});

        int offset = ng_c[2];
        dptr_r_plane = r_plane_tab.table();
        ParallelFor(ncell, [=] AMREX_GPU_DEVICE (int k) noexcept
        {
            dptr_r_plane(k-offset) = dptr_r[k];
        });

        // Rho * Theta
        PlaneAverage t_ave(&(S_data[IntVars::cons]), geom, solverChoice.ave_plane, true);
        t_ave.compute_averages(ZDir(), t_ave.field());

        Gpu::HostVector<    Real> t_plane_h(ncell);
        Gpu::DeviceVector<  Real> t_plane_d(ncell);

        t_ave.line_average(RhoTheta_comp, t_plane_h);

        Gpu::copyAsync(Gpu::hostToDevice, t_plane_h.begin(), t_plane_h.end(), t_plane_d.begin());

        Real* dptr_t = t_plane_d.data();

        t_plane_tab.resize({tdomain.smallEnd(2)}, {tdomain.bigEnd(2)});

        dptr_t_plane = t_plane_tab.table();
        ParallelFor(ncell, [=] AMREX_GPU_DEVICE (int k) noexcept
        {
            dptr_t_plane(k-offset) = dptr_t[k];
        });

        if (solverChoice.moisture_type != MoistureType::None)
        {
            Gpu::HostVector<  Real> qv_plane_h(ncell), qc_plane_h(ncell);
            Gpu::DeviceVector<Real> qv_plane_d(ncell), qc_plane_d(ncell);

            // Water vapor
            PlaneAverage qv_ave(&(S_data[IntVars::cons]), geom, solverChoice.ave_plane, true);
            qv_ave.compute_averages(ZDir(), qv_ave.field());
            qv_ave.line_average(RhoQ1_comp, qv_plane_h);
            Gpu::copyAsync(Gpu::hostToDevice, qv_plane_h.begin(), qv_plane_h.end(), qv_plane_d.begin());

            // Cloud water
            PlaneAverage qc_ave(&(S_data[IntVars::cons]), geom, solverChoice.ave_plane, true);
            qc_ave.compute_averages(ZDir(), qc_ave.field());
            qc_ave.line_average(RhoQ2_comp, qc_plane_h);
            Gpu::copyAsync(Gpu::hostToDevice, qc_plane_h.begin(), qc_plane_h.end(), qc_plane_d.begin());

            Real* dptr_qv = qv_plane_d.data();
            Real* dptr_qc = qc_plane_d.data();

            qv_plane_tab.resize({tdomain.smallEnd(2)}, {tdomain.bigEnd(2)});
            qc_plane_tab.resize({tdomain.smallEnd(2)}, {tdomain.bigEnd(2)});

            dptr_qv_plane = qv_plane_tab.table();
            dptr_qc_plane = qc_plane_tab.table();
            ParallelFor(ncell, [=] AMREX_GPU_DEVICE (int k) noexcept
            {
                dptr_qv_plane(k-offset) = dptr_qv[k];
                dptr_qc_plane(k-offset) = dptr_qc[k];
            });
        }
    }

    // *****************************************************************************
    // Define source term for cell-centered conserved variables, from
    //    1. user-defined source terms for (rho theta) and (rho q_t)
    //    2. radiation           for (rho theta)
    //    3. Rayleigh damping    for (rho theta)
    //    4. custom forcing      for (rho theta) and (rho Q1)
    //    5. custom subsidence   for (rho theta) and (rho Q1)
    //    6. numerical diffusion for (rho theta)
    //    7. sponging
    //    8. turbulent perturbation
    //    9. nudging towards input sounding values (only for theta)
    // *****************************************************************************

    // ***********************************************************************************************
    // Add remaining source terms
    // ***********************************************************************************************
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
    for ( MFIter mfi(S_data[IntVars::cons],TileNoZ()); mfi.isValid(); ++mfi)
    {
        Box bx  = mfi.tilebox();

        const Array4<const Real> & cell_data  = S_data[IntVars::cons].array(mfi);
        const Array4<const Real> & cell_prim  = S_prim.array(mfi);
        const Array4<Real>       & cell_src   = source.array(mfi);

        const Array4<const Real>& z_cc_arr = (use_terrain) ? z_phys_cc->const_array(mfi) : Array4<Real>{};

#ifdef ERF_USE_RRTMGP
        // *************************************************************************************
        // 2. Add radiation source terms to (rho theta)
        // *************************************************************************************
        {
            auto const& qheating_arr = qheating_rates->const_array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // Short-wavelength and long-wavelength radiation source terms
                cell_src(i,j,k,RhoTheta_comp) += qheating_arr(i,j,k,0) + qheating_arr(i,j,k,1);
            });
        }

#endif

        // *************************************************************************************
        // 3. Add Rayleigh damping for (rho theta)
        // *************************************************************************************
        Real zlo      = geom.ProbLo(2);
        Real dz       = geom.CellSize(2);
        Real ztop     = solverChoice.rayleigh_ztop;
        Real zdamp    = solverChoice.rayleigh_zdamp;
        Real dampcoef = solverChoice.rayleigh_dampcoef;

        if (solverChoice.rayleigh_damp_T) {
            int n  = RhoTheta_comp;
            int nr = Rho_comp;
            int np = PrimTheta_comp;
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real zcc = (z_cc_arr) ? z_cc_arr(i,j,k) : zlo + (k+0.5)*dz;
                Real zfrac = 1 - (ztop - zcc) / zdamp;
                if (zfrac > 0) {
                    Real theta = cell_prim(i,j,k,np);
                    Real sinefac = std::sin(PIoTwo*zfrac);
                    cell_src(i, j, k, n) -= dampcoef*sinefac*sinefac * (theta - thetabar[k]) * cell_data(i,j,k,nr);
                }
            });
        }

        // *************************************************************************************
        // 4. Add custom forcing for (rho theta)
        // *************************************************************************************
        if (solverChoice.custom_rhotheta_forcing) {
            const int n = RhoTheta_comp;
            if (solverChoice.custom_forcing_prim_vars) {
                const int nr = Rho_comp;
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    cell_src(i, j, k, n) += cell_data(i,j,k,nr) * dptr_rhotheta_src[k];
                });
            } else {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    cell_src(i, j, k, n) += dptr_rhotheta_src[k];
                });
            }
        }

        // *************************************************************************************
        // 4. Add custom forcing for RhoQ1
        // *************************************************************************************
        if (solverChoice.custom_moisture_forcing) {
            const int n = RhoQ1_comp;
            if (solverChoice.custom_forcing_prim_vars) {
                const int nr = Rho_comp;
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    cell_src(i, j, k, n) += cell_data(i,j,k,nr) * dptr_rhoqt_src[k];
                });
            } else {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    cell_src(i, j, k, n) += dptr_rhoqt_src[k];
                });
            }
        }

        // *************************************************************************************
        // 5. Add custom subsidence for (rho theta)
        // *************************************************************************************
        if (solverChoice.custom_w_subsidence) {
            const int n = RhoTheta_comp;
            if (solverChoice.custom_forcing_prim_vars) {
                const int nr = Rho_comp;
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real dzInv = (z_cc_arr) ? 1.0/ (z_cc_arr(i,j,k+1) - z_cc_arr(i,j,k-1)) : 0.5*dxInv[2];
                    Real T_hi = dptr_t_plane(k+1) / dptr_r_plane(k+1);
                    Real T_lo = dptr_t_plane(k-1) / dptr_r_plane(k-1);
                    Real wbar_cc = 0.5 * (dptr_wbar_sub[k] + dptr_wbar_sub[k+1]);
                    cell_src(i, j, k, n) -= cell_data(i,j,k,nr) * wbar_cc * (T_hi - T_lo) * dzInv;
                });
            } else {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real dzInv = (z_cc_arr) ? 1.0/ (z_cc_arr(i,j,k+1) - z_cc_arr(i,j,k-1)) : 0.5*dxInv[2];
                    Real T_hi = dptr_t_plane(k+1) / dptr_r_plane(k+1);
                    Real T_lo = dptr_t_plane(k-1) / dptr_r_plane(k-1);
                    Real wbar_cc = 0.5 * (dptr_wbar_sub[k] + dptr_wbar_sub[k+1]);
                    cell_src(i, j, k, n) -= wbar_cc * (T_hi - T_lo) * dzInv;
                });
            }
        }

        // *************************************************************************************
        // 5. Add custom subsidence for RhoQ1 and RhoQ2
        // *************************************************************************************
        if (solverChoice.custom_w_subsidence && (solverChoice.moisture_type != MoistureType::None)) {
            const int nv = RhoQ1_comp;
            if (solverChoice.custom_forcing_prim_vars) {
                const int nr = Rho_comp;
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real dzInv = (z_cc_arr) ? 1.0/ (z_cc_arr(i,j,k+1) - z_cc_arr(i,j,k-1)) : 0.5*dxInv[2];
                    Real Qv_hi = dptr_qv_plane(k+1) / dptr_r_plane(k+1);
                    Real Qv_lo = dptr_qv_plane(k-1) / dptr_r_plane(k-1);
                    Real Qc_hi = dptr_qc_plane(k+1) / dptr_r_plane(k+1);
                    Real Qc_lo = dptr_qc_plane(k-1) / dptr_r_plane(k-1);
                    Real wbar_cc = 0.5 * (dptr_wbar_sub[k] + dptr_wbar_sub[k+1]);
                    cell_src(i, j, k, nv  ) -= cell_data(i,j,k,nr) * wbar_cc * (Qv_hi - Qv_lo) * dzInv;
                    cell_src(i, j, k, nv+1) -= cell_data(i,j,k,nr) * wbar_cc * (Qc_hi - Qc_lo) * dzInv;
                });
            } else {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real dzInv = (z_cc_arr) ? 1.0/ (z_cc_arr(i,j,k+1) - z_cc_arr(i,j,k-1)) : 0.5*dxInv[2];
                    Real Qv_hi = dptr_qv_plane(k+1) / dptr_r_plane(k+1);
                    Real Qv_lo = dptr_qv_plane(k-1) / dptr_r_plane(k-1);
                    Real Qc_hi = dptr_qc_plane(k+1) / dptr_r_plane(k+1);
                    Real Qc_lo = dptr_qc_plane(k-1) / dptr_r_plane(k-1);
                    Real wbar_cc = 0.5 * (dptr_wbar_sub[k] + dptr_wbar_sub[k+1]);
                    cell_src(i, j, k, nv  ) -= wbar_cc * (Qv_hi - Qv_lo) * dzInv;
                    cell_src(i, j, k, nv+1) -= wbar_cc * (Qc_hi - Qc_lo) * dzInv;
                });
            }
        }

        // *************************************************************************************
        // 6. Add numerical diffuion for rho and (rho theta)
        // *************************************************************************************
        if (l_use_ndiff) {
            int sc;
            int nc;

            const Array4<const Real>& mf_m   = mapfac_m->const_array(mfi);

            // Rho is a special case
            NumericalDiffusion_Scal(bx, sc=0, nc=1, dt, solverChoice.NumDiffCoeff,
                                    cell_data, cell_data, cell_src, mf_m);

            // Other scalars proceed as normal
            NumericalDiffusion_Scal(bx, sc=1, nc=1, dt, solverChoice.NumDiffCoeff,
                                    cell_prim, cell_data, cell_src, mf_m);


            if (l_use_KE && l_diff_KE) {
                NumericalDiffusion_Scal(bx, sc=RhoKE_comp, nc=1, dt, solverChoice.NumDiffCoeff,
                                        cell_prim, cell_data, cell_src, mf_m);
            }

            NumericalDiffusion_Scal(bx, sc=RhoScalar_comp, nc=NSCALARS, dt, solverChoice.NumDiffCoeff,
                                    cell_prim, cell_data, cell_src, mf_m);
        }

        // *************************************************************************************
        // 7. Add sponging
        // *************************************************************************************
        if(!(solverChoice.spongeChoice.sponge_type == "input_sponge")){
            ApplySpongeZoneBCsForCC(solverChoice.spongeChoice, geom, bx, cell_src, cell_data);
        }

        // *************************************************************************************
        // 8. Add perturbation
        // *************************************************************************************
        if (solverChoice.pert_type == PerturbationType::Source) {
            auto m_ixtype = S_data[IntVars::cons].boxArray().ixType(); // Conserved term
            const amrex::Array4<const amrex::Real>& pert_cell = turbPert.pb_cell.const_array(mfi);
            turbPert.apply_tpi(level, bx, RhoTheta_comp, m_ixtype, cell_src, pert_cell); // Applied as source term
        }

        // *************************************************************************************
        // 9. Add nudging towards value specified in input sounding
        // *************************************************************************************
        if (solverChoice.nudging_from_input_sounding)
        {
            int itime_n    = 0;
            int itime_np1  = 0;
            Real coeff_n   = Real(1.0);
            Real coeff_np1 = Real(0.0);

            Real tau_inv = Real(1.0) / input_sounding_data.tau_nudging;

            int n_sounding_times = input_sounding_data.input_sounding_time.size();

            for (int nt = 1; nt < n_sounding_times; nt++) {
                if (time > input_sounding_data.input_sounding_time[nt]) itime_n = nt;
            }
            if (itime_n == n_sounding_times-1) {
                itime_np1 = itime_n;
            } else {
                itime_np1 = itime_n+1;
                coeff_np1 = (time                                               - input_sounding_data.input_sounding_time[itime_n]) /
                            (input_sounding_data.input_sounding_time[itime_np1] - input_sounding_data.input_sounding_time[itime_n]);
                coeff_n   = Real(1.0) - coeff_np1;
            }

            const Real* theta_inp_sound_n   = input_sounding_data.theta_inp_sound_d[itime_n].dataPtr();
            const Real* theta_inp_sound_np1 = input_sounding_data.theta_inp_sound_d[itime_np1].dataPtr();

            const int n  = RhoTheta_comp;
            const int nr = Rho_comp;

            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real nudge = (coeff_n*theta_inp_sound_n[k] + coeff_np1*theta_inp_sound_np1[k]) - (dptr_t_plane(k)/dptr_r_plane(k));
                nudge *= tau_inv;
                cell_src(i, j, k, n) += cell_data(i, j, k, nr) * nudge;
            });
        }
    } // mfi
    } // OMP
}
