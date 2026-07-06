#ifndef ERF_FARSITE_ELLIPSE_H
#define ERF_FARSITE_ELLIPSE_H

/**
 * @file ERF_FarsiteEllipse.H
 * @brief FARSITE elliptical fire growth model and level-set propagation
 *
 * Richards (1990) elliptical fire growth model.
 *
 * The fire front is represented by a signed level-set field phi:
 *   phi < 0  : burned
 *   phi > 0  : unburned
 *   phi = 0  : fire front
 *
 * Propagation is two-pass:
 *   Pass 1 (GPU): compute spread vector at each front cell from ROS field
 *                 and Richards ellipse coefficients. Spread is accumulated
 *                 into disp_accum (persistent across timesteps) so that
 *                 sub-cell displacements are not lost.
 *   Pass 2 (host): collect cells whose accumulated displacement >= cell_size,
 *                  MPI_Allgatherv, stamp phi, reset those accumulators.
 */

#include <AMReX_Config.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_ccse-mpi.H>
#include <AMReX_Gpu.H>
#include <cmath>
#include <vector>

/**
 * @struct FarsiteParams
 * @brief Configuration parameters for FARSITE level-set propagation
 */
struct FarsiteParams
{
    amrex::Real phi_threshold   = 0.1;  ///< front detection level in phi
    amrex::Real coeff_a         = 0.5;  ///< Richards head coefficient
    amrex::Real coeff_b         = 0.25; ///< Richards flank coefficient
    amrex::Real coeff_c         = 0.1;  ///< Richards backing coefficient
    int         use_anderson_lw = 1;    ///< derive a,b,c from Anderson L/W ratio
    amrex::Real gaussian_sigma  = -1.0; ///< phi stamping radius: <0 = single cell, 0 = auto, >0 = user [m]
    amrex::Real cfl_fire        = 0.5;  ///< fire CFL number for subcycle timestep control
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real anderson_LW_ratio(amrex::Real wind_speed_mph) noexcept
{
    if (wind_speed_mph < 1.0) { return 1.0; }
    amrex::Real lw = 0.936 * std::exp(0.2566 * wind_speed_mph) - 0.397 * std::sqrt(wind_speed_mph);
    return amrex::max(1.0, amrex::min(lw, 8.0));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void LW_ratio_to_richards_coefficients(amrex::Real LW_ratio,
    amrex::Real& coeff_a, amrex::Real& coeff_b, amrex::Real& coeff_c) noexcept
{
    coeff_a = 1.0;
    coeff_c = 0.2 * coeff_a;
    coeff_b = (coeff_a + coeff_c) / (2.0 * LW_ratio);
}

/**
 * @brief Advance fire front phi by one fire substep dt_fire using Richards (1990)
 * elliptical expansion with sub-cell displacement accumulation.
 *
 * KEY DESIGN: spread distances per atmospheric timestep (~0.024 m) are far
 * smaller than the fire grid cell size (~25 m). Without accumulation, the
 * Huygens points are always stamped back into their origin cell and the front
 * never moves. disp_accum (comp 0 = x, comp 1 = y) stores the unresolved
 * displacement carried forward across timesteps. A cell stamps a new position
 * only once the accumulated displacement magnitude reaches the cell diagonal.
 *
 * Pass 1 (GPU): Compute the spread vector for each front cell and ADD it to
 *               disp_accum. When |disp_accum| >= stamp_threshold, write the
 *               accumulated target position to farsite_work and zero-out
 *               disp_accum for that cell.
 * Pass 2 (host/MPI): Collect nonzero farsite_work entries, all-gather, stamp phi.
 *
 * @param[in,out] phi          Level-set field
 * @param[in,out] farsite_work Scratch for ready-to-stamp positions (2 comp)
 * @param[in,out] disp_accum   Persistent sub-cell displacement accumulator (2 comp)
 * @param[in]     vel_eff      Effective wind field [m/s], 2 comp
 * @param[in]     R_mf         ROS field [m/s]
 * @param[in]     arrival_time Fire arrival time: >= 0 burned, < 0 unburned
 * @param[in]     geom_fire    Fire grid geometry
 * @param[in]     dt_fire      Fire substep [s]
 * @param[in]     fp           FARSITE parameters
 */
inline
void advance_farsite_one_step(
    amrex::MultiFab& phi,
    amrex::MultiFab& farsite_work,
    amrex::MultiFab& disp_accum,
    const amrex::MultiFab& vel_eff,
    const amrex::MultiFab& R_mf,
    const amrex::MultiFab& arrival_time,
    const amrex::Geometry& geom_fire,
    amrex::Real dt_fire,
    const FarsiteParams& fp)
{
    using namespace amrex;

    const auto* prob_lo = geom_fire.ProbLo();
    const auto* dx      = geom_fire.CellSize();
    const Real dxi = 1.0 / dx[0];
    const Real dyi = 1.0 / dx[1];

    // Stamp threshold: fire front stamps a new cell once accumulated
    // displacement reaches one cell width in either direction.
    // Using min(dx,dy) ensures we stamp before skipping a cell.
    const Real stamp_threshold = amrex::min(dx[0], dx[1]);

    // Clear farsite_work — only cells that cross the threshold this step
    // will write nonzero values into it.
    farsite_work.setVal(0.0_rt);

    // -------------------------------------------------------------------------
    // Pass 1: GPU — accumulate spread; write to farsite_work when threshold met
    // -------------------------------------------------------------------------
    for (MFIter mfi(R_mf); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto phi_arr    = phi.array(mfi);
        auto spread_arr = farsite_work.array(mfi);
        auto disp_arr   = disp_accum.array(mfi);
        auto vel_arr    = vel_eff.const_array(mfi);
        auto ros_arr    = R_mf.const_array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept
        {
            int i = iv[0], j = iv[1], k = iv[2];
            Real phi_val = phi_arr(iv);

            // Only process front cells
            if (phi_val > fp.phi_threshold) {
                return;
            }

            // Compute gradient of phi
            Real dphi_dx = (phi_arr(i+1, j, k) - phi_arr(i-1, j, k)) * dxi * 0.5;
            Real dphi_dy = (phi_arr(i, j+1, k) - phi_arr(i, j-1, k)) * dyi * 0.5;
            Real grad_mag = std::sqrt(dphi_dx*dphi_dx + dphi_dy*dphi_dy);

            if (grad_mag < 1.0e-50) { return; }  // burned interior

            // Outward normal
            Real nx = dphi_dx / grad_mag;
            Real ny = dphi_dy / grad_mag;

            // Wind
            Real ux = vel_arr(iv, 0);
            Real uy = vel_arr(iv, 1);
            Real wind_mag = std::sqrt(ux*ux + uy*uy);

            // Richards ellipse scale factor
            Real scale = 1.0;
            if (wind_mag > 1.0e-10) {
                Real wx = ux / wind_mag;
                Real wy = uy / wind_mag;
                Real cos_theta = amrex::max(-1.0, amrex::min(nx*wx + ny*wy, 1.0));
                Real sin_theta = std::sqrt(1.0 - cos_theta*cos_theta);
                Real a = fp.coeff_a, b = fp.coeff_b, c = fp.coeff_c;
                if (fp.use_anderson_lw != 0) {
                    LW_ratio_to_richards_coefficients(
                        anderson_LW_ratio(wind_mag * 2.237), a, b, c);
                }
                scale = (cos_theta > 0.0)
                    ? a * cos_theta + b * sin_theta
                    : b * sin_theta - c * cos_theta;
            }

            // Incremental spread [m] this substep
            Real ds = amrex::max(scale, 0.0) * ros_arr(iv) * dt_fire;

            // Accumulate displacement
            disp_arr(iv, 0) += ds * nx;
            disp_arr(iv, 1) += ds * ny;

            // Check accumulated magnitude against threshold
            Real accum_mag = std::sqrt(disp_arr(iv,0)*disp_arr(iv,0)
                                     + disp_arr(iv,1)*disp_arr(iv,1));
            if (accum_mag >= stamp_threshold) {
                // Write accumulated target position to farsite_work
                // (physical coords of the source cell + accumulated vector)
                Real x_src = prob_lo[0] + (i + 0.5) * dx[0];
                Real y_src = prob_lo[1] + (j + 0.5) * dx[1];
                spread_arr(iv, 0) = x_src + disp_arr(iv, 0);
                spread_arr(iv, 1) = y_src + disp_arr(iv, 1);

                // Reset accumulator for this cell
                disp_arr(iv, 0) = 0.0;
                disp_arr(iv, 1) = 0.0;
            }
        });
    }

    // Synchronize GPU after Pass 1
    amrex::Gpu::streamSynchronize();

    // -------------------------------------------------------------------------
    // Pass 2: Host — collect stamping positions, MPI gather
    // -------------------------------------------------------------------------
    std::vector<Real> xarray, yarray;

    for (MFIter mfi(farsite_work); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto spread_arr = farsite_work.const_array(mfi);

        for (int k = bx.smallEnd()[2]; k <= bx.bigEnd()[2]; ++k) {
            for (int j = bx.smallEnd()[1]; j <= bx.bigEnd()[1]; ++j) {
                for (int i = bx.smallEnd()[0]; i <= bx.bigEnd()[0]; ++i) {
                    Real sx = spread_arr(i, j, k, 0);
                    Real sy = spread_arr(i, j, k, 1);
                    // farsite_work stores absolute target coords when threshold met
                    if (sx != 0.0 || sy != 0.0) {
                        xarray.push_back(sx);
                        yarray.push_back(sy);
                    }
                }
            }
        }
    }

    // MPI all-gather
    int npts_local = static_cast<int>(xarray.size());
    std::vector<int> npts_all(ParallelDescriptor::NProcs());
    ParallelAllGather::AllGather(&npts_local, 1, npts_all.data(),
                                 ParallelDescriptor::Communicator());

    std::vector<int> displs(ParallelDescriptor::NProcs() + 1, 0);
    for (int r = 0; r < ParallelDescriptor::NProcs(); ++r) {
        displs[r+1] = displs[r] + npts_all[r];
    }
    int npts_global = displs[ParallelDescriptor::NProcs()];

    std::vector<Real> xarray_all(npts_global), yarray_all(npts_global);

#ifdef AMREX_USE_MPI
    MPI_Allgatherv(xarray.data(), npts_local,
                   ParallelDescriptor::Mpi_typemap<Real>::type(),
                   xarray_all.data(), npts_all.data(), displs.data(),
                   ParallelDescriptor::Mpi_typemap<Real>::type(),
                   ParallelDescriptor::Communicator());
    MPI_Allgatherv(yarray.data(), npts_local,
                   ParallelDescriptor::Mpi_typemap<Real>::type(),
                   yarray_all.data(), npts_all.data(), displs.data(),
                   ParallelDescriptor::Mpi_typemap<Real>::type(),
                   ParallelDescriptor::Communicator());
#else
    xarray_all = xarray;
    yarray_all = yarray;
#endif

    // -------------------------------------------------------------------------
    // Step A: reset phi to 0
    // Step B: re-stamp burned interior from arrival_time
    // Step C: stamp new front positions (only if any perimeter cells crossed threshold)
    // -------------------------------------------------------------------------
    phi.setVal(0.0_rt);

    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto p  = phi.array(mfi);
        auto at = arrival_time.const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
            if (at(iv) >= 0.0_rt) p(iv) = -1.0_rt;
        });
    }
    amrex::Gpu::streamSynchronize();

    if (npts_global > 0) {
        if (fp.gaussian_sigma < 0.0) {
            // Single-cell stamping
            amrex::Gpu::DeviceVector<Real> d_x(npts_global), d_y(npts_global);
            amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                             xarray_all.begin(), xarray_all.end(), d_x.begin());
            amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                             yarray_all.begin(), yarray_all.end(), d_y.begin());
            const Real* xp = d_x.data();
            const Real* yp = d_y.data();
            int npts = npts_global;

            for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.tilebox();
                auto p = phi.array(mfi);
                amrex::ParallelFor(npts, [=] AMREX_GPU_DEVICE (int n) noexcept {
                    int ii = static_cast<int>(std::round((xp[n] - prob_lo[0]) / dx[0] - 0.5_rt));
                    int jj = static_cast<int>(std::round((yp[n] - prob_lo[1]) / dx[1] - 0.5_rt));
                    if (bx.contains(amrex::IntVect(ii, jj, 0))) {
                        p(ii, jj, 0) = amrex::min(p(ii, jj, 0), -1.0_rt);
                    }
                });
            }
        } else {
            // Gaussian/SDF stamping
            Real sigma = (fp.gaussian_sigma > 0.0) ? fp.gaussian_sigma : dx[0];
            const Real* x_ptr = xarray_all.data();
            const Real* y_ptr = yarray_all.data();
            int npts = npts_global;

            for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
                auto p = phi.array(mfi);
                const Box& bx = mfi.validbox();
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
                    Real x_cell = prob_lo[0] + (iv[0] + 0.5) * dx[0];
                    Real y_cell = prob_lo[1] + (iv[1] + 0.5) * dx[1];
                    for (int n = 0; n < npts; ++n) {
                        Real dist = std::sqrt((x_cell-x_ptr[n])*(x_cell-x_ptr[n])
                                            + (y_cell-y_ptr[n])*(y_cell-y_ptr[n]));
                        p(iv) = amrex::min(p(iv), amrex::max(dist - sigma, -1.0_rt));
                    }
                });
            }
        }
    }
}

/**
 * @brief Compute fire CFL-limited substep.
 */
inline
amrex::Real compute_fire_dt(
    const amrex::MultiFab& R_mf,
    const amrex::Geometry& geom_fire,
    amrex::Real dt_atm,
    amrex::Real cfl_fire)
{
    using namespace amrex;
    Real max_ros = R_mf.max(0);
    if (max_ros < 1.0e-10) { return dt_atm; }
    const auto* dx = geom_fire.CellSize();
    return cfl_fire * amrex::min(dx[0], dx[1]) / max_ros;
}

/**
 * @brief Advance the fire front through multiple substeps to cover one atmospheric timestep.
 *
 * @param[in,out] phi          Level-set field
 * @param[in,out] farsite_work Scratch for ready-to-stamp positions (2 comp)
 * @param[in,out] disp_accum   Persistent sub-cell displacement accumulator (2 comp)
 * @param[in]     vel_eff      Effective wind field [m/s], 2 comp
 * @param[in]     R_mf         ROS field [m/s]
 * @param[in]     arrival_time Fire arrival time (>= 0 burned, < 0 unburned)
 * @param[in]     geom_fire    Fire grid geometry
 * @param[in]     dt_atm       Atmospheric timestep [s]
 * @param[in]     fp           FARSITE parameters
 * @return Number of fire substeps taken
 */
inline
int advance_fire_subcycle(
    amrex::MultiFab& phi,
    amrex::MultiFab& farsite_work,
    amrex::MultiFab& disp_accum,
    const amrex::MultiFab& vel_eff,
    const amrex::MultiFab& R_mf,
    const amrex::MultiFab& arrival_time,
    const amrex::Geometry& geom_fire,
    amrex::Real dt_atm,
    const FarsiteParams& fp)
{
    using namespace amrex;

    Real time_remaining = dt_atm;
    int n_substeps = 0;

    while (time_remaining > 1.0e-14) {
        Real dt_fire = amrex::min(
            compute_fire_dt(R_mf, geom_fire, time_remaining, fp.cfl_fire),
            time_remaining);

        advance_farsite_one_step(phi, farsite_work, disp_accum,
                                 vel_eff, R_mf, arrival_time,
                                 geom_fire, dt_fire, fp);

        time_remaining -= dt_fire;
        n_substeps++;

        if (n_substeps > 1000) {
            amrex::Abort("advance_fire_subcycle: too many substeps");
        }
    }

    return n_substeps;
}


#endif // ERF_FARSITE_ELLIPSE_H