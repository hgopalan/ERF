// Richards (1990) elliptical fire growth model.
// Richards, G.D. (1990). An elliptical growth model of forest fire fronts
//   and its numerical solution. Int. J. Numer. Meth. Eng. 30(6):1163-1179.
//
// Ported from hgopalan/wildfire_levelset src/farsite_ellipse.H.
// The fire front is represented by a signed level-set field phi:
//   phi < 0  : burned
//   phi > 0  : unburned
//   phi = 0  : fire front
//
// Propagation is two-pass:
//   Pass 1 (GPU): compute spread vector at each front cell from ROS field
//                 and Richards ellipse coefficients.
//   Pass 2 (host): collect propagated positions, MPI_Allgatherv so every
//                  rank holds the global front, then stamp phi.
//
// The MPI_Allgatherv call transfers O(perimeter / dx_fire) Real values.
// For a 2 km perimeter on a 10 m fire grid this is ~200 points per rank,
// and the all-gather transfers ~200 * nranks * 8 bytes — negligible overhead.
//
// Assumptions:
//   - Single AMR level (no AMR refinement in z between fire and atm grids).
//   - The fire grid uses the same x-y MPI decomposition as the atmospheric
//     level-0 grid; each rank owns the same x-y columns in both.
//   - fire_phi has 1 ghost cell in x and y for gradient stencils.
//   - wind_eff has no ghost cells required (read at owned cells only).

#ifndef ERF_FARSITE_ELLIPSE_H
#define ERF_FARSITE_ELLIPSE_H

#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.h>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <cmath>

// Struct to hold FARSITE propagation parameters
struct FarsiteParams {
    amrex::Real phi_threshold   = 0.0_rt;  // front detection level in phi
    amrex::Real coeff_a         = 0.5_rt;  // Richards head coefficient
    amrex::Real coeff_b         = 0.25_rt; // Richards flank coefficient
    amrex::Real coeff_c         = 0.1_rt;  // Richards backing coefficient
    int         use_anderson_lw = 1;       // derive a,b,c from Anderson (1983) L/W ratio
    amrex::Real gaussian_sigma  = -1.0_rt; // phi stamping radius: <0 = single cell, 0 = auto, >0 = user [m]
    amrex::Real cfl_fire        = 0.5_rt;  // fire CFL number for subcycle timestep control
};

// Anderson (1983) L/W ratio as a function of midflame wind speed [mph].
// Anderson, H.E. (1983). Predicting wind-driven wildland fire size and shape.
//   USDA Forest Service Research Paper INT-305.
// Returns L/W in [1.0, 8.0].
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real anderson_LW_ratio(amrex::Real wind_speed_mph) noexcept
{
    using namespace amrex;

    if (wind_speed_mph < Real(0.01)) {
        return Real(1.0);  // Nearly circular for no wind
    }

    Real LW = Real(0.936) * std::exp(Real(0.2566) * wind_speed_mph)
            + Real(0.461) * std::exp(Real(-0.1548) * wind_speed_mph)
            - Real(0.397);

    // Ensure physically reasonable bounds
    return amrex::max(Real(1.0), amrex::min(LW, Real(8.0)));
}

// Convert Anderson L/W ratio to Richards (1990) ellipse coefficients.
// Sets coeff_a=1.0, coeff_c=0.2*coeff_a, coeff_b=(a+c)/(2*LW).
// From FARSITE parameterisation.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void LW_ratio_to_richards_coefficients(amrex::Real LW_ratio,
    amrex::Real& coeff_a, amrex::Real& coeff_b, amrex::Real& coeff_c) noexcept
{
    using namespace amrex;

    coeff_a = Real(1.0);              // Head fire at full spread rate
    coeff_c = Real(0.2) * coeff_a;    // Backing fire at 20% of head fire
    coeff_b = (coeff_a + coeff_c) / (Real(2.0) * LW_ratio);

    // Ensure physically reasonable bounds
    coeff_b = amrex::max(coeff_b, Real(0.1));
    coeff_b = amrex::min(coeff_b, coeff_a);
}

// Advance fire front phi by one fire substep dt_fire using Richards (1990)
// elliptical expansion.
//
// Inputs:
//   phi          — level-set field on fire grid (C*nx x C*ny x 1), 1 ghost cell
//   vel_eff      — effective midflame wind on fire grid, 2 comp, no ghosts
//   R_mf         — ROS field from Phase 2 Rothermel [m/s], no ghosts
//   farsite_work — scratch MultiFab, same BA/DM as phi, 2 comp, no ghosts
//   geom_fire    — fire grid geometry
//   dt_fire      — fire substep [s]
//   fp           — FARSITE parameters (coeff_a/b/c, phi_threshold, etc.)
//
// The function is NOT GPU-only: Pass 1 is GPU (ParallelFor),
// Pass 2 is host (loop + MPI_Allgatherv + GPU stamp).
//
// phi is modified in-place: reset to 0, then fire-front positions stamped to -1.
inline void advance_farsite_one_step(
    amrex::MultiFab& phi,
    amrex::MultiFab& farsite_work,
    const amrex::MultiFab& vel_eff,
    const amrex::MultiFab& R_mf,
    const amrex::Geometry& geom_fire,
    amrex::Real dt_fire,
    const FarsiteParams& fp)
{
    using namespace amrex;

    const auto* prob_lo = geom_fire.ProbLo();
    const auto* dx = geom_fire.CellSize();

    const Real phi_thresh = fp.phi_threshold;
    const int use_anderson_LW = fp.use_anderson_lw;
    Real coeff_a = fp.coeff_a;
    Real coeff_b = fp.coeff_b;
    Real coeff_c = fp.coeff_c;

    // Dynamic arrays to collect positions where phi < threshold
    std::vector<Real> xarray;
    std::vector<Real> yarray;

    // -----------------------------------------------------------------------
    // Pass 1: GPU spread vector computation (ParallelFor over fire grid)
    // -----------------------------------------------------------------------
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto const p = phi.const_array(mfi);
        auto const v = vel_eff.const_array(mfi);
        auto spread = farsite_work.array(mfi);
        auto const R_arr = R_mf.const_array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            // Check if this cell is near the fire front (phi < threshold)
            Real phi_val = p(i, j, k);

            // Initialize spread to zero
            spread(i, j, k, 0) = 0.0_rt;
            spread(i, j, k, 1) = 0.0_rt;

            // Only compute spread for cells at the fire front
            if (phi_val > phi_thresh) {
                return;
            }

            // Compute centred-difference gradient of phi to get outward normal
            Real dphi_dx = (p(i + 1, j, k) - p(i - 1, j, k)) / (2.0_rt * dx[0]);
            Real dphi_dy = (p(i, j + 1, k) - p(i, j - 1, k)) / (2.0_rt * dx[1]);
            Real grad_mag = std::sqrt(dphi_dx * dphi_dx + dphi_dy * dphi_dy);

            // Avoid division by zero
            if (grad_mag < 1.0e-50_rt) {
                return;
            }

            // Normal vector (points toward unburned region, phi > 0)
            Real nx = dphi_dx / grad_mag;
            Real ny = dphi_dy / grad_mag;

            // Get wind velocity components
            Real ux = v(i, j, k, 0);
            Real uy = v(i, j, k, 1);

            // Wind speed magnitude
            Real wind_mag = std::sqrt(ux * ux + uy * uy);

            // Read ROS from pre-computed field
            Real R_head = R_arr(i, j, k);

            // Compute spread distance
            Real spread_dist = 0.0_rt;
            if (wind_mag > 1.0e-10_rt) {
                // Wind unit vector
                Real wx = ux / wind_mag;
                Real wy = uy / wind_mag;

                // cos(theta) = dot(normal, wind)
                Real cos_theta = nx * wx + ny * wy;
                cos_theta = amrex::max(-1.0_rt, amrex::min(1.0_rt, cos_theta));
                Real sin_theta = std::sqrt(1.0_rt - cos_theta * cos_theta);

                // Richards' elliptical scale factor
                Real scale_factor = 0.0_rt;
                if (use_anderson_LW == 1) {
                    // Convert wind_mag (m/s) to ft/min
                    // Assuming wind_mag is in m/s, convert to ft/min: multiply by 196.85
                    Real U_ftmin = wind_mag * 196.85_rt;
                    Real U_mph = U_ftmin * (60.0_rt / 5280.0_rt);  // ft/min to mph
                    Real LW = anderson_LW_ratio(U_mph);
                    Real local_a = 1.0_rt;
                    Real local_b = 0.0_rt;
                    Real local_c = 0.0_rt;
                    LW_ratio_to_richards_coefficients(LW, local_a, local_b, local_c);
                    if (cos_theta > 0.0_rt) {
                        scale_factor = local_a * cos_theta + local_b * sin_theta;
                    } else {
                        scale_factor = local_b * sin_theta - local_c * cos_theta;
                    }
                } else {
                    // Use fixed coefficients
                    if (cos_theta > 0.0_rt) {
                        scale_factor = coeff_a * cos_theta + coeff_b * sin_theta;
                    } else {
                        scale_factor = coeff_b * sin_theta - coeff_c * cos_theta;
                    }
                }

                spread_dist = amrex::max(scale_factor, 0.0_rt) * R_head * dt_fire;
            } else {
                spread_dist = R_head * dt_fire;
            }

            spread(i, j, k, 0) = spread_dist * nx;
            spread(i, j, k, 1) = spread_dist * ny;
        });
    }

    // GPU synchronize so results are visible on host
    amrex::Gpu::streamSynchronize();

    // -----------------------------------------------------------------------
    // Pass 2: Host collection, MPI gather, phi stamp
    // -----------------------------------------------------------------------
    for (MFIter mfi(farsite_work); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto const spread = farsite_work.const_array(mfi);

        const auto lo = amrex::lbound(bx);
        const auto hi = amrex::ubound(bx);

        for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    if (std::abs(spread(i, j, k, 0)) > 0.0_rt ||
                        std::abs(spread(i, j, k, 1)) > 0.0_rt) {
                        Real x = prob_lo[0] + (i + 0.5_rt) * dx[0];
                        Real y = prob_lo[1] + (j + 0.5_rt) * dx[1];
                        Real x_new = x + spread(i, j, k, 0);
                        Real y_new = y + spread(i, j, k, 1);
                        xarray.push_back(x_new);
                        yarray.push_back(y_new);
                    }
                }
            }
        }
    }

#ifdef AMREX_USE_MPI
    // Gather spread positions from all MPI ranks
    {
        int local_n = static_cast<int>(xarray.size());
        int nprocs = amrex::ParallelDescriptor::NProcs();
        std::vector<int> counts(nprocs), displs(nprocs);
        MPI_Allgather(&local_n, 1, MPI_INT,
                      counts.data(), 1, MPI_INT,
                      amrex::ParallelDescriptor::Communicator());
        int total = 0;
        for (int r = 0; r < nprocs; ++r) { displs[r] = total; total += counts[r]; }

        std::vector<Real> all_x(total), all_y(total);
        MPI_Allgatherv(xarray.data(), local_n,
                       amrex::ParallelDescriptor::Mpi_typemap<Real>::type(),
                       all_x.data(), counts.data(), displs.data(),
                       amrex::ParallelDescriptor::Mpi_typemap<Real>::type(),
                       amrex::ParallelDescriptor::Communicator());
        MPI_Allgatherv(yarray.data(), local_n,
                       amrex::ParallelDescriptor::Mpi_typemap<Real>::type(),
                       all_y.data(), counts.data(), displs.data(),
                       amrex::ParallelDescriptor::Mpi_typemap<Real>::type(),
                       amrex::ParallelDescriptor::Communicator());
        xarray = std::move(all_x);
        yarray = std::move(all_y);
    }
#endif

    // Reset phi everywhere, then stamp new fire-front positions
    phi.setVal(0.0_rt);
    const Real phi_new = -1.0_rt;  // burned region marker
    const int npts = static_cast<int>(xarray.size());

    if (npts > 0) {
        // Copy positions to device vectors
        amrex::Gpu::DeviceVector<Real> d_xarray(npts);
        amrex::Gpu::DeviceVector<Real> d_yarray(npts);

        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, xarray.begin(), xarray.end(),
                              d_xarray.begin());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, yarray.begin(), yarray.end(),
                              d_yarray.begin());
        amrex::Gpu::streamSynchronize();

        const Real* px = d_xarray.dataPtr();
        const Real* py = d_yarray.dataPtr();

        // Determine Gaussian stamping mode
        const Real raw_gsigma = fp.gaussian_sigma;
        const bool use_gaussian = (raw_gsigma >= 0.0_rt);
        Real eff_sigma = raw_gsigma;
        if (use_gaussian && eff_sigma == 0.0_rt) {
            eff_sigma = 3.0_rt * amrex::min(dx[0], dx[1]);
        }

        for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto p = phi.array(mfi);

            if (!use_gaussian) {
                // Single-cell stamping
                ParallelFor(npts, [=] AMREX_GPU_DEVICE(int idx) noexcept {
                    Real x_pos = px[idx];
                    Real y_pos = py[idx];
                    int ii = static_cast<int>(std::round((x_pos - prob_lo[0]) / dx[0] - 0.5_rt));
                    int jj = static_cast<int>(std::round((y_pos - prob_lo[1]) / dx[1] - 0.5_rt));
                    int kk = 0;
                    if (bx.contains(amrex::IntVect(AMREX_D_DECL(ii, jj, kk)))) {
                        p(ii, jj, kk) = phi_new;
                    }
                });
            } else {
                // SDF Gaussian stamping (future enhancement for smooth level-set)
                // For now, fall back to single-cell
                ParallelFor(npts, [=] AMREX_GPU_DEVICE(int idx) noexcept {
                    Real x_pos = px[idx];
                    Real y_pos = py[idx];
                    int ii = static_cast<int>(std::round((x_pos - prob_lo[0]) / dx[0] - 0.5_rt));
                    int jj = static_cast<int>(std::round((y_pos - prob_lo[1]) / dx[1] - 0.5_rt));
                    int kk = 0;
                    if (bx.contains(amrex::IntVect(AMREX_D_DECL(ii, jj, kk)))) {
                        p(ii, jj, kk) = phi_new;
                    }
                });
            }
        }
    }
}

// Compute the maximum fire substep satisfying the CFL condition:
//   dt_fire = cfl * min(dx, dy) / max(R_mf)
// Returns dt_atm if R_mf is zero everywhere (no active fire).
inline amrex::Real compute_fire_dt(
    const amrex::MultiFab& R_mf,
    const amrex::Geometry& geom_fire,
    amrex::Real dt_atm,
    amrex::Real cfl_fire)
{
    using namespace amrex;

    const auto* dx = geom_fire.CellSize();
    Real dx_min = amrex::min(dx[0], dx[1]);

    Real R_max = R_mf.max(0);
    if (R_max < 1.0e-10_rt) {
        return dt_atm;  // No active fire
    }

    Real dt_fire = cfl_fire * dx_min / R_max;
    return amrex::min(dt_fire, dt_atm);
}

// Advance the fire front through multiple substeps to cover one atmospheric
// timestep dt_atm. Returns the number of fire substeps taken.
//
// Each substep:
//   1. Calls advance_farsite_one_step().
//   2. Recomputes dt_fire from updated R_mf (ROS is held constant within
//      the atmospheric step; this recompute only adjusts for CFL).
inline int advance_fire_subcycle(
    amrex::MultiFab& phi,
    amrex::MultiFab& farsite_work,
    const amrex::MultiFab& vel_eff,
    const amrex::MultiFab& R_mf,
    const amrex::Geometry& geom_fire,
    amrex::Real dt_atm,
    const FarsiteParams& fp)
{
    using namespace amrex;

    int n_substeps = 0;
    Real t_elapsed = 0.0_rt;

    while (t_elapsed < dt_atm) {
        Real dt_fire = compute_fire_dt(R_mf, geom_fire, dt_atm - t_elapsed, fp.cfl_fire);

        if (dt_fire < 1.0e-10_rt) {
            break;  // No progress possible
        }

        // Ensure we don't overshoot dt_atm
        dt_fire = amrex::min(dt_fire, dt_atm - t_elapsed);

        advance_farsite_one_step(phi, farsite_work, vel_eff, R_mf, geom_fire, dt_fire, fp);

        t_elapsed += dt_fire;
        ++n_substeps;

        // Safety check to prevent infinite loops
        if (n_substeps > 10000) {
            amrex::Print() << "[FIRE] WARNING: subcycle limit reached at t_elapsed=" << t_elapsed << "\n";
            break;
        }
    }

    return n_substeps;
}

#endif // ERF_FARSITE_ELLIPSE_H
