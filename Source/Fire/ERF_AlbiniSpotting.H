#ifndef ERF_ALBINI_SPOTTING_H
#define ERF_ALBINI_SPOTTING_H

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Gpu.H>
#include <AMReX_ccse-mpi.H>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <limits>
#include <vector>
#include <random>
#include <string>
#include <memory>

#include <ERF_FireParams.H>
#include <ERF_FireGrid.H>
#include <ERF_FuelModels.H>
#include <ERF_FireDiagnostics.H>
#include <ERF_ScottSpottingTable.H>
#include <ERF_HostFabView.H>


/**
 * @file ERF_AlbiniSpotting.H
 * @brief Albini (1983) stochastic firebrand spotting model.
 *
 * Implements the lofting height, stochastic launch, forward-Euler trajectory
 * integration, and phi-negative stamping for ember spotting.
 *
 * Key references:
 *   Albini, F.A. (1983). "Potential spotting distance from wind-driven surface fires."
 *   USDA Forest Service Research Paper INT-309.
 *
 * Ported from: wildfire_levelset/src/albini_spotting.H
 * Changes from source:
 *   - InputParameters::AlbiniSpottingParams → FireParams::SpottingParams
 *   - the launch intensity is the fire layer's own Byram intensity field
 *     (h w0 R with the cell's own fuel, ERF_FireDiagnostics.H), not a second
 *     formula on the uniform fuel; the Scott distance cap and the re-entry
 *     fraction are the landing cell's own as well (2026-10)
 *   - the brand drifts on the reference-height wind scaled with a neutral
 *     log profile to its height, as Albini's model has it, not on the
 *     WAF-reduced midflame wind (2026-10)
 *   - a landing stamps phi only; it no longer removes 95 % of the fuel of
 *     every cell of the spot disc (2026-10)
 *   - only the launch set is gathered over the ranks; the landing filters
 *     (non-burnable mask, re-entry fuel fraction) run on the device (2026-10)
 *   - weise_data/WeiseBigingParams removed
 *
 * Diagnostic MultiFab layout (4 components):
 *   comp 0: lofting height H_z [m] at source fire-front cells
 *   comp 1: firebrand count launched per source cell
 *   comp 2: maximum landing distance [m] from source cell
 *   comp 3: active flag (1.0) at cells that received a spot ignition
 */

// ─── Albini lofting height ──────────────────────────────────────────────────

/**
 * @brief Compute firebrand lofting height [m].
 *
 * H_z = 12.2 × I_B^(1/3)   [Albini 1983 INT-309, eq. 1]
 *
 * Returns 0 when I_B_kW is below I_B_min_kW.
 *
 * @param[in] I_B_kW       Byram fireline intensity [kW/m]
 * @param[in] I_B_min_kW   Minimum intensity threshold [kW/m]
 * @return Lofting height H_z [m], or 0.0 if below threshold.
 */
inline amrex::Real
albini_lofting_height(amrex::Real I_B_kW, amrex::Real I_B_min_kW = 100.0)
{
    if (I_B_kW < I_B_min_kW) { return 0.0; }
    return 12.2 * std::cbrt(I_B_kW);
}

/**
 * @brief Factor that brings the reference-height wind to the brand's height
 *        above ground with a neutral log profile: ln(z/z0) / ln(z_ref/z0),
 *        with z floored at z0 (the factor is then 0) and the brand height
 *        floored at the reference height's own factor of 1 only through the
 *        profile itself (a brand above z_ref sees a larger wind). Albini
 *        (1983) drifts the brand with the wind at its height; the midflame
 *        wind used until 2026-10 was 0.36x the 6.1 m wind for grass and
 *        carried the terrain factors, so a brand lofted to 120 m travelled a
 *        quarter of the distance the profile gives.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real albini_wind_profile_factor(amrex::Real z_agl, amrex::Real z_ref, amrex::Real z0) noexcept
{
    const amrex::Real z0s = amrex::max(z0, amrex::Real(1.0e-6));
    const amrex::Real zr  = amrex::max(z_ref, z0s * amrex::Real(1.000001));
    const amrex::Real z   = amrex::max(z_agl, z0s);
    return std::log(z / z0s) / std::log(zr / z0s);
}

// ─── Bilinear interpolation of host fields ──────────────────────────────────

/**
 * @brief Bilinear interpolation of a scalar field at continuous coordinates.
 *
 * Used for the terrain elevation under a firebrand in flight. Positions outside
 * the domain are clamped to the edge, since a brand leaving the domain is
 * discarded by the wind interpolation on the same step.
 *
 * @param[in] f          Flat host array, indexed j * nx + i
 * @param[in] nx, ny     Grid dimensions
 * @param[in] dx, dy     Cell sizes [m]
 * @param[in] prob_lo_x  Domain lower x [m]
 * @param[in] prob_lo_y  Domain lower y [m]
 * @param[in] x, y       Query position [m]
 * @return interpolated value
 */
inline amrex::Real
albini_interp_scalar(const std::vector<amrex::Real>& f,
                     int nx, int ny,
                     amrex::Real dx, amrex::Real dy,
                     amrex::Real prob_lo_x, amrex::Real prob_lo_y,
                     amrex::Real x, amrex::Real y)
{
    amrex::Real gx = (x - prob_lo_x) / dx - amrex::Real(0.5);
    amrex::Real gy = (y - prob_lo_y) / dy - amrex::Real(0.5);
    int i0 = static_cast<int>(std::floor(gx));
    int j0 = static_cast<int>(std::floor(gy));
    amrex::Real wx = gx - static_cast<amrex::Real>(i0);
    amrex::Real wy = gy - static_cast<amrex::Real>(j0);

    auto at = [&](int i, int j) -> amrex::Real {
        i = std::max(0, std::min(nx - 1, i));
        j = std::max(0, std::min(ny - 1, j));
        return f[static_cast<std::size_t>(j) * nx + i];
    };

    return (1.0 - wx) * (1.0 - wy) * at(i0,     j0)
         +        wx  * (1.0 - wy) * at(i0 + 1, j0)
         + (1.0 - wx) *        wy  * at(i0,     j0 + 1)
         +        wx  *        wy  * at(i0 + 1, j0 + 1);
}

/**
 * @brief Bilinear interpolation of 2-component wind at continuous coordinates.
 *
 * @param[in]  wind_u     u-component array (comp 0 of wind MultiFab)
 * @param[in]  wind_v     v-component array (comp 1 of wind MultiFab)
 * @param[in]  nx         Number of cells in x
 * @param[in]  ny         Number of cells in y
 * @param[in]  dx         Cell size in x [m]
 * @param[in]  dy         Cell size in y [m]
 * @param[in]  prob_lo_x  Domain lower x [m]
 * @param[in]  prob_lo_y  Domain lower y [m]
 * @param[in]  x          Query x position [m]
 * @param[in]  y          Query y position [m]
 * @param[out] u_out      Interpolated u-wind [m/s]
 * @param[out] v_out      Interpolated v-wind [m/s]
 * @return true if position is within domain, false if outside (exits trajectory).
 */
inline bool
albini_interp_wind(const std::vector<amrex::Real>& wind_u,
                   const std::vector<amrex::Real>& wind_v,
                   int nx, int ny,
                   amrex::Real dx, amrex::Real dy,
                   amrex::Real prob_lo_x, amrex::Real prob_lo_y,
                   amrex::Real x, amrex::Real y,
                   amrex::Real& u_out, amrex::Real& v_out)
{
    // Cell-centre coordinates: cell (i,j) centre at prob_lo + (i+0.5)*dx
    amrex::Real fx = (x - prob_lo_x) / dx - 0.5;
    amrex::Real fy = (y - prob_lo_y) / dy - 0.5;

    int i0 = static_cast<int>(std::floor(fx));
    int j0 = static_cast<int>(std::floor(fy));

    // Domain exit check
    if (i0 < 0 || i0 >= nx - 1 || j0 < 0 || j0 >= ny - 1) {
        u_out = 0.0;
        v_out = 0.0;
        return false;
    }

    amrex::Real ax = fx - i0;
    amrex::Real ay = fy - j0;

    auto idx = [&](int ii, int jj) { return jj * nx + ii; };

    u_out = (1.0 - ax) * (1.0 - ay) * wind_u[idx(i0,   j0  )]
          + (      ax) * (1.0 - ay) * wind_u[idx(i0+1, j0  )]
          + (1.0 - ax) * (      ay) * wind_u[idx(i0,   j0+1)]
          + (      ax) * (      ay) * wind_u[idx(i0+1, j0+1)];

    v_out = (1.0 - ax) * (1.0 - ay) * wind_v[idx(i0,   j0  )]
          + (      ax) * (1.0 - ay) * wind_v[idx(i0+1, j0  )]
          + (1.0 - ax) * (      ay) * wind_v[idx(i0,   j0+1)]
          + (      ax) * (      ay) * wind_v[idx(i0+1, j0+1)];

    return true;
}

namespace albini_detail {

/// Gather one component of a fire-grid field into a flat host array on every rank
inline void gather_field(std::vector<amrex::Real>& h, const amrex::MultiFab& mf, int comp, int nx)
{
    using namespace amrex;
    std::fill(h.begin(), h.end(), Real(0.0));
    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        const ERFHostFabView v(mf[mfi]);
        auto a = v.array();
        LoopOnCpu(bx, [&](int i, int j, int /*k*/) { h[static_cast<std::size_t>(j) * nx + i] = a(i, j, 0, comp); });
    }
    ParallelDescriptor::ReduceRealSum(h.data(), static_cast<int>(h.size()));
}

/// Launch-cell record: cell, intensity, distance cap
struct LaunchCell {
    int  i, j;
    amrex::Real I_B;      ///< Byram intensity [kW/m]
    amrex::Real cap;      ///< maximum spotting distance of the cell's fuel [m]
};

} // namespace albini_detail

// ─── Main spotting function ─────────────────────────────────────────────────

/**
 * @brief Apply Albini (1983) stochastic firebrand spotting to the fire level-set.
 *
 * Algorithm:
 *  1. Launch set on the device: burned cells (phi < 0) with a Byram intensity
 *     of at least I_B_min (with spotting.launch_from = front only those within
 *     front_band cells of an unburned cell), plus the extra launch sites of
 *     burning structures. Each rank's list is gathered to every rank and sorted
 *     by cell, so the random draws below do not depend on the decomposition.
 *  2. The lofting height H_z = 12.2 × I_B^(1/3) of each launch cell.
 *  3. Stochastic launch: draw uniform random; launch if draw < P_launch.
 *  4. Forward-Euler trajectory over n_traj_steps sub-steps, drifting on the
 *     reference-height wind scaled to the brand's height above ground with a
 *     neutral log profile of roughness z0 (albini_wind_profile_factor); the
 *     wind and terrain fields are gathered to every rank for it.
 *  5. The landing distance is capped at the Scott/Albini maximum distance of
 *     the source cell's fuel.
 *  6. Catching probability filter P_catch.
 *  7. On the device, per landing: a brand on a non-burnable cell is dropped
 *     (counted in ember_landings when given); one on a cell whose remaining
 *     fuel is below reentry_fuel_thresh of that cell's initial load is dropped.
 *  8. Stamp phi negative within spot_radius of every accepted landing.
 *  9. Record diagnostics in albini_data.
 *
 * @param[in,out] phi              Fire level-set MultiFab (modified at landing sites)
 * @param[in,out] albini_data      4-component diagnostic MultiFab (comp 0-3)
 * @param[in]     wind_ref         2-component reference-height wind on the fire grid [m/s]
 * @param[in]     intensity        Byram fireline intensity [kW/m] (fill_fire_diagnostics)
 * @param[in]     geom             Fire grid geometry
 * @param[in]     sp               Spotting parameters (FireParams::SpottingParams)
 * @param[in]     step             Current simulation step (for deterministic seeding)
 * @param[in]     fuel_load        Remaining fuel [kg/m2] (the re-entry filter); nullptr skips it
 * @param[in]     fuel_w0          Initial fuel load per cell [kg/m2] (the re-entry fraction's
 *                                 denominator); nullptr takes fuel_load_initial_kg_m2
 * @param[in]     fuel_load_initial_kg_m2  Uniform initial load [kg/m2] when fuel_w0 is null
 * @param[in]     fuel_model       Per-cell fuel code for the distance cap; nullptr takes fuel_model_id
 * @param[in]     fuel_system      Fuel system string ("13" or "40") for distance table
 * @param[in]     fuel_model_id    Uniform fuel code for the distance cap
 * @param[in]     z_ref            Height of the reference wind above ground [m]
 * @param[in]     z0               Roughness length of the log profile [m]
 * @param[in]     fire_debug       Print the launch summary
 * @param[in]     terrain_z        Ground elevation per fire cell [m]; nullptr = flat
 * @param[in]     nonburnable      Non-burnable mask; a brand landing in it is dropped
 * @param[in]     ember_landings   Brands dropped on mask cells, counted per cell (exposure)
 * @param[in]     phi_normalized   FARSITE [-1, 1] stamp convention when true, metres when false
 * @param[in]     extra_launch_intensity  Optional equivalent fireline intensity [kW/m] of
 *                                 cells that launch brands although they are not burning fuel
 *                                 (the footprint cells of a burning structure); a cell with a
 *                                 positive value is a launch site with that intensity
 */
inline void
compute_albini_spotting(amrex::MultiFab&                  phi,
                        amrex::MultiFab&                  albini_data,
                        const amrex::MultiFab&            wind_ref,
                        const amrex::MultiFab&            intensity,
                        const amrex::Geometry&            geom,
                        const FireParams::SpottingParams& sp,
                        int                               step,
                        const amrex::MultiFab*            fuel_load,
                        const amrex::MultiFab*            fuel_w0,
                        amrex::Real                       fuel_load_initial_kg_m2,
                        const amrex::MultiFab*            fuel_model,
                        const std::string&                fuel_system,
                        int                               fuel_model_id,
                        amrex::Real                       z_ref,
                        amrex::Real                       z0,
                        bool                              fire_debug = false,
                        const amrex::MultiFab*            terrain_z = nullptr,
                        const amrex::MultiFab*            nonburnable = nullptr,
                        amrex::MultiFab*                  ember_landings = nullptr,
                        bool                              phi_normalized = true,
                        const amrex::MultiFab*            extra_launch_intensity = nullptr)
{
    using namespace amrex;

    // GpuArray by value: CellSize()/ProbLo() hand back pointers into host
    // Geometry members, which the stamping kernel below cannot dereference.
    const auto dx      = geom.CellSizeArray();
    const auto prob_lo = geom.ProbLoArray();

    const Box domain = geom.Domain();
    const int nx = domain.length(0);
    const int ny = domain.length(1);
    const int ncells = nx * ny;

    // ---------------------------------------------------------------------
    // 1. Launch set, on the device: intensity (or the extra launch intensity)
    //    where a cell launches, 0 elsewhere, and the cell's fuel code.
    // ---------------------------------------------------------------------
    const bool front_only = (sp.launch_from == "front");
    const int  band       = sp.front_band;
    const Real I_min      = sp.I_B_min;
    const bool has_extra  = (extra_launch_intensity != nullptr);
    const bool has_codes  = (fuel_model != nullptr);

    // phi with enough ghost cells for the front-band test (its own three
    // suffice for the default band of 2; a wider band gets a copy)
    const MultiFab* phi_g = &phi;
    std::unique_ptr<MultiFab> phi_copy;
    if (front_only && band > phi.nGrow(0)) {
        phi_copy = std::make_unique<MultiFab>(phi.boxArray(), phi.DistributionMap(), 1, IntVect(band, band, 0));
        MultiFab::Copy(*phi_copy, phi, 0, 0, 1, 0);
        fire_fill_boundary(*phi_copy, geom);
        phi_g = phi_copy.get();
    }
    const IntVect dlo = domain.smallEnd();
    const IntVect dhi = domain.bigEnd();

    MultiFab launch(phi.boxArray(), phi.DistributionMap(), 2, 0);
    for (MFIter mfi(launch); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto L   = launch.array(mfi);
        auto p   = phi_g->const_array(mfi);
        auto ib  = intensity.const_array(mfi);
        Array4<const Real> ex, code;
        if (has_extra) { ex = extra_launch_intensity->const_array(mfi); }
        if (has_codes) { code = fuel_model->const_array(mfi); }
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real I = 0.0;
            if (p(i, j, k) < 0.0 && ib(i, j, k) >= I_min) {
                bool near_unburned = !front_only;
                if (front_only) {
                    for (int dj = -band; dj <= band && !near_unburned; ++dj) {
                        for (int di = -band; di <= band; ++di) {
                            const int ii = i + di, jj = j + dj;
                            if (ii < dlo[0] || ii > dhi[0] || jj < dlo[1] || jj > dhi[1]) { continue; }
                            if (p(ii, jj, k) >= 0.0) { near_unburned = true; break; }
                        }
                    }
                }
                if (near_unburned) { I = ib(i, j, k); }
            } else if (has_extra && ex(i, j, k) > 0.0) {
                // A burning structure: not fuel, not behind the front, but a
                // launch site with the equivalent intensity it was given.
                I = ex(i, j, k);
            }
            L(i, j, k, 0) = I;
            L(i, j, k, 1) = has_codes ? code(i, j, k) : Real(-1.0);
        });
    }
    Gpu::streamSynchronize();

    // This rank's launch cells, packed as (i, j, I_B, cap) for the gather
    std::vector<Real> local_rec;
    for (MFIter mfi(launch); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        const ERFHostFabView Lv(launch[mfi]);
        auto L = Lv.array();
        LoopOnCpu(bx, [&](int i, int j, int /*k*/) {
            const Real I = L(i, j, 0, 0);
            if (I <= 0.0) { return; }
            const int code = (L(i, j, 0, 1) >= 0.0) ? static_cast<int>(L(i, j, 0, 1) + Real(0.5)) : fuel_model_id;
            local_rec.push_back(static_cast<Real>(i));
            local_rec.push_back(static_cast<Real>(j));
            local_rec.push_back(I);
            local_rec.push_back(scott_max_spot_dist(code, fuel_system));
        });
    }

    // Gather every rank's list; sort by (j, i) so the random draws below are
    // consumed in the same order whatever the decomposition.
    const int nranks = ParallelDescriptor::NProcs();
    std::vector<int> n_all(nranks, 0), displs(nranks + 1, 0);
    int n_local = static_cast<int>(local_rec.size());
#ifdef AMREX_USE_MPI
    MPI_Allgather(&n_local, 1, MPI_INT, n_all.data(), 1, MPI_INT, ParallelDescriptor::Communicator());
#else
    n_all[0] = n_local;
#endif
    for (int r = 0; r < nranks; ++r) { displs[r + 1] = displs[r] + n_all[r]; }
    std::vector<Real> all_rec(displs[nranks]);
#ifdef AMREX_USE_MPI
    MPI_Allgatherv(local_rec.data(), n_local, ParallelDescriptor::Mpi_typemap<Real>::type(),
                   all_rec.data(), n_all.data(), displs.data(),
                   ParallelDescriptor::Mpi_typemap<Real>::type(), ParallelDescriptor::Communicator());
#else
    all_rec = local_rec;
#endif
    std::vector<albini_detail::LaunchCell> front_cells;
    front_cells.reserve(all_rec.size() / 4);
    for (std::size_t n = 0; n + 3 < all_rec.size(); n += 4) {
        front_cells.push_back({static_cast<int>(all_rec[n]), static_cast<int>(all_rec[n + 1]), all_rec[n + 2], all_rec[n + 3]});
    }
    std::sort(front_cells.begin(), front_cells.end(),
              [](const albini_detail::LaunchCell& a, const albini_detail::LaunchCell& b) {
                  return (a.j != b.j) ? (a.j < b.j) : (a.i < b.i);
              });
    const int n_front = static_cast<int>(front_cells.size());
    if (n_front == 0) { return; }

    // ---------------------------------------------------------------------
    // 2. The fields the trajectory reads at arbitrary points: the reference
    //    wind and the ground, gathered to every rank.
    // ---------------------------------------------------------------------
    std::vector<Real> h_u(ncells, 0.0), h_v(ncells, 0.0), h_terrain(ncells, 0.0);
    albini_detail::gather_field(h_u, wind_ref, 0, nx);
    albini_detail::gather_field(h_v, wind_ref, 1, nx);
    if (terrain_z) { albini_detail::gather_field(h_terrain, *terrain_z, 0, nx); }
    const Real z_ground_min = *std::min_element(h_terrain.begin(), h_terrain.end());

    // RNG: fixed seed when sp.random_seed > 0, otherwise time+step based.
    unsigned seed = (sp.random_seed > 0)
                    ? static_cast<unsigned>(sp.random_seed + step)
                    : static_cast<unsigned>(std::time(nullptr)) + static_cast<unsigned>(step);
    // Broadcast seed so all ranks use identical RNG state.
    int iseed = static_cast<int>(seed);
    ParallelDescriptor::Bcast(&iseed, 1, ParallelDescriptor::IOProcessorNumber());
    seed = static_cast<unsigned>(iseed);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<Real> uniform(0.0, 1.0);

    struct LandingSite {
        Real x, y;
        int  src_i, src_j;
        Real dist;
        Real H_z;
    };
    std::vector<LandingSite> landings;

    for (int fc = 0; fc < n_front; ++fc) {
        const int  si  = front_cells[fc].i;
        const int  sj  = front_cells[fc].j;
        const Real I_B = front_cells[fc].I_B;
        const Real H_z = albini_lofting_height(I_B, sp.I_B_min);
        if (H_z <= 0.0) { continue; }
        const Real cx  = prob_lo[0] + (si + 0.5) * dx[0];
        const Real cy  = prob_lo[1] + (sj + 0.5) * dx[1];
        const Real cz  = h_terrain[static_cast<std::size_t>(sj) * nx + si];

        // Stochastic launch: P_launch = P_base × min(1, I_B / I_B_min)
        const Real P_launch = sp.P_base * std::min(Real(1.0), I_B / sp.I_B_min);
        if (uniform(rng) >= P_launch) { continue; }

        // Forward-Euler trajectory. The brand is lofted H_z above the ground at
        // the source and descends at its terminal velocity until it meets the
        // ground beneath it, drifting on the reference wind scaled to its
        // height above that ground. The step size is set from the longest fall
        // the brand could make, down to the lowest ground in the domain, so the
        // fixed step count always covers the descent.
        Real z_brand  = cz + H_z;
        Real t_max    = (sp.terminal_velocity > 0.0)
                      ? (z_brand - z_ground_min) / sp.terminal_velocity : 0.0;
        Real dt_traj  = (sp.n_traj_steps > 0) ? t_max / sp.n_traj_steps : 0.0;
        Real dz_step  = sp.terminal_velocity * dt_traj;

        Real px = cx;
        Real py = cy;
        bool in_domain = true;

        for (int ns = 0; ns < sp.n_traj_steps && in_domain; ++ns) {
            Real u_w = 0.0, v_w = 0.0;
            in_domain = albini_interp_wind(h_u, h_v, nx, ny, dx[0], dx[1],
                                           prob_lo[0], prob_lo[1],
                                           px, py, u_w, v_w);
            if (!in_domain) { break; }
            const Real z_ground_here = albini_interp_scalar(h_terrain, nx, ny, dx[0], dx[1],
                                                            prob_lo[0], prob_lo[1], px, py);
            const Real f = albini_wind_profile_factor(z_brand - z_ground_here, z_ref, z0);
            px += f * u_w * dt_traj;
            py += f * v_w * dt_traj;
            z_brand -= dz_step;

            const Real z_ground = albini_interp_scalar(h_terrain, nx, ny, dx[0], dx[1],
                                                       prob_lo[0], prob_lo[1], px, py);
            if (z_brand <= z_ground) { break; }
        }

        if (!in_domain) { continue; }

        // Maximum distance cap of the source cell's fuel
        Real dist = std::sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy));
        const Real max_dist_cap = front_cells[fc].cap;
        if (dist > max_dist_cap && dist > 0.0) {
            px = cx + (px - cx) / dist * max_dist_cap;
            py = cy + (py - cy) / dist * max_dist_cap;
            dist = max_dist_cap;
        }

        // Catching probability filter
        if (sp.P_catch < 1.0 && uniform(rng) >= sp.P_catch) { continue; }

        landings.push_back({px, py, si, sj, dist, H_z});
    }

    const int nland = static_cast<int>(landings.size());
    if (nland == 0) {
        if (fire_debug) {
            amrex::Print() << "[FIRE SPOTTING] front_cells=" << n_front << "  launched=0\n";
        }
        return;
    }

    // ---------------------------------------------------------------------
    // 3. Landing filters on the device: the mask and the re-entry fraction of
    //    the landing cell, decided by the rank that owns it and agreed by all.
    // ---------------------------------------------------------------------
    Gpu::DeviceVector<Real> d_lx(nland), d_ly(nland), d_ldist(nland), d_lhz(nland);
    Gpu::DeviceVector<int>  d_lsi(nland), d_lsj(nland), d_li(nland), d_lj(nland), d_ok(nland);
    {
        std::vector<Real> h_lx(nland), h_ly(nland), h_ldist(nland), h_lhz(nland);
        std::vector<int>  h_lsi(nland), h_lsj(nland), h_li(nland), h_lj(nland), h_ok(nland, 0);
        for (int n = 0; n < nland; ++n) {
            const auto& ls = landings[n];
            h_lx[n] = ls.x;  h_ly[n] = ls.y;  h_ldist[n] = ls.dist;  h_lhz[n] = ls.H_z;
            h_lsi[n] = ls.src_i;  h_lsj[n] = ls.src_j;
            int li = static_cast<int>((ls.x - prob_lo[0]) / dx[0]);
            int lj = static_cast<int>((ls.y - prob_lo[1]) / dx[1]);
            h_li[n] = std::max(0, std::min(nx - 1, li));
            h_lj[n] = std::max(0, std::min(ny - 1, lj));
        }
        Gpu::copy(Gpu::hostToDevice, h_lx.begin(),    h_lx.end(),    d_lx.begin());
        Gpu::copy(Gpu::hostToDevice, h_ly.begin(),    h_ly.end(),    d_ly.begin());
        Gpu::copy(Gpu::hostToDevice, h_ldist.begin(), h_ldist.end(), d_ldist.begin());
        Gpu::copy(Gpu::hostToDevice, h_lhz.begin(),   h_lhz.end(),   d_lhz.begin());
        Gpu::copy(Gpu::hostToDevice, h_lsi.begin(),   h_lsi.end(),   d_lsi.begin());
        Gpu::copy(Gpu::hostToDevice, h_lsj.begin(),   h_lsj.end(),   d_lsj.begin());
        Gpu::copy(Gpu::hostToDevice, h_li.begin(),    h_li.end(),    d_li.begin());
        Gpu::copy(Gpu::hostToDevice, h_lj.begin(),    h_lj.end(),    d_lj.begin());
        Gpu::copy(Gpu::hostToDevice, h_ok.begin(),    h_ok.end(),    d_ok.begin());
    }
    const Real* p_lx    = d_lx.data();
    const Real* p_ly    = d_ly.data();
    const Real* p_ldist = d_ldist.data();
    const Real* p_lhz   = d_lhz.data();
    const int*  p_lsi   = d_lsi.data();
    const int*  p_lsj   = d_lsj.data();
    const int*  p_li    = d_li.data();
    const int*  p_lj    = d_lj.data();
    int*        p_ok    = d_ok.data();

    const bool has_mask  = (nonburnable != nullptr);
    const bool has_fuel  = (fuel_load != nullptr);
    const bool has_w0    = (fuel_w0 != nullptr);
    const bool count_emb = (ember_landings != nullptr);
    const Real reentry   = sp.reentry_fuel_thresh;
    const Real w0_uniform = fuel_load_initial_kg_m2;

    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        Array4<const Real> mask_arr, fuel_arr, w0_arr;
        if (has_mask) { mask_arr = nonburnable->const_array(mfi); }
        if (has_fuel) { fuel_arr = fuel_load->const_array(mfi); }
        if (has_w0)   { w0_arr   = fuel_w0->const_array(mfi); }
        Array4<Real> emb_arr;
        if (count_emb) { emb_arr = ember_landings->array(mfi); }
        // one thread per landing: the owning box decides
        ParallelFor(nland, [=] AMREX_GPU_DEVICE (int n) noexcept {
            const int li = p_li[n], lj = p_lj[n];
            if (!bx.contains(IntVect(li, lj, 0))) { return; }
            if (has_mask && mask_arr(li, lj, 0) > 0.5) {
                // A brand on a non-burnable cell (structure, non-burnable fuel
                // code, masked firebreak) does not ignite; counted for the
                // exposure diagnostics when they are on.
                if (count_emb) { Gpu::Atomic::AddNoRet(&emb_arr(li, lj, 0), Real(1.0)); }
                return;
            }
            if (has_fuel) {
                // reentry_fuel_thresh is a fraction of the load the landing
                // cell started with (its own on a fuel map): brands landing in
                // the burned area or on consumed fuel are discarded.
                const Real w0 = has_w0 ? w0_arr(li, lj, 0) : w0_uniform;
                const Real frac = (w0 > 0.0) ? fuel_arr(li, lj, 0) / amrex::max(w0, std::numeric_limits<Real>::min())
                                             : Real(1.0);
                if (frac < reentry) { return; }
            }
            p_ok[n] = 1;
        });
    }
    Gpu::streamSynchronize();
    std::vector<int> h_ok(nland, 0);
    Gpu::copy(Gpu::deviceToHost, d_ok.begin(), d_ok.end(), h_ok.begin());
    ParallelDescriptor::ReduceIntMax(h_ok.data(), nland);
    Gpu::copy(Gpu::hostToDevice, h_ok.begin(), h_ok.end(), d_ok.begin());
    int n_accepted = 0;
    Real max_dist = 0.0;
    for (int n = 0; n < nland; ++n) { if (h_ok[n]) { ++n_accepted; max_dist = std::max(max_dist, landings[n].dist); } }

    // ---------------------------------------------------------------------
    // 4. Stamp phi at the accepted landings and record the source diagnostics,
    //    in one pass over the cells against the landing list.
    // ---------------------------------------------------------------------
    if (n_accepted > 0) {
        const Real spot_radius = sp.spot_radius;
        for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto phi_arr  = phi.array(mfi);
            auto alb_arr  = albini_data.array(mfi);
            Array4<const Real> mask_arr;
            if (has_mask) { mask_arr = nonburnable->const_array(mfi); }

            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) noexcept
            {
                // Non-burnable cells never receive a spot stamp, even inside the
                // disc of a landing that fell on a burnable neighbour.
                if (has_mask && mask_arr(i, j, 0) > 0.5) { return; }
                const Real ccx = prob_lo[0] + (i + 0.5) * dx[0];
                const Real ccy = prob_lo[1] + (j + 0.5) * dx[1];

                Real phi_new  = phi_arr(i, j, 0, 0);
                bool spotted  = false;
                Real hz_here  = 0.0;
                Real n_brands = 0.0;
                Real max_d    = alb_arr(i, j, 0, 2);

                for (int n = 0; n < nland; ++n) {
                    if (p_ok[n] == 0) { continue; }
                    const Real ddx = ccx - p_lx[n];
                    const Real ddy = ccy - p_ly[n];
                    const Real d   = std::sqrt(ddx*ddx + ddy*ddy);
                    if (d <= spot_radius) {
                        // Signed distance of the spot disc: normalized by the
                        // radius in the [-1, 1] convention (as initialize_ignition
                        // does), in metres on the level-set path.
                        const Real sdf_val = phi_normalized ? (d - spot_radius) / spot_radius
                                                            : (d - spot_radius);
                        phi_new = amrex::min(phi_new, sdf_val);
                        spotted = true;
                    }
                    if (i == p_lsi[n] && j == p_lsj[n]) {
                        hz_here   = p_lhz[n];
                        n_brands += 1.0;
                        max_d     = amrex::max(max_d, p_ldist[n]);
                    }
                }

                phi_arr(i, j, 0, 0) = phi_new;
                if (spotted) { alb_arr(i, j, 0, 3) = 1.0; }   // active spot flag; the fuel is left as it is
                if (n_brands > 0.0) {
                    alb_arr(i, j, 0, 0)  = hz_here;    // lofting height
                    alb_arr(i, j, 0, 1) += n_brands;   // brand count
                    alb_arr(i, j, 0, 2)  = max_d;      // max landing dist
                }
            });
        }
    }
    // The kernels above read the device vectors, which are freed on return.
    Gpu::streamSynchronize();

    // Clamp phi to [-1, 1] after all spotting stamps. Only in the normalized
    // convention: on the level-set path phi is a signed distance in metres.
    if (phi_normalized && n_accepted > 0) {
        for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi) {
            auto p = phi.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (const amrex::IntVect& iv) noexcept {
                p(iv) = amrex::max(static_cast<amrex::Real>(-1.0), static_cast<amrex::Real>(amrex::min(static_cast<amrex::Real>(p(iv)), static_cast<amrex::Real>(1.0))));
            });
        }
    }

    // Every fire-grid exchange goes through fire_fill_boundary, which also
    // copies the nearest interior value into the ghosts outside a
    // non-periodic face; FillBoundary alone leaves those untouched.
    fire_fill_boundary(phi, geom);

    if (fire_debug) {
        amrex::Print() << "[FIRE SPOTTING] front_cells=" << n_front
                       << "  launched=" << n_accepted
                       << "  max_landing_dist=" << max_dist << " m"
                       << "  spot_radius=" << sp.spot_radius << " m"
                       << "  phi_min_after=" << phi.min(0) << "\n";
    }
}

#endif
