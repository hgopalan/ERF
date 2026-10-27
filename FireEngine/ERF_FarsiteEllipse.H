#ifndef ERF_FARSITE_ELLIPSE_H
#define ERF_FARSITE_ELLIPSE_H

#include <AMReX_Config.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_ccse-mpi.H>
#include <AMReX_Gpu.H>
#include <ERF_FireGrid.H>
#include <cmath>
#include <limits>
#include <vector>

namespace farsite_front {
    /// erf.fire.farsite.front_update
    constexpr int legacy     = 0;
    constexpr int front_cell = 1;
}

namespace farsite_shape {
    /// erf.fire.farsite.shape
    constexpr int rectangle = 0;   ///< support function of [-c, a] x [-b, b] (this code before 2026-10)
    constexpr int ellipse   = 1;   ///< Richards (1990) / Alexander (1985) Huygens ellipse (default)
}

struct FarsiteParams
{
    amrex::Real phi_threshold   = 0.1;
    amrex::Real coeff_a         = 0.5;
    amrex::Real coeff_b         = 0.25;
    amrex::Real coeff_c         = 0.1;
    int         use_anderson_lw = 1;
    amrex::Real gaussian_sigma  = -1.0;
    amrex::Real cfl_fire        = 0.5;
    int         front_update    = farsite_front::front_cell;
    int         shape           = farsite_shape::ellipse;
};

/// Shape of the spread ellipse from the head rate and the midflame wind
/// (shared by the FARSITE path and erf.fire.levelset.ellipse).
struct SpreadEllipse
{
    amrex::Real a = 0.0;   ///< semi-minor rate (flank) [m/s]
    amrex::Real b = 0.0;   ///< semi-major rate [m/s]
    amrex::Real c = 0.0;   ///< centre offset rate [m/s]
    amrex::Real LB = 1.0;  ///< length-to-width ratio
    amrex::Real HB = 1.0;  ///< head-to-back ratio
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real anderson_LW_ratio(amrex::Real wind_speed_mph) noexcept
{
    // Anderson (1983) as fitted in FARSITE (Finney 1998, eq. 8), U in mi/h:
    // L/W = 0.936 e^{0.2566 U} + 0.461 e^{-0.1548 U} - 0.397, which is 1 at U = 0.
    // The second term used to be -0.397 sqrt(U), 22 % narrower at 5 mi/h.
    if (wind_speed_mph <= 0.0) { return 1.0; }
    amrex::Real lw = 0.936 * std::exp(0.2566 * wind_speed_mph)
                   + 0.461 * std::exp(-0.1548 * wind_speed_mph)
                   - 0.397;
    return amrex::max(static_cast<amrex::Real>(1.0), static_cast<amrex::Real>(amrex::min(static_cast<amrex::Real>(lw), static_cast<amrex::Real>(8.0))));
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
 * @brief Ellipse rates for a head rate R and a length-to-width ratio
 *
 * Alexander's (1985) head-to-back ratio HB = (LB + sqrt(LB^2 - 1)) / (LB -
 * sqrt(LB^2 - 1)) and the Richards (1990) / Finney (1998) axes b = (R + R/HB) / 2
 * along the wind, a = b / LB across it and the centre offset c = b - R/HB, so the
 * head runs at b + c = R, the back at b - c = R/HB and the flanks at a. LB = 1
 * is the disc of radius R.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
SpreadEllipse spread_ellipse_from_lw(amrex::Real R, amrex::Real LB) noexcept
{
    SpreadEllipse e;
    LB = amrex::max(amrex::Real(1.0), LB);
    e.LB = LB;
    const amrex::Real s = std::sqrt(amrex::max(static_cast<amrex::Real>(LB * LB - 1.0),  amrex::Real(0.0)));
    // (LB + s)/(LB - s) = (LB + s)^2 (since (LB - s)(LB + s) = 1): no cancelling
    // difference in the divisor, so a large LB cannot give 0/0 in single precision
    e.HB = (LB - s > 1.0e-12) ? (LB + s) * (LB + s) : amrex::Real(1.0e12);
    const amrex::Real Rb = R / e.HB;
    e.b = 0.5 * (R + Rb);
    e.a = e.b / LB;
    e.c = e.b - Rb;
    return e;
}

/**
 * @brief Ellipse rates for a head rate R and a midflame wind speed.
 * @param R        Head rate of spread [m/s]
 * @param U_mps    Midflame wind speed [m/s]
 * @param lw_fixed Length-to-width ratio to use when > 0, else Anderson (1983) from the wind
 * @param lw_max   Cap on the length-to-width ratio (Anderson caps at 8)
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
SpreadEllipse spread_ellipse(amrex::Real R, amrex::Real U_mps,
                             amrex::Real lw_fixed, amrex::Real lw_max) noexcept
{
    amrex::Real LB = (lw_fixed > 0.0) ? lw_fixed : anderson_LW_ratio(U_mps * 2.23694);
    LB = amrex::max(amrex::Real(1.0), amrex::min(LB, lw_max));
    return spread_ellipse_from_lw(R, LB);
}

/// Normal speed of the ellipse for an outward normal at angle theta to the wind.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ellipse_normal_speed(const SpreadEllipse& e, amrex::Real cos_th, amrex::Real sin_th) noexcept
{
    return e.c * cos_th + std::sqrt(e.b * e.b * cos_th * cos_th + e.a * e.a * sin_th * sin_th);
}

namespace farsite_front {

/// Larger than any arrival time
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
constexpr amrex::Real never () noexcept { return std::numeric_limits<amrex::Real>::max(); }

/**
 * Spread shape of one front cell at unit head rate.
 *
 * With erf.fire.farsite.shape = ellipse (default) and wind, the shape is the
 * Huygens ellipse of Richards (1990): semi-axes b along the wind and a across
 * it, centre offset c from the ignition point along the wind, all at unit head
 * rate (b + c = 1), from Anderson's length-to-width ratio and Alexander's
 * head-to-back ratio (spread_ellipse). With shape = rectangle the normal
 * speed is a cos + b |sin| ahead and b |sin| - c cos behind the wind, the
 * support function of the rectangle [-c, a] x [-b, b] in the wind frame, the
 * reading of the Richards coefficients this code used until 2026-10. Without
 * wind the normal speed is the head rate in every direction and the shape is
 * the unit disc.
 */
struct Shape
{
    amrex::Real wx, wy;   ///< unit wind direction
    amrex::Real a, b, c;  ///< rectangle: head, flank and backing coefficients; ellipse: semi-minor, semi-major and offset, all > 0
    amrex::Real sx, sy;   ///< terrain slope dz/dx, dz/dy
    bool isotropic;       ///< no wind: the unit disc
    bool ellipse;         ///< the ellipse (true) or the rectangle (false) with wind
};

/**
 * Time the shape takes to cover the map vector (ddx, ddy) at unit head rate:
 * its gauge, lengthened by the slope, since a step ds along the ground up a
 * slope of angle theta in that direction covers ds cos(theta) in map view.
 *
 * For the ellipse the gauge is the t with (par - c t)^2 / (b t)^2 + per^2 /
 * (a t)^2 = 1, the positive root of a^2 (b^2 - c^2) t^2 + 2 a^2 c par t -
 * (a^2 par^2 + b^2 per^2) = 0: par / (b + c) = par straight ahead, |par| / (b - c)
 * = |par| HB straight behind.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real gauge (amrex::Real ddx, amrex::Real ddy, const Shape& s) noexcept
{
    using namespace amrex::literals;
    const amrex::Real len = std::sqrt(ddx * ddx + ddy * ddy);
    if (len <= 0.0_rt) { return 0.0_rt; }
    amrex::Real g = len;
    if (!s.isotropic) {
        const amrex::Real par = ddx * s.wx + ddy * s.wy;
        const amrex::Real per = std::abs(ddy * s.wx - ddx * s.wy);
        if (s.ellipse) {
            const amrex::Real a2 = s.a * s.a;
            const amrex::Real b2 = s.b * s.b;
            // > 0 (b > c); b - c = R/HB carries the same cancellation as b^2 - c^2
            // (7e-6 relative in float at L/W = 8, 1e-14 in double: harmless)
            const amrex::Real A  = a2 * (s.b - s.c) * (s.b + s.c);
            const amrex::Real B  = a2 * s.c * par;            // half the linear coefficient
            const amrex::Real C  = a2 * par * par + b2 * per * per;
            const amrex::Real D  = std::sqrt(B * B + A * C);
            // the root without the cancelling difference ahead of the wind
            // (B > 0, A C << B^2 at a large L/W)
            g = (B >= 0.0_rt) ? C / (B + D) : (D - B) / A;
        } else {
            g = amrex::max(amrex::max(par / s.a, -par / s.c), per / s.b);
        }
    }
    const amrex::Real tan_theta = (s.sx * ddx + s.sy * ddy) / len;
    return g * std::sqrt(1.0_rt + tan_theta * tan_theta);
}

/// Zero of (1 - l) g0 + l g1 strictly inside (0, 1), or -1 if there is none
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real crossing (amrex::Real g0, amrex::Real g1) noexcept
{
    using namespace amrex::literals;
    if ((g0 < 0.0_rt && g1 > 0.0_rt) || (g0 > 0.0_rt && g1 < 0.0_rt)) { return g0 / (g0 - g1); }
    return -1.0_rt;
}

/**
 * Arrival time at a cell from one quadrant of its neighbours.
 *
 * Tx, Ty are the arrival times of the quadrant's x and y neighbours (negative
 * while unburned), hx, hy the signed offsets from those neighbours to the cell,
 * and rate the head rate of spread (> 0). Returns never() if neither burned.
 *
 * This is the Hopf-Lax (Huygens) update: the front passed each point of the
 * segment joining the two neighbour centres at the linearly interpolated time,
 * and the cell burns when the spread shape grown from the first of those points
 * reaches it. For a planar front this is exact: the time between rows is the
 * row spacing along the normal over the normal speed, one row at a time. Along
 * the segment the objective is convex; for the rectangle it is piecewise linear,
 * so the minimum lies at an end or at a kink, and every kink is tried; for the
 * disc the stationary point has a closed form; for the ellipse it is smooth and
 * found by a golden-section search. The slope is left out of the choice of the
 * point and applied to the travel time from it.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real quadrant_arrival (amrex::Real Tx, amrex::Real Ty,
                              amrex::Real hx, amrex::Real hy,
                              const Shape& s, amrex::Real rate) noexcept
{
    using namespace amrex::literals;
    const bool has_x = (Tx >= 0.0_rt);
    const bool has_y = (Ty >= 0.0_rt);
    const amrex::Real inv_rate = 1.0_rt / rate;
    amrex::Real best = never();
    if (has_x) { best = amrex::min(best, Tx + gauge(hx, 0.0_rt, s) * inv_rate); }
    if (has_y) { best = amrex::min(best, Ty + gauge(0.0_rt, hy, s) * inv_rate); }
    if (!(has_x && has_y)) { return best; }

    amrex::Real lam[6];
    int n = 0;
    if (!s.isotropic && s.ellipse) {
        // Smooth convex objective on the flat gauge: golden-section search on
        // (0, 1), 32 steps (a bracket of 1e-7 of the segment); the slope is
        // then applied to the travel time from the point found, as the other
        // branches do (inside the search the slope factor would make the
        // objective non-convex).
        Shape flat = s;
        flat.sx = 0.0_rt; flat.sy = 0.0_rt;
        const amrex::Real gr = 0.6180339887498949_rt;
        amrex::Real lo = 0.0_rt, hi = 1.0_rt;
        amrex::Real x1 = hi - gr * (hi - lo), x2 = lo + gr * (hi - lo);
        auto obj = [&] (amrex::Real l) {
            return Tx + l * (Ty - Tx) + gauge((1.0_rt - l) * hx, l * hy, flat) * inv_rate;
        };
        amrex::Real f1 = obj(x1), f2 = obj(x2);
        for (int it = 0; it < 32; ++it) {
            if (f1 < f2) { hi = x2; x2 = x1; f2 = f1; x1 = hi - gr * (hi - lo); f1 = obj(x1); }
            else         { lo = x1; x1 = x2; f1 = f2; x2 = lo + gr * (hi - lo); f2 = obj(x2); }
        }
        const amrex::Real l = (f1 < f2) ? x1 : x2;
        return amrex::min(best, Tx + l * (Ty - Tx) + gauge((1.0_rt - l) * hx, l * hy, s) * inv_rate);
    }
    if (s.isotropic) {
        // Tx + l (Ty - Tx) + |d(l)| / rate with d(l) = ((1 - l)|hx|, l |hy|) is
        // stationary where |hx| cos(p) - |hy| sin(p) = rate (Ty - Tx), p the angle of d.
        const amrex::Real ax = std::abs(hx);
        const amrex::Real ay = std::abs(hy);
        const amrex::Real q  = amrex::max(-1.0_rt, amrex::min(rate * (Ty - Tx) / std::sqrt(ax * ax + ay * ay), 1.0_rt));
        const amrex::Real p  = std::acos(q) - std::atan2(ay, ax);
        if (p > 0.0_rt && p < 0.5_rt * amrex::Math::pi<amrex::Real>()) {
            lam[n++] = ax * std::sin(p) / (ay * std::cos(p) + ax * std::sin(p));
        }
    } else {
        // Kinks of max(par / a, -par / c, |per| / b) along the segment, where par
        // and per (along and across the wind) are linear in l.
        const amrex::Real p0 =  hx * s.wx;
        const amrex::Real p1 =  hy * s.wy;
        const amrex::Real q0 = -hx * s.wy;
        const amrex::Real q1 =  hy * s.wx;
        lam[n++] = crossing(p0, p1);
        lam[n++] = crossing(q0, q1);
        lam[n++] = crossing( p0 / s.a - q0 / s.b,  p1 / s.a - q1 / s.b);
        lam[n++] = crossing( p0 / s.a + q0 / s.b,  p1 / s.a + q1 / s.b);
        lam[n++] = crossing(-p0 / s.c - q0 / s.b, -p1 / s.c - q1 / s.b);
        lam[n++] = crossing(-p0 / s.c + q0 / s.b, -p1 / s.c + q1 / s.b);
    }
    for (int m = 0; m < n; ++m) {
        const amrex::Real l = lam[m];
        if (l <= 0.0_rt || l >= 1.0_rt) { continue; }
        best = amrex::min(best, Tx + l * (Ty - Tx) + gauge((1.0_rt - l) * hx, l * hy, s) * inv_rate);
    }
    return best;
}

} // namespace farsite_front

/**
 * @brief One FARSITE substep, erf.fire.farsite.front_update = "legacy".
 *
 * Every cell with phi <= phi_threshold and a nonzero central-difference gradient
 * accumulates R dt along its normal and, once that reaches one cell, stamps a
 * burned target one cell ahead. phi is then rebuilt as -1 on burned cells and 0
 * everywhere else, so from the first substep on every unburned cell sits below
 * the threshold: the last burned row and the first unburned row both accumulate,
 * both stamp, and the front advances two rows per cell of travel, about twice
 * the rate of spread (Exec/RegTests/FarsiteFrontUpdate). Kept to reproduce
 * results from before 2026-09.
 */
inline
void advance_farsite_one_step_legacy(
    amrex::MultiFab& phi,
    amrex::MultiFab& farsite_work,
    amrex::MultiFab& disp_accum,
    amrex::MultiFab& arrival_time,
    const amrex::MultiFab& vel_eff,
    const amrex::MultiFab& R_mf,
    const amrex::Geometry& geom_fire,
    amrex::Real dt_fire,
    amrex::Real current_time,
    const FarsiteParams& fp,
    const amrex::MultiFab* slopes = nullptr,
    const amrex::MultiFab* nonburnable = nullptr,
    bool debug = false)
{
    using namespace amrex;

    const auto prob_lo = geom_fire.ProbLoArray();
    const auto dx      = geom_fire.CellSizeArray();
    const bool use_mask = (nonburnable != nullptr);
    const Real dxi = 1.0 / dx[0];
    const Real dyi = 1.0 / dx[1];
    const Real stamp_threshold = amrex::min(dx[0], dx[1]);

    farsite_work.setVal(0.0_rt);

    const bool use_terrain = (slopes != nullptr);

    // -------------------------------------------------------------------------
    // Pass 1: accumulate displacement; stamp absolute target when threshold met
    // -------------------------------------------------------------------------
    for (MFIter mfi(R_mf, false); mfi.isValid(); ++mfi) {
        const Box& bx   = mfi.validbox();
        auto phi_arr    = phi.array(mfi);
        auto spread_arr = farsite_work.array(mfi);
        auto disp_arr   = disp_accum.array(mfi);
        auto vel_arr    = vel_eff.const_array(mfi);
        auto ros_arr    = R_mf.const_array(mfi);

        amrex::Array4<const Real> slope_arr;
        if (use_terrain) slope_arr = slopes->const_array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept
        {
            int i = iv[0], j = iv[1], k = iv[2];

            if (phi_arr(iv) > fp.phi_threshold) { return; }

            Real dphi_dx = (phi_arr(i+1,j,k) - phi_arr(i-1,j,k)) * dxi * 0.5;
            Real dphi_dy = (phi_arr(i,j+1,k) - phi_arr(i,j-1,k)) * dyi * 0.5;
            Real grad_mag = std::sqrt(dphi_dx*dphi_dx + dphi_dy*dphi_dy);
            // unit vectors from floored magnitudes, before the early returns
            // (away from the front grad = 0, on calm cells wind = 0, and the
            // unselected 0/0 would be speculated under the fpe traps)
            const Real inv_grad = Real(1.0) / amrex::max(grad_mag, Real(1.0e-10));
            if (grad_mag < 1.0e-10) { return; }

            Real nx = dphi_dx * inv_grad;
            Real ny = dphi_dy * inv_grad;

            Real ux = vel_arr(iv, 0);
            Real uy = vel_arr(iv, 1);
            Real wind_mag = std::sqrt(ux*ux + uy*uy);
            const Real inv_wind = Real(1.0) / amrex::max(wind_mag, Real(1.0e-10));

            Real scale = 1.0;
            if (wind_mag > 1.0e-10) {
                Real wx = ux * inv_wind;
                Real wy = uy * inv_wind;
                Real cos_t = amrex::max(static_cast<amrex::Real>(-1.0), static_cast<amrex::Real>(amrex::min(static_cast<amrex::Real>(nx*wx + ny*wy), static_cast<amrex::Real>(1.0))));
                Real sin_t = std::sqrt(1.0 - cos_t*cos_t);
                Real a = fp.coeff_a, b = fp.coeff_b, c = fp.coeff_c;
                if (fp.use_anderson_lw != 0) {
                    LW_ratio_to_richards_coefficients(
                        anderson_LW_ratio(wind_mag * 2.237), a, b, c);
                }
                if (fp.shape == farsite_shape::ellipse) {
                    // the ellipse of the Anderson ratio, or of the ratio the
                    // given coefficients imply, (a + c) / (2 b)
                    const Real LB = (fp.use_anderson_lw != 0) ? anderson_LW_ratio(wind_mag * 2.237)
                                                              : (a + c) / (2.0 * amrex::max(b, Real(1.0e-6)));
                    scale = ellipse_normal_speed(spread_ellipse_from_lw(Real(1.0), LB), cos_t, sin_t);
                } else {
                    scale = (cos_t > 0.0)
                        ? a * cos_t + b * sin_t
                        : b * sin_t - c * cos_t;
                }
            }

            // Distance travelled along the terrain surface
            Real ds = amrex::max(static_cast<amrex::Real>(scale), static_cast<amrex::Real>(0.0)) * ros_arr(iv) * dt_fire;

            // Project onto the map. A surface step ds up a slope of angle theta
            // in the spread direction covers ds cos(theta) horizontally, with
            // tan(theta) = grad(z) . n.
            if (use_terrain) {
                Real s_n = slope_arr(i,j,k,0) * nx + slope_arr(i,j,k,1) * ny;
                ds /= std::sqrt(1.0 + s_n * s_n);
            }

            disp_arr(iv, 0) += ds * nx;
            disp_arr(iv, 1) += ds * ny;

            Real accum_mag = std::sqrt(disp_arr(iv,0)*disp_arr(iv,0)
                                     + disp_arr(iv,1)*disp_arr(iv,1));
            if (accum_mag >= stamp_threshold) {
                Real x_src = prob_lo[0] + (i + 0.5) * dx[0];
                Real y_src = prob_lo[1] + (j + 0.5) * dx[1];
                spread_arr(iv, 0) = x_src + disp_arr(iv, 0);
                spread_arr(iv, 1) = y_src + disp_arr(iv, 1);
                disp_arr(iv, 0) = 0.0;
                disp_arr(iv, 1) = 0.0;
            }
        });
    }

    amrex::Gpu::streamSynchronize();

    // -------------------------------------------------------------------------
    // Pass 2: collect stamped positions on host
    // -------------------------------------------------------------------------
    std::vector<Real> xarray, yarray;

    for (MFIter mfi(farsite_work, false); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();

#ifdef AMREX_USE_GPU
        // farsite_work lives in device memory (The_Arena is unmanaged by default),
        // so the loop below cannot read it directly. Stage the box into a pinned
        // host FAB first. The staging FAB carries the device FAB's own box, so the
        // Array4 below is indexed with the same global (i,j,k) either way.
        const FArrayBox& dev_fab = farsite_work[mfi];
        FArrayBox host_fab(dev_fab.box(), dev_fab.nComp(), The_Pinned_Arena());
        Gpu::dtoh_memcpy_async(host_fab.dataPtr(), dev_fab.dataPtr(),
                               dev_fab.size() * sizeof(Real));
        Gpu::streamSynchronize();
        auto spread_arr = host_fab.const_array();
#else
        auto spread_arr = farsite_work.const_array(mfi);
#endif

        for (int k = bx.smallEnd()[2]; k <= bx.bigEnd()[2]; ++k)
            for (int j = bx.smallEnd()[1]; j <= bx.bigEnd()[1]; ++j)
                for (int i = bx.smallEnd()[0]; i <= bx.bigEnd()[0]; ++i) {
                    Real sx = spread_arr(i,j,k,0);
                    Real sy = spread_arr(i,j,k,1);
                    if (sx != 0.0_rt || sy != 0.0_rt) {
                        xarray.push_back(sx);
                        yarray.push_back(sy);
                    }
                }
    }

    // -------------------------------------------------------------------------
    // MPI all-gather: use MPI_Allgather directly to avoid AMReX wrapper issues.
    // Falls back to a simple copy on non-MPI builds.
    // -------------------------------------------------------------------------
    int npts_local = static_cast<int>(xarray.size());
    int nranks     = ParallelDescriptor::NProcs();

    std::vector<int> npts_all(nranks, 0);
    std::vector<int> displs(nranks + 1, 0);

#ifdef AMREX_USE_MPI
    // Gather per-rank counts
    MPI_Allgather(&npts_local, 1, MPI_INT,
                  npts_all.data(), 1, MPI_INT,
                  ParallelDescriptor::Communicator());
#else
    npts_all[0] = npts_local;
#endif

    for (int r = 0; r < nranks; ++r)
        displs[r+1] = displs[r] + npts_all[r];
    int npts_global = displs[nranks];

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

    if (debug) {
        amrex::Print() << "[FIRE DEBUG] FARSITE legacy: stamping " << npts_global << " new perimeter points\n";
    }

    // -------------------------------------------------------------------------
    // Step A: reset phi to 0
    // Step B: restore burned interior from arrival_time
    // Step C: stamp new perimeter positions
    // Step D: update arrival_time for newly burned cells
    // -------------------------------------------------------------------------
    phi.setVal(0.0_rt);

    for (MFIter mfi(phi, false); mfi.isValid(); ++mfi) {
        auto p  = phi.array(mfi);
        auto at = arrival_time.const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
            if (at(iv) >= 0.0_rt) p(iv) = -1.0_rt;
        });
    }
    amrex::Gpu::streamSynchronize();

    if (npts_global > 0) {
        if (fp.gaussian_sigma < 0.0) {
            amrex::Gpu::DeviceVector<Real> d_x(npts_global), d_y(npts_global);
            amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                             xarray_all.begin(), xarray_all.end(), d_x.begin());
            amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                             yarray_all.begin(), yarray_all.end(), d_y.begin());
            const Real* xp = d_x.data();
            const Real* yp = d_y.data();
            int npts = npts_global;

            for (MFIter mfi(phi, false); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.validbox();
                auto p = phi.array(mfi);
                Array4<const Real> m;
                if (use_mask) { m = nonburnable->const_array(mfi); }
                amrex::ParallelFor(npts, [=] AMREX_GPU_DEVICE (int n) noexcept {
                    int ii = static_cast<int>(
                        std::round((xp[n] - prob_lo[0]) / dx[0] - 0.5_rt));
                    int jj = static_cast<int>(
                        std::round((yp[n] - prob_lo[1]) / dx[1] - 0.5_rt));
                    if (bx.contains(amrex::IntVect(ii, jj, 0))) {
                        // A target inside a non-burnable cell is dropped: the
                        // front stops at the footprint instead of crossing it.
                        if (use_mask && m(ii, jj, 0) > 0.5_rt) { return; }
                        p(ii, jj, 0) = amrex::min(p(ii, jj, 0), -1.0_rt);
                    }
                });
            }
            // ParallelFor is asynchronous while the Gpu::DeviceVector frees its device
            // allocation at scope exit; let the kernels finish reading it first.
            amrex::Gpu::streamSynchronize();
        } else {
            Real sigma = (fp.gaussian_sigma > 0.0) ? fp.gaussian_sigma : dx[0];
            // xarray_all/yarray_all are host std::vectors; their raw pointers cannot
            // be dereferenced from a device lambda. Stage them on the device first,
            // exactly as the gaussian_sigma < 0 branch above does.
            amrex::Gpu::DeviceVector<Real> d_xg(npts_global), d_yg(npts_global);
            amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                             xarray_all.begin(), xarray_all.end(), d_xg.begin());
            amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                             yarray_all.begin(), yarray_all.end(), d_yg.begin());
            const Real* x_ptr = d_xg.data();
            const Real* y_ptr = d_yg.data();
            int npts = npts_global;
            for (MFIter mfi(phi, false); mfi.isValid(); ++mfi) {
                auto p = phi.array(mfi);
                const Box& bx = mfi.validbox();
                Array4<const Real> m;
                if (use_mask) { m = nonburnable->const_array(mfi); }
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
                    if (use_mask && m(iv) > 0.5_rt) { return; }
                    Real xc = prob_lo[0] + (iv[0]+0.5)*dx[0];
                    Real yc = prob_lo[1] + (iv[1]+0.5)*dx[1];
                    for (int n = 0; n < npts; ++n) {
                        Real dist = std::sqrt((xc-x_ptr[n])*(xc-x_ptr[n])
                                           + (yc-y_ptr[n])*(yc-y_ptr[n]));
                        p(iv) = amrex::min(p(iv), amrex::max(dist-sigma,-1.0_rt));
                    }
                });
            }
            // ParallelFor is asynchronous while the Gpu::DeviceVector frees its device
            // allocation at scope exit; let the kernels finish reading it first.
            amrex::Gpu::streamSynchronize();
        }
    }

    // Step D: record arrival time for newly burned cells
    {
        const Real t_now = current_time;
        for (MFIter mfi(phi, false); mfi.isValid(); ++mfi) {
            auto p  = phi.const_array(mfi);
            auto at = arrival_time.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
                if (p(iv) < 0.0_rt && at(iv) < 0.0_rt) {
                    at(iv) = t_now;
                }
            });
        }
    }
}

/**
 * @brief One FARSITE substep, erf.fire.farsite.front_update = "front_cell".
 *
 * Front cells are the unburned, burnable cells with a burned 4-neighbour (with
 * the ellipse, with any burned cell within nine cells). Each
 * gets an arrival time from the arrival times of those neighbours with the
 * Hopf-Lax update of farsite_front::quadrant_arrival, using the spread shape
 * (the Richards ellipse, or the rectangle) of its own midflame wind (the disc
 * without wind) and its slope, and burns
 * in this substep if that time falls inside it, taking that time, not the
 * substep's, as its arrival time. A planar front therefore moves one row per
 * row spacing along the normal over R times the Richards normal speed (exact
 * for a front aligned with the grid, within one discrete-source offset of a
 * cell otherwise), and the
 * burned region of a uniform case grows as the ignition swept by the spread
 * shape, which keeps a finite line's head flat and a disc convex.
 *
 * With the ellipse every burned cell within nine cells is also tried as a
 * direct Huygens source: the cell burns when the distance its own rate has
 * accumulated since the source burned covers gauge(cell - src). The quadrant
 * update interpolates the arrival time linearly along the segment between two
 * face neighbours, which lags where the shape's characteristic is far from
 * the grid directions: at L/W = 2.9 the flank of the ellipse grows from a
 * source about three cells upwind (the direction (c, a) = (2.7, 1)), and the
 * segment stencil alone gave a third of the flank rate. A direct source gives
 * the exact point-source arrival, and nine cells cover the source offset of
 * the widest Anderson ellipse (L/W = 8); a source whose straight path to the
 * cell crosses a non-burnable or zero-rate cell is not used. The rate over the path is taken from
 * a distance clock, component 2 of disp_accum, which every cell accumulates
 * as its own R dt from the start of the run; a cell that burns records the
 * clock at its arrival in component 3 (-1 until the update stamps it, so a
 * cell dated by another writer, a spot landing or a scheduled polygon, takes
 * the clock at its own arrival rather than the run's start), so the distance
 * available to a source
 * S is clock(cell, t) - clock(S, T_S): the integral of the rate over the time
 * since S burned, exact for a rate that varies in time (the acceleration
 * clocks, a lull, a retardant factor) and uniform in space, rather than the
 * current rate applied over the whole path. Where the two cells' rates
 * differ the difference credits the source's history to the target, so it
 * is bounded by the source's own integral since it burned, clock(S, t) -
 * clock(S, T_S), which a path through the slower fuel cannot beat and which
 * stays exact for any uniform time dependence (a two-fuel strip keeps the
 * fast fuel's cells at T_boundary + k h / R_fast, ERF_GTestFarsiteShape).
 * The rectangle and the disc keep the quadrant update alone, whose paths are
 * grid-aligned or closed-form.
 *
 * The head rate R is the larger of the cell's own rate and those of its burned
 * neighbours: the burned side carries the crown-fire rate, which is only set in
 * burned cells, and the cell's own rate keeps the front moving where a burned
 * cell's rate has dropped. Its mean since the first neighbour burned is what
 * the update uses, accumulated in disp_accum (component 0 the distance R dt
 * summed, component 1 set to 1 while the cell is a front cell), so a rate that
 * varies in time is averaged over the crossing of the cell rather than applied
 * retroactively. Unburned cells get phi = +1, burned cells -1.
 */
inline
void advance_farsite_front_cells(
    amrex::MultiFab& phi,
    amrex::MultiFab& disp_accum,
    amrex::MultiFab& arrival_time,
    const amrex::MultiFab& vel_eff,
    const amrex::MultiFab& R_mf,
    const amrex::Geometry& geom_fire,
    amrex::Real dt_fire,
    amrex::Real current_time,
    const FarsiteParams& fp,
    const amrex::MultiFab* slopes = nullptr,
    const amrex::MultiFab* nonburnable = nullptr)
{
    using namespace amrex;

    const auto dx = geom_fire.CellSizeArray();
    const Real t0 = current_time;
    const Real t1 = current_time + dt_fire;
    const bool use_terrain = (slopes != nullptr);
    const bool use_mask    = (nonburnable != nullptr);
    const int  use_anderson_lw = fp.use_anderson_lw;
    const bool ellipse = (fp.shape == farsite_shape::ellipse);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(disp_accum.nComp() >= 4,
        "[FIRE] the FARSITE front-cell update needs a 4-component disp_accum (distance, front flag, distance clock, clock at burn)");

    // Coefficients divide the gauge; a zero one would mean no spread that way.
    const Real coef_min = 1.0e-6_rt;
    const Real coef_a = amrex::max(fp.coeff_a, coef_min);
    const Real coef_b = amrex::max(fp.coeff_b, coef_min);
    const Real coef_c = amrex::max(fp.coeff_c, coef_min);

    // An ignition, a spot landing, a scheduled polygon or a restart can leave a
    // cell burned (phi < 0) without an arrival time, or dated by its writer
    // without the clock at its burn (component 3 stays -1 until this update
    // stamps it; only this update writes it). A cell without a time burns from
    // the start of this substep; the clock at burn is the clock now (before
    // this substep's accumulation) less the rate over the time since the
    // arrival, the clock at t0 for a cell dated t0.
    for (MFIter mfi(arrival_time); mfi.isValid(); ++mfi) {
        auto p   = phi.const_array(mfi);
        auto at  = arrival_time.array(mfi);
        auto acc = disp_accum.array(mfi);
        auto R   = R_mf.const_array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (p(i, j, k) < 0.0_rt && (at(i, j, k) < 0.0_rt || acc(i, j, k, 3) < -0.5_rt)) {
                if (at(i, j, k) < 0.0_rt) { at(i, j, k) = t0; }
                const Real since = amrex::max(t0 - at(i, j, k), 0.0_rt);
                acc(i, j, k, 3) = amrex::max(acc(i, j, k, 2) - amrex::max(R(i, j, k), 0.0_rt) * since, 0.0_rt);
            }
        });
    }

    // The distance clock: every cell accumulates its own R dt (to t1), so
    // that the distance available to a direct source is the clock difference
    // since the source burned (see the comment above).
    for (MFIter mfi(disp_accum); mfi.isValid(); ++mfi) {
        auto acc = disp_accum.array(mfi);
        auto R   = R_mf.const_array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            acc(i, j, k, 2) += amrex::max(R(i, j, k), 0.0_rt) * dt_fire;
        });
    }

    // Arrival time and rate of spread with ghost cells: each update reads its
    // four neighbours, and with the ellipse the direct sources within r_src
    // cells, across box and periodic boundaries.
    const int r_src = ellipse ? 9 : 1;
    const IntVect dlo = geom_fire.Domain().smallEnd();
    const IntVect dhi = geom_fire.Domain().bigEnd();
    const bool per_x = geom_fire.isPeriodic(0);
    const bool per_y = geom_fire.isPeriodic(1);
    MultiFab nbr(arrival_time.boxArray(), arrival_time.DistributionMap(), 5, IntVect(r_src, r_src, 0));
    MultiFab::Copy(nbr, arrival_time, 0, 0, 1, 0);
    MultiFab::Copy(nbr, R_mf,         0, 1, 1, 0);
    MultiFab::Copy(nbr, disp_accum,   3, 2, 1, 0);   // the clock at burn of every burned cell
    MultiFab::Copy(nbr, disp_accum,   2, 3, 1, 0);   // and its clock now
    if (use_mask) { MultiFab::Copy(nbr, *nonburnable, 0, 4, 1, 0); } else { nbr.setVal(0.0_rt, 4, 1, 0); }   // the mask, within the source stencil
    fire_fill_boundary(nbr, geom_fire);

    // With the ellipse a cell is a candidate when any burned cell lies within
    // r_src cells, not only a face neighbour: under an oblique wind the ellipse
    // reaches the cells on its axis long before a face neighbour of theirs
    // burns (at L/W = 5 and a wind at 45 degrees to the grid the face test
    // alone burned no cell of a point source in 1200 s). Two separable passes
    // of the burned flag over the stencil, the first on the box grown in y.
    MultiFab near_x(arrival_time.boxArray(), arrival_time.DistributionMap(), 1, IntVect(0, r_src, 0));
    MultiFab near_f(arrival_time.boxArray(), arrival_time.DistributionMap(), 1, 0);
    if (ellipse) {
        for (MFIter mfi(near_x); mfi.isValid(); ++mfi) {
            auto nb = nbr.const_array(mfi);
            auto nx = near_x.array(mfi);
            ParallelFor(mfi.growntilebox(IntVect(0, r_src, 0)), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                Real f = 0.0_rt;
                for (int di = -r_src; di <= r_src; ++di) { if (nb(i + di, j, k, 0) >= 0.0_rt) { f = 1.0_rt; } }
                nx(i, j, k) = f;
            });
        }
        for (MFIter mfi(near_f); mfi.isValid(); ++mfi) {
            auto nx = near_x.const_array(mfi);
            auto nf = near_f.array(mfi);
            ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                Real f = 0.0_rt;
                for (int dj = -r_src; dj <= r_src; ++dj) { if (nx(i, j + dj, k) > 0.5_rt) { f = 1.0_rt; } }
                nf(i, j, k) = f;
            });
        }
    } else {
        near_f.setVal(0.0_rt);
    }

    for (MFIter mfi(arrival_time); mfi.isValid(); ++mfi) {
        auto nb  = nbr.const_array(mfi);
        auto at  = arrival_time.array(mfi);
        auto acc = disp_accum.array(mfi);
        auto vel = vel_eff.const_array(mfi);
        auto nf  = near_f.const_array(mfi);
        Array4<const Real> slp;
        if (use_terrain) { slp = slopes->const_array(mfi); }
        Array4<const Real> msk;
        if (use_mask) { msk = nonburnable->const_array(mfi); }

        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const Real Txm = nb(i-1, j, k, 0);
            const Real Txp = nb(i+1, j, k, 0);
            const Real Tym = nb(i, j-1, k, 0);
            const Real Typ = nb(i, j+1, k, 0);
            const bool has_face = (Txm >= 0.0_rt || Txp >= 0.0_rt || Tym >= 0.0_rt || Typ >= 0.0_rt);
            const bool near     = ellipse && (nf(i, j, k) > 0.5_rt);
            const bool front = (nb(i, j, k, 0) < 0.0_rt)
                            && !(use_mask && msk(i, j, k) > 0.5_rt)
                            && (has_face || near);
            if (!front) {
                acc(i, j, k, 0) = 0.0_rt;
                acc(i, j, k, 1) = 0.0_rt;
                return;
            }

            Real t_first = farsite_front::never();
            Real rate_now = nb(i, j, k, 1);
            if (Txm >= 0.0_rt) { t_first = amrex::min(t_first, Txm); rate_now = amrex::max(rate_now, nb(i-1, j, k, 1)); }
            if (Txp >= 0.0_rt) { t_first = amrex::min(t_first, Txp); rate_now = amrex::max(rate_now, nb(i+1, j, k, 1)); }
            if (Tym >= 0.0_rt) { t_first = amrex::min(t_first, Tym); rate_now = amrex::max(rate_now, nb(i, j-1, k, 1)); }
            if (Typ >= 0.0_rt) { t_first = amrex::min(t_first, Typ); rate_now = amrex::max(rate_now, nb(i, j+1, k, 1)); }
            rate_now = amrex::max(rate_now, 0.0_rt);

            // Distance covered at the head rate since the first neighbour
            // burned, for the quadrant paths; a cell with no burned face
            // neighbour is reached by the direct sources only
            Real rate = 0.0_rt;
            if (has_face) {
                if (acc(i, j, k, 1) != 1.0_rt) {
                    acc(i, j, k, 0) = rate_now * amrex::max(t0 - t_first, 0.0_rt);
                    acc(i, j, k, 1) = 1.0_rt;
                }
                acc(i, j, k, 0) += rate_now * dt_fire;
                rate = acc(i, j, k, 0) / (t1 - t_first);
            } else {
                acc(i, j, k, 0) = 0.0_rt;
                acc(i, j, k, 1) = 0.0_rt;
            }
            if (!(rate > 0.0_rt) && !ellipse) { return; }

            farsite_front::Shape s;
            const Real ux = vel(i, j, k, 0);
            const Real uy = vel(i, j, k, 1);
            const Real wind_mag = std::sqrt(ux * ux + uy * uy);
            s.isotropic = !(wind_mag > 1.0e-10_rt);
            s.ellipse = ellipse;
            s.wx = 1.0_rt; s.wy = 0.0_rt;
            s.a = 1.0_rt; s.b = 1.0_rt; s.c = 1.0_rt;
            // unit wind from the floored magnitude (calm cells give 0/0 in the
            // unselected arm, which the optimiser evaluates)
            const Real inv_wind = 1.0_rt / amrex::max(wind_mag, 1.0e-10_rt);
            if (!s.isotropic) {
                s.wx = ux * inv_wind;
                s.wy = uy * inv_wind;
                Real a = coef_a, b = coef_b, c = coef_c;
                if (use_anderson_lw != 0) {
                    LW_ratio_to_richards_coefficients(anderson_LW_ratio(wind_mag * 2.237_rt), a, b, c);
                }
                if (ellipse) {
                    // Anderson's ratio, or the ratio the given coefficients imply
                    const Real LB = (use_anderson_lw != 0) ? anderson_LW_ratio(wind_mag * 2.237_rt)
                                                           : (a + c) / (2.0_rt * amrex::max(b, coef_min));
                    const SpreadEllipse e = spread_ellipse_from_lw(1.0_rt, LB);
                    s.a = amrex::max(e.a, coef_min);
                    s.b = amrex::max(e.b, coef_min);
                    s.c = amrex::max(e.c, 0.0_rt);
                } else {
                    s.a = amrex::max(a, coef_min);
                    s.b = amrex::max(b, coef_min);
                    s.c = amrex::max(c, coef_min);
                }
            }
            s.sx = use_terrain ? slp(i, j, k, 0) : 0.0_rt;
            s.sy = use_terrain ? slp(i, j, k, 1) : 0.0_rt;

            Real T_new = farsite_front::never();
            if (rate > 0.0_rt) {
                T_new = amrex::min(T_new, farsite_front::quadrant_arrival(Txm, Tym,  dx[0],  dx[1], s, rate));
                T_new = amrex::min(T_new, farsite_front::quadrant_arrival(Txp, Tym, -dx[0],  dx[1], s, rate));
                T_new = amrex::min(T_new, farsite_front::quadrant_arrival(Txm, Typ,  dx[0], -dx[1], s, rate));
                T_new = amrex::min(T_new, farsite_front::quadrant_arrival(Txp, Typ, -dx[0], -dx[1], s, rate));
            }
            // a target with no rate of its own (a drop at factor 0, a fuel-free
            // cell) is never a direct target: its clock can be ahead of the
            // sources' from an earlier, faster rate, and the 1e-30 floor below
            // would date it at the step's start
            if (ellipse && nb(i, j, k, 1) > 0.0_rt) {
                // direct Huygens sources, for the calm disc as well so that a
                // vanishing wind changes nothing (the quadrant paths alone lag
                // the disc's diagonals too): a burned cell within r_src cells and
                // inside the domain (a non-periodic ghost is an extrapolated
                // copy). The cell burns when the distance its clock has
                // accumulated since the source burned covers the gauge; the
                // time inside this substep comes from its current rate.
                const Real clock_x = acc(i, j, k, 2);
                const Real r_now   = amrex::max(nb(i, j, k, 1), 1.0e-30_rt);
                for (int dj = -r_src; dj <= r_src; ++dj) {
                    const int jj = j + dj;
                    if (!per_y && (jj < dlo[1] || jj > dhi[1])) { continue; }
                    for (int di = -r_src; di <= r_src; ++di) {
                        const int ii = i + di;
                        if (!per_x && (ii < dlo[0] || ii > dhi[0])) { continue; }
                        if (nb(ii, jj, k, 0) < 0.0_rt) { continue; }
                        // The straight path from the source must not cross a
                        // non-burnable or zero-rate cell (a fire line, a
                        // retardant drop at factor 0, a fuel-free strip): the
                        // direct sources know nothing of what lies between,
                        // and the candidates nine cells out jumped a one-cell
                        // line. The cells nearest the segment, one per step.
                        {
                            const int nstep = amrex::max(std::abs(di), std::abs(dj));
                            bool blocked = false;
                            for (int m = 1; m < nstep && !blocked; ++m) {
                                const int im = i + (2 * di * m + (di >= 0 ? nstep : -nstep)) / (2 * nstep);
                                const int jm = j + (2 * dj * m + (dj >= 0 ? nstep : -nstep)) / (2 * nstep);
                                if (nb(im, jm, k, 4) > 0.5_rt || !(nb(im, jm, k, 1) > 0.0_rt)) { blocked = true; }
                            }
                            if (blocked) { continue; }
                        }
                        // The clock difference is the distance when both cells ran
                        // the same rate history; with a rate contrast (a fuel
                        // boundary, head against flank wind) it credits the
                        // source's history to the target, so it is bounded by the
                        // source's own integral since it burned, which a path
                        // through slower fuel cannot beat and which stays exact
                        // for any uniform time dependence (a current-rate bound
                        // would lose the direct paths under a falling rate).
                        const Real clock_s_burn = nb(ii, jj, k, 2);
                        const Real covered = amrex::min(clock_x - clock_s_burn, nb(ii, jj, k, 3) - clock_s_burn);
                        const Real need    = farsite_front::gauge(-di * dx[0], -dj * dx[1], s);
                        if (covered >= need) { T_new = amrex::min(T_new, t1 - (covered - need) / r_now); }
                    }
                }
            }
            if (T_new <= t1) {
                const Real at_new = amrex::max(T_new, t0);
                at(i, j, k) = at_new;
                acc(i, j, k, 0) = 0.0_rt;
                acc(i, j, k, 1) = 0.0_rt;
                // the clock at the arrival, read back from the end of the substep
                acc(i, j, k, 3) = amrex::max(acc(i, j, k, 2) - amrex::max(nb(i, j, k, 1), 0.0_rt) * (t1 - at_new), 0.0_rt);
            }
        });
    }

    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto p  = phi.array(mfi);
        auto at = arrival_time.const_array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            p(i, j, k) = (at(i, j, k) >= 0.0_rt) ? -1.0_rt : 1.0_rt;
        });
    }
    // The kernels above read nbr, which is freed on return.
    Gpu::streamSynchronize();
}

inline
void advance_farsite_one_step(
    amrex::MultiFab& phi,
    amrex::MultiFab& farsite_work,
    amrex::MultiFab& disp_accum,
    amrex::MultiFab& arrival_time,
    const amrex::MultiFab& vel_eff,
    const amrex::MultiFab& R_mf,
    const amrex::Geometry& geom_fire,
    amrex::Real dt_fire,
    amrex::Real current_time,
    const FarsiteParams& fp,
    const amrex::MultiFab* slopes = nullptr,
    const amrex::MultiFab* nonburnable = nullptr,
    bool debug = false)
{
    if (fp.front_update == farsite_front::legacy) {
        advance_farsite_one_step_legacy(phi, farsite_work, disp_accum, arrival_time,
                                        vel_eff, R_mf, geom_fire, dt_fire,
                                        current_time, fp, slopes, nonburnable, debug);
    } else {
        advance_farsite_front_cells(phi, disp_accum, arrival_time,
                                    vel_eff, R_mf, geom_fire, dt_fire,
                                    current_time, fp, slopes, nonburnable);
    }
}

inline
amrex::Real compute_fire_dt(
    const amrex::MultiFab& R_mf,
    const amrex::Geometry& geom_fire,
    amrex::Real dt_atm,
    amrex::Real cfl_fire)
{
    using namespace amrex;
    Real max_ros = R_mf.max(0);
    const auto dx = geom_fire.CellSizeArray();
    // floored rate before the early return (no burnable fuel: 0 -> x/0 speculated)
    const Real dt_cfl = cfl_fire * amrex::min(dx[0], dx[1]) / amrex::max(max_ros, Real(1.0e-10));
    if (max_ros < 1.0e-10) { return dt_atm; }
    return dt_cfl;
}

inline
int advance_fire_subcycle(
    amrex::MultiFab& phi,
    amrex::MultiFab& farsite_work,
    amrex::MultiFab& disp_accum,
    amrex::MultiFab& arrival_time,
    const amrex::MultiFab& vel_eff,
    const amrex::MultiFab& R_mf,
    const amrex::Geometry& geom_fire,
    amrex::Real dt_atm,
    amrex::Real current_time,
    const FarsiteParams& fp,
    const amrex::MultiFab* slopes = nullptr,
    const amrex::MultiFab* nonburnable = nullptr)
{
    using namespace amrex;

    Real time_remaining = dt_atm;
    int n_substeps = 0;

    while (time_remaining > 1.0e-14) {
        Real dt_fire = amrex::min(
            compute_fire_dt(R_mf, geom_fire, time_remaining, fp.cfl_fire),
            time_remaining);

        advance_farsite_one_step(phi, farsite_work, disp_accum,
                                 arrival_time,
                                 vel_eff, R_mf,
                                 geom_fire, dt_fire,
                                 current_time + (dt_atm - time_remaining),
                                 fp, slopes, nonburnable);

        // The step rebuilds phi in the valid region only (setVal then the
        // perimeter stamps), so the front normal of the next substep read
        // zeros in the ghosts along box and domain edges. The layer refills
        // after the last substep; the ones in between need it here.
        fire_fill_boundary(phi, geom_fire);

        time_remaining -= dt_fire;
        ++n_substeps;

        if (n_substeps > 1000) {
            amrex::Abort("advance_fire_subcycle: too many substeps");
        }
    }

    return n_substeps;
}

#endif // ERF_FARSITE_ELLIPSE_H
