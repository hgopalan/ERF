#ifndef ERF_FIRE_BOUNDARY_GUARD_H_
#define ERF_FIRE_BOUNDARY_GUARD_H_

/**
 * @file ERF_FireBoundaryGuard.H
 * @brief The fire at the edge of the fire grid: the guard band of WRF-SFIRE
 *        (fire_boundary_guard) and the distance from the burning region to
 *        each edge for the start-up reach estimate.
 *
 * Nothing outside the fire grid burns. A front that reaches a non-periodic edge
 * of the grid stops there, with the ghost cells taking the nearest interior
 * value (fire_fill_boundary), so a fire that was going to leave the domain is
 * clipped instead. These kernels report that situation:
 *
 *  - count_burning_in_edge_band(): the number of burning cells (phi < 0) within
 *    `band` cells of any non-periodic edge, and which edges are touched. Zero
 *    while the fire is well inside the grid.
 *  - burning_distance_to_edges(): for each non-periodic edge, the distance in
 *    metres from the nearest burning cell centre to that edge. With the largest
 *    rate of spread and the time left in the run this is the reach estimate
 *    printed once at ignition.
 *
 * Periodic edges are skipped: the fire wraps across them, and a burning cell
 * next to one is not at an edge of anything. Both kernels are device
 * reductions over the valid cells and return the same value on every rank.
 */

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_Reduce.H>
#include <AMReX_ParallelDescriptor.H>
#include <array>
#include <limits>

namespace erf_fire_edge {

/// Edge indices of the results below, in the order the messages list them.
enum Face : int { West = 0, East = 1, South = 2, North = 3 };

inline const char* face_name(int f)
{
    static const char* names[4] = {"west (x lo)", "east (x hi)", "south (y lo)", "north (y hi)"};
    return (f >= 0 && f < 4) ? names[f] : "?";
}

/// Burning cells within `band` fire cells of a non-periodic edge. `touched[f]`
/// is the count of such cells at edge f, and the return value their total; a
/// cell in a corner counts once in the total and once at each edge it touches.
/// A band of zero or less checks nothing.
inline amrex::Long
count_burning_in_edge_band(const amrex::MultiFab& phi,
                           const amrex::Geometry& geom,
                           int band,
                           std::array<amrex::Long, 4>& touched)
{
    using namespace amrex;
    touched = {0, 0, 0, 0};
    if (band <= 0) return 0;

    const IntVect lo = geom.Domain().smallEnd();
    const IntVect hi = geom.Domain().bigEnd();
    // Captured by value: Geometry::isPeriodic reads a host member.
    const bool per_x = geom.isPeriodic(0);
    const bool per_y = geom.isPeriodic(1);
    const int  xlo_band = lo[0] + band - 1;   // i <= xlo_band is within the west band
    const int  xhi_band = hi[0] - band + 1;
    const int  ylo_band = lo[1] + band - 1;
    const int  yhi_band = hi[1] - band + 1;

    ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum> reduce_op;
    ReduceData<unsigned long long, unsigned long long, unsigned long long,
               unsigned long long, unsigned long long> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto const p  = phi.const_array(mfi);
        reduce_op.eval(bx, reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            unsigned long long w = 0, e = 0, s = 0, n = 0;
            if (p(i, j, k, 0) < Real(0.0)) {
                if (!per_x && i <= xlo_band) w = 1;
                if (!per_x && i >= xhi_band) e = 1;
                if (!per_y && j <= ylo_band) s = 1;
                if (!per_y && j >= yhi_band) n = 1;
            }
            const unsigned long long any = (w + e + s + n > 0) ? 1 : 0;
            return {any, w, e, s, n};
        });
    }
    ReduceTuple hv = reduce_data.value(reduce_op);
    Long total = static_cast<Long>(amrex::get<0>(hv));
    touched[West]  = static_cast<Long>(amrex::get<1>(hv));
    touched[East]  = static_cast<Long>(amrex::get<2>(hv));
    touched[South] = static_cast<Long>(amrex::get<3>(hv));
    touched[North] = static_cast<Long>(amrex::get<4>(hv));
    ParallelDescriptor::ReduceLongSum(total);
    ParallelDescriptor::ReduceLongSum(touched.data(), 4);
    return total;
}

/// Distance [m] from the nearest burning cell centre (phi < 0) to each
/// non-periodic edge of the fire grid; a periodic edge, or a grid with no
/// burning cell, reports a negative value. Returns the number of burning cells.
inline amrex::Long
burning_distance_to_edges(const amrex::MultiFab& phi,
                          const amrex::Geometry& geom,
                          std::array<amrex::Real, 4>& dist)
{
    using namespace amrex;
    const IntVect lo = geom.Domain().smallEnd();
    const IntVect hi = geom.Domain().bigEnd();
    const auto dx = geom.CellSizeArray();
    const Real big = std::numeric_limits<Real>::max();

    ReduceOps<ReduceOpSum, ReduceOpMin, ReduceOpMin, ReduceOpMin, ReduceOpMin> reduce_op;
    ReduceData<unsigned long long, Real, Real, Real, Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto const p  = phi.const_array(mfi);
        reduce_op.eval(bx, reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            if (p(i, j, k, 0) < Real(0.0)) {
                return {1ull,
                        (Real(i - lo[0]) + Real(0.5)) * dx[0],
                        (Real(hi[0] - i) + Real(0.5)) * dx[0],
                        (Real(j - lo[1]) + Real(0.5)) * dx[1],
                        (Real(hi[1] - j) + Real(0.5)) * dx[1]};
            }
            return {0ull, big, big, big, big};
        });
    }
    ReduceTuple hv = reduce_data.value(reduce_op);
    Long n_burning = static_cast<Long>(amrex::get<0>(hv));
    dist[West]  = amrex::get<1>(hv);
    dist[East]  = amrex::get<2>(hv);
    dist[South] = amrex::get<3>(hv);
    dist[North] = amrex::get<4>(hv);
    ParallelDescriptor::ReduceLongSum(n_burning);
    ParallelDescriptor::ReduceRealMin(dist.data(), 4);
    for (int f = 0; f < 4; ++f) {
        const bool periodic = geom.isPeriodic(f < 2 ? 0 : 1);
        if (periodic || n_burning == 0) dist[f] = Real(-1.0);
    }
    return n_burning;
}

} // namespace erf_fire_edge

#endif // ERF_FIRE_BOUNDARY_GUARD_H_
