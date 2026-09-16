#include <gtest/gtest.h>
#include <AMReX_REAL.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <array>

#include "ERF_FireBoundaryGuard.H"

/**
 * @file ERF_GTestFireBoundaryGuard.cpp
 * @brief The two kernels behind erf.fire.boundary_guard_* and
 *        erf.fire.edge_reach_check: the count of burning cells within a band
 *        of every non-periodic edge, and the distance from the burning region
 *        to each edge.
 *
 * A 16 x 8 fire grid of 2 m cells split into four boxes, so every count and
 * distance crosses box boundaries, with the periodicity chosen per test.
 */

using namespace amrex;
using namespace erf_fire_edge;

namespace {

constexpr int NX = 16, NY = 8;
constexpr Real DX = 2.0;

struct Grid {
    BoxArray ba; DistributionMapping dm; Geometry geom;
    Grid(bool per_x, bool per_y)
    {
        Box domain(IntVect(0, 0, 0), IntVect(NX - 1, NY - 1, 0));
        ba = BoxArray(domain);
        ba.maxSize(IntVect(8, 4, 1));
        dm = DistributionMapping(ba);
        RealBox rb(0.0_rt, 0.0_rt, 0.0_rt, Real(NX) * DX, Real(NY) * DX, 1.0_rt);
        geom = Geometry(domain, rb, CoordSys::cartesian, {per_x ? 1 : 0, per_y ? 1 : 0, 0});
    }
};

/// phi = +1 everywhere, -1 on the listed cells.
MultiFab burning(const Grid& g, std::initializer_list<std::pair<int,int>> cells)
{
    MultiFab phi(g.ba, g.dm, 1, 0);
    phi.setVal(1.0_rt);
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto p = phi.array(mfi);
        for (auto c : cells) {
            IntVect iv(c.first, c.second, 0);
            if (bx.contains(iv)) p(iv) = -1.0_rt;
        }
    }
    return phi;
}

} // namespace

TEST(FireBoundaryGuard, NothingBurningCountsNothing)
{
    Grid g(false, false);
    MultiFab phi = burning(g, {});
    std::array<Long, 4> touched;
    EXPECT_EQ(count_burning_in_edge_band(phi, g.geom, 2, touched), 0);
    for (int f = 0; f < 4; ++f) EXPECT_EQ(touched[f], 0);
    std::array<Real, 4> dist;
    EXPECT_EQ(burning_distance_to_edges(phi, g.geom, dist), 0);
    for (int f = 0; f < 4; ++f) EXPECT_LT(dist[f], 0.0_rt);
}

TEST(FireBoundaryGuard, InteriorFireIsOutsideEveryBand)
{
    Grid g(false, false);
    MultiFab phi = burning(g, {{7, 3}, {8, 3}, {7, 4}, {8, 4}});
    std::array<Long, 4> touched;
    EXPECT_EQ(count_burning_in_edge_band(phi, g.geom, 2, touched), 0);
    // A band of 4 on 8 rows reaches every cell: j = 3 is the 4th row from the
    // south and j = 4 the 4th from the north, each cell counted once in the total.
    EXPECT_EQ(count_burning_in_edge_band(phi, g.geom, 4, touched), 4);
    EXPECT_EQ(touched[South], 2);
    EXPECT_EQ(touched[North], 2);
    EXPECT_EQ(touched[West], 0);
    EXPECT_EQ(touched[East], 0);
    EXPECT_EQ(count_burning_in_edge_band(phi, g.geom, 0, touched), 0);   // band off
}

TEST(FireBoundaryGuard, BandCountsPerEdgeAndCornersOnceInTotal)
{
    Grid g(false, false);
    // one cell in the west band, one in the east band, one in the south-west corner
    MultiFab phi = burning(g, {{1, 4}, {15, 5}, {0, 0}});
    std::array<Long, 4> touched;
    EXPECT_EQ(count_burning_in_edge_band(phi, g.geom, 2, touched), 3);
    EXPECT_EQ(touched[West], 2);
    EXPECT_EQ(touched[East], 1);
    EXPECT_EQ(touched[South], 1);
    EXPECT_EQ(touched[North], 0);
}

TEST(FireBoundaryGuard, PeriodicEdgesAreSkipped)
{
    Grid g(true, false);
    MultiFab phi = burning(g, {{0, 4}, {15, 4}, {5, 0}});
    std::array<Long, 4> touched;
    EXPECT_EQ(count_burning_in_edge_band(phi, g.geom, 2, touched), 1);
    EXPECT_EQ(touched[West], 0);
    EXPECT_EQ(touched[East], 0);
    EXPECT_EQ(touched[South], 1);
    std::array<Real, 4> dist;
    EXPECT_EQ(burning_distance_to_edges(phi, g.geom, dist), 3);
    EXPECT_LT(dist[West], 0.0_rt);
    EXPECT_LT(dist[East], 0.0_rt);
    EXPECT_NEAR(dist[South], 0.5 * DX, 1e-12);
    EXPECT_NEAR(dist[North], (Real(NY - 1 - 4) + 0.5) * DX, 1e-12);
}

TEST(FireBoundaryGuard, DistanceIsFromTheNearestBurningCellCentre)
{
    Grid g(false, false);
    MultiFab phi = burning(g, {{3, 2}, {10, 6}});
    std::array<Real, 4> dist;
    EXPECT_EQ(burning_distance_to_edges(phi, g.geom, dist), 2);
    EXPECT_NEAR(dist[West],  (3 + 0.5) * DX, 1e-12);            // cell 3
    EXPECT_NEAR(dist[East],  (NX - 1 - 10 + 0.5) * DX, 1e-12);  // cell 10
    EXPECT_NEAR(dist[South], (2 + 0.5) * DX, 1e-12);            // cell (3,2)
    EXPECT_NEAR(dist[North], (NY - 1 - 6 + 0.5) * DX, 1e-12);   // cell (10,6)
}
