#include <gtest/gtest.h>
#include <AMReX_REAL.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_BoxList.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <cmath>
#include <vector>

#include "ERF_FireGrid.H"
#include "ERF_FireAtmCoupling.H"
#include "ERF_TerrainSlope.H"
#include "ERF_FireWindExtract.H"

/**
 * @file ERF_GTestFireAnchorLevel.cpp
 * @brief The fire grid over a refined region (erf.fire.anchor_level) and the
 *        maps from fire cells back to atmospheric columns.
 *
 * The atmospheric level is 32 x 24 x 4 cells of 25 m over 800 x 600 x 100 m,
 * periodic in x and y. The refined region is columns 8-19 by 4-11, covered by
 * two boxes of unequal width, so every map crosses a box boundary, and the
 * fire grid refines it by C = 4. Every expected value is computed from the
 * physical position of the fire cell (the column holding x is floor(x / dx)),
 * never from the index formula i / C + atm_lo under test, and the fields are
 * linear or quadratic in position, so a map that lands on the wrong column
 * gives a different value. A full-domain level checks that level 0 is built
 * exactly as before.
 */

using namespace amrex;

namespace {

constexpr int  C  = 4;
constexpr Real DX = 25.0;

struct AtmLevel
{
    BoxArray            ba;
    DistributionMapping dm;
    Geometry            geom;
};

AtmLevel make_level (const std::vector<Box>& boxes, bool periodic_xy)
{
    AtmLevel L;
    BoxList bl;
    for (const auto& b : boxes) { bl.push_back(b); }
    L.ba = BoxArray(bl);
    L.dm = DistributionMapping(L.ba);
    Box dom(IntVect(0, 0, 0), IntVect(31, 23, 3));
    RealBox rb(0.0_rt, 0.0_rt, 0.0_rt, 800.0_rt, 600.0_rt, 100.0_rt);
    L.geom = Geometry(dom, rb, CoordSys::cartesian, {periodic_xy ? 1 : 0, periodic_xy ? 1 : 0, 0});
    return L;
}

/// Columns 8-19 by 4-11 in two boxes of 8 and 4 columns
AtmLevel region_level ()
{
    return make_level({Box(IntVect(8, 4, 0), IntVect(15, 11, 3)),
                       Box(IntVect(16, 4, 0), IntVect(19, 11, 3))}, true);
}

BoxArray k0_slabs (const BoxArray& ba)
{
    BoxList bl;
    for (int n = 0; n < ba.size(); ++n) {
        Box b = ba[n];
        b.setSmall(2, 0);
        b.setBig(2, 0);
        bl.push_back(b);
    }
    return BoxArray(bl);
}

AMREX_GPU_HOST_DEVICE inline Real terrain (Real x, Real y) noexcept
{
    return x * x / 1000.0_rt + y * y / 2000.0_rt;
}

/// Nodal terrain height z(x, y) at every node, ghosts included
void fill_terrain_nodes (MultiFab& z)
{
    for (MFIter mfi(z); mfi.isValid(); ++mfi) {
        auto a = z.array(mfi);
        ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            a(i, j, k) = terrain(Real(i) * DX, Real(j) * DX);
        });
    }
}

/// Fire-grid field x + 1000 y at the cell centres
void fill_fire_position (MultiFab& q, const Geometry& g)
{
    const auto plo = g.ProbLoArray();
    const auto dx  = g.CellSizeArray();
    for (MFIter mfi(q); mfi.isValid(); ++mfi) {
        auto a = q.array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            a(i, j, k) = (plo[0] + (Real(i) + 0.5_rt) * dx[0]) + 1000.0_rt * (plo[1] + (Real(j) + 0.5_rt) * dx[1]);
        });
    }
}

/// k = 0 atmospheric field carrying its own column index, i + 1000 j
void fill_column_index (MultiFab& a0)
{
    for (MFIter mfi(a0); mfi.isValid(); ++mfi) {
        auto a = a0.array(mfi);
        ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            a(i, j, k) = Real(i) + 1000.0_rt * Real(j);
        });
    }
}

/// Face velocities whose cell averages are u = x + y / 2 and v = y + x / 4,
/// and cell-centre heights; ghosts included, as FillPatch leaves them
void fill_linear_wind (MultiFab& xvel, MultiFab& yvel, MultiFab& zcc)
{
    for (MFIter mfi(xvel); mfi.isValid(); ++mfi) {
        auto u = xvel.array(mfi);
        ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            u(i, j, k) = Real(i) * DX + 0.5_rt * (Real(j) + 0.5_rt) * DX;
        });
    }
    for (MFIter mfi(yvel); mfi.isValid(); ++mfi) {
        auto v = yvel.array(mfi);
        ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            v(i, j, k) = Real(j) * DX + 0.25_rt * (Real(i) + 0.5_rt) * DX;
        });
    }
    for (MFIter mfi(zcc); mfi.isValid(); ++mfi) {
        auto z = zcc.array(mfi);
        ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            z(i, j, k) = (Real(k) + 0.5_rt) * DX;
        });
    }
}

/// Column of level `L` holding a physical coordinate
int column_of (Real x) { return static_cast<int>(std::floor(x / DX)); }

Real node_mean (int ia, int ja)
{
    return 0.25_rt * (terrain(ia * DX, ja * DX) + terrain((ia + 1) * DX, ja * DX)
                    + terrain(ia * DX, (ja + 1) * DX) + terrain((ia + 1) * DX, (ja + 1) * DX));
}

/// Largest relative error of a fire-grid field against expected(x, y, comp)
template <typename F>
Real max_rel_error (const MultiFab& mf, const FireGrid& fg, int comp, F&& expected)
{
    const Geometry& g = fg.geom;
    Real err = 0.0;
    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const auto a = mf.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                const Real x = g.ProbLo(0) + (Real(i) + 0.5_rt) * g.CellSize(0);
                const Real y = g.ProbLo(1) + (Real(j) + 0.5_rt) * g.CellSize(1);
                const Real e = expected(x, y);
                err = std::max(err, std::abs(a(i, j, 0, comp) - e) / std::max(Real(1.0), std::abs(e)));
            }
        }
    }
    ParallelDescriptor::ReduceRealMax(err);
    return err;
}

const Real TOL = (sizeof(Real) == 8) ? 1.0e-12 : 1.0e-5;

} // namespace

TEST(FireAnchorLevel, RegionGridStartsAtTheRegionCorner)
{
    AtmLevel L = region_level();
    FireGrid fg = create_fire_grid(L.ba, L.dm, L.geom, C);

    EXPECT_EQ(fg.atm_lo, IntVect(8, 4, 0));
    EXPECT_EQ(fg.geom.Domain(), Box(IntVect(0, 0, 0), IntVect(47, 31, 0)));
    EXPECT_NEAR(fg.geom.ProbLo(0), 200.0, TOL);
    EXPECT_NEAR(fg.geom.ProbHi(0), 500.0, TOL);
    EXPECT_NEAR(fg.geom.ProbLo(1), 100.0, TOL);
    EXPECT_NEAR(fg.geom.ProbHi(1), 300.0, TOL);
    EXPECT_NEAR(fg.geom.CellSize(0), 6.25, TOL);
    // The region has edges of its own, whatever the domain's periodicity
    EXPECT_FALSE(fg.geom.isPeriodic(0));
    EXPECT_FALSE(fg.geom.isPeriodic(1));
    ASSERT_EQ(fg.ba.size(), 2);
    EXPECT_EQ(fg.ba[0], Box(IntVect(0, 0, 0), IntVect(31, 31, 0)));
    EXPECT_EQ(fg.ba[1], Box(IntVect(32, 0, 0), IntVect(47, 31, 0)));
    EXPECT_EQ(fg.dm, L.dm);
}

TEST(FireAnchorLevel, FullDomainGridIsThePlainRefinement)
{
    AtmLevel L = make_level({Box(IntVect(0, 0, 0), IntVect(31, 23, 3))}, true);
    L.ba.maxSize(IntVect(16, 8, 4));
    L.dm = DistributionMapping(L.ba);
    FireGrid fg = create_fire_grid(L.ba, L.dm, L.geom, C);

    EXPECT_EQ(fg.atm_lo, IntVect(0, 0, 0));
    EXPECT_EQ(fg.geom.Domain(), Box(IntVect(0, 0, 0), IntVect(127, 95, 0)));
    // The domain's own bounds, to the bit
    EXPECT_EQ(fg.geom.ProbLo(0), L.geom.ProbLo(0));
    EXPECT_EQ(fg.geom.ProbHi(0), L.geom.ProbHi(0));
    EXPECT_EQ(fg.geom.ProbLo(1), L.geom.ProbLo(1));
    EXPECT_EQ(fg.geom.ProbHi(1), L.geom.ProbHi(1));
    EXPECT_TRUE(fg.geom.isPeriodic(0));
    EXPECT_TRUE(fg.geom.isPeriodic(1));
    BoxArray expect = k0_slabs(L.ba);
    expect.refine(IntVect(C, C, 1));
    EXPECT_EQ(fg.ba, expect);
}

TEST(FireAnchorLevel, RegionSpanningOneDirectionKeepsThatPeriodicity)
{
    AtmLevel L = make_level({Box(IntVect(8, 0, 0), IntVect(19, 23, 3))}, true);
    FireGrid fg = create_fire_grid(L.ba, L.dm, L.geom, C);
    EXPECT_EQ(fg.atm_lo, IntVect(8, 0, 0));
    EXPECT_FALSE(fg.geom.isPeriodic(0));
    EXPECT_TRUE(fg.geom.isPeriodic(1));
    EXPECT_EQ(fg.geom.ProbLo(1), 0.0);
    EXPECT_EQ(fg.geom.ProbHi(1), 600.0);
}

TEST(FireAnchorLevel, TerrainMapsReadTheColumnUnderTheFireCell)
{
    AtmLevel L = region_level();
    FireGrid fg = create_fire_grid(L.ba, L.dm, L.geom, C);

    // Two ghost layers, as the atmospheric level carries: a map one column off
    // then reads a valid but wrong node and fails the comparison.
    MultiFab z_nd(convert(L.ba, IntVect(1, 1, 1)), L.dm, 1, 2);
    fill_terrain_nodes(z_nd);

    MultiFab zs(fg.ba, fg.dm, 1, 0);
    compute_fire_surface_height(zs, &z_nd, L.geom, fg);
    EXPECT_LT(max_rel_error(zs, fg, 0, [](Real x, Real y) {
        return node_mean(column_of(x), column_of(y)); }), TOL);

    MultiFab slopes(fg.ba, fg.dm, 2, 1);
    compute_terrain_slopes(slopes, &z_nd, L.geom, fg, "");
    EXPECT_LT(max_rel_error(slopes, fg, 0, [](Real x, Real) {
        const Real xl = column_of(x) * DX;
        return ((xl + DX) * (xl + DX) - xl * xl) / 1000.0_rt / DX; }), TOL);
    EXPECT_LT(max_rel_error(slopes, fg, 1, [](Real, Real y) {
        const Real yl = column_of(y) * DX;
        return ((yl + DX) * (yl + DX) - yl * yl) / 2000.0_rt / DX; }), TOL);

    MultiFab grounds(fg.ba, fg.dm, 4, 0);
    compute_fire_column_grounds(grounds, &z_nd, L.geom, fg);
    for (int c = 0; c < 4; ++c) {
        EXPECT_LT(max_rel_error(grounds, fg, c, [c](Real x, Real y) {
            const int i0 = column_of(x - 0.5_rt * DX);
            const int j0 = column_of(y - 0.5_rt * DX);
            return node_mean(i0 + (c & 1), j0 + ((c >> 1) & 1)); }), TOL) << "column " << c;
    }
}

TEST(FireAnchorLevel, AtmosphericSurfaceFieldsReachTheirFireCells)
{
    AtmLevel L = region_level();
    FireGrid fg = create_fire_grid(L.ba, L.dm, L.geom, C);

    // One ghost layer, filled with its own column index: a map one column off
    // reads a valid but wrong column and fails the comparison.
    MultiFab a0(k0_slabs(L.ba), L.dm, 1, IntVect(1, 1, 0));
    fill_column_index(a0);
    MultiFab f(fg.ba, fg.dm, 1, 0);
    fill_fire_from_atm_k0(f, a0, fg);
    EXPECT_LT(max_rel_error(f, fg, 0, [](Real x, Real y) {
        return Real(column_of(x)) + 1000.0_rt * Real(column_of(y)); }), TOL);
}

TEST(FireAnchorLevel, FluxAveragesOntoTheColumnsItCameFrom)
{
    for (bool region : {true, false}) {
        AtmLevel L = region ? region_level()
                            : make_level({Box(IntVect(0, 0, 0), IntVect(31, 23, 3))}, true);
        if (!region) { L.ba.maxSize(IntVect(16, 8, 4)); L.dm = DistributionMapping(L.ba); }
        FireGrid fg = create_fire_grid(L.ba, L.dm, L.geom, C);

        MultiFab q(fg.ba, fg.dm, 1, 0);
        fill_fire_position(q, fg.geom);
        MultiFab q_atm(k0_slabs(L.ba), L.dm, 1, 0);
        q_atm.setVal(-1.0e30_rt);
        coarsen_fire_flux_to_atm(q_atm, q, L.geom, fg);

        // A field linear in position averages to its value at the column centre
        Real err = 0.0;
        for (MFIter mfi(q_atm); mfi.isValid(); ++mfi) {
            const auto a = q_atm.const_array(mfi);
            const Box& bx = mfi.validbox();
            for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                    const Real e = (Real(i) + 0.5_rt) * DX + 1000.0_rt * (Real(j) + 0.5_rt) * DX;
                    err = std::max(err, std::abs(a(i, j, 0) - e) / e);
                }
            }
        }
        ParallelDescriptor::ReduceRealMax(err);
        EXPECT_LT(err, TOL) << (region ? "refined region" : "full domain");
    }
}

TEST(FireAnchorLevel, WindIsSampledFromTheSurroundingColumns)
{
    AtmLevel L = region_level();
    FireGrid fg = create_fire_grid(L.ba, L.dm, L.geom, C);

    MultiFab xvel(convert(L.ba, IntVect(1, 0, 0)), L.dm, 1, 2);
    MultiFab yvel(convert(L.ba, IntVect(0, 1, 0)), L.dm, 1, 2);
    MultiFab zcc(L.ba, L.dm, 1, 2);
    fill_linear_wind(xvel, yvel, zcc);

    MultiFab zs(fg.ba, fg.dm, 1, 0);
    zs.setVal(0.0);
    MultiFab grounds(fg.ba, fg.dm, 4, 0);
    grounds.setVal(0.0);
    MultiFab wind(fg.ba, fg.dm, 2, 0);
    MultiFab ez(fg.ba, fg.dm, 1, 0);

    // Bilinear: a field linear in position is reproduced at the fire cell centre
    fill_fire_wind_from_interpolation(wind, ez, xvel, yvel, zcc, zs, grounds, fg, 10.0, 4);
    EXPECT_LT(max_rel_error(wind, fg, 0, [](Real x, Real y) { return x + 0.5_rt * y; }), 1.0e3 * TOL);
    EXPECT_LT(max_rel_error(wind, fg, 1, [](Real x, Real y) { return y + 0.25_rt * x; }), 1.0e3 * TOL);

    // Nearest: the value at the centre of the column holding the fire cell
    fill_fire_wind_from_interpolation(wind, ez, xvel, yvel, zcc, zs, grounds, fg, 10.0, 4,
                                      nullptr, nullptr, 0, 0);
    EXPECT_LT(max_rel_error(wind, fg, 0, [](Real x, Real y) {
        return (column_of(x) + 0.5_rt) * DX + 0.5_rt * (column_of(y) + 0.5_rt) * DX; }), 1.0e3 * TOL);
    EXPECT_LT(max_rel_error(wind, fg, 1, [](Real x, Real y) {
        return (column_of(y) + 0.5_rt) * DX + 0.25_rt * (column_of(x) + 0.5_rt) * DX; }), 1.0e3 * TOL);
}
