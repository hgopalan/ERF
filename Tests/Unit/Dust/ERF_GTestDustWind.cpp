#include <gtest/gtest.h>
#include <AMReX_REAL.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <cmath>
#include <type_traits>

#include "ERF_DustGrid.H"
#include "ERF_DustWindExtract.H"

/**
 * @file ERF_GTestDustWind.cpp
 * @brief The wind handed to the dust grid, written to fail on the code before the
 *        October 2026 validation:
 *  - a reference height below the first cell centre takes the first cell's
 *    wind (the linear scan fell through to the second-highest cell);
 *  - the terrain speed factor scales the surface layer's u* instead of
 *    replacing it by a neutral log law on another roughness.
 */

using namespace amrex;

constexpr double REAL_RTOL = std::is_same<amrex::Real, float>::value ? 1.0e-6 : 1.0e-12;

namespace {

constexpr int NZ = 8;
constexpr Real DZ = 20.0;   // cell centres at 10, 30, ..., 150 m

/// One column: u(k) = k + 1 on the x faces, v = 0, z_cc(k) = 10 + 20 k.
struct WindColumn {
    Box domain{IntVect(0, 0, 0), IntVect(0, 0, NZ - 1)};
    Geometry geom{domain, RealBox({0.0, 0.0, 0.0}, {10.0, 10.0, NZ * DZ}), 0, {0, 0, 0}};
    BoxArray ba{domain};
    DistributionMapping dm{ba};
    MultiFab xvel{convert(ba, IntVect(1, 0, 0)), dm, 1, 0};
    MultiFab yvel{convert(ba, IntVect(0, 1, 0)), dm, 1, 0};
    MultiFab zcc{ba, dm, 1, 1};
    DustGrid dg;

    WindColumn ()
    {
        for (MFIter mfi(xvel); mfi.isValid(); ++mfi) {
            auto u = xvel.array(mfi);
            ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                u(i, j, k) = Real(k + 1);
            });
        }
        yvel.setVal(0.0);
        for (MFIter mfi(zcc); mfi.isValid(); ++mfi) {
            auto z = zcc.array(mfi);
            ParallelFor(mfi.growntilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                z(i, j, k) = (k + Real(0.5)) * DZ;
            });
        }
        // the dust grid is the 2-D slab of the same column (grid_ratio 1)
        Box slab = domain; slab.setBig(2, 0);
        dg.ba = BoxArray(slab);
        dg.dm = dm;
        dg.geom = Geometry(slab, RealBox({0.0, 0.0, 0.0}, {10.0, 10.0, 1.0}), 0, {0, 0, 0});
        dg.grid_ratio = 1;
    }
};

Real first_value (const MultiFab& mf, int comp)
{
    Real v = 0.0;
    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const Box b(IntVect(0, 0, 0), IntVect(0, 0, 0));
        if (!mfi.validbox().contains(b.smallEnd())) { continue; }
        FArrayBox h(b, 1, The_Pinned_Arena());
        h.copy<RunOn::Device>(mf[mfi], b, comp, b, 0, 1);
        Gpu::streamSynchronize();
        v = h(b.smallEnd());
    }
    ParallelDescriptor::ReduceRealSum(v);
    return v;
}

Real wind_at (WindColumn& c, Real zref)
{
    MultiFab wind(c.dg.ba, c.dg.dm, 2, 0);
    fill_dust_wind_from_interpolation(wind, c.xvel, c.yvel, c.zcc, c.dg, zref, NZ);
    return first_value(wind, 0);
}

} // namespace

TEST(DustWind, ReferenceHeightBelowTheFirstCellCentreTakesTheFirstCellsWind)
{
    WindColumn c;
    // z_target = z_surf + zref with z_surf = z_cc(0) - dz/2 = 0 here
    EXPECT_NEAR(wind_at(c, 5.0),  1.0, REAL_RTOL) << "below the first centre: the first cell's wind (the scan gave u(nz-2) = 7)";
    EXPECT_NEAR(wind_at(c, 10.0), 1.0, REAL_RTOL) << "exactly at the first centre";
    EXPECT_NEAR(wind_at(c, 25.0), 1.75, REAL_RTOL) << "three quarters of the way to the second centre";
    EXPECT_NEAR(wind_at(c, 70.0), 4.0, REAL_RTOL) << "at the fourth centre";
    EXPECT_NEAR(wind_at(c, 1000.0), Real(NZ), REAL_RTOL) << "above the top: the top cell's wind";
}

TEST(DustWind, TerrainSpeedFactorScalesTheSurfaceLayerFrictionVelocity)
{
    WindColumn c;
    MultiFab ustar(c.dg.ba, c.dg.dm, 1, 0), raw(c.dg.ba, c.dg.dm, 2, 0), corr(c.dg.ba, c.dg.dm, 2, 0);
    ustar.setVal(0.37);
    raw.setVal(3.0, 0, 1, 0); raw.setVal(4.0, 1, 1, 0);      // |U| = 5
    // a flat cell: factor 1 keeps the surface layer's u* (the log law gave 0.70x)
    MultiFab::Copy(corr, raw, 0, 0, 2, 0);
    scale_dust_ustar_by_wind_ratio(ustar, corr, raw);
    EXPECT_NEAR(first_value(ustar, 0), 0.37, REAL_RTOL * 0.37);
    // a windward face: the 1.5 speed-up is the u* factor
    corr.mult(1.5, 0, 2, 0);
    scale_dust_ustar_by_wind_ratio(ustar, corr, raw);
    EXPECT_NEAR(first_value(ustar, 0), 0.37 * 1.5, REAL_RTOL * 0.555);
    // a calm raw wind: nothing to scale by
    raw.setVal(0.0);
    scale_dust_ustar_by_wind_ratio(ustar, corr, raw);
    EXPECT_NEAR(first_value(ustar, 0), 0.37 * 1.5, REAL_RTOL * 0.555);
}
