#include <gtest/gtest.h>
#include <AMReX_REAL.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <cmath>

#include <ERF_Constants.H>
#include <ERF_IndexDefines.H>
#include <ERF_EOS.H>
#include "ERF_FireAtmCoupling.H"
#include "ERF_FireSmokeEmission.H"

/**
 * @file ERF_GTestFireAtmInjection.cpp
 * @brief What the fire puts into the atmospheric column: with
 *        erf.fire.heat_tendency_exner the rho theta tendency carries 1 / Pi, so
 *        the enthalpy c_p Pi d(rho theta)/dt integrated over the column is the
 *        surface flux (without it the column is Pi times short, 4.5 % at 850
 *        hPa); the smoke source divides the mass flux by the thickness of the
 *        k = 0 cell from its nodal heights, not by the centre-to-centre spacing.
 */

using namespace amrex;

namespace {

constexpr int  NX = 4, NZ = 50;
constexpr Real DXY = 100.0, DZ = 10.0;
/// Relative round-off tolerance of the build precision
constexpr double REL = (sizeof(Real) == 8) ? 1.0e-9 : 1.0e-4;

struct Column
{
    BoxArray            ba;
    DistributionMapping dm;
    Geometry            geom;
    Column ()
    {
        Box domain(IntVect(0, 0, 0), IntVect(NX - 1, NX - 1, NZ - 1));
        ba = BoxArray(domain);
        dm = DistributionMapping(ba);
        geom = Geometry(domain, RealBox(0.0, 0.0, 0.0, NX * DXY, NX * DXY, NZ * DZ), CoordSys::cartesian, {1, 1, 0});
    }
};

} // namespace

// nvcc refuses an extended __device__ lambda whose enclosing function is
// private, as gtest's TestBody is: the bodies that launch kernels are
// namespace-scope functions, and each TEST calls its own.
namespace {
void ExnerTendency_TheColumnEnthalpyIsTheSurfaceFlux ()
{
    Column c;
    const Real q = 5.0e4_rt, alfg = 10.0_rt, p = 8.5e4_rt;
    MultiFab src_off(c.ba, c.dm, NVAR_max, 0), src_on(c.ba, c.dm, NVAR_max, 0);
    MultiFab Q(c.ba, c.dm, 1, 0), zcc(c.ba, c.dm, 1, 1), S(c.ba, c.dm, NVAR_max, 0);
    src_off.setVal(0.0_rt); src_on.setVal(0.0_rt);
    Q.setVal(0.0_rt);
    Q.setVal(q, makeSlab(c.geom.Domain(), 2, 0), 0, 1);
    S.setVal(0.0_rt);
    S.setVal(1.0_rt, Rho_comp, 1);
    S.setVal(getRhoThetagivenP(p), RhoTheta_comp, 1);
    for (MFIter mfi(zcc); mfi.isValid(); ++mfi) {
        auto z = zcc.array(mfi);
        ParallelFor(mfi.growntilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept { z(i, j, k) = (k + 0.5_rt) * DZ; });
    }
    // d(rho theta)/dt = -(1/c_p) dQ/dz without the density factor; alfg = 10 m
    // puts the whole profile inside the 500 m column (1 - e^{-50})
    apply_fire_tendency_to_cc_source(src_off, Q, nullptr, zcc, S, c.geom, alfg, 1.0_rt, false,
                                     false, false, nullptr, nullptr, false, false, false);
    apply_fire_tendency_to_cc_source(src_on,  Q, nullptr, zcc, S, c.geom, alfg, 1.0_rt, false,
                                     false, false, nullptr, nullptr, false, false, true);

    const Real exner = std::pow(p / p_0, R_d / Cp_d);
    constexpr Real Cp_fire = 1004.64;   // the coupling's c_p
    EXPECT_NEAR(exner, 0.9546, 5.0e-4);
    Real err_ratio = 0.0, sum_off = 0.0, sum_on = 0.0;
    for (MFIter mfi(src_on); mfi.isValid(); ++mfi) {
        auto a = src_off.const_array(mfi, RhoTheta_comp);
        auto b = src_on.const_array(mfi, RhoTheta_comp);
        for (int k = 0; k < NZ; ++k) {
            const Real toff = a(1, 2, k), ton = b(1, 2, k);
            if (toff > 0.0_rt) { err_ratio = amrex::max(err_ratio, std::abs(ton / toff - 1.0_rt / exner)); }
            sum_off += Cp_fire * exner * toff * DZ;   // the enthalpy the plain tendency carries
            sum_on  += Cp_fire * exner * ton * DZ;
        }
    }
    EXPECT_LT(err_ratio, REL) << "every cell's tendency is divided by Pi";
    EXPECT_NEAR(sum_on,  q, 1.0e3 * REL * q) << "c_p Pi d(rho theta)/dt integrates to the flux";
    EXPECT_NEAR(sum_off, exner * q, 1.0e3 * REL * q) << "without the factor the column enthalpy is Pi q";
}
}  // namespace

TEST(ExnerTendency, TheColumnEnthalpyIsTheSurfaceFlux)
{
    ExnerTendency_TheColumnEnthalpyIsTheSurfaceFlux();
}

namespace {
void SmokeInjection_SourceDividesByTheCellThickness ()
{
    Column c;
    const Real heat = 1.0e5_rt, ef = 0.02_rt, hoc = 1.8e7_rt;
    MultiFab src(c.ba, c.dm, 1, 0), Q(c.ba, c.dm, 1, 0);
    Q.setVal(0.0_rt);
    Q.setVal(heat, makeSlab(c.geom.Domain(), 2, 0), 0, 1);
    // nodal heights: z(i, j, k) = 0.1 i + k (8 + 0.2 i + 0.2 j), so the k = 0
    // cell is 8 + 0.2 (i + 1/2) + 0.2 (j + 1/2) thick (the mean of its four
    // node pairs) while the uniform dz is 10 and the centre-to-centre spacing
    // of a stretched column would differ again
    MultiFab znd(convert(c.ba, IntVect(1, 1, 1)), c.dm, 1, 1);
    for (MFIter mfi(znd); mfi.isValid(); ++mfi) {
        auto z = znd.array(mfi);
        ParallelFor(mfi.growntilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            z(i, j, k) = 0.1_rt * i + k * (8.0_rt + 0.2_rt * i + 0.2_rt * j);
        });
    }
    const Real flux = ef * heat / hoc;
    src.setVal(0.0_rt);
    inject_smoke_from_fire(src, Q, &znd, c.geom, ef, hoc, 0, false, 0);
    for (MFIter mfi(src); mfi.isValid(); ++mfi) {
        auto s = src.const_array(mfi);
        for (int j = 0; j < NX; ++j) {
            for (int i = 0; i < NX; ++i) {
                const Real dz_cell = 8.0_rt + 0.2_rt * (i + 0.5_rt) + 0.2_rt * (j + 0.5_rt);
                EXPECT_NEAR(s(i, j, 0), flux / dz_cell, REL * flux / dz_cell) << "column " << i << "," << j;
                EXPECT_EQ(s(i, j, 1), 0.0_rt) << "only k = 0 takes the source";
            }
        }
    }
    // without terrain: the uniform dz
    src.setVal(0.0_rt);
    inject_smoke_from_fire(src, Q, nullptr, c.geom, ef, hoc, 0, false, 0);
    EXPECT_NEAR(src.max(0), flux / DZ, REL * flux / DZ);
    EXPECT_NEAR(src.min(0), 0.0, 1.0e-30);
}
}  // namespace

TEST(SmokeInjection, SourceDividesByTheCellThickness)
{
    SmokeInjection_SourceDividesByTheCellThickness();
}
