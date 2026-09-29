// Contracts of the prescribed uniform-Ct disk: its point areas sum to pi R^2, its sampling
// points sit one diameter upstream along the normal, its force on the fluid is -T n with
// T = 1/2 rho Ct U^2 pi R^2 from the area-weighted upstream normal speed, and the ct_disk
// inputs are parsed with their ranges.

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <gtest/gtest.h>

#include "ERF_MovingBodiesInputs.H"
#include "ERF_PrescribedCtDisk.H"

namespace {

using amrex::Real;
constexpr Real pi = 3.14159265358979323846;
// the disk's sums are exact up to roundoff: 1e-9 relative in double, a few 1e-7 in single
constexpr Real rel_tol = (sizeof(Real) == 8) ? Real(1.0e-9) : Real(2.0e-5);
// positions are sums of a few products of order 1e2-1e3 m
constexpr Real abs_tol = (sizeof(Real) == 8) ? Real(1.0e-9) : Real(1.0e-3);

MovingBodyInputs disk_inputs (Real yaw_deg)
{
    MovingBodyInputs b;
    b.name = "D1";
    b.type = "ct_disk";
    b.base_pos = {{750.0, 600.0, 10.0}};
    b.rotor_radius = 120.0;
    b.hub_height = 150.0;
    b.ct = 0.75;
    b.yaw_deg = yaw_deg;
    b.num_points_r = 6;
    b.num_points_t = 12;
    b.sample_diameters_upstream = 1.0;
    b.air_density = 1.2;
    b.output_root = "D1";
    return b;
}

} // namespace

TEST(PrescribedCtDisk, PointsCoverTheDiskAndSampleUpstream)
{
    const erf_actuator::PrescribedCtDisk d(disk_inputs(0.0));
    EXPECT_EQ(d.num_points(), 6 * 12);
    EXPECT_NEAR(d.area(), pi * 120.0 * 120.0, rel_tol * pi * 120.0 * 120.0);
    const auto n = d.normal();
    EXPECT_DOUBLE_EQ(n[0], 1.0);
    EXPECT_DOUBLE_EQ(n[1], 0.0);
    const auto& p = d.disk_points();
    const auto& s = d.sample_points();
    ASSERT_EQ(p.size(), s.size());
    for (int k = 0; k < d.num_points(); ++k) {
        SCOPED_TRACE("point " + std::to_string(k));
        // every disk point lies in the plane x = 750 within the disk, centred at (750, 600, 160)
        EXPECT_NEAR(p[3*k], 750.0, abs_tol);
        const Real r = std::hypot(p[3*k+1] - 600.0, p[3*k+2] - 160.0);
        EXPECT_LT(r, 120.0);
        EXPECT_GT(r, 0.0);
        // the sampling point is the same point one diameter (240 m) upstream
        EXPECT_NEAR(s[3*k], 750.0 - 240.0, abs_tol);
        EXPECT_NEAR(s[3*k+1], p[3*k+1], abs_tol);
        EXPECT_NEAR(s[3*k+2], p[3*k+2], abs_tol);
    }
}

TEST(PrescribedCtDisk, ForceIsMinusThrustAlongTheNormalFromTheUpstreamSpeed)
{
    erf_actuator::PrescribedCtDisk d(disk_inputs(0.0));
    const int n = d.num_points();
    // upstream 10 m/s along +x, disk 8 m/s: the thrust uses the upstream speed, the power the disk speed
    std::vector<Real> vs(3 * n, 0.0), vd(3 * n, 0.0);
    for (int k = 0; k < n; ++k) { vs[3*k] = 10.0; vd[3*k] = 8.0; vd[3*k+1] = 1.0; }
    d.update(vs, vd);
    const Real T = 0.5 * 1.2 * 0.75 * 100.0 * pi * 120.0 * 120.0;
    EXPECT_NEAR(d.free_stream_speed(), 10.0, rel_tol * 10.0);
    EXPECT_NEAR(d.disk_speed(), 8.0, rel_tol * 8.0);          // only the normal component counts
    EXPECT_NEAR(d.thrust(), T, rel_tol * T);
    EXPECT_NEAR(d.power(), T * 8.0, rel_tol * T * 8.0);
    std::array<Real,3> sum{{0.0, 0.0, 0.0}};
    for (int k = 0; k < n; ++k) { for (int c = 0; c < 3; ++c) { sum[c] += d.forces()[3*k+c]; } }
    EXPECT_NEAR(sum[0], -T, rel_tol * T);
    EXPECT_NEAR(sum[1], 0.0, rel_tol * T);
    EXPECT_NEAR(sum[2], 0.0, rel_tol * T);
}

TEST(PrescribedCtDisk, YawTurnsTheNormalAndTheForce)
{
    erf_actuator::PrescribedCtDisk d(disk_inputs(30.0));
    const auto nrm = d.normal();
    EXPECT_NEAR(nrm[0], std::cos(pi / 6.0), rel_tol);
    EXPECT_NEAR(nrm[1], std::sin(pi / 6.0), rel_tol);
    const int n = d.num_points();
    // a flow of 10 m/s along the normal: the same thrust as unyawed, turned with the normal
    std::vector<Real> v(3 * n, 0.0);
    for (int k = 0; k < n; ++k) { v[3*k] = 10.0 * nrm[0]; v[3*k+1] = 10.0 * nrm[1]; }
    d.update(v, v);
    const Real T = 0.5 * 1.2 * 0.75 * 100.0 * pi * 120.0 * 120.0;
    EXPECT_NEAR(d.free_stream_speed(), 10.0, rel_tol * 10.0);
    std::array<Real,3> sum{{0.0, 0.0, 0.0}};
    for (int k = 0; k < n; ++k) { for (int c = 0; c < 3; ++c) { sum[c] += d.forces()[3*k+c]; } }
    EXPECT_NEAR(sum[0], -T * nrm[0], rel_tol * T);
    EXPECT_NEAR(sum[1], -T * nrm[1], rel_tol * T);
    EXPECT_NEAR(sum[2], 0.0, rel_tol * T);
    // the sampling points are upstream along the yawed normal
    EXPECT_NEAR(d.sample_points()[0], d.disk_points()[0] - 240.0 * nrm[0], abs_tol);
    EXPECT_NEAR(d.sample_points()[1], d.disk_points()[1] - 240.0 * nrm[1], abs_tol);
}

TEST(MovingBodiesInputs, ReadsACtDiskBlockWithDefaults)
{
    {
        amrex::ParmParse pp("erf.moving_bodies");
        pp.addarr("bodies", std::vector<std::string>{"DA"});
    }
    {
        amrex::ParmParse pa("erf.moving_bodies.DA");
        pa.add("type", std::string("ct_disk"));
        pa.addarr("base_pos", std::vector<Real>{750.0, 600.0, 0.0});
        pa.add("rotor_radius", 120.0);
        pa.add("hub_height", 150.0);
        pa.add("ct", 0.75);
        pa.add("epsilon", 3.0);
    }
    const MovingBodiesInputs in = MovingBodiesInputs::read();
    ASSERT_EQ(in.bodies.size(), 1u);
    const MovingBodyInputs& b = in.bodies[0];
    EXPECT_EQ(b.type, "ct_disk");
    EXPECT_EQ(b.rotor_radius, static_cast<Real>(120.0));
    EXPECT_EQ(b.hub_height, static_cast<Real>(150.0));
    EXPECT_EQ(b.ct, static_cast<Real>(0.75));
    EXPECT_EQ(b.epsilon, static_cast<Real>(3.0));
    EXPECT_EQ(b.yaw_deg, static_cast<Real>(0.0));
    EXPECT_EQ(b.num_points_r, 8);
    EXPECT_EQ(b.num_points_t, 16);
    EXPECT_EQ(b.sample_diameters_upstream, static_cast<Real>(1.0));
    EXPECT_EQ(b.air_density, static_cast<Real>(1.225));
    EXPECT_EQ(b.output_root, "moving_bodies/DA");
}
