// Contract of erf_actuator::sample_velocity: each velocity component is interpolated from its
// own staggered grid, linearly in the physical height, so a field linear in x, y and physical
// z is reproduced exactly on a uniform-dz mesh, on a stretched mesh (z_phys_nd with equal
// columns) and on a terrain-following mesh (z_phys_nd varying with x and y), at points inside
// the cells, on faces, near the ground and near the top; and every point is found by exactly
// one box, however the domain is split.

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_RealBox.H>

#include <gtest/gtest.h>

#include "ERF_ActuatorSampling.H"

namespace {

using amrex::Real;

// The linear field: (u,v,w) = c0 + cx x + cy y + cz z_phys, different coefficients per component
struct LinearField {
    std::array<Real,4> u{{ 3.0,  0.010, -0.020, 0.030}};
    std::array<Real,4> v{{-1.0, -0.005,  0.015, 0.010}};
    std::array<Real,4> w{{ 0.5,  0.002,  0.003, -0.004}};
    static Real eval (const std::array<Real,4>& c, Real x, Real y, Real z) { return c[0] + c[1]*x + c[2]*y + c[3]*z; }
};

// BTF-style terrain-following heights: z(x,y,eta) = h(x,y) + (H - h(x,y)) * eta, with eta the
// stretched nominal level fraction; h = 0 gives a flat stretched mesh
struct MeshSpec {
    int nx = 12, ny = 10, nz = 8;
    Real Lx = 1200.0, Ly = 1000.0, H = 400.0;
    Real stretch = 1.15;         // 1.0: uniform nominal levels
    Real hill = 0.0;             // terrain height amplitude (m); 0: flat
    std::array<int,3> max_grid{{1024, 1024, 1024}};

    Real dx () const { return Lx / nx; }
    Real dy () const { return Ly / ny; }
    Real h (Real x, Real y) const
    {
        constexpr Real pi = 3.14159265358979323846;
        return hill * (0.5 + 0.5 * std::cos(2.0 * pi * x / Lx)) * (0.5 + 0.5 * std::sin(2.0 * pi * y / Ly));
    }
    Real eta (int k) const
    {
        if (stretch == 1.0) { return Real(k) / nz; }
        return (std::pow(stretch, k) - 1.0) / (std::pow(stretch, nz) - 1.0);
    }
    Real z_node (int i, int j, int k) const
    {
        const Real x = i * dx(), y = j * dy();
        const Real hs = h(x, y);
        return hs + (H - hs) * eta(k);
    }
};

struct Fields {
    amrex::Geometry geom;
    amrex::BoxArray ba;
    amrex::DistributionMapping dm;
    amrex::MultiFab u, v, w;
    std::unique_ptr<amrex::MultiFab> znd;
    MeshSpec spec;
    LinearField lin;

    // build the mesh and fill the face velocities with the linear field at the face centres;
    // uniform_dz leaves z_phys_nd null so the sampler takes the nominal heights
    Fields (const MeshSpec& m, bool uniform_dz)
        : spec(m)
    {
        const amrex::Box domain(amrex::IntVect(0, 0, 0), amrex::IntVect(m.nx-1, m.ny-1, m.nz-1));
        const amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)}, {AMREX_D_DECL(m.Lx, m.Ly, m.H)});
        const std::array<int,3> periodic{{0, 0, 0}};
        geom = amrex::Geometry(domain, &rb, 0, periodic.data());
        ba = amrex::BoxArray(domain);
        ba.maxSize(amrex::IntVect(m.max_grid[0], m.max_grid[1], m.max_grid[2]));
        dm = amrex::DistributionMapping(ba);
        const int ng = 2;
        u.define(amrex::convert(ba, amrex::IntVect(1,0,0)), dm, 1, ng);
        v.define(amrex::convert(ba, amrex::IntVect(0,1,0)), dm, 1, ng);
        w.define(amrex::convert(ba, amrex::IntVect(0,0,1)), dm, 1, ng);
        if (!uniform_dz) {
            znd = std::make_unique<amrex::MultiFab>(amrex::convert(ba, amrex::IntVect(1,1,1)), dm, 1, ng);
        }
        const Real dz = m.H / m.nz;
        auto node_z = [&](int i, int j, int k) { return uniform_dz ? k * dz : m.z_node(i, j, k); };
        for (amrex::MFIter mfi(u, false); mfi.isValid(); ++mfi) {
            // fill valid and ghost cells alike: the sampler reads one ghost row across box faces
            auto ua = u.array(mfi); auto va = v.array(mfi); auto wa = w.array(mfi);
            amrex::Array4<Real> za;
            if (znd) { za = znd->array(mfi); }
            const amrex::Box gbx = amrex::grow(mfi.validbox(), ng);   // x-face box grown
            const amrex::Box cbx = amrex::enclosedCells(gbx);
            amrex::LoopOnCpu(amrex::surroundingNodes(cbx), [&](int i, int j, int k) {
                if (za) { za(i,j,k) = node_z(i, j, k); }
            });
            amrex::LoopOnCpu(amrex::convert(cbx, amrex::IntVect(1,0,0)), [&](int i, int j, int k) {
                const Real x = i * m.dx(), y = (j + 0.5) * m.dy();
                const Real z = 0.25 * (node_z(i,j,k) + node_z(i,j+1,k) + node_z(i,j,k+1) + node_z(i,j+1,k+1));
                ua(i,j,k) = LinearField::eval(lin.u, x, y, z);
            });
            amrex::LoopOnCpu(amrex::convert(cbx, amrex::IntVect(0,1,0)), [&](int i, int j, int k) {
                const Real x = (i + 0.5) * m.dx(), y = j * m.dy();
                const Real z = 0.25 * (node_z(i,j,k) + node_z(i+1,j,k) + node_z(i,j,k+1) + node_z(i+1,j,k+1));
                va(i,j,k) = LinearField::eval(lin.v, x, y, z);
            });
            amrex::LoopOnCpu(amrex::convert(cbx, amrex::IntVect(0,0,1)), [&](int i, int j, int k) {
                const Real x = (i + 0.5) * m.dx(), y = (j + 0.5) * m.dy();
                const Real z = 0.25 * (node_z(i,j,k) + node_z(i+1,j,k) + node_z(i,j+1,k) + node_z(i+1,j+1,k));
                wa(i,j,k) = LinearField::eval(lin.w, x, y, z);
            });
        }
    }

    // sample the points and check every component against the analytic field
    void check (const std::vector<Real>& pos, Real tol) const
    {
        std::vector<Real> vel;
        erf_actuator::sample_velocity(u, v, w, znd.get(), geom, pos, vel);
        ASSERT_EQ(vel.size(), pos.size());
        for (std::size_t p = 0; p < pos.size() / 3; ++p) {
            SCOPED_TRACE("point " + std::to_string(p) + " at (" + std::to_string(pos[3*p]) + ", " +
                         std::to_string(pos[3*p+1]) + ", " + std::to_string(pos[3*p+2]) + ")");
            const Real x = pos[3*p], y = pos[3*p+1], z = pos[3*p+2];
            const Real eu = LinearField::eval(lin.u, x, y, z);
            const Real ev = LinearField::eval(lin.v, x, y, z);
            const Real ew = LinearField::eval(lin.w, x, y, z);
            EXPECT_NEAR(vel[3*p],   eu, tol * std::max(Real(1.0), std::abs(eu)));
            EXPECT_NEAR(vel[3*p+1], ev, tol * std::max(Real(1.0), std::abs(ev)));
            EXPECT_NEAR(vel[3*p+2], ew, tol * std::max(Real(1.0), std::abs(ew)));
        }
    }
};

// Points inside cells, on cell faces, on the terrain-following mesh's lowest and highest
// centres, and at the hub-height of a rotor: all inside the domain.
std::vector<Real> test_points (const MeshSpec& m)
{
    std::vector<Real> pos;
    auto add = [&](Real x, Real y, Real z) { pos.push_back(x); pos.push_back(y); pos.push_back(z); };
    add(0.37 * m.Lx, 0.61 * m.Ly, 0.55 * m.H);   // interior, generic
    add(0.50 * m.Lx, 0.50 * m.Ly, 0.50 * m.H);   // on x and y cell faces
    add(3.0 * m.dx(), 2.0 * m.dy(), 0.25 * m.H); // on a cell corner column
    add(0.12 * m.Lx, 0.90 * m.Ly, 0.02 * m.H + m.h(0.12 * m.Lx, 0.90 * m.Ly) * 0.98); // just above the ground
    add(0.81 * m.Lx, 0.23 * m.Ly, 0.97 * m.H);   // near the top
    add(0.66 * m.Lx, 0.44 * m.Ly, 0.375 * m.H);  // rotor hub height
    add(0.66 * m.Lx, 0.44 * m.Ly + 0.3 * m.H, 0.375 * m.H); // rotor tip in y
    return pos;
}

// relative tolerance: each interpolation weight is exact for a linear field, so only the
// roundoff of a few dozen operations on the face values remains: far below 1e-9 in double,
// and a few units of 1e-7 in single precision, where 2e-4 leaves a wide margin
constexpr Real tol = (sizeof(Real) == 8) ? Real(1.0e-9) : Real(2.0e-4);

} // namespace

TEST(ActuatorSampling, ExactForLinearFieldOnUniformMesh)
{
    MeshSpec m;
    Fields f(m, true);
    f.check(test_points(m), tol);
}

TEST(ActuatorSampling, ExactForLinearFieldOnStretchedMesh)
{
    MeshSpec m;
    m.stretch = 1.15;
    Fields f(m, false);
    f.check(test_points(m), tol);
}

TEST(ActuatorSampling, ExactForLinearFieldOverTerrain)
{
    MeshSpec m;
    m.stretch = 1.1;
    m.hill = 60.0;
    Fields f(m, false);
    f.check(test_points(m), tol);
}

// The same points sampled with the domain in one box and split into several boxes agree with
// the analytic field, so a point on or near a box face is found once and read through the
// ghost cells rather than from a neighbour's valid data.
TEST(ActuatorSampling, IndependentOfBoxDecomposition)
{
    MeshSpec m;
    m.hill = 60.0;
    m.max_grid = {{4, 5, 1024}};     // 3 x 2 uneven boxes, never split in z
    Fields f(m, false);
    std::vector<Real> pos = test_points(m);
    // points exactly on the box faces x = 4 dx and y = 5 dy
    auto add = [&](Real x, Real y, Real z) { pos.push_back(x); pos.push_back(y); pos.push_back(z); };
    add(4 * m.dx(), 5 * m.dy(), Real(0.5) * m.H);
    add(4 * m.dx(), Real(0.31) * m.Ly, Real(0.3) * m.H);
    add(Real(0.73) * m.Lx, 5 * m.dy(), Real(0.7) * m.H);
    f.check(pos, tol);
}

TEST(ActuatorSampling, EmptyPointListIsAllowed)
{
    MeshSpec m;
    Fields f(m, true);
    std::vector<Real> vel{1.0, 2.0, 3.0};
    erf_actuator::sample_velocity(f.u, f.v, f.w, nullptr, f.geom, {}, vel);
    EXPECT_TRUE(vel.empty());
}
