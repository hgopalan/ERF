// Contract of the actuator sampler (Core/ERF_ActuatorSampling).
//
// - sample_velocity: each velocity component is interpolated from its own staggered grid,
//   linearly in the physical height, so a field linear in x, y and physical z is reproduced
//   exactly on a uniform-dz mesh, on a stretched mesh (z_phys_nd with equal columns) and on a
//   terrain-following mesh (z_phys_nd varying with x and y), at points inside the cells, on faces,
//   near the ground and near the top; every point is found by exactly one box, however the
//   domain is split, in z too; a point outside the domain aborts, naming it.
//   A point on the domain's upper x or y face belongs to the last cell; in a periodic direction a
//   point beyond the domain is sampled where its image lies inside (sample_velocity, sample_cell_scalar
//   and terrain_heights alike).
// - points_covered_by: the reach counts, and wraps across a periodic seam; on a terrain or stretched mesh
//   (CoverZ::Footprint) the boxes' footprints must hold the reach across, at any height, and for the
//   spreading (CoverZ::Column) the grids must hold it in every cell the heights within reach may lie in
//   (from mesh_z_bounds, exact on a flat uniform mesh), or over the whole height without the bounds.
// - wrap_periodic: a point a few ulps below a non-zero prob_lo lands on prob_lo, not on prob_hi.
// - terrain_heights: the bilinear k = 0 node surface.
// - min_cell_heights: the shortest physical cell (z_phys_nd node spacing) under a region, over the
//   region's heights only, across periodic seams.
// - tip_cell_heights: the cell height that limits a blade tip, each cell weighed by the tip speed that can cross it.
// - tip_cell_heights_batched: the same heights as tip_cell_heights, every region in one reduction, whatever
//   the regions (empty, non-finite, zero radius, across a periodic seam, beyond the domain) and their order.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_BoxList.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_RealBox.H>

#include <gtest/gtest.h>

#include "ERF_ActuatorSampling.H"
#include "ERF_GTestThrowOnAbort.H"

namespace {

using amrex::Real;

// The linear field: (u,v,w) = c0 + cx x + cy y + cz z_phys, different coefficients per component
struct LinearField {
    std::array<Real,4> u{{ 3.0,  0.010, -0.020, 0.030}};
    std::array<Real,4> v{{-1.0, -0.005,  0.015, 0.010}};
    std::array<Real,4> w{{ 0.5,  0.002,  0.003, -0.004}};
    static Real eval (const std::array<Real,4>& c, Real x, Real y, Real z) { return c[0] + c[1]*x + c[2]*y + c[3]*z; }
};

// basic terrain-following (BTF) heights: z(x,y,eta) = h(x,y) + (H - h(x,y)) * eta, with eta the
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
    m.max_grid = {{4, 5, 1024}};     // 3 x 2 boxes of 4 x 5 cells, never split in z
    Fields f(m, false);
    std::vector<Real> pos = test_points(m);
    // points exactly on the box faces x = 4 dx and y = 5 dy
    auto add = [&](Real x, Real y, Real z) { pos.push_back(x); pos.push_back(y); pos.push_back(z); };
    add(4 * m.dx(), 5 * m.dy(), Real(0.5) * m.H);
    add(4 * m.dx(), Real(0.31) * m.Ly, Real(0.3) * m.H);
    add(Real(0.73) * m.Lx, 5 * m.dy(), Real(0.7) * m.H);
    f.check(pos, tol);
    // split in z as well (boxes of 3, 3 and 2 cells: a fine level stopping below the top is the lowest of
    // them): each box searches only its own column and first ghost layer, and the points on and near the
    // boxes' z faces sample as in one whole column
    MeshSpec mz = m;
    mz.max_grid = {{4, 5, 3}};
    Fields fz(mz, false);
    for (int k : {3, 6}) {
        const Real zf = mz.z_node(5, 5, k);
        add(5.5 * mz.dx(), 5.5 * mz.dy(), zf);
        add(5.5 * mz.dx(), 5.5 * mz.dy(), zf - Real(0.01) * mz.H);
        add(5.5 * mz.dx(), 5.5 * mz.dy(), zf + Real(0.01) * mz.H);
    }
    fz.check(pos, tol);
}

// The local sampler with buffers kept from call to call: one kernel per box, the same answers as with
// its own buffers, and a call with fewer points after one with more (the buffers keep their size) gets
// zeros and no counts at the points its boxes do not hold, not the previous call's answers
TEST(ActuatorSampling, KeptBuffersGiveTheSameAnswersAndOneKernelPerBox)
{
    MeshSpec m;
    m.hill = 60.0;
    m.max_grid = {{4, 5, 1024}};
    Fields f(m, false);
    const std::vector<Real> many = test_points(m);
    // the first two points: the second call reads only the start of the buffers the first one filled
    const std::vector<Real> few(many.begin(), many.begin() + 6);
    erf_actuator::ActuatorDeviceBuffers buf;
    int nlocal = 0;
    for (amrex::MFIter mfi(f.u, false); mfi.isValid(); ++mfi) { ++nlocal; }
    for (const std::vector<Real>* pts : {&many, &few, &many}) {
        std::vector<Real> vel_kept, vel_own;
        std::vector<int> cnt_kept, cnt_own;
        const long before = buf.kernel_launches;
        erf_actuator::sample_velocity_local(f.u, f.v, f.w, f.znd.get(), f.geom, *pts, vel_kept, cnt_kept, &buf);
        EXPECT_EQ(buf.kernel_launches - before, nlocal);
        erf_actuator::sample_velocity_local(f.u, f.v, f.w, f.znd.get(), f.geom, *pts, vel_own, cnt_own);
        EXPECT_EQ(vel_kept, vel_own);
        EXPECT_EQ(cnt_kept, cnt_own);
        EXPECT_GE(buf.sample_vel.device.size(), many.size());
        EXPECT_GE(buf.sample_cnt.device.size(), many.size() / 3);
    }
}

TEST(ActuatorSampling, AnEmptyPointListGivesNoVelocities)
{
    MeshSpec m;
    Fields f(m, true);
    std::vector<Real> vel{1.0, 2.0, 3.0};
    erf_actuator::sample_velocity(f.u, f.v, f.w, nullptr, f.geom, {}, vel);
    EXPECT_TRUE(vel.empty());
}

// On a refined level the grids are not the whole domain: a point is usable only if the cells it
// and its reach need are all on the level.
TEST(ActuatorSampling, CoverageByALevelIncludesTheReach)
{
    // a 3000 x 1200 x 600 m domain on 50 m cells, periodic in x and y; the level covers only
    // x in [500, 1500), y in [300, 900), z in [0, 450) as two boxes
    amrex::Box domain(amrex::IntVect(0, 0, 0), amrex::IntVect(59, 23, 11));
    amrex::RealBox rb({0.0, 0.0, 0.0}, {3000.0, 1200.0, 600.0});
    amrex::Geometry geom(domain, rb, 0, {1, 1, 0});
    amrex::BoxList bl;
    bl.push_back(amrex::Box(amrex::IntVect(10, 6, 0), amrex::IntVect(19, 17, 8)));
    bl.push_back(amrex::Box(amrex::IntVect(20, 6, 0), amrex::IntVect(29, 17, 8)));
    amrex::BoxArray ba(bl);
    std::string outside;
    // a point well inside with a 150 m reach: covered
    EXPECT_TRUE(erf_actuator::points_covered_by(ba, geom, {750.0, 600.0, 150.0}, 150.0, outside));
    EXPECT_TRUE(outside.empty());
    // the same point with a reach that crosses the level's x edge (500 m): not covered, named
    EXPECT_FALSE(erf_actuator::points_covered_by(ba, geom, {750.0, 600.0, 150.0}, 300.0, outside));
    EXPECT_NE(outside.find("750"), std::string::npos);
    // a point across the two boxes' shared face: the union covers it
    EXPECT_TRUE(erf_actuator::points_covered_by(ba, geom, {1000.0, 600.0, 150.0}, 100.0, outside));
    // a point outside the level, and one above its top
    EXPECT_FALSE(erf_actuator::points_covered_by(ba, geom, {2000.0, 600.0, 150.0}, 0.0, outside));
    EXPECT_FALSE(erf_actuator::points_covered_by(ba, geom, {750.0, 600.0, 500.0}, 0.0, outside));
    // several points: the first outside is the one named
    EXPECT_FALSE(erf_actuator::points_covered_by(ba, geom, {750.0, 600.0, 150.0, 2500.0, 600.0, 150.0, 2600.0, 600.0, 150.0}, 0.0, outside));
    EXPECT_NE(outside.find("2500"), std::string::npos);
    // a level covering the whole domain covers everything, reach or not
    amrex::BoxArray whole(domain);
    EXPECT_TRUE(erf_actuator::points_covered_by(whole, geom, {10.0, 10.0, 10.0, 2990.0, 1190.0, 590.0}, 400.0, outside));
    // an empty point list is covered
    EXPECT_TRUE(erf_actuator::points_covered_by(ba, geom, {}, 100.0, outside));
}

// Whether the grids cover a point does not depend on how the level is cut into boxes: the same region as one
// box and as unequal, non-mirror boxes (at most 7 x 5 x 3 cells) gives the same answer for every point of a sweep
// across its edges and seams, every reach and every z rule
TEST(ActuatorSampling, CoverageIsIndependentOfTheBoxDecomposition)
{
    amrex::Box domain(amrex::IntVect(0, 0, 0), amrex::IntVect(59, 23, 11));
    amrex::RealBox rb({0.0, 0.0, 0.0}, {3000.0, 1200.0, 600.0});
    amrex::Geometry geom(domain, rb, 0, {1, 1, 0});
    const amrex::Box region(amrex::IntVect(10, 6, 0), amrex::IntVect(29, 17, 8));
    const amrex::BoxArray one(region);
    amrex::BoxArray cut(region);
    cut.maxSize(amrex::IntVect(7, 5, 3));
    ASSERT_GT(cut.size(), 20);
    int points = 0, covered = 0;
    for (const auto z : {erf_actuator::CoverZ::Reach, erf_actuator::CoverZ::Footprint, erf_actuator::CoverZ::Column}) {
        for (const amrex::Real reach : {amrex::Real(0.0), amrex::Real(60.0), amrex::Real(130.0)}) {
            for (amrex::Real x = 380.0; x < 1650.0; x += 37.0) {
                for (amrex::Real y = 190.0; y < 1010.0; y += 41.0) {
                    for (amrex::Real h = 5.0; h < 560.0; h += 53.0) {
                        std::string a, b;
                        const bool ca = erf_actuator::points_covered_by(one, geom, {x, y, h}, reach, a, z);
                        const bool cb = erf_actuator::points_covered_by(cut, geom, {x, y, h}, reach, b, z);
                        ASSERT_EQ(ca, cb) << "(" << x << ", " << y << ", " << h << ") reach " << reach << " rule " << static_cast<int>(z);
                        EXPECT_EQ(a, b);
                        ++points;
                        if (ca) { ++covered; }
                    }
                }
            }
        }
    }
    // the sweep holds points on both sides of the answer
    EXPECT_GT(covered, points / 10);
    EXPECT_LT(covered, points - points / 10);
}

// The level's boxes built once (CoverageGrids) and reused for every test of a check give the answer, and name
// the point, that building them afresh for each test gives: over a sweep across the level's edges and a
// periodic seam, with every reach and every z rule, on boxes cut unevenly
TEST(ActuatorSampling, CoverageOnGridsBuiltOnceIsCoverageOnGridsBuiltEachTime)
{
    amrex::Box domain(amrex::IntVect(0, 0, 0), amrex::IntVect(59, 23, 11));
    amrex::RealBox rb({0.0, 0.0, 0.0}, {3000.0, 1200.0, 600.0});
    amrex::Geometry geom(domain, rb, 0, {1, 1, 0});
    // a level across the x seam (cells 50..59 and 0..9) and an island, cut unevenly
    amrex::BoxList bl;
    bl.push_back(amrex::Box(amrex::IntVect(50, 4, 0), amrex::IntVect(59, 19, 8)));
    bl.push_back(amrex::Box(amrex::IntVect(0, 4, 0), amrex::IntVect(9, 19, 8)));
    bl.push_back(amrex::Box(amrex::IntVect(25, 8, 0), amrex::IntVect(34, 15, 11)));
    amrex::BoxArray ba(bl);
    ba.maxSize(amrex::IntVect(7, 5, 3));
    const erf_actuator::CoverageGrids grids(ba, geom);
    // the grids themselves: the cell-centred boxes, and each flattened onto the bottom layer, built here apart
    ASSERT_EQ(grids.cells.size(), ba.size());
    ASSERT_EQ(grids.footprints.size(), ba.size());
    for (int b = 0; b < static_cast<int>(ba.size()); ++b) {
        EXPECT_EQ(grids.cells[b], ba[b]);
        amrex::Box flat = ba[b];
        flat.setSmall(2, 0);
        flat.setBig(2, 0);
        EXPECT_EQ(grids.footprints[b], flat);
    }
    erf_actuator::ZBounds zb;
    zb.ground_lo = 0.0; zb.ground_hi = 0.0; zb.dz_lo = 50.0; zb.dz_hi = 50.0;
    int points = 0, covered = 0;
    for (const auto z : {erf_actuator::CoverZ::Reach, erf_actuator::CoverZ::Footprint, erf_actuator::CoverZ::Column}) {
        for (const amrex::Real reach : {amrex::Real(0.0), amrex::Real(60.0), amrex::Real(130.0)}) {
            for (amrex::Real x = -90.0; x < 3090.0; x += 71.0) {
                for (amrex::Real y = 110.0; y < 1110.0; y += 83.0) {
                    for (amrex::Real h = 5.0; h < 590.0; h += 97.0) {
                        const std::vector<amrex::Real> pos{x, y, h, 1500.0, 600.0, 300.0};
                        std::string a, b;
                        const bool once = erf_actuator::points_covered_by(grids, geom, pos, reach, a, z, &zb);
                        const bool fresh = erf_actuator::points_covered_by(ba, geom, pos, reach, b, z, &zb);
                        ASSERT_EQ(once, fresh) << "(" << x << ", " << y << ", " << h << ") reach " << reach << " rule " << static_cast<int>(z);
                        EXPECT_EQ(a, b);
                        ++points;
                        if (once) { ++covered; }
                    }
                }
            }
        }
    }
    // the sweep is not vacuous: some points are covered and some are not
    EXPECT_GT(covered, 0);
    EXPECT_LT(covered, points);
}


// In a periodic direction the cells a point needs wrap across the seam: a level that covers part
// of the periodic extent must also hold the cells on the far side of the seam
TEST(ActuatorSampling, CoverageWrapsAcrossAPeriodicSeam)
{
    // 3000 m in x on 50 m cells (60 cells), periodic in x and y
    amrex::Box domain(amrex::IntVect(0, 0, 0), amrex::IntVect(59, 23, 11));
    amrex::RealBox rb({0.0, 0.0, 0.0}, {3000.0, 1200.0, 600.0});
    amrex::Geometry geom(domain, rb, 0, {1, 1, 0});
    std::string outside;
    // the level holds the first half in x: a point 25 m from the seam with a 100 m reach needs the
    // cells 58 and 59 across it, which the level does not hold
    const amrex::BoxArray half(amrex::Box(amrex::IntVect(0, 0, 0), amrex::IntVect(29, 23, 11)));
    EXPECT_FALSE(erf_actuator::points_covered_by(half, geom, {25.0, 600.0, 150.0}, 100.0, outside));
    EXPECT_NE(outside.find("25"), std::string::npos) << outside;
    // the same point well inside the half: covered
    EXPECT_TRUE(erf_actuator::points_covered_by(half, geom, {750.0, 600.0, 150.0}, 100.0, outside));
    // a level holding both sides of the seam covers the point, from either side
    amrex::BoxList bl;
    bl.push_back(amrex::Box(amrex::IntVect(0, 0, 0), amrex::IntVect(9, 23, 11)));
    bl.push_back(amrex::Box(amrex::IntVect(50, 0, 0), amrex::IntVect(59, 23, 11)));
    const amrex::BoxArray seam(bl);
    EXPECT_TRUE(erf_actuator::points_covered_by(seam, geom, {25.0, 600.0, 150.0}, 100.0, outside));
    EXPECT_TRUE(erf_actuator::points_covered_by(seam, geom, {2975.0, 600.0, 150.0}, 100.0, outside));
    // a reach wider than the period needs the whole periodic extent
    EXPECT_FALSE(erf_actuator::points_covered_by(seam, geom, {25.0, 600.0, 150.0}, 4000.0, outside));
    // the wrap is in y too, and a non-periodic z is clamped to the domain
    const amrex::BoxArray low_y(amrex::Box(amrex::IntVect(0, 0, 0), amrex::IntVect(59, 11, 11)));
    EXPECT_FALSE(erf_actuator::points_covered_by(low_y, geom, {750.0, 25.0, 10.0}, 100.0, outside));
    EXPECT_TRUE(erf_actuator::points_covered_by(amrex::BoxArray(domain), geom, {750.0, 600.0, 10.0}, 100.0, outside));
}

// On a terrain-following or stretched mesh a point's k is not its height over dz: for the sampler a level
// whose grids stand over the point's column, at any height, covers it, and one beside it does not; for the
// spreading the grids must cover the whole height
TEST(ActuatorSampling, CoverageWithoutTheZCheckNeedsTheFootprints)
{
    amrex::Box domain(amrex::IntVect(0, 0, 0), amrex::IntVect(59, 23, 11));
    amrex::RealBox rb({0.0, 0.0, 0.0}, {3000.0, 1200.0, 600.0});
    amrex::Geometry geom(domain, rb, 0, {1, 1, 0});
    // two stacked boxes over x in [500, 1500), the upper one not reaching the ground, and one beside them
    amrex::BoxList bl;
    bl.push_back(amrex::Box(amrex::IntVect(10, 6, 0), amrex::IntVect(29, 17, 3)));
    bl.push_back(amrex::Box(amrex::IntVect(10, 6, 4), amrex::IntVect(29, 17, 8)));
    const amrex::BoxArray ba(bl);
    std::string outside;
    // 500 m up, above the level's top (450 m) by its nominal cells: refused with the z check, not without it
    using erf_actuator::CoverZ;
    EXPECT_FALSE(erf_actuator::points_covered_by(ba, geom, {750.0, 600.0, 500.0}, 100.0, outside));
    EXPECT_TRUE(erf_actuator::points_covered_by(ba, geom, {750.0, 600.0, 500.0}, 100.0, outside, CoverZ::Footprint));
    // across, the reach still counts
    EXPECT_FALSE(erf_actuator::points_covered_by(ba, geom, {750.0, 600.0, 500.0}, 300.0, outside, CoverZ::Footprint));
    // a box raised off the ground over the column covers it too
    const amrex::BoxArray raised(amrex::Box(amrex::IntVect(10, 6, 5), amrex::IntVect(29, 17, 8)));
    EXPECT_TRUE(erf_actuator::points_covered_by(raised, geom, {750.0, 600.0, 50.0}, 100.0, outside, CoverZ::Footprint));
    EXPECT_FALSE(erf_actuator::points_covered_by(raised, geom, {1750.0, 600.0, 50.0}, 100.0, outside, CoverZ::Footprint));
    // for the spreading neither covers it: the stacked boxes stop at 450 m, the raised one starts at 250 m
    EXPECT_FALSE(erf_actuator::points_covered_by(ba, geom, {750.0, 600.0, 100.0}, 100.0, outside, CoverZ::Column));
    EXPECT_FALSE(erf_actuator::points_covered_by(raised, geom, {750.0, 600.0, 300.0}, 100.0, outside, CoverZ::Column));
    // two boxes stacked over the whole height do
    amrex::BoxList full;
    full.push_back(amrex::Box(amrex::IntVect(10, 6, 0), amrex::IntVect(29, 17, 5)));
    full.push_back(amrex::Box(amrex::IntVect(10, 6, 6), amrex::IntVect(29, 17, 11)));
    EXPECT_TRUE(erf_actuator::points_covered_by(amrex::BoxArray(full), geom, {750.0, 600.0, 100.0}, 100.0, outside, CoverZ::Column));
    // with the mesh's bounds: on this flat 50 m mesh the cells within reach of 300 m (k 4..8) are on the stacked
    // boxes (to k 8), as with the reach in z, and those of 400 m (k 6..10) are not; with ground from 0 to 200 m and
    // levels 30 to 50 m apart the heights within reach of 300 m may lie in any of k 0..11, above the boxes' top
    erf_actuator::ZBounds flat;
    flat.ground_lo = 0.0; flat.ground_hi = 0.0; flat.dz_lo = 50.0; flat.dz_hi = 50.0;
    EXPECT_TRUE(erf_actuator::points_covered_by(ba, geom, {750.0, 600.0, 300.0}, 100.0, outside, CoverZ::Column, &flat));
    EXPECT_FALSE(erf_actuator::points_covered_by(ba, geom, {750.0, 600.0, 400.0}, 100.0, outside, CoverZ::Column, &flat));
    erf_actuator::ZBounds hilly;
    hilly.ground_lo = 0.0; hilly.ground_hi = 200.0; hilly.dz_lo = 30.0; hilly.dz_hi = 50.0;
    EXPECT_FALSE(erf_actuator::points_covered_by(ba, geom, {750.0, 600.0, 300.0}, 100.0, outside, CoverZ::Column, &hilly));
}

// mesh_z_bounds: the lowest and highest ground and the smallest and largest level spacing of a
// terrain-following mesh z = h + (H - h) k / nz over a ramp h = 4 i (m), found across boxes
TEST(ActuatorSampling, TheMeshsZBoundsAreItsGroundAndSpacingExtremes)
{
    const int nx = 8, nz = 10;
    const Real H = 500.0;
    const amrex::Box domain(amrex::IntVect(0, 0, 0), amrex::IntVect(nx - 1, 3, nz - 1));
    const amrex::RealBox rb({0.0, 0.0, 0.0}, {800.0, 400.0, 500.0});
    const amrex::Geometry geom(domain, rb, 0, {0, 0, 0});
    amrex::BoxArray ba(domain);
    ba.maxSize(4);
    const amrex::DistributionMapping dm(ba);
    amrex::MultiFab znd(amrex::convert(ba, amrex::IntVect(1, 1, 1)), dm, 1, 0);
    for (amrex::MFIter mfi(znd, false); mfi.isValid(); ++mfi) {
        auto z = znd.array(mfi);
        amrex::LoopOnCpu(mfi.validbox(), [&](int i, int j, int k) {
            const Real h = Real(4.0) * Real(i);
            z(i,j,k) = h + (H - h) * Real(k) / Real(nz);
            (void) j;
        });
    }
    const erf_actuator::ZBounds b = erf_actuator::mesh_z_bounds(znd, geom);
    EXPECT_NEAR(b.ground_lo, 0.0, 1.0e-4);
    EXPECT_NEAR(b.ground_hi, 4.0 * nx, 1.0e-4);
    EXPECT_NEAR(b.dz_lo, (H - Real(4.0 * nx)) / nz, 1.0e-3);
    EXPECT_NEAR(b.dz_hi, H / nz, 1.0e-3);
}

// min_cell_heights: the shortest physical cell under a region, from z_phys_nd's node spacing; over a
// hill the cells are shorter than the nominal dz, so a rotor over the summit counts more cells than
// one over the flat; the nominal dz on a uniform-dz mesh, for an empty region, and on every rank
TEST(ActuatorSampling, TheShortestCellUnderARegionIsPhysical)
{
    MeshSpec m; m.stretch = 1.0; m.hill = 100.0; m.max_grid = {{4, 5, 3}};   // split in z too
    Fields f(m, false);
    const Real dz0 = m.H / m.nz;
    // the reference: every column whose node lies in [ilo, ihi] x [jlo, jhi], every spacing overlapping [zl, zh]
    auto brute = [&](int ilo, int ihi, int jlo, int jhi, Real zl, Real zh) {
        Real best = 1.0e30;
        for (int i = std::max(ilo, 0); i <= std::min(ihi, m.nx); ++i) {
            for (int j = std::max(jlo, 0); j <= std::min(jhi, m.ny); ++j) {
                for (int k = 0; k < m.nz; ++k) {
                    const Real z0 = m.z_node(i, j, k), z1 = m.z_node(i, j, k + 1);
                    if (z1 > zl && z0 < zh) { best = std::min(best, z1 - z0); }
                }
            }
        }
        return best;
    };
    // the summit (the hill's top at x = 0, y = Ly / 4) and a low corner (x = Lx / 2, y = 3 Ly / 4)
    const Real r = 120.0, zc = 200.0;
    const std::vector<Real> regions{
        -r, Real(0.25) * m.Ly - r, zc - r, r, Real(0.25) * m.Ly + r, zc + r,
        Real(0.5) * m.Lx - r, Real(0.75) * m.Ly - r, zc - r, Real(0.5) * m.Lx + r, Real(0.75) * m.Ly + r, zc + r,
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
    std::vector<Real> dz;
    erf_actuator::min_cell_heights(f.znd.get(), f.geom, regions, dz);
    ASSERT_EQ(dz.size(), 3u);
    auto node = [](Real x, Real d) { return static_cast<int>(std::floor(x / d)); };
    const Real want0 = brute(node(-r, m.dx()), node(r, m.dx()) + 1, node(Real(0.25) * m.Ly - r, m.dy()),
                             node(Real(0.25) * m.Ly + r, m.dy()) + 1, zc - r, zc + r);
    const Real want1 = brute(node(Real(0.5) * m.Lx - r, m.dx()), node(Real(0.5) * m.Lx + r, m.dx()) + 1,
                             node(Real(0.75) * m.Ly - r, m.dy()), node(Real(0.75) * m.Ly + r, m.dy()) + 1, zc - r, zc + r);
    const Real tol = (sizeof(Real) == 8) ? Real(1.0e-9) : Real(1.0e-4);
    EXPECT_NEAR(dz[0], want0, tol);
    EXPECT_NEAR(dz[1], want1, tol);
    // closed form: the highest node column, x = 0 and y = 200 m (or 300 m), stands h = 100 (0.5 + 0.5 sin(0.4 pi))
    // = 97.553 m high, so its cells are (400 - h) / 8 = 37.80590 m, against the nominal 50 m
    EXPECT_NEAR(dz[0], Real(37.8058968), Real(1.0e-4));
    EXPECT_LT(dz[0], Real(0.8) * dz0);
    EXPECT_GT(dz[0], Real(0.0));
    EXPECT_GT(dz[1], dz[0] + Real(5.0)); // away from the summit the cells are taller
    EXPECT_DOUBLE_EQ(dz[2], dz0);        // an empty region
    // periodic in x: a region beyond the domain's lower x face finds the columns of its image near the
    // upper face (nodes nx - 3 to nx - 1); without the periodicity it lies outside and gets the nominal dz
    const std::array<int,3> periodic{{1, 0, 0}};
    const amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)}, {AMREX_D_DECL(m.Lx, m.Ly, m.H)});
    const amrex::Geometry pgeom(f.geom.Domain(), &rb, 0, periodic.data());
    const std::vector<Real> beyond{Real(-2.5) * m.dx(), Real(0.25) * m.Ly - r, zc - r,
                                   Real(-1.5) * m.dx(), Real(0.25) * m.Ly + r, zc + r};
    std::vector<Real> pdz;
    erf_actuator::min_cell_heights(f.znd.get(), pgeom, beyond, pdz);
    ASSERT_EQ(pdz.size(), 1u);
    EXPECT_NEAR(pdz[0], brute(m.nx - 3, m.nx - 1, node(Real(0.25) * m.Ly - r, m.dy()), node(Real(0.25) * m.Ly + r, m.dy()) + 1,
                              zc - r, zc + r), tol);
    EXPECT_LT(pdz[0], dz0);
    erf_actuator::min_cell_heights(f.znd.get(), f.geom, beyond, pdz);
    EXPECT_DOUBLE_EQ(pdz[0], dz0);
    // a flat uniform mesh as ERF builds it, z_phys_nd = k dz with an inexact dz (400 m / 7): dz to rounding
    MeshSpec mf = m; mf.hill = 0.0; mf.nz = 7;
    Fields flat(mf, false);
    const Real dzf = mf.H / mf.nz;
    erf_actuator::min_cell_heights(flat.znd.get(), flat.geom, regions, dz);
    ASSERT_EQ(dz.size(), 3u);
    for (const Real v : dz) { EXPECT_NEAR(v, dzf, Real(1.0e-4) * dzf); }
    // a null z_phys_nd (outside ERF): the nominal dz
    Fields nominal(m, true);
    erf_actuator::min_cell_heights(nominal.znd.get(), nominal.geom, regions, dz);
    for (const Real v : dz) { EXPECT_DOUBLE_EQ(v, dz0); }
}

// min_cell_heights on a stretched mesh, where a column's cells differ: only the spacings whose span
// overlaps the region's heights count (the bottom cells under a band high in the column do not), a
// spacing at the top of a z box counts, a region straddling a periodic seam takes the columns on both
// sides, a non-finite bound keeps the nominal dz, and a folded cell is not a height
TEST(ActuatorSampling, TheShortestCellCountsOnlyTheRegionsHeightsAndBothSidesOfASeam)
{
    MeshSpec m; m.hill = 100.0; m.max_grid = {{4, 5, 3}};   // stretch 1.15; z boxes of cells 0-2, 3-5, 6-7
    Fields f(m, false);
    const Real dz0 = m.H / m.nz;
    const Real tol = (sizeof(Real) == 8) ? Real(1.0e-9) : Real(1.0e-4);
    auto brute = [&](int ilo, int ihi, int jlo, int jhi, Real zl, Real zh) {
        Real best = 1.0e30;
        for (int i = std::max(ilo, 0); i <= std::min(ihi, m.nx); ++i) {
            for (int j = std::max(jlo, 0); j <= std::min(jhi, m.ny); ++j) {
                for (int k = 0; k < m.nz; ++k) {
                    const Real z0 = m.z_node(i, j, k), z1 = m.z_node(i, j, k + 1);
                    if (z1 > zl && z0 < zh) { best = std::min(best, z1 - z0); }
                }
            }
        }
        return best;
    };
    // a band high in the summit's columns (x = -1.5 dx .. 1.5 dx, y = 2 dy .. 3 dy: nodes -2 .. 2 and 2 .. 4,
    // clipped to i = 0 .. 2 without periodicity): the bottom cells, the shortest of the stretched column,
    // lie below it and must not count
    const Real zl = 250.0, zh = 300.0;
    const std::vector<Real> band{Real(-1.5) * m.dx(), Real(2.0) * m.dy(), zl, Real(1.5) * m.dx(), Real(3.0) * m.dy(), zh};
    std::vector<Real> dz;
    erf_actuator::min_cell_heights(f.znd.get(), f.geom, band, dz);
    ASSERT_EQ(dz.size(), 1u);
    const Real want_band = brute(0, 2, 2, 4, zl, zh);
    EXPECT_NEAR(dz[0], want_band, tol);
    EXPECT_GT(dz[0], brute(0, 2, 2, 4, Real(0.0), m.H) + Real(5.0));   // the whole column's shortest is far below
    // the spacing at the top of the lowest z box (node 2 to node 3) on a flat stretched mesh, alone in the band
    MeshSpec ms = m; ms.hill = 0.0;
    Fields flat(ms, false);
    const Real z2 = ms.z_node(0, 0, 2), z3 = ms.z_node(0, 0, 3);
    const std::vector<Real> top{Real(100.0), Real(100.0), z2 + Real(0.1), Real(500.0), Real(500.0), z3 - Real(0.1)};
    erf_actuator::min_cell_heights(flat.znd.get(), flat.geom, top, dz);
    EXPECT_NEAR(dz[0], z3 - z2, tol);
    EXPECT_GT(dz[0], ms.z_node(0, 0, 1) - ms.z_node(0, 0, 0) + Real(1.0));   // not the bottom cell's height
    // periodic in y: a region straddling y = 0 (nodes -1 .. 2) takes j = ny - 1, ny and j = 0 .. 2. The hill
    // rises from y = 0 towards y = Ly / 4 and falls below it near y = Ly, so the shortest cells are on the
    // y = 0 side; the columns near y = Ly alone give taller ones
    const std::array<int,3> periodic_y{{0, 1, 0}};
    const amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)}, {AMREX_D_DECL(m.Lx, m.Ly, m.H)});
    const amrex::Geometry pgeom(f.geom.Domain(), &rb, 0, periodic_y.data());
    const std::vector<Real> seam{Real(100.0), Real(-0.5) * m.dy(), Real(0.0), Real(300.0), Real(1.5) * m.dy(), m.H};
    erf_actuator::min_cell_heights(f.znd.get(), pgeom, seam, dz);
    const Real low_side = brute(1, 4, 0, 2, Real(0.0), m.H), high_side = brute(1, 4, m.ny - 1, m.ny, Real(0.0), m.H);
    EXPECT_NEAR(dz[0], std::min(low_side, high_side), tol);
    EXPECT_LT(low_side, high_side - Real(0.5));
    // a region wider than the periodic domain: every column
    const std::vector<Real> wide{Real(100.0), -m.Ly, Real(0.0), Real(300.0), Real(2.0) * m.Ly, m.H};
    erf_actuator::min_cell_heights(f.znd.get(), pgeom, wide, dz);
    EXPECT_NEAR(dz[0], brute(1, 4, 0, m.ny, Real(0.0), m.H), tol);
    // a huge but finite bound (a diverged hub position, 1e12 m): clamped a domain length out, so the
    // region covers every column across, as one exactly the domain wide does
    const std::vector<Real> huge{Real(-1.0e12), Real(100.0), Real(0.0), Real(1.0e12), Real(300.0), m.H};
    const std::vector<Real> whole{Real(0.0), Real(100.0), Real(0.0), m.Lx, Real(300.0), m.H};
    std::vector<Real> dzw;
    erf_actuator::min_cell_heights(f.znd.get(), f.geom, huge, dz);
    erf_actuator::min_cell_heights(f.znd.get(), f.geom, whole, dzw);
    EXPECT_EQ(dz[0], dzw[0]);
    EXPECT_LT(dz[0], dz0 - Real(5.0));
    // a non-finite bound (a diverged hub position): the nominal dz, no trap
    const std::vector<Real> bad{std::numeric_limits<Real>::quiet_NaN(), Real(0.0), Real(0.0), Real(100.0), Real(100.0), Real(100.0)};
    erf_actuator::min_cell_heights(f.znd.get(), f.geom, bad, dz);
    EXPECT_DOUBLE_EQ(dz[0], dz0);
    // a folded cell: node (1, 1, 1) lifted above node (1, 1, 2) on the flat stretched mesh. Its negative
    // spacing is skipped, so the region over columns 0 .. 2 still finds the other columns' bottom cell
    // (before, the negative minimum fell back to the nominal dz)
    for (amrex::MFIter mfi(*flat.znd, false); mfi.isValid(); ++mfi) {
        auto za = flat.znd->array(mfi);
        amrex::LoopOnCpu(amrex::grow(mfi.validbox(), 2), [&](int i, int j, int k) {
            if (i == 1 && j == 1 && k == 1) { za(i,j,k) = z2 + Real(1.0); }
        });
    }
    const std::vector<Real> fold{Real(0.5) * ms.dx(), Real(0.5) * ms.dy(), Real(0.0), Real(1.5) * ms.dx(), Real(1.5) * ms.dy(), ms.H};
    erf_actuator::min_cell_heights(flat.znd.get(), flat.geom, fold, dz);
    EXPECT_NEAR(dz[0], ms.z_node(0, 0, 1) - ms.z_node(0, 0, 0), tol);
    EXPECT_LT(dz[0], dz0 - Real(5.0));
}

// tip_cell_heights: each cell of the disk counts with the share of the tip speed that can cross it,
// sqrt(R^2 - d^2) / R for a cell whose nearest point is d from the hub height on a flat mesh, plus the
// slope G of its faces on a sloping one (at most sqrt(1 + G^2)). On a stretched column the short cells low
// in the disk count less than their height alone; the slack moves the hub height by up to that much; on a
// planar slope a cell low in the disk can limit through its slope; a zero radius keeps the nominal dz
TEST(ActuatorSampling, TheTipCellHeightWeighsEachCellByTheTipSpeedThatCrossesIt)
{
    MeshSpec ms; ms.max_grid = {{4, 5, 3}};   // flat, stretch 1.15: spacings 29.1 m at the ground to 77.5 m at the top
    Fields flat(ms, false);
    const Real dz0 = ms.H / ms.nz;
    const Real tol = (sizeof(Real) == 8) ? Real(1.0e-9) : Real(1.0e-4);
    // the reference, over one column (every column of the flat mesh is the same)
    auto brute = [&](Real zl, Real zh, Real zc, Real R, Real slack) {
        Real best = 1.0e30;
        for (int k = 0; k < ms.nz; ++k) {
            const Real z0 = ms.z_node(0, 0, k), z1 = ms.z_node(0, 0, k + 1);
            if (!(z1 > zl && z0 < zh)) { continue; }
            const Real gap = (zc < z0) ? z0 - zc : ((zc > z1) ? zc - z1 : Real(0.0));
            const Real d = std::max(gap - slack, Real(0.0));
            if (d >= R) { continue; }
            best = std::min(best, (z1 - z0) / (std::sqrt(R * R - d * d) / R));
        }
        return best;
    };
    const Real zc = 200.0, R = 120.0;
    const std::vector<Real> rows{100.0, 100.0, zc - R, 500.0, 500.0, zc + R, zc, R, 0.0};
    std::vector<Real> dz, plain;
    erf_actuator::tip_cell_heights(flat.znd.get(), flat.geom, rows, dz);
    ASSERT_EQ(dz.size(), 1u);
    EXPECT_NEAR(dz[0], brute(zc - R, zc + R, zc, R, Real(0.0)), tol);
    // hand values: the cell 145.5 .. 196.5 m (51.0 m, 3.5 m below the hub) gives 50.99 m; the one
    // 101.2 .. 145.5 m (44.3 m, 54.5 m below) gives 44.32 / 0.891 = 49.74 m, the limit
    EXPECT_NEAR(dz[0], Real(49.743), Real(0.005));
    const std::vector<Real> box(rows.begin(), rows.begin() + 6);
    erf_actuator::min_cell_heights(flat.znd.get(), flat.geom, box, plain);
    EXPECT_GT(dz[0], plain[0] + Real(10.0));   // the 38.5 m cell 99 m below the hub counts as 67.9 m
    // a slack of 60 m brings every cell 60 m nearer the hub height: the 38.5 m cell 98.8 m below now
    // counts as 38.54 / 0.946 = 40.73 m and is the limit
    const std::vector<Real> slack{100.0, 100.0, zc - R, 500.0, 500.0, zc + R, zc, R, 60.0};
    erf_actuator::tip_cell_heights(flat.znd.get(), flat.geom, slack, dz);
    EXPECT_NEAR(dz[0], brute(zc - R, zc + R, zc, R, Real(60.0)), tol);
    EXPECT_NEAR(dz[0], Real(40.727), Real(0.005));
    // a zero radius (no blades): the nominal dz
    const std::vector<Real> bare{100.0, 100.0, zc - R, 500.0, 500.0, zc + R, zc, 0.0, 0.0};
    erf_actuator::tip_cell_heights(flat.znd.get(), flat.geom, bare, dz);
    EXPECT_DOUBLE_EQ(dz[0], dz0);
    // a planar slope, ground h = 0.25 x under equal cells (z = h + (400 - h) k / 8): face k slopes as
    // 0.25 (1 - k / 8), so a tip moving sideways crosses it as well. The limit is the bottom cell of the
    // column x = 800 m (200 .. 225 m, 75 m below a hub at 300 m): 25 / (0.7806 + 0.25) = 24.257 m, where
    // the vertical speed alone (q) would give the 25 m of the cell at hub height
    MeshSpec mp = ms; mp.stretch = 1.0;
    Fields plane(mp, false);
    for (amrex::MFIter mfi(*plane.znd, false); mfi.isValid(); ++mfi) {
        auto za = plane.znd->array(mfi);
        amrex::LoopOnCpu(amrex::grow(mfi.validbox(), 2), [&](int i, int j, int k) {
            amrex::ignore_unused(j);
            const Real h = Real(0.25) * i * mp.dx();
            za(i,j,k) = h + (mp.H - h) * Real(k) / mp.nz;
        });
    }
    const std::vector<Real> prow{500.0, 300.0, 180.0, 700.0, 700.0, 420.0, 300.0, 120.0, 0.0};
    erf_actuator::tip_cell_heights(plane.znd.get(), plane.geom, prow, dz);
    EXPECT_NEAR(dz[0], Real(25.0) / (std::sqrt(Real(1.0) - Real(0.625) * Real(0.625)) + Real(0.25)), Real(1.0e-4));
    EXPECT_NEAR(dz[0], Real(24.2571), Real(1.0e-3));
    erf_actuator::min_cell_heights(plane.znd.get(), plane.geom, std::vector<Real>(prow.begin(), prow.begin() + 6), plain);
    EXPECT_NEAR(plain[0], Real(25.0), Real(1.0e-4));   // the plain vertical spacing is not weighted
    // a region at the domain's high x side (not periodic): the ghost nodes beyond it are not read, so a
    // fill value there (a fine level can leave one) changes nothing
    const std::vector<Real> side{1000.0, 300.0, 280.0, 1200.0, 700.0, 400.0, 340.0, 120.0, 0.0};
    std::vector<Real> clean;
    erf_actuator::tip_cell_heights(plane.znd.get(), plane.geom, side, clean);
    for (amrex::MFIter mfi(*plane.znd, false); mfi.isValid(); ++mfi) {
        auto za = plane.znd->array(mfi);
        amrex::LoopOnCpu(amrex::grow(mfi.validbox(), 2), [&](int i, int j, int k) {
            if (i < 0 || i > mp.nx) { za(i,j,k) = Real(1.0e30); }
        });
    }
    erf_actuator::tip_cell_heights(plane.znd.get(), plane.geom, side, dz);
    EXPECT_EQ(dz[0], clean[0]);
    EXPECT_GT(dz[0], Real(10.0));
}

// tip_cell_heights_batched: one reduction over every region gives each region the height the per-region
// search gives it, on a flat stretched mesh, over the hill, with periodic seams in x and y and boxes split
// in every direction, for regions empty, non-finite, without blades, straddling a seam or far outside, in
// any order and any subset (a cache searches only the rotors that moved)
TEST(ActuatorSampling, TheBatchedTipCellHeightsMatchThePerRegionSearch)
{
    // equal to the last bit on the host, where both take the same expression per cell; to rounding on a
    // device, which may contract the two kernels differently
    auto same = [](Real a, Real b) {
#ifdef AMREX_USE_GPU
        return a == b || std::abs(a - b) <= Real(8.0) * std::numeric_limits<Real>::epsilon() * std::max(std::abs(a), std::abs(b));
#else
        return a == b;
#endif
    };
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    for (const Real hill : {Real(0.0), Real(100.0)}) {
        for (const std::array<int,3>& periodic : {std::array<int,3>{{0, 0, 0}}, std::array<int,3>{{1, 1, 0}}}) {
            MeshSpec m; m.hill = hill; m.max_grid = {{4, 5, 3}};
            Fields f(m, false);
            const amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)}, {AMREX_D_DECL(m.Lx, m.Ly, m.H)});
            const amrex::Geometry geom(f.geom.Domain(), &rb, 0, periodic.data());
            // the hand-picked regions, then a sweep of rotors over the domain and beyond its sides, with
            // tilted, yawed and coned shapes (box widths, slack) that differ from row to row
            std::vector<Real> rows{
                100.0, 100.0, 80.0, 500.0, 500.0, 320.0, 200.0, 120.0, 0.0,               // the middle
                -150.0, 400.0, 30.0, 90.0, 640.0, 270.0, 150.0, 120.0, 4.0,               // across x = 0
                1000.0, 850.0, 20.0, 1300.0, 1150.0, 260.0, 140.0, 120.0, 0.0,            // across both upper sides
                1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,                              // empty: no line
                nan, 100.0, 80.0, 500.0, 500.0, 320.0, 200.0, 120.0, 0.0,                 // a diverged hub
                100.0, 100.0, 80.0, 500.0, 500.0, 320.0, 200.0, 0.0, 0.0,                 // no blades
                -1.0e12, 100.0, 0.0, 1.0e12, 300.0, 400.0, 200.0, 120.0, 0.0,             // huge but finite
                5000.0, 5000.0, 80.0, 5200.0, 5200.0, 320.0, 200.0, 120.0, 0.0,           // far outside
                300.0, 200.0, 350.0, 900.0, 800.0, 500.0, 420.0, 120.0, 10.0};            // above the top
            for (int s = 0; s < 24; ++s) {
                const Real cx = Real(-200.0) + Real(70.0) * s, cy = Real(-150.0) + Real(53.0) * s;
                const Real cz = Real(120.0) + Real(7.0) * (s % 9), rad = Real(60.0) + Real(5.0) * (s % 7);
                const Real hx = rad * Real(0.1) * (s % 5), hy = rad * (Real(1.0) - Real(0.05) * (s % 3));
                rows.insert(rows.end(), {cx - hx, cy - hy, cz - rad, cx + hx, cy + hy, cz + rad, cz, rad, Real(0.5) * (s % 4)});
            }
            const int n = static_cast<int>(rows.size() / 9);
            std::vector<Real> one, batched;
            erf_actuator::tip_cell_heights(f.znd.get(), geom, rows, one);
            erf_actuator::tip_cell_heights_batched(f.znd.get(), geom, rows, batched);
            ASSERT_EQ(batched.size(), one.size());
            int limited = 0;   // regions whose height is a cell's, not the nominal dz
            for (int r = 0; r < n; ++r) {
                EXPECT_TRUE(same(batched[r], one[r])) << "hill " << hill << " periodic " << periodic[0] << " region " << r
                                                      << ": batched " << batched[r] << ", per region " << one[r];
                if (one[r] != m.H / m.nz) { ++limited; }
            }
            EXPECT_GT(limited, n / 2);
            // any subset, in any order: a region's height does not depend on the others searched with it
            std::vector<int> pick;
            for (int r = n - 1; r >= 0; r -= 3) { pick.push_back(r); }
            std::vector<Real> sub, part;
            for (const int r : pick) { sub.insert(sub.end(), rows.begin() + 9 * r, rows.begin() + 9 * r + 9); }
            erf_actuator::tip_cell_heights_batched(f.znd.get(), geom, sub, part);
            ASSERT_EQ(part.size(), pick.size());
            for (std::size_t k = 0; k < pick.size(); ++k) { EXPECT_TRUE(same(part[k], one[pick[k]])) << "region " << pick[k]; }
        }
    }
    // no regions, and a null z_phys_nd (the nominal dz), as for the per-region search
    MeshSpec m;
    Fields nominal(m, true);
    std::vector<Real> dz{1.0};
    erf_actuator::tip_cell_heights_batched(nominal.znd.get(), nominal.geom, std::vector<Real>{}, dz);
    EXPECT_TRUE(dz.empty());
    const std::vector<Real> row{100.0, 100.0, 80.0, 500.0, 500.0, 320.0, 200.0, 120.0, 0.0};
    erf_actuator::tip_cell_heights_batched(nominal.znd.get(), nominal.geom, row, dz);
    ASSERT_EQ(dz.size(), 1u);
    EXPECT_DOUBLE_EQ(dz[0], m.H / m.nz);
}

// A point a few ulps below a non-zero prob_lo wraps to lo + (len - tiny), which rounds onto prob_hi, owned by
// no box: it lands on prob_lo instead
TEST(ActuatorSampling, AWrappedPointNeverLandsOnTheUpperFace)
{
    const Real lo = Real(-177.507), hi = Real(8056.149);
    const amrex::Box domain(amrex::IntVect(0, 0, 0), amrex::IntVect(63, 7, 7));
    const amrex::RealBox rb({lo, 0.0, 0.0}, {hi, 800.0, 800.0});
    const amrex::Geometry geom(domain, rb, 0, {1, 0, 0});
    const Real below = std::nextafter(static_cast<Real>(lo), static_cast<Real>(-1.0e30));
    const std::vector<Real> p = erf_actuator::wrap_periodic({below, 400.0, 400.0}, geom);
    EXPECT_LT(p[0], static_cast<Real>(geom.ProbHi(0))) << "wrapped onto the upper face";
    EXPECT_GE(p[0], static_cast<Real>(geom.ProbLo(0)));
    // every point a few ulps either side of the seam stays inside [lo, hi)
    Real x = static_cast<Real>(lo);
    for (int n = 0; n < 8; ++n) { x = std::nextafter(x, static_cast<Real>(-1.0e30)); }
    for (int n = 0; n < 16; ++n, x = std::nextafter(x, static_cast<Real>(1.0e30))) {
        const Real w = erf_actuator::wrap_periodic({x, 400.0, 400.0}, geom)[0];
        EXPECT_TRUE(w >= static_cast<Real>(geom.ProbLo(0)) && w < static_cast<Real>(geom.ProbHi(0))) << n << " " << w;
    }
}

// A point outside the domain is found by no box: the sampler aborts and names it
TEST(ActuatorSampling, TheUpperFaceIsTheLastCellAndPeriodicPointsTheirImage)
{
    MeshSpec m;
    m.stretch = 1.0;
    m.max_grid = {{5, 4, 1024}};
    Fields f(m, true);
    // on the domain's upper x and y faces (an end of a line placed on the boundary)
    f.check({m.Lx, Real(0.37) * m.Ly, Real(0.5) * m.H, Real(0.41) * m.Lx, m.Ly, Real(0.5) * m.H, m.Lx, m.Ly, Real(0.25) * m.H}, tol);
    // and on its top face (a vertical wake line that reaches the top), on the uniform and the stretched mesh,
    // with the column in one box and split in z
    const std::vector<Real> top{Real(0.41) * m.Lx, Real(0.37) * m.Ly, m.H, m.Lx, m.Ly, m.H};
    f.check(top, tol);
    {
        MeshSpec ms = m;
        ms.stretch = 1.15;
        Fields fs(ms, false);
        fs.check(top, tol);
        MeshSpec mz = ms;
        mz.max_grid = {{5, 4, 4}};
        Fields fz(mz, false);
        fz.check(top, tol);
        MeshSpec mu = m;
        mu.max_grid = {{5, 4, 4}};
        Fields fu(mu, true);
        fu.check(top, tol);
    }
    {
        // 1000 m on 88 cells: prob_hi / dx rounds one ulp above 88 in double, as 5000 m on 96 cells does in float
        MeshSpec m88 = m;
        m88.Lx = (sizeof(Real) == 8) ? 1000.0 : 5000.0;
        m88.nx = (sizeof(Real) == 8) ? 88 : 96;
        Fields f88(m88, true);
        f88.check({m88.Lx, Real(0.37) * m88.Ly, Real(0.5) * m88.H}, tol);
    }
    // periodic in x: a point a third of a cell past either seam takes its image's value
    const std::array<int,3> periodic{{1, 0, 0}};
    const amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)}, {AMREX_D_DECL(m.Lx, m.Ly, m.H)});
    f.geom = amrex::Geometry(f.geom.Domain(), &rb, 0, periodic.data());
    const Real past = m.dx() / 3.0;
    const std::vector<Real> outside{m.Lx + past, Real(0.37) * m.Ly, Real(0.5) * m.H, -past, Real(0.61) * m.Ly, Real(0.25) * m.H};
    const std::vector<Real> image{past, Real(0.37) * m.Ly, Real(0.5) * m.H, m.Lx - past, Real(0.61) * m.Ly, Real(0.25) * m.H};
    std::vector<Real> vel, ref;
    erf_actuator::sample_velocity(f.u, f.v, f.w, nullptr, f.geom, outside, vel);
    erf_actuator::sample_velocity(f.u, f.v, f.w, nullptr, f.geom, image, ref);
    ASSERT_EQ(vel.size(), ref.size());
    for (std::size_t i = 0; i < vel.size(); ++i) { EXPECT_NEAR(vel[i], ref[i], tol * std::max(Real(1.0), std::abs(ref[i]))) << i; }
    const auto wrapped = erf_actuator::wrap_periodic(outside, f.geom);
    for (std::size_t i = 0; i < image.size(); ++i) { EXPECT_NEAR(wrapped[i], image[i], 1.0e-4 * m.Lx) << i; }
}

TEST(ActuatorSampling, APointOutsideTheDomainIsRefusedNamingIt)
{
    MeshSpec m;
    Fields f(m, true);
    std::vector<Real> vel;
    const std::string msg = erf_gtest::abort_message([&] {
        erf_actuator::sample_velocity(f.u, f.v, f.w, nullptr, f.geom, {Real(5000.0), Real(500.0), Real(100.0)}, vel);
    });
    EXPECT_NE(msg.find("5000"), std::string::npos) << msg;
    EXPECT_NE(msg.find("sampled by 0 boxes"), std::string::npos) << msg;
}

// The terrain surface under a point: the k = 0 node plane of z_phys_nd, bilinear between the four
// nodes around (x, y); prob_lo z on a uniform-dz mesh
TEST(ActuatorSampling, TerrainHeightIsTheBilinearNodeSurface)
{
    MeshSpec m; m.hill = 80.0; m.max_grid = {{4, 5, 1024}};   // several boxes: each point has one owner
    Fields f(m, false);
    std::vector<Real> pos, expect;
    // at nodes: exact
    for (int i : {0, 3, 7, m.nx}) { for (int j : {0, 4, m.ny}) {
        pos.insert(pos.end(), {i * m.dx(), j * m.dy(), 999.0}); expect.push_back(m.z_node(i, j, 0));
    } }
    // at a cell centre and at general points: bilinear between the four nodes
    auto bilin = [&](Real x, Real y) {
        const int i = static_cast<int>(std::floor(x / m.dx())), j = static_cast<int>(std::floor(y / m.dy()));
        const Real wx = x / m.dx() - i, wy = y / m.dy() - j;
        return (1 - wy) * ((1 - wx) * m.z_node(i, j, 0) + wx * m.z_node(i+1, j, 0)) + wy * ((1 - wx) * m.z_node(i, j+1, 0) + wx * m.z_node(i+1, j+1, 0));
    };
    for (auto xy : {std::array<Real,2>{{250.0, 350.0}}, std::array<Real,2>{{612.5, 137.25}}, std::array<Real,2>{{1199.0, 999.0}}}) {
        pos.insert(pos.end(), {xy[0], xy[1], 0.0}); expect.push_back(bilin(xy[0], xy[1]));
    }
    std::vector<Real> h;
    erf_actuator::terrain_heights(f.znd.get(), f.geom, pos, h);
    ASSERT_EQ(h.size(), expect.size());
    // heights up to 80 m: 1e-9 m in double, a few float spacings (8e-6 m at 80 m) in single precision
    const Real htol = (sizeof(Real) == 8) ? Real(1.0e-9) : Real(5.0e-5);
    for (std::size_t p = 0; p < h.size(); ++p) { EXPECT_NEAR(h[p], expect[p], htol) << "point " << p; }
    EXPECT_GT(*std::max_element(h.begin(), h.end()), 40.0);   // the hill is really there
    // a uniform-dz mesh: the domain floor everywhere
    Fields flat(m, true);
    erf_actuator::terrain_heights(flat.znd.get(), flat.geom, pos, h);
    for (const Real v : h) { EXPECT_DOUBLE_EQ(v, 0.0); }
    // no points: nothing to do
    erf_actuator::terrain_heights(f.znd.get(), f.geom, {}, h);
    EXPECT_TRUE(h.empty());
}

// On a slope the ground under a point (bilinear between the four nodes of its cell) lies below the
// cell's bottom face (their mean) on one side: a point just above the ground there is sampled from
// the bottom cell, by the column's extrapolation, still exactly for a linear field, on any split
TEST(ActuatorSampling, APointJustAboveSlopingGroundIsSampled)
{
    MeshSpec m; m.stretch = 1.1; m.hill = 120.0; m.max_grid = {{4, 5, 1024}};
    Fields f(m, false);
    std::vector<Real> xy;
    for (int i = 0; i < 40; ++i) { for (int j = 0; j < 30; ++j) { xy.insert(xy.end(), {Real((i + 0.37) * m.Lx / 40), Real((j + 0.61) * m.Ly / 30), Real(0.0)}); } }
    std::vector<Real> h;
    erf_actuator::terrain_heights(f.znd.get(), f.geom, xy, h);
    std::vector<Real> pos;
    int below_face = 0;
    for (std::size_t p = 0; p < h.size(); ++p) {
        const Real x = xy[3*p], y = xy[3*p+1];
        pos.insert(pos.end(), {x, y, h[p] + Real(0.5)});
        const int i = static_cast<int>(std::floor(x / m.dx())), j = static_cast<int>(std::floor(y / m.dy()));
        const Real face = 0.25 * (m.z_node(i, j, 0) + m.z_node(i+1, j, 0) + m.z_node(i, j+1, 0) + m.z_node(i+1, j+1, 0));
        if (h[p] + 0.5 < face) { ++below_face; }
    }
    EXPECT_GT(below_face, 50) << "the slope must put points below their cell's bottom face";
    f.check(pos, tol);
}
