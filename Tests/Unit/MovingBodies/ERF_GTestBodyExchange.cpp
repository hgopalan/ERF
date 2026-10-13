// Contract of erf_actuator::BodyExchange and its helpers (Core/ERF_BodyExchange):
//
// - domain_images, cells_within, ranks_touching: a range of cells wraps across a periodic seam and
//   is clipped at a non-periodic side; a fitted mesh takes the whole column.
// - choose_owners: a body's owner is a rank whose boxes overlap its footprint (not rank i modulo the
//   rank count) as long as one of them is among the least loaded ranks whose bodies share its time
//   step (then the most cells, then the lowest rank); otherwise the nearest least loaded such rank, so
//   that the bodies stay one per rank while there are as many ranks; a footprint no box overlaps goes
//   to the nearest box.
// - exchange_buffers: each rank gets what was sent to it, sizes unknown beforehand.
// - sample_velocity and spread_forces: the owners' points sampled and spread through point-to-point
//   messages give exactly what the global sampler and spreader give over all the points (bit for bit
//   on one and two ranks), on a uniform and on a terrain-following mesh with unequal boxes, a body's
//   owner need not take part in its exchanges, and a rank whose boxes no kernel reaches receives no
//   point at all; a body whose points leave its envelope gets a larger one, and the results do not change.
// - a point outside the domain aborts on its owner, naming it, as the global sampler does.
//
// Compiled into erf_unit_tests (one rank) and erf_parallel_tests (1, 2 and 4 ranks).

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
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
#include "ERF_ActuatorSpreading.H"
#include "ERF_BodyExchange.H"
#include "ERF_GTestThrowOnAbort.H"

namespace {

using amrex::Real;

// A level split along x into one slab per rank (slab r on rank r, each cut in two boxes along y, so
// box faces lie inside every slab), periodic in x, with a smooth but not linear velocity field.
struct SlabMesh {
    int nx = 64, ny = 16, nz = 10;
    Real Lx = 3200.0, Ly = 800.0, H = 500.0;
    Real hill = 0.0;   // basic terrain-following amplitude (m); 0: a uniform-dz mesh (no z_phys_nd)

    amrex::Geometry geom;
    amrex::BoxArray ba;
    amrex::DistributionMapping dm;
    amrex::MultiFab u, v, w;
    std::unique_ptr<amrex::MultiFab> znd, detj;

    Real dx () const { return Lx / nx; }
    Real dy () const { return Ly / ny; }
    Real h (Real x, Real y) const
    {
        constexpr Real pi = 3.14159265358979323846;
        return hill * (Real(0.5) + Real(0.5) * std::cos(Real(2.0) * pi * x / Lx)) * (Real(0.5) + Real(0.5) * std::sin(Real(2.0) * pi * y / Ly));
    }
    Real z_node (int i, int j, int k) const { const Real hs = h(i * dx(), j * dy()); return hs + (H - hs) * Real(k) / Real(nz); }
    static Real field (int c, Real x, Real y, Real z)
    {
        return std::sin(Real(0.003) * x + Real(0.7) * c) * std::cos(Real(0.011) * y) + Real(0.002) * z * (c + 1) + Real(0.3) * c;
    }

    explicit SlabMesh (Real hill_m = 0.0) : hill(hill_m)
    {
        const int np = amrex::ParallelDescriptor::NProcs();
        const amrex::Box domain(amrex::IntVect(0, 0, 0), amrex::IntVect(nx-1, ny-1, nz-1));
        const amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)}, {AMREX_D_DECL(Lx, Ly, H)});
        const std::array<int,3> periodic{{1, 0, 0}};
        geom = amrex::Geometry(domain, &rb, 0, periodic.data());
        amrex::BoxList bl;
        amrex::Vector<int> pmap;
        for (int r = 0; r < np; ++r) {
            const int i0 = r * nx / np, i1 = (r + 1) * nx / np - 1;
            const int jm = ny / 2 - 3;   // unequal halves in y
            bl.push_back(amrex::Box(amrex::IntVect(i0, 0, 0), amrex::IntVect(i1, jm, nz-1)));
            bl.push_back(amrex::Box(amrex::IntVect(i0, jm + 1, 0), amrex::IntVect(i1, ny-1, nz-1)));
            pmap.push_back(r);
            pmap.push_back(r);
        }
        ba = amrex::BoxArray(bl);
        dm = amrex::DistributionMapping(pmap);
        const int ng = 2;
        u.define(amrex::convert(ba, amrex::IntVect(1,0,0)), dm, 1, ng);
        v.define(amrex::convert(ba, amrex::IntVect(0,1,0)), dm, 1, ng);
        w.define(amrex::convert(ba, amrex::IntVect(0,0,1)), dm, 1, ng);
        if (hill > Real(0.0)) {
            znd = std::make_unique<amrex::MultiFab>(amrex::convert(ba, amrex::IntVect(1,1,1)), dm, 1, ng);
            detj = std::make_unique<amrex::MultiFab>(ba, dm, 1, ng);
        }
        for (amrex::MFIter mfi(u, false); mfi.isValid(); ++mfi) {
            auto ua = u.array(mfi); auto va = v.array(mfi); auto wa = w.array(mfi);
            const amrex::Box cbx = amrex::grow(amrex::enclosedCells(mfi.validbox()), ng);
            auto zn = [&](int i, int j, int k) { return znd ? z_node(i, j, k) : H * Real(k) / Real(nz); };
            if (znd) {
                auto za = znd->array(mfi);
                auto ja = detj->array(mfi);
                amrex::LoopOnCpu(amrex::surroundingNodes(cbx), [&](int i, int j, int k) { za(i,j,k) = z_node(i, j, k); });
                amrex::LoopOnCpu(cbx, [&](int i, int j, int k) {
                    ja(i,j,k) = (H - h((i + Real(0.5)) * dx(), (j + Real(0.5)) * dy())) / H; (void) k; });
            }
            amrex::LoopOnCpu(amrex::convert(cbx, amrex::IntVect(1,0,0)), [&](int i, int j, int k) {
                ua(i,j,k) = field(0, i * dx(), (j + Real(0.5)) * dy(), Real(0.5) * (zn(i,j,k) + zn(i,j,k+1)));
            });
            amrex::LoopOnCpu(amrex::convert(cbx, amrex::IntVect(0,1,0)), [&](int i, int j, int k) {
                va(i,j,k) = field(1, (i + Real(0.5)) * dx(), j * dy(), Real(0.5) * (zn(i,j,k) + zn(i,j,k+1)));
            });
            amrex::LoopOnCpu(amrex::convert(cbx, amrex::IntVect(0,0,1)), [&](int i, int j, int k) {
                wa(i,j,k) = field(2, (i + Real(0.5)) * dx(), (j + Real(0.5)) * dy(), zn(i,j,k));
            });
        }
    }

    void define_sources (amrex::MultiFab& sx, amrex::MultiFab& sy, amrex::MultiFab& sz) const
    {
        sx.define(amrex::convert(ba, amrex::IntVect(1,0,0)), dm, 1, 0);
        sy.define(amrex::convert(ba, amrex::IntVect(0,1,0)), dm, 1, 0);
        sz.define(amrex::convert(ba, amrex::IntVect(0,0,1)), dm, 1, 0);
        sx.setVal(0.0); sy.setVal(0.0); sz.setVal(0.0);
    }
};

// A cluster of n points about (x, y, z), spread over +- half across and +- halfz in height (m), with a
// fixed pseudo-random pattern.
std::vector<Real> cluster (Real x, Real y, Real z, Real half, Real halfz, int n, int seed)
{
    std::vector<Real> p;
    unsigned s = 12345u + 977u * static_cast<unsigned>(seed);
    auto rnd = [&s] () { s = 1103515245u * s + 12345u; return Real((s >> 8) & 0xffff) / Real(65535.0); };
    for (int k = 0; k < n; ++k) {
        p.push_back(x + half * (Real(2.0) * rnd() - Real(1.0)));
        p.push_back(y + half * (Real(2.0) * rnd() - Real(1.0)));
        p.push_back(z + halfz * (Real(2.0) * rnd() - Real(1.0)));
    }
    return p;
}

// Three bodies: one across the periodic seam at x = 0, one on a box face inside the middle of the
// domain, one near the top; their owners are not the ranks round-robin would give.
struct Farm {
    std::vector<std::vector<Real>> pts;
    std::vector<int> owners;
    explicit Farm (const SlabMesh& m)
    {
        const int np = amrex::ParallelDescriptor::NProcs();
        pts.push_back(cluster(Real(0.0), Real(0.45) * m.Ly, Real(0.4) * m.H, Real(3.0) * m.dx(), Real(50.0), 40, 1));
        // about the y box face (j = ny/2 - 2) and the x slab face (i = nx/2) of the middle of the domain, and a point on both
        std::vector<Real> mid = cluster(Real(0.5) * m.Lx, (m.ny / 2 - 2) * m.dy(), Real(0.5) * m.H, Real(2.5) * m.dx(), Real(50.0), 40, 2);
        mid.insert(mid.end(), {Real(0.5) * m.Lx, (m.ny / 2 - 2) * m.dy(), Real(0.5) * m.H});
        // kernels (this body's width is 1.5 dx) whose reach of 3 widths ends, or starts, exactly on the face
        // x = Lx/2 between two slabs (on 2 and 4 ranks), at a cell centre in y and z so that the face at that
        // distance has a weight: the left slab's copy of that face (its box's high face, which only the
        // spreader's second pass writes) must still get the point to its right
        const Real yc = Real(4.5) * m.dy(), zc = Real(4.5) * m.H / Real(m.nz);
        mid.insert(mid.end(), {Real(0.5) * m.Lx - Real(4.5) * m.dx(), yc, zc,
                               Real(0.5) * m.Lx + Real(4.5) * m.dx(), yc, zc});
        pts.push_back(mid);
        std::vector<Real> top = cluster(Real(0.8) * m.Lx, Real(0.3) * m.Ly, Real(0.9) * m.H, Real(1.5) * m.dx(), Real(30.0), 30, 3);
        // on the domain's upper y face (not periodic), which belongs to the last cell: its rank must be asked
        top.insert(top.end(), {Real(0.8) * m.Lx + Real(0.3) * m.dx(), m.Ly, Real(0.85) * m.H});
        pts.push_back(top);
        for (int b = 0; b < 3; ++b) { owners.push_back((np - 1 - b + 3 * np) % np); }
    }
};

constexpr Real near_tol = (sizeof(Real) == 8) ? Real(1.0e-12) : Real(1.0e-5);

} // namespace

TEST(BodyExchange, RangesWrapAtPeriodicSeamsAndAreClippedElsewhere)
{
    const amrex::Box domain(amrex::IntVect(0, 0, 0), amrex::IntVect(31, 15, 7));
    const amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)}, {AMREX_D_DECL(320.0, 160.0, 80.0)});
    const std::array<int,3> periodic{{1, 0, 0}};
    const amrex::Geometry geom(domain, &rb, 0, periodic.data());
    // inside: unchanged
    auto in = erf_actuator::domain_images(amrex::Box(amrex::IntVect(4, 2, 1), amrex::IntVect(9, 5, 3)), geom);
    ASSERT_EQ(in.size(), 1u);
    EXPECT_EQ(in[0], amrex::Box(amrex::IntVect(4, 2, 1), amrex::IntVect(9, 5, 3)));
    // across the low x seam: the part below 0 wraps to the top end; y below 0 is clipped
    auto seam = erf_actuator::domain_images(amrex::Box(amrex::IntVect(-3, -2, 0), amrex::IntVect(2, 3, 7)), geom);
    ASSERT_EQ(seam.size(), 2u);
    std::sort(seam.begin(), seam.end(), [] (const amrex::Box& a, const amrex::Box& b) { return a.smallEnd(0) < b.smallEnd(0); });
    EXPECT_EQ(seam[0], amrex::Box(amrex::IntVect(0, 0, 0), amrex::IntVect(2, 3, 7)));
    EXPECT_EQ(seam[1], amrex::Box(amrex::IntVect(29, 0, 0), amrex::IntVect(31, 3, 7)));
    // a whole period or more: all of x
    auto wide = erf_actuator::domain_images(amrex::Box(amrex::IntVect(-40, 0, 0), amrex::IntVect(40, 1, 1)), geom);
    ASSERT_EQ(wide.size(), 1u);
    EXPECT_EQ(wide[0].smallEnd(0), 0);
    EXPECT_EQ(wide[0].bigEnd(0), 31);
    // wholly beyond a non-periodic side: nothing
    EXPECT_TRUE(erf_actuator::domain_images(amrex::Box(amrex::IntVect(0, 20, 0), amrex::IntVect(3, 25, 1)), geom).empty());
    // cells_within: floor of the coordinates over dx, padded; a fitted mesh takes the whole column
    const amrex::Box c = erf_actuator::cells_within({{15.0, 25.0, 41.0}}, {{34.9, 25.0, 49.0}}, geom, 1, false);
    EXPECT_EQ(c, amrex::Box(amrex::IntVect(0, 1, 3), amrex::IntVect(4, 3, 5)));
    const amrex::Box f = erf_actuator::cells_within({{15.0, 25.0, 41.0}}, {{34.9, 25.0, 49.0}}, geom, 0, true);
    EXPECT_EQ(f, amrex::Box(amrex::IntVect(1, 2, 0), amrex::IntVect(3, 2, 7)));
}

TEST(BodyExchange, OwnersComeFromTheRanksTheFootprintOverlapsWhileThatKeepsTheBalance)
{
    // four boxes along a periodic x, on ranks 3, 1, 2 and 0 (not in box order); choose_owners reads the
    // map only, so the ranks need not be this run's
    const amrex::Box domain(amrex::IntVect(0, 0, 0), amrex::IntVect(31, 7, 3));
    const amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)}, {AMREX_D_DECL(320.0, 80.0, 40.0)});
    const std::array<int,3> periodic{{1, 0, 0}};
    const amrex::Geometry geom(domain, &rb, 0, periodic.data());
    amrex::BoxList bl;
    for (int b = 0; b < 4; ++b) { bl.push_back(amrex::Box(amrex::IntVect(8*b, 0, 0), amrex::IntVect(8*b + 7, 7, 3))); }
    const amrex::BoxArray ba(bl);
    const amrex::DistributionMapping dm(amrex::Vector<int>{3, 1, 2, 0});
    auto column = [] (int i0, int i1) { return amrex::Box(amrex::IntVect(i0, 2, 0), amrex::IntVect(i1, 5, 3)); };
    auto owners = [&] (const std::vector<amrex::Box>& fp, std::vector<Real> dt = {}) {
        if (dt.empty()) { dt.assign(fp.size(), Real(0.01)); }
        return erf_actuator::choose_owners(fp, dt, ba, dm, geom, 4);
    };
    // one body inside the third box: its rank (2), where round-robin would give rank 0
    EXPECT_EQ(owners({column(18, 21)}), std::vector<int>({2}));
    // over the second (3 columns) and third (5 columns) boxes: the one with more of it
    EXPECT_EQ(owners({column(13, 20)}), std::vector<int>({2}));
    // two bodies there: the second goes to the other rank it stands on, which owns none yet
    EXPECT_EQ(owners({column(13, 20), column(13, 20)}), std::vector<int>({2, 1}));
    // a third, on the second box only, whose rank already owns one while ranks 0 and 3 own none: the
    // nearer of those (rank 3's box ends 2 columns away, rank 0's begins 15 away)
    EXPECT_EQ(owners({column(13, 20), column(13, 20), column(9, 10)}), std::vector<int>({2, 1, 3}));
    // as many bodies as ranks, all on one box: one each, the nearest first; a fifth joins the rank it stands on
    EXPECT_EQ(owners({column(18, 21), column(18, 21), column(18, 21), column(18, 21)}), std::vector<int>({2, 1, 0, 3}));
    EXPECT_EQ(owners({column(18, 21), column(18, 21), column(18, 21), column(18, 21), column(18, 21)}),
              std::vector<int>({2, 1, 0, 3, 2}));
    // across the periodic seam: 3 columns in the last box (rank 0) and 3 in the first (rank 3): the lower rank;
    // one column more on the first box: rank 3
    EXPECT_EQ(owners({column(-3, 2)}), std::vector<int>({0}));
    EXPECT_EQ(owners({column(-3, 3)}), std::vector<int>({3}));
    // a body with another time step does not join a rank stepping the first one's, while one is free;
    // an unknown time step (<= 0) joins only another unknown one
    EXPECT_EQ(owners({column(18, 21), column(18, 21)}, {Real(0.01), Real(0.02)}), std::vector<int>({2, 1}));
    EXPECT_EQ(owners({column(18, 21), column(18, 21), column(18, 21), column(18, 21), column(18, 21)},
                     {Real(0.01), Real(0.02), Real(0.02), Real(0.02), Real(0.02)}),
              std::vector<int>({2, 1, 0, 3, 1}));
    EXPECT_EQ(owners({column(18, 21), column(18, 21), column(18, 21), column(18, 21), column(18, 21)},
                     {Real(0.01), Real(-1.0), Real(-1.0), Real(-1.0), Real(-1.0)}),
              std::vector<int>({2, 1, 0, 3, 1}));
    // an empty rank is kept for a time step that has no rank yet: ten bodies of one step on four ranks
    // leave a rank for the eleventh, of another step, wherever it comes in the list
    {
        std::vector<amrex::Box> fp(11, column(18, 21));
        std::vector<Real> dt(11, Real(0.01));
        dt[10] = Real(0.02);
        const std::vector<int> o = owners(fp, dt);
        for (int b = 0; b < 10; ++b) { EXPECT_NE(o[b], o[10]) << "body " << b; }
        std::vector<int> per(4, 0);
        for (int r : o) { ++per[r]; }
        EXPECT_EQ(*std::max_element(per.begin(), per.end()), 4);   // ten on three ranks: at most four each
        std::swap(dt[0], dt[10]);
        const std::vector<int> f = owners(fp, dt);
        for (int b = 1; b < 11; ++b) { EXPECT_NE(f[b], f[0]) << "body " << b; }
    }
    // on one rank two time steps cannot be kept apart: both there (the driver's lockstep check then stops the run)
    const amrex::DistributionMapping dm_one(amrex::Vector<int>{0, 0, 0, 0});
    EXPECT_EQ(erf_actuator::choose_owners({column(18, 21), column(1, 2)}, {Real(0.01), Real(0.02)}, ba, dm_one, geom, 1),
              std::vector<int>({0, 0}));
    // a refined level with a gap: a footprint no box overlaps goes to the nearest box's rank
    amrex::BoxList gap;
    gap.push_back(amrex::Box(amrex::IntVect(0, 0, 0), amrex::IntVect(7, 7, 3)));
    gap.push_back(amrex::Box(amrex::IntVect(24, 0, 0), amrex::IntVect(31, 7, 3)));
    const amrex::DistributionMapping dm_gap(amrex::Vector<int>{1, 0});
    EXPECT_EQ(erf_actuator::choose_owners({column(14, 15)}, {Real(0.01)}, amrex::BoxArray(gap), dm_gap, geom, 2), std::vector<int>({1}));
    EXPECT_EQ(erf_actuator::choose_owners({column(17, 18)}, {Real(0.01)}, amrex::BoxArray(gap), dm_gap, geom, 2), std::vector<int>({0}));
    // a map naming a rank the caller does not count is refused
    const std::string msg = erf_gtest::abort_message([&] { erf_actuator::choose_owners({column(18, 21)}, {Real(0.01)}, ba, dm, geom, 3); });
    EXPECT_NE(msg.find("outside 0 .. nranks - 1"), std::string::npos) << msg;
}

TEST(BodyExchange, BuffersReachTheRanksTheyAreSentTo)
{
    const int np = amrex::ParallelDescriptor::NProcs();
    const int me = amrex::ParallelDescriptor::MyProc();
    // each rank sends me + 1 copies of its rank to the next rank (to itself on one rank), whose size the
    // receiver does not know, and, on four ranks or more, an empty buffer to the rank after that
    std::map<int, std::vector<Real>> out, in;
    out[(me + 1) % np] = std::vector<Real>(static_cast<std::size_t>(me + 1), Real(me));
    std::vector<int> from{(me - 1 + np) % np};
    if (np >= 4) {
        out[(me + 2) % np];
        from.push_back((me - 2 + np) % np);
        std::sort(from.begin(), from.end());
    }
    erf_actuator::exchange_buffers(out, from, in);
    const int prev = (me - 1 + np) % np;
    ASSERT_EQ(in.count(prev), 1u);
    ASSERT_EQ(in[prev].size(), static_cast<std::size_t>(prev + 1));
    for (Real x : in[prev]) { EXPECT_EQ(x, Real(prev)); }
    if (np >= 4) {
        ASSERT_EQ(in.count((me - 2 + np) % np), 1u);
        EXPECT_TRUE(in[(me - 2 + np) % np].empty());
    }
}

namespace {

// the farm sampled through the exchange and, point list by point list, by the global sampler
void expect_sampling_matches (const SlabMesh& m)
{
    const Farm farm(m);
    erf_actuator::BodyExchange ex(farm.owners);
    std::vector<std::vector<Real>> pts(3), vel;
    for (int b = 0; b < 3; ++b) { if (ex.owns(b)) { pts[b] = farm.pts[b]; } }
    ex.sample_velocity(m.u, m.v, m.w, m.znd.get(), m.geom, pts, std::vector<Real>(3, Real(0.0)), vel);
    ASSERT_EQ(vel.size(), 3u);
    for (int b = 0; b < 3; ++b) {
        std::vector<Real> ref;
        erf_actuator::sample_velocity(m.u, m.v, m.w, m.znd.get(), m.geom, farm.pts[b], ref);
        if (!ex.owns(b)) {
            EXPECT_TRUE(vel[b].empty());
            continue;
        }
        ASSERT_EQ(vel[b].size(), ref.size());
        for (std::size_t q = 0; q < ref.size(); ++q) {
            // the one value of the one box holding the point: the same bits
            EXPECT_EQ(vel[b][q], ref[q]) << "body " << b << " value " << q;
        }
    }
}

// the farm spread through the exchange against all its points spread by the global spreader
void expect_spreading_matches (const SlabMesh& m)
{
    const Farm farm(m);
    const int np = amrex::ParallelDescriptor::NProcs();
    erf_actuator::BodyExchange ex(farm.owners);
    std::vector<std::vector<Real>> pos(3), force(3), eps(3);
    std::vector<Real> all_pos, all_force, all_eps;
    for (int b = 0; b < 3; ++b) {
        const std::vector<Real>& p = farm.pts[b];
        std::vector<Real> f, e;
        for (std::size_t k = 0; k < p.size() / 3; ++k) {
            f.insert(f.end(), {Real(1.0e4) * (b + 1) + Real(13.0) * k, Real(-2.0e3) * k, Real(500.0) * (b - 1)});
            e.push_back(Real(1.0 + 0.5 * b) * m.dx());
        }
        all_pos.insert(all_pos.end(), p.begin(), p.end());
        all_force.insert(all_force.end(), f.begin(), f.end());
        all_eps.insert(all_eps.end(), e.begin(), e.end());
        if (ex.owns(b)) { pos[b] = p; force[b] = f; eps[b] = e; }
    }
    amrex::MultiFab ax, ay, az, gx, gy, gz;
    m.define_sources(ax, ay, az);
    m.define_sources(gx, gy, gz);
    ex.spread_forces(pos, force, eps, std::vector<Real>(3, Real(0.0)), m.znd.get(), m.detj.get(), m.geom, ax, ay, az);
    erf_actuator::spread_forces(all_pos, all_force, all_eps, m.znd.get(), m.detj.get(), m.geom, gx, gy, gz);
    const std::array<std::pair<amrex::MultiFab*, amrex::MultiFab*>,3> pairs{{{&ax, &gx}, {&ay, &gy}, {&az, &gz}}};
    for (int d = 0; d < 3; ++d) {
        const Real scale = pairs[d].second->norm0();
        ASSERT_GT(scale, Real(0.0));
        amrex::MultiFab::Subtract(*pairs[d].first, *pairs[d].second, 0, 0, 1, 0);
        const Real diff = pairs[d].first->norm0();
        if (np <= 2) {
            // a normalisation is the same sum of the same parts on two ranks or one
            EXPECT_EQ(diff, Real(0.0)) << "component " << d;
        } else {
            // an MPI reduction may add more than two parts in another order than rank order
            EXPECT_LE(diff, near_tol * scale) << "component " << d;
        }
    }
}

} // namespace

TEST(BodyExchange, SamplingIsTheGlobalSamplersOnAUniformMesh)
{
    SlabMesh m;
    expect_sampling_matches(m);
}

TEST(BodyExchange, SamplingIsTheGlobalSamplersOverTerrain)
{
    SlabMesh m(60.0);
    expect_sampling_matches(m);
}

TEST(BodyExchange, SpreadingIsTheGlobalSpreadersOnAUniformMesh)
{
    SlabMesh m;
    expect_spreading_matches(m);
}

TEST(BodyExchange, SpreadingIsTheGlobalSpreadersOverTerrain)
{
    SlabMesh m(60.0);
    expect_spreading_matches(m);
}

TEST(BodyExchange, ARankNoKernelReachesReceivesNothing)
{
    SlabMesh m;
    const int np = amrex::ParallelDescriptor::NProcs();
    const int me = amrex::ParallelDescriptor::MyProc();
    // a small body inside the first slab (rank 0), owned by the last rank, which therefore takes no part
    // in its exchanges when there are several ranks
    const std::vector<Real> p = cluster(Real(8.0) * m.dx(), Real(0.5) * m.Ly, Real(0.5) * m.H, m.dx(), Real(40.0), 20, 7);
    erf_actuator::BodyExchange ex({np - 1});
    std::vector<std::vector<Real>> pts(1), vel, pos(1), force(1), eps(1);
    if (ex.owns(0)) {
        pts[0] = p;
        pos[0] = p;
        for (std::size_t k = 0; k < p.size() / 3; ++k) { force[0].insert(force[0].end(), {Real(1.0e4), Real(0.0), Real(0.0)}); }
        eps[0].assign(p.size() / 3, m.dx());
    }
    ex.sample_velocity(m.u, m.v, m.w, nullptr, m.geom, pts, {Real(0.0)}, vel);
    amrex::MultiFab sx, sy, sz;
    m.define_sources(sx, sy, sz);
    ex.spread_forces(pos, force, eps, {Real(0.0)}, nullptr, nullptr, m.geom, sx, sy, sz);
    // the kernels (one cell wide, 3 cells of reach, 2 of padding) stay within the first slab
    EXPECT_EQ(ex.participants(0), std::vector<int>({0}));
    if (me != 0) {
        EXPECT_EQ(ex.points_received(), 0L);
        EXPECT_EQ(sx.norm0(0, 0, true), Real(0.0));
    } else if (np > 1) {
        // every point twice from the owner: to be sampled, then to be spread
        EXPECT_EQ(ex.points_received(), static_cast<long>(2 * (p.size() / 3)));
    }
    // and the force is all there
    const Real total = erf_actuator::integrate_source(0, sx, nullptr, m.geom);
    EXPECT_NEAR(total, Real(1.0e4) * Real(p.size() / 3), near_tol * Real(1.0e4) * Real(p.size() / 3));
}

TEST(BodyExchange, AnEnvelopeGrowsWhenItsPointsLeaveIt)
{
    SlabMesh m;
    const int np = amrex::ParallelDescriptor::NProcs();
    erf_actuator::BodyExchange ex({0});
    const std::vector<Real> a = cluster(Real(8.0) * m.dx(), Real(0.5) * m.Ly, Real(0.5) * m.H, m.dx(), Real(40.0), 10, 11);
    const std::vector<Real> b = cluster(m.Lx - Real(8.0) * m.dx(), Real(0.5) * m.Ly, Real(0.5) * m.H, m.dx(), Real(40.0), 10, 12);
    Real a_xmin = a[0];
    for (std::size_t k = 0; k < a.size(); k += 3) { a_xmin = std::min(a_xmin, a[k]); }
    auto sample = [&] (const std::vector<Real>& p) {
        std::vector<std::vector<Real>> pts(1), vel;
        if (ex.owns(0)) { pts[0] = p; }
        ex.sample_velocity(m.u, m.v, m.w, nullptr, m.geom, pts, {Real(10.0)}, vel);
        std::vector<Real> ref;
        erf_actuator::sample_velocity(m.u, m.v, m.w, nullptr, m.geom, p, ref);
        if (ex.owns(0)) { EXPECT_EQ(vel[0], ref); }
    };
    sample(a);
    EXPECT_EQ(ex.num_envelope_updates(), 1);
    const auto env_a = ex.envelope(0);
    // the envelope holds the points with the margin; only the first slab's rank takes part
    EXPECT_EQ(env_a[0], a_xmin - Real(10.0));
    EXPECT_EQ(ex.participants(0), std::vector<int>({0}));
    sample(a);
    EXPECT_EQ(ex.num_envelope_updates(), 1);
    // the body moves to the last slab: the envelope grows to hold both places, every rank learns it
    sample(b);
    EXPECT_EQ(ex.num_envelope_updates(), 2);
    EXPECT_EQ(ex.envelope(0)[0], env_a[0]);
    EXPECT_GE(ex.envelope(0)[3], m.Lx - Real(9.0) * m.dx());
    EXPECT_EQ(ex.participants(0).back(), np - 1);
    sample(a);
    EXPECT_EQ(ex.num_envelope_updates(), 2);
}

TEST(BodyExchange, APointOutsideTheDomainAbortsNamingIt)
{
    SlabMesh m;
    // the owner aborts once the exchanges are over, so the other ranks are not left waiting
    erf_actuator::BodyExchange ex({0});
    std::vector<std::vector<Real>> pts(1), vel;
    if (ex.owns(0)) { pts[0] = {Real(100.0), Real(100.0), Real(100.0), Real(200.0), m.Ly + Real(30.0), Real(100.0)}; }
    const std::string msg = erf_gtest::abort_message([&] {
        ex.sample_velocity(m.u, m.v, m.w, nullptr, m.geom, pts, {Real(0.0)}, vel);
    });
    if (ex.owns(0)) {
        EXPECT_NE(msg.find("sampled by 0 boxes"), std::string::npos) << msg;
        EXPECT_NE(msg.find("830"), std::string::npos) << msg;
    } else {
        EXPECT_TRUE(msg.empty()) << msg;
    }
}
