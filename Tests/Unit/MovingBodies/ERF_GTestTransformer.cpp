// Contract of erf_conductors::Transformer: the line ends whose x, y lie on a transformer's
// footprint are dead-ended on it, and an end on two footprints, an end not above the box's top
// or a transformer no line ends on is refused by name; the box stands on the terrain under its
// centre; and the lines' load on it is the sum of their pulls and their moment about the centre
// of the base, flagged when the horizontal force or the overturning moment exceeds its allowable
// value, a zero allowable being left unchecked.

#include <array>
#include <cmath>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "ERF_Transformer.H"

using amrex::Real;
using erf_conductors::LineInputs;
using erf_conductors::Transformer;
using erf_conductors::TransformerInputs;
using P3 = std::array<Real,3>;

namespace {

constexpr Real tol = std::is_same<Real, float>::value ? Real(1.0e-3) : Real(1.0e-9);

TransformerInputs box (const std::string& name, Real x, Real y)
{
    TransformerInputs t;
    t.name = name;
    t.position = {{x, y}};
    t.size = {{8.0, 5.0, 6.0}};
    return t;
}

// a line at its absolute heights: ends 10 m above ground at 50 m
LineInputs line (const std::string& name, const P3& a, const P3& b)
{
    LineInputs s;
    s.name = name;
    s.end_a = a;
    s.end_b = b;
    s.lengths = {Real(1.01) * std::abs(b[0] - a[0])};
    return s;
}

} // namespace

TEST(Transformer, TheBoxStandsOnTheTerrainUnderItsCentre)
{
    const Transformer t(box("T", 100.0, 200.0), 50.0);
    EXPECT_EQ(t.base(), (P3{{100.0, 200.0, 50.0}}));
    EXPECT_EQ(t.box_lo(), (P3{{96.0, 197.5, 50.0}}));
    EXPECT_EQ(t.box_hi(), (P3{{104.0, 202.5, 56.0}}));
}

TEST(Transformer, LineEndsOnAFootprintAreDeadEndedOnIt)
{
    std::vector<Transformer> ts{Transformer(box("T1", 100.0, 200.0), 50.0), Transformer(box("T2", 400.0, 200.0), 50.0)};
    // L1 from T1 to T2, L2 from T1's edge to open ground, L3 nowhere near either
    const std::vector<LineInputs> lines{line("L1", {{102.0, 200.0, 60.0}}, {{398.0, 200.0, 60.0}}),
                                        line("L2", {{104.0, 202.5, 58.0}}, {{300.0, 400.0, 60.0}}),
                                        line("L3", {{200.0, 600.0, 60.0}}, {{500.0, 600.0, 60.0}})};
    EXPECT_EQ(erf_conductors::attach_line_ends(ts, lines), "");
    ASSERT_EQ(ts[0].ends().size(), 2u);
    EXPECT_EQ(ts[0].ends()[0].line, 0u); EXPECT_EQ(ts[0].ends()[0].end, 0);
    EXPECT_EQ(ts[0].ends()[1].line, 1u); EXPECT_EQ(ts[0].ends()[1].end, 0);
    ASSERT_EQ(ts[1].ends().size(), 1u);
    EXPECT_EQ(ts[1].ends()[0].line, 0u); EXPECT_EQ(ts[1].ends()[0].end, 1);
}

TEST(Transformer, MisplacedEndsAndIdleTransformersAreRefusedByName)
{
    {
        // overlapping footprints
        std::vector<Transformer> ts{Transformer(box("T1", 100.0, 200.0), 50.0), Transformer(box("T2", 106.0, 200.0), 50.0)};
        const std::string err = erf_conductors::attach_line_ends(ts, {line("L1", {{103.0, 200.0, 60.0}}, {{400.0, 200.0, 60.0}})});
        EXPECT_NE(err.find("L1.end_a lies on the footprints of both T1 and T2"), std::string::npos) << err;
    }
    {
        // an end inside the box, below its top at 56 m
        std::vector<Transformer> ts{Transformer(box("T1", 100.0, 200.0), 50.0)};
        const std::string err = erf_conductors::attach_line_ends(ts, {line("L1", {{400.0, 200.0, 60.0}}, {{100.0, 200.0, 55.0}})});
        EXPECT_NE(err.find("L1.end_b ends on T1"), std::string::npos) << err;
        EXPECT_NE(err.find("not above the transformer's top"), std::string::npos) << err;
    }
    {
        // a transformer no line ends on
        std::vector<Transformer> ts{Transformer(box("T1", 100.0, 200.0), 50.0), Transformer(box("T9", 900.0, 900.0), 50.0)};
        const std::string err = erf_conductors::attach_line_ends(ts, {line("L1", {{100.0, 200.0, 60.0}}, {{400.0, 200.0, 60.0}})});
        EXPECT_NE(err.find("erf.conductors.T9: no line ends on it"), std::string::npos) << err;
    }
}

TEST(Transformer, TheLoadIsTheSumOfThePullsAndTheirMomentAboutTheBase)
{
    Transformer t(box("T", 100.0, 200.0), 50.0);
    // two lines pulling along +x and +y from 10 m above the base, offset on the top
    const std::vector<P3> at{{{102.0, 200.0, 60.0}}, {{100.0, 201.0, 60.0}}};
    const std::vector<P3> f{{{3000.0, 0.0, -1000.0}}, {{0.0, 4000.0, -800.0}}};
    const auto L = t.load(at, f);
    EXPECT_NEAR(L.force[0], 3000.0, tol); EXPECT_NEAR(L.force[1], 4000.0, tol); EXPECT_NEAR(L.force[2], -1800.0, tol);
    EXPECT_NEAR(L.horizontal_force, 5000.0, tol * 10);
    // M = sum r x F with r = (2, 0, 10) and (0, 1, 10) from the base centre
    const P3 M{{0.0 * -1000.0 - 10.0 * 0.0 + (1.0 * -800.0 - 10.0 * 4000.0),
                10.0 * 3000.0 - 2.0 * -1000.0 + (10.0 * 0.0 - 0.0 * -800.0),
                2.0 * 0.0 - 0.0 * 3000.0 + (0.0 * 4000.0 - 1.0 * 0.0)}};
    for (int d = 0; d < 3; ++d) { EXPECT_NEAR(L.moment[d], M[d], tol * 1e3) << d; }
    EXPECT_NEAR(L.overturning_moment, std::sqrt(M[0] * M[0] + M[1] * M[1]), tol * 1e3);
    EXPECT_FALSE(L.over_allowable) << "no allowable given: not checked";
}

TEST(Transformer, TheLoadIsFlaggedOverEitherAllowable)
{
    const std::vector<P3> at{{{100.0, 200.0, 60.0}}};
    const std::vector<P3> f{{{5000.0, 0.0, 0.0}}};   // 5 kN at 10 m: 50 kN m
    auto flagged = [&](Real force, Real moment) {
        TransformerInputs in = box("T", 100.0, 200.0);
        in.allowable_force = force;
        in.allowable_moment = moment;
        return Transformer(in, 50.0).load(at, f).over_allowable;
    };
    EXPECT_TRUE(flagged(4000.0, 0.0));
    EXPECT_FALSE(flagged(6000.0, 0.0));
    EXPECT_TRUE(flagged(0.0, 4.0e4));
    EXPECT_FALSE(flagged(0.0, 6.0e4));
    EXPECT_TRUE(flagged(6000.0, 4.0e4)) << "the moment alone is over";
    EXPECT_FALSE(flagged(6000.0, 6.0e4));
}
