#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

/// Relative round-off tolerance of the build precision.
static constexpr double REL = (sizeof(amrex::Real) == 8) ? 1e-10 : 1e-4;

#include "ERF_FireParams.H"   // ERF_CheneyGouldModel.H reads FireParams without including it
#include "ERF_DirectionalRos.H"

/**
 * @file ERF_GTestDirectionalShape.cpp
 * @brief The two shapes of the directional rate. For a peaked wind factor the
 *        Wulff shape of the projection, which is what the level set spreads
 *        from a point, falls short of the projection's own head rate; the
 *        ellipse built from the same model keeps the model's head, back and
 *        flank rates, is its own Wulff shape, and adds misaligned wind and
 *        slope as vectors.
 */

namespace {

/// Rothermel coefficients close to fuel model 1 at 5.5 % moisture, with the
/// wind cap lifted so the wind factor is a pure power law.
DirectionalRosState rothermel_state ()
{
    DirectionalRosState st;
    st.model = DIRECTIONAL_ROS_ROTHERMEL;
    st.rc = RothermelComputed{};
    st.rc.R0           = 0.0239;
    st.rc.C            = 5.4e-5;
    st.rc.B            = 2.07;
    st.rc.beta_ratio_E = 1.32;
    st.rc.beta         = 0.0010625;
    st.rc.phi_s_const  = 41.1;
    st.rc.U_max_ftmin  = 1.0e6;
    st.rc.wind_conv    = 196.85;
    st.rc.ros_conv     = 1.0;
    st.rc.I_R          = 0.0;
    return st;
}

/// Extent along +x of the Wulff shape of a normal speed F: the minimum over
/// |theta| < 90 degrees of F(theta) / cos(theta), sampled every 0.01 degree.
template <class Speed>
double wulff_extent_x (Speed&& F)
{
    const double pi = amrex::Math::pi<double>();
    double best = 1.0e300;
    for (int n = -8999; n <= 8999; ++n) {
        const double th = n * 0.01 * pi / 180.0;
        best = std::min(best, static_cast<double>(F(std::cos(th), std::sin(th))) / std::cos(th));
    }
    return best;
}

}

TEST(DirectionalShape, ProjectionWulffHeadFallsShort)
{
    const DirectionalRosState st = rothermel_state();
    const amrex::Real U = 1.5;
    auto projection = [&](double nx, double) {
        return directional_ros_cell(st, static_cast<amrex::Real>(U * nx), amrex::Real(0.0));
    };
    const double head = projection(1.0, 0.0);
    const double R0 = st.rc.R0;
    const double phi_w = head / R0 - 1.0;
    const double B = st.rc.B;
    ASSERT_GT(phi_w * (B - 1.0), 1.0);

    // min over u of R0 (1 + phi_w u^B) / u, at u = (phi_w (B - 1))^(-1/B)
    const double tip = R0 * B / (B - 1.0) * std::pow(phi_w * (B - 1.0), 1.0 / B);
    const double wulff = wulff_extent_x(projection);
    EXPECT_NEAR(wulff, tip, 1e-3 * tip);
    EXPECT_LT(wulff, 0.7 * head);
}

TEST(DirectionalShape, EllipseKeepsModelRatesAligned)
{
    const DirectionalRosState st = rothermel_state();
    const amrex::Real U = 1.5, s = 0.3;
    const DirectionalEllipse de = directional_ellipse_cell(st, U, 0.0, s, 0.0);
    const double head = directional_ros_cell(st, U, s);
    const double R0 = directional_ros_cell(st, 0.0, 0.0);
    EXPECT_NEAR(de.hx, 1.0, REL);
    EXPECT_NEAR(de.hy, 0.0, REL);
    // The projection's own rates at the head, back and flank normals
    EXPECT_NEAR(directional_ellipse_speed(de, 1.0, 0.0), head, REL * head);
    EXPECT_NEAR(directional_ellipse_speed(de, -1.0, 0.0), R0, REL * head);
    EXPECT_NEAR(directional_ellipse_speed(de, 0.0, 1.0), R0, REL * head);
    EXPECT_NEAR(directional_ellipse_speed(de, 0.0, -1.0), R0, REL * head);
    EXPECT_NEAR(de.e.b + de.e.c, head, REL * head);
}

TEST(DirectionalShape, EllipseIsItsOwnWulffShape)
{
    const DirectionalRosState st = rothermel_state();
    const DirectionalEllipse de = directional_ellipse_cell(st, 1.5, 0.0, 0.0, 0.0);
    auto ellipse = [&](double nx, double ny) {
        return directional_ellipse_speed(de, static_cast<amrex::Real>(nx), static_cast<amrex::Real>(ny));
    };
    const double head = ellipse(1.0, 0.0);
    EXPECT_NEAR(wulff_extent_x(ellipse), head, 10.0 * REL * head);
    // backing: the same along -x
    auto reversed = [&](double nx, double ny) { return ellipse(-nx, -ny); };
    EXPECT_NEAR(wulff_extent_x(reversed), ellipse(-1.0, 0.0), 10.0 * REL * head);
}

TEST(DirectionalShape, EllipseAddsMisalignedWindAndSlope)
{
    const DirectionalRosState st = rothermel_state();
    const amrex::Real U = 1.0, s = 0.4;
    // wind along +x, upslope along +y
    const DirectionalEllipse de = directional_ellipse_cell(st, U, 0.0, 0.0, s);
    const double R0 = directional_ros_cell(st, 0.0, 0.0);
    const double dR_w = directional_ros_cell(st, U, 0.0) - R0;
    const double dR_s = directional_ros_cell(st, 0.0, s) - R0;
    const double P = std::sqrt(dR_w * dR_w + dR_s * dR_s);
    EXPECT_NEAR(de.hx, dR_w / P, REL);
    EXPECT_NEAR(de.hy, dR_s / P, REL);
    const double head = R0 + P;   // Rothermel is additive, so g = 1
    EXPECT_NEAR(directional_ellipse_speed(de, de.hx, de.hy), head, REL * head);
    EXPECT_NEAR(directional_ellipse_speed(de, -de.hx, -de.hy), R0, REL * head);
    EXPECT_NEAR(directional_ellipse_speed(de, -de.hy, de.hx), R0, REL * head);
}

TEST(DirectionalShape, EllipseCalmIsCircle)
{
    const DirectionalRosState st = rothermel_state();
    const DirectionalEllipse de = directional_ellipse_cell(st, 0.0, 0.0, 0.0, 0.0);
    const double R0 = st.rc.R0;
    const double pi = amrex::Math::pi<double>();
    for (int n = 0; n < 360; n += 15) {
        const double th = n * pi / 180.0;
        EXPECT_NEAR(directional_ellipse_speed(de, static_cast<amrex::Real>(std::cos(th)),
                                              static_cast<amrex::Real>(std::sin(th))), R0, REL * R0);
    }
}

TEST(DirectionalShape, EllipseMacArthurHeadAndBacking)
{
    // MacArthur takes no slope: head macarthur_ros(U), back and flanks its
    // no-wind rate.
    DirectionalRosState st;
    st.model = DIRECTIONAL_ROS_MACARTHUR;
    const amrex::Real U = 2.0;
    const DirectionalEllipse de = directional_ellipse_cell(st, 0.0, U, 0.5, 0.0);
    const double head = macarthur_ros(U);
    const double back = macarthur_ros(0.0);
    EXPECT_NEAR(de.hy, 1.0, REL);
    EXPECT_NEAR(directional_ellipse_speed(de, 0.0, 1.0), head, REL * head);
    EXPECT_NEAR(directional_ellipse_speed(de, 0.0, -1.0), back, REL * head);
    EXPECT_NEAR(directional_ellipse_speed(de, 1.0, 0.0), back, REL * head);
}

/// Relative tolerance of an inverted wind factor (a power law or a bisection).
static constexpr double REL_INV = (sizeof(amrex::Real) == 8) ? 1e-7 : 1e-3;

TEST(DirectionalShape, AndersonFlankKeepsHeadAndBack)
{
    const DirectionalRosState st = rothermel_state();
    const amrex::Real U = 1.5;
    const DirectionalEllipse dm = directional_ellipse_cell(st, U, 0.0, 0.0, 0.0);
    const DirectionalEllipse da = directional_ellipse_cell(st, U, 0.0, 0.0, 0.0,
                                                           DIRECTIONAL_ELLIPSE_LW_ANDERSON, 8.0);
    const double head = directional_ros_cell(st, U, 0.0);
    const double R0 = directional_ros_cell(st, 0.0, 0.0);
    const double LB = anderson_LW_ratio(static_cast<amrex::Real>(U * 2.23694));
    EXPECT_NEAR(da.e.LB, LB, REL_INV * LB);
    // head and back as with the model's flanks; only the flank rate changes
    EXPECT_NEAR(directional_ellipse_speed(da, 1.0, 0.0), head, REL * head);
    EXPECT_NEAR(directional_ellipse_speed(da, -1.0, 0.0), R0, REL * head);
    EXPECT_NEAR(directional_ellipse_speed(da, 0.0, 1.0), dm.e.b / LB, REL_INV * head);
    EXPECT_GT(directional_ellipse_speed(da, 0.0, 1.0), directional_ellipse_speed(dm, 0.0, 1.0));
}

TEST(DirectionalShape, EffectiveWindInvertsTheWindFactor)
{
    const DirectionalRosState st = rothermel_state();
    // wind alone: the effective wind is the wind
    const amrex::Real U = 1.5;
    const double R_w = directional_ros_cell(st, U, 0.0);
    EXPECT_NEAR(directional_effective_wind(st, static_cast<amrex::Real>(R_w)), U, REL_INV * U);
    // slope alone: the wind that gives the same head rate
    const double R_s = directional_ros_cell(st, 0.0, 0.4);
    const double U_eff = directional_effective_wind(st, static_cast<amrex::Real>(R_s));
    EXPECT_NEAR(directional_ros_cell(st, static_cast<amrex::Real>(U_eff), 0.0), R_s, REL_INV * R_s);
    // no wind or slope: none
    EXPECT_NEAR(directional_effective_wind(st, st.rc.R0), 0.0, 1e-12);
    // the model's wind limit caps it: the slope's effective wind is about
    // 250 ft/min, so a 200 ft/min limit binds
    DirectionalRosState capped = st;
    capped.rc.U_max_ftmin = 200.0;
    EXPECT_NEAR(directional_effective_wind(capped, static_cast<amrex::Real>(R_s)), 200.0 / 196.85, REL_INV);
}

TEST(DirectionalShape, EffectiveWindBisectionAndCap)
{
    DirectionalRosState st;
    st.model = DIRECTIONAL_ROS_MACARTHUR;
    const amrex::Real U = 2.0;
    EXPECT_NEAR(directional_effective_wind(st, macarthur_ros(U)), U, 1e-4);
    // a strong wind saturates Anderson's fit at 8; lw_max caps it lower
    const DirectionalRosState ro = rothermel_state();
    const DirectionalEllipse d8 = directional_ellipse_cell(ro, 20.0, 0.0, 0.0, 0.0,
                                                           DIRECTIONAL_ELLIPSE_LW_ANDERSON, 8.0);
    const DirectionalEllipse d3 = directional_ellipse_cell(ro, 20.0, 0.0, 0.0, 0.0,
                                                           DIRECTIONAL_ELLIPSE_LW_ANDERSON, 3.0);
    EXPECT_NEAR(d8.e.LB, 8.0, REL);
    EXPECT_NEAR(d3.e.LB, 3.0, REL);
    EXPECT_NEAR(directional_ellipse_speed(d3, 0.0, 1.0), d3.e.b / 3.0, REL * d3.e.b);
}
