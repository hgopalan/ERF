#include <ERF_FireEnginePrerequisites.H>
#include <ERF_NumericalSchemes.H>
#include <ERF.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using namespace amrex;

void verify_levelset_viscous_stability(const FireParams& fp, Real dx, Real dy)
{
    if (fp.propagation_method != "levelset") { return; }
    using fire_levelset::LevelSetViscousStability;
    auto num = [] (Real v) { std::ostringstream o; o << std::setprecision(4) << v; return o.str(); };
    const Real h = amrex::min(dx, dy);
    const Real eps_v = fp.levelset_eps_visc;
    const Real eps_f = fp.levelset_eps_visc_front;
    const bool front_hybrid = (fp.levelset_gradient == "weno5z_front");
    const bool band_on      = front_hybrid && fp.levelset_weno_band_cells > 0;
    // Where each kind of difference acts (FireLayer: band = weno_band_cells h,
    // WENO for |phi| <= band; LevelSetGradient::eps_at: eps_visc_front for
    // |phi| <= d0, eps_visc for |phi| >= d1, linear in between).
    const bool all_first = (fp.levelset_gradient == "upwind");
    const bool all_weno  = (fp.levelset_gradient == "weno5z") || (front_hybrid && !band_on);

    struct Coef { std::string key; Real eps; bool weno; std::string where; bool blended; };
    std::vector<Coef> coefs;
    if (all_first || all_weno) {
        coefs.push_back({"eps_visc", eps_v, all_weno, "away from the front", false});
        if (eps_f >= 0.0) {
            coefs.push_back({"eps_visc_front", eps_f, all_weno, "within erf.fire.levelset.visc_front_cells of the front", false});
        }
    } else {
        // weno5z_front: first order beyond the band, where the coefficient
        // runs from its value at the band's edge to eps_visc; WENO5 inside,
        // where it runs from eps_visc_front to that same edge value. First
        // order is the tighter limit at every cfl, so a value checked with it
        // needs no WENO5 check as well.
        coefs.push_back({"eps_visc", eps_v, false, "away from the front", false});
        if (eps_f >= 0.0) {
            const Real band = fp.levelset_weno_band_cells * h;
            const Real d0   = fp.levelset_visc_front_cells * h;
            const Real d1   = d0 + fp.levelset_visc_transition_cells * h;
            // the coefficient just beyond the band's edge
            Real eps_edge = eps_v;
            if (band < d0)      { eps_edge = eps_f; }
            else if (band < d1) { eps_edge = eps_f + (band - d0) / (d1 - d0) * (eps_v - eps_f); }
            coefs.push_back({"eps_visc_front", eps_f, true, "near the front, inside the WENO band", false});
            if (eps_edge > eps_v) {
                const std::string where = (band < d0)
                    ? "within erf.fire.levelset.visc_front_cells of the front, beyond the WENO band "
                      "(weno_band_cells < visc_front_cells)"
                    : "just beyond the WENO band (weno_band_cells = " + std::to_string(fp.levelset_weno_band_cells)
                      + "), inside the blend to eps_visc that ends visc_front_cells + visc_transition_cells = "
                      + num(fp.levelset_visc_front_cells + fp.levelset_visc_transition_cells) + " cells from the front";
                coefs.push_back({"eps_visc_front", eps_edge, false, where, eps_edge != eps_f});
            }
        }
    }
    // With erf.fire.levelset.wall_extrapolate, one_sided_derivatives falls
    // back to first-order differences wherever its stencil reaches a
    // non-burnable cell, so a coefficient that acts with WENO5 elsewhere also
    // acts with first-order differences next to the mask.
    if (fp.levelset_wall_extrapolate) {
        const std::size_t n = coefs.size();
        for (std::size_t i = 0; i < n; ++i) {
            if (!coefs[i].weno) { continue; }
            Coef c = coefs[i];
            c.weno = false;
            c.where += ", next to a non-burnable cell (erf.fire.levelset.wall_extrapolate = true falls back to "
                       "first-order differences there)";
            coefs.push_back(c);
        }
    }
    for (const Coef& c : coefs) {
        if (fire_levelset::levelset_substep_gain(fp.levelset_cfl, c.eps, dx, dy, c.weno)
            <= 1.0 + fire_levelset::LEVELSET_GAIN_TOL) { continue; }
        const LevelSetViscousStability st =
            fire_levelset::levelset_viscous_stability(fp.levelset_cfl, c.eps, dx, dy, c.weno, true);
        const std::string what = (c.blended ? "the viscosity " + num(c.eps) + " m, blended from erf.fire.levelset."
                                              + c.key + " = " + num(eps_f) + " m,"
                                            : "erf.fire.levelset." + c.key + " = " + num(c.eps) + " m") + " " + c.where
            + " (" + (c.weno ? "WENO5" : "first-order") + " differences there, erf.fire.levelset.gradient = "
            + fp.levelset_gradient + ") with erf.fire.levelset.cfl = " + num(fp.levelset_cfl) + " on "
            + num(dx) + " x " + num(dy) + " m fire cells";
        const std::string limits = (st.eps_max > 0.0)
            ? "Keep that viscosity <= " + num(st.eps_max) + " m (" + num(st.eps_max / h)
              + " fire cells), or erf.fire.levelset.cfl <= " + num(st.cfl_max) + "."
            : "No viscosity is stable at this cfl with these differences; keep erf.fire.levelset.cfl <= "
              + num(st.cfl_max) + ".";
        Abort("[FIRE] The level-set substep is unstable: " + what + ". The term R eps lap(phi) and the "
              "upwind |grad phi|, stepped with RK3 at dt = cfl h / R_max, grow a short wave by a factor "
              + num(st.gain) + " per substep on the least favourable front direction (near the cell "
              "diagonal), which every curved front and every windless fire has. " + limits);
    }
}

void verify_native_fire_prerequisites(const ERF& erf,
                                      int lev,
                                      const FireParams& fire_params)
{
    // Check 11: the dust layer and the fire-dust coupling of this repository live
    // on level 0 and share the fire grid's index space; they cannot follow a
    // finer fire grid. (Upstream ERF's shared checks know nothing of the dust
    // layer, so the check lives with the native engine.)
    if (lev > 0) {
        ParmParse pp_dust("erf.dust");
        bool dust_enable = false;
        pp_dust.query("enable", dust_enable);
        if (dust_enable) {
            Abort("[FIRE] The dust layer and the fire-dust coupling run on level 0, so they cannot "
                  "run with the fire grid on level " + std::to_string(lev) + " (erf.dust.enable = true). "
                  "Set erf.fire.anchor_level = 0, or turn the dust layer off.");
        }
    }

    // Check 12: Domain depth > the height the wind is sampled at: the target
    // height, or erf.fire.wind_sample_ht when the wind is sampled there and
    // brought down. The depth, not the top: a height above ground compared
    // with the absolute top let a domain from z = 156 to 160 m pass a 6.1 m
    // sampling height. Necessary, not sufficient: over terrain the column
    // above the highest ground is shallower than the depth.
    Real depth_z = erf.Geom(lev).ProbHi(2) - erf.Geom(lev).ProbLo(2);
    Real sample_z = amrex::max(fire_params.sampled_wind_ht(), fire_params.wind_sample_ht);
    std::string msg12 = std::string("[FIRE] Domain depth ")
                      + std::to_string(depth_z)
                      + " m (geometry.prob_hi(2) - geometry.prob_lo(2)) should exceed the wind's sampling height "
                      + std::to_string(sample_z)
                      + " m (erf.fire.wind_ref_ht, or 6.096 m with erf.fire.use_per_fuel_wind_ht,"
                        " or erf.fire.wind_sample_ht when set)";
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(depth_z > sample_z, msg12.c_str());

    // Check 13: divisors and step sizes read straight from the inputs file.
    // ParmParse applies no bounds, and a zero reaches the solver as an integer
    // division by zero (SIGFPE) or, for cfl, a zero-length subcycle that never
    // terminates. Each is only checked on the path that actually uses it.
    if (fire_params.propagation_method == "levelset") {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            fire_params.levelset_reinit_every >= 1,
            "[FIRE] erf.fire.levelset.reinit_every must be >= 1; it is used as a "
            "modulus divisor, so 0 is an integer division by zero");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            fire_params.levelset_cfl > 0.0,
            "[FIRE] erf.fire.levelset.cfl must be > 0; a non-positive value gives a "
            "zero-length subcycle and the level-set loop never advances");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            fire_params.levelset_reinit_iters >= 0,
            "[FIRE] erf.fire.levelset.reinit_iters must be >= 0");
        // the artificial viscosity against the substep's stability limit, on
        // the fire cells (grid_ratio >= 1 is checked above)
        verify_levelset_viscous_stability(fire_params,
                                          erf.Geom(lev).CellSize(0) / fire_params.grid_ratio,
                                          erf.Geom(lev).CellSize(1) / fire_params.grid_ratio);
    }

    if (fire_params.spotting.enable) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            fire_params.spotting.spotting_interval >= 1,
            "[FIRE] erf.fire.spotting.spotting_interval must be >= 1; it is used as "
            "a modulus divisor, so 0 is an integer division by zero");
    }

    // Check 14: the FARSITE subcycle divides by cfl_fire in the same way.
    if (fire_params.propagation_method != "levelset") {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            fire_params.farsite_cfl_fire > 0.0,
            "[FIRE] erf.fire.farsite.cfl_fire must be > 0; a non-positive value "
            "makes every fire substep zero-length");
    }

    amrex::Print() << "[FIRE] All prerequisites verified successfully" << std::endl;
}
