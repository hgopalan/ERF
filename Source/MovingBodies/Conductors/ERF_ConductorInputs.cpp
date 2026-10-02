#include "ERF_ConductorInputs.H"

#include <cmath>
#include <limits>
#include <set>

#include <AMReX.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

namespace erf_conductors {

const std::array<Real,3>& SpanInputs::point (int k) const
{
    if (k == 0) { return end_a; }
    if (k == num_spans()) { return end_b; }
    return towers[static_cast<std::size_t>(k - 1)];
}

std::array<Real,3>& SpanInputs::point (int k)
{
    if (k == 0) { return end_a; }
    if (k == num_spans()) { return end_b; }
    return towers[static_cast<std::size_t>(k - 1)];
}

Real SpanInputs::chord (int k) const
{
    const auto& a = point(k);
    const auto& b = point(k + 1);
    Real c2 = 0.0;
    for (int d = 0; d < 3; ++d) { c2 += (b[d] - a[d]) * (b[d] - a[d]); }
    return std::sqrt(c2);
}

std::array<Real,3> SpanInputs::conductor_point (int k) const
{
    std::array<Real,3> p = point(k);
    if (has_insulators() && k > 0 && k < num_spans()) { p[2] -= insulator_length; }
    return p;
}

void SpanInputs::lengths_from_stringing_tension (Real w)
{
    if (!(stringing_tension > 0.0)) { return; }
    lengths.resize(static_cast<std::size_t>(num_spans()));
    const Real H = stringing_tension;
    for (int k = 0; k < num_spans(); ++k) {
        const auto a = conductor_point(k);
        const auto b = conductor_point(k + 1);
        const Real h = std::sqrt((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]));
        const Real c = std::sqrt(h * h + (b[2] - a[2]) * (b[2] - a[2]));
        const Real extra = w * w * h * h * h * h / (Real(24.0) * H * H * c);
        // a vertical span (h = 0) has no parabola: it hangs straight under its weight's stretch
        const Real stretch = (h > 0.0) ? H * c / (axial_stiffness * h) : Real(0.0);
        lengths[static_cast<std::size_t>(k)] = (c + extra) / (Real(1.0) + stretch);
    }
}

Real SpanInputs::catenary_sag (int k) const
{
    const Real c = chord(k);
    const Real L = lengths[static_cast<std::size_t>(k)];
    return (L > c) ? std::sqrt(Real(3.0) * c * (L - c) / Real(8.0)) : Real(0.0);
}

int SpanInputs::num_line_nodes () const
{
    const int strings = has_insulators() ? static_cast<int>(towers.size()) : 0;
    return num_spans() * (segments + 1) + strings * (insulator_segments + 1);
}

std::string SpanInputs::span_root (int k) const
{
    return (num_spans() == 1) ? output_root : output_root + "_span" + std::to_string(k + 1);
}

std::string SpanInputs::span_name (int k) const
{
    return (num_spans() == 1) ? name : name + "_span" + std::to_string(k + 1);
}

Catenary elastic_catenary (Real chord, Real length, Real w, Real EA)
{
    Catenary cat;
    if (!(chord > 0.0 && length > 0.0 && w > 0.0 && EA > 0.0)) { return cat; }
    // for a horizontal tension H the catenary parameter is a = H / w, the stretched length
    // L_s = 2 a sinh(c / 2a), and the unstretched length L_s - (H / EA) (c/2 + (a/2) sinh(c/a));
    // the latter falls as H grows, slack line or taut, so the H that gives the line's length is
    // found by bisection (on log H, in double precision whatever Real is)
    const double c = chord, L = length;
    auto unstretched = [&](double H) {
        const double a = H / static_cast<double>(w);
        const double x = c / (2.0 * a);
        if (x > 300.0) { return std::numeric_limits<double>::max(); }   // so slack that sinh overflows
        return 2.0 * a * std::sinh(x) - H / static_cast<double>(EA) * (0.5 * c + 0.5 * a * std::sinh(c / a));
    };
    double lo = std::log(1.0e-6 * static_cast<double>(w) * c), hi = std::log(1.0e3 * static_cast<double>(EA));
    for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (unstretched(std::exp(mid)) > L) { lo = mid; } else { hi = mid; }
    }
    const double H = std::exp(0.5 * (lo + hi));
    const double a = H / static_cast<double>(w);
    cat.horizontal_tension = static_cast<Real>(H);
    cat.sag = static_cast<Real>(a * (std::cosh(c / (2.0 * a)) - 1.0));
    cat.end_tension = static_cast<Real>(H * std::cosh(c / (2.0 * a)));
    cat.stretched_length = static_cast<Real>(2.0 * a * std::sinh(c / (2.0 * a)));
    return cat;
}

std::string ConductorInputs::validate_slack (const SpanInputs& s, bool on_terrain)
{
    if (s.stringing_tension > 0.0) { return std::string(); }
    for (int k = 0; k < s.num_spans(); ++k) {
        const Real c = s.chord(k);
        const std::string which = (s.num_spans() == 1) ? "" : " of span " + std::to_string(k + 1);
        const Real L = s.lengths[static_cast<std::size_t>(k)];
        if (!(L > c)) {
            return "erf.conductors." + s.name + ".length" + which + " (" + std::to_string(L) +
                   " m) must exceed the distance between its ends (" + std::to_string(c) + " m" +
                   (on_terrain ? std::string(" where they stand on the terrain") : std::string()) + "): a span hangs with slack";
        }
    }
    return std::string();
}

std::string ConductorInputs::validate_span (const SpanInputs& s, bool check_slack)
{
    const std::string key = "erf.conductors." + s.name + ".";
    if (s.stringing_tension < 0.0) { return key + "stringing_tension must be positive (N), or 0 with length"; }
    if (s.stringing_tension > 0.0 && !s.lengths.empty()) { return key + "length and stringing_tension both given: give one of them"; }
    if (s.stringing_tension > 0.0) {
        // the lengths come from the chords once the attachments stand on the terrain
    } else if (static_cast<int>(s.lengths.size()) != s.num_spans()) {
        return key + "length needs one unstretched length per span (" + std::to_string(s.num_spans()) + " for " +
               std::to_string(s.towers.size()) + " tower(s)), " + std::to_string(s.lengths.size()) + " given";
    }
    for (int k = 0; k < s.num_spans(); ++k) {
        const std::string which = (s.num_spans() == 1) ? "" : " of span " + std::to_string(k + 1);
        if (!(s.chord(k) > 0.0)) {
            return key + (s.num_spans() == 1 ? std::string("end_a and end_b must be distinct points")
                                             : "the attachment points" + which + " must be distinct (end_a, towers, end_b)");
        }
    }
    if (check_slack) {
        const std::string err = validate_slack(s);
        if (!err.empty()) { return err; }
    }
    if (!(s.diameter > 0.0)) { return key + "diameter must be positive (m)"; }
    if (!(s.mass_per_length > 0.0)) { return key + "mass_per_length must be positive (kg/m)"; }
    if (!(s.axial_stiffness > 0.0)) { return key + "axial_stiffness must be positive (N)"; }
    if (s.drag_coefficient < 0.0) { return key + "drag_coefficient must be >= 0"; }
    if (!(s.damping_ratio > 0.0 && s.damping_ratio <= 1.0)) { return key + "damping_ratio must be in (0, 1] (fraction of critical)"; }
    if (s.segments < 2) { return key + "segments must be >= 2"; }
    if (s.insulator_length < 0.0) { return key + "insulator_length must be >= 0 (m; 0: the conductor is clamped at the towers)"; }
    if (s.insulator_length > 0.0) {
        if (s.towers.empty()) {
            return key + "insulator_length needs towers: a line is dead-ended at end_a and end_b and hangs from "
                         "insulator strings only at the towers between them";
        }
        if (!(s.insulator_mass > 0.0)) { return key + "insulator_mass must be positive (kg per string) with insulator_length"; }
        if (!(s.insulator_diameter > 0.0)) { return key + "insulator_diameter must be positive (m)"; }
        for (std::size_t t = 0; t < s.towers.size(); ++t) {
            if (!(s.towers[t][2] > s.insulator_length)) {
                return key + "insulator_length (" + std::to_string(s.insulator_length) + " m) must be less than the height of tower " +
                       std::to_string(t + 1) + " above the terrain (" + std::to_string(s.towers[t][2]) + " m)";
            }
        }
    }
    return std::string();
}

std::string ConductorInputs::validate_transformer (const TransformerInputs& t)
{
    const std::string key = "erf.conductors." + t.name + ".";
    if (!(t.size[0] > 0.0 && t.size[1] > 0.0 && t.size[2] > 0.0)) { return key + "size needs a positive length, width and height (m)"; }
    if (t.allowable_force < 0.0) { return key + "allowable_force must be >= 0 (N; 0: not checked)"; }
    if (t.allowable_moment < 0.0) { return key + "allowable_moment must be >= 0 (N m; 0: not checked)"; }
    return std::string();
}

std::string ConductorInputs::validate_settings (const ConductorInputs& in)
{
    if (in.diagnostics_int < 1) { return "erf.conductors.diagnostics_int must be >= 1"; }
    if (in.anchor_level < -1) { return "erf.conductors.anchor_level must be a level (0 .. amr.max_level) or -1 for the finest"; }
    if (!(in.air_density > 0.0)) { return "erf.conductors.air_density must be positive (kg/m^3)"; }
    if (in.substeps < 1) { return "erf.conductors.substeps must be >= 1"; }
    if (in.moordyn_dt < 0.0) { return "erf.conductors.moordyn_dt must be >= 0 (s; 0: no bound beyond moordyn_cfl)"; }
    if (!(in.moordyn_cfl > 0.0 && in.moordyn_cfl <= max_moordyn_cfl)) {
        return "erf.conductors.moordyn_cfl must be in (0, " + std::to_string(max_moordyn_cfl) +
               "]: larger Courant factors make MoorDyn's line integration diverge or give a wrong sag";
    }
    if (in.moordyn_log_level < 0 || in.moordyn_log_level > 3) { return "erf.conductors.moordyn_log_level must be 0 (debug) to 3 (errors only)"; }
    if (!(in.surface_offset > 0.0)) { return "erf.conductors.surface_offset must be positive (m)"; }
    if (in.stats_start < 0.0) { return "erf.conductors.stats_start must be >= 0 (s)"; }
    if (in.node_output_int < 0) { return "erf.conductors.node_output_int must be >= 0 (0: no node output)"; }
    if (!(in.epsilon > 0.0)) { return "erf.conductors.epsilon must be positive (cells)"; }
    if (!(in.flashover_distance > 0.0)) { return "erf.conductors.flashover_distance must be positive (m)"; }
    for (const SpanInputs& s : in.spans) {
        for (int k = 0; k <= s.num_spans(); ++k) {
            if (s.point(k)[2] >= in.surface_offset) {
                return "erf.conductors." + s.name + ": the attachment heights must stay below surface_offset (" +
                       std::to_string(in.surface_offset) + " m), MoorDyn's free surface";
            }
        }
    }
    return std::string();
}

std::string ConductorInputs::validate_solver (int max_level, int anchor_level, bool fpe_traps)
{
    if (anchor_level < 0 || anchor_level > max_level) {
        return "erf.conductors.anchor_level = " + std::to_string(anchor_level) + " is not a level of this run (0 .. " +
               std::to_string(max_level) + ")";
    }
    if (fpe_traps) {
        return "erf.conductors: MoorDyn's initial-condition solver overflows an intermediate value and is killed by a "
               "floating-point trap; switch off amrex.fpe_trap_invalid, amrex.fpe_trap_zero and amrex.fpe_trap_overflow";
    }
    return std::string();
}

ConductorInputs ConductorInputs::read ()
{
    ConductorInputs in;
    ParmParse pp("erf.conductors");

    std::vector<std::string> names, tnames;
    pp.queryarr("spans", names);
    pp.queryarr("transformers", tnames);
    in.active = !names.empty();
    if (!in.active) {
        if (!tnames.empty()) { Abort("erf.conductors.transformers needs lines ending on them (erf.conductors.spans)"); }
        return in;
    }

    pp.query("diagnostics_dir", in.diagnostics_dir);
    pp.query("diagnostics_int", in.diagnostics_int);
    pp.query("anchor_level", in.anchor_level);
    pp.query("air_density", in.air_density);
    pp.query("substeps", in.substeps);
    pp.query("moordyn_dt", in.moordyn_dt);
    pp.query("moordyn_cfl", in.moordyn_cfl);
    pp.query("moordyn_log_level", in.moordyn_log_level);
    pp.query("surface_offset", in.surface_offset);
    pp.query("stats_start", in.stats_start);
    pp.query("node_output_int", in.node_output_int);
    pp.query("drag_on_flow", in.drag_on_flow);
    pp.query("epsilon", in.epsilon);
    pp.query("flashover_distance", in.flashover_distance);
    std::vector<Real> vel;
    if (pp.queryarr("prescribed_velocity", vel)) {
        if (vel.size() != 3) { Abort("erf.conductors.prescribed_velocity needs three components (m/s)"); }
        in.has_prescribed_velocity = true;
        for (int d = 0; d < 3; ++d) { in.prescribed_velocity[d] = vel[d]; }
    }

    std::set<std::string> seen;
    for (const std::string& name : names) {
        if (!seen.insert(name).second) { Abort("erf.conductors.spans lists '" + name + "' twice"); }
        SpanInputs s;
        s.name = name;
        ParmParse ps("erf.conductors." + name);
        std::vector<Real> a, b;
        ps.getarr("end_a", a);
        ps.getarr("end_b", b);
        if (a.size() != 3 || b.size() != 3) { Abort("erf.conductors." + name + ".end_a and end_b need three components (m)"); }
        for (int d = 0; d < 3; ++d) { s.end_a[d] = a[d]; s.end_b[d] = b[d]; }
        std::vector<Real> t;
        ps.queryarr("towers", t);
        if (t.size() % 3 != 0) { Abort("erf.conductors." + name + ".towers needs three components (m) per tower"); }
        for (std::size_t i = 0; i < t.size(); i += 3) { s.towers.push_back({{t[i], t[i+1], t[i+2]}}); }
        ps.queryarr("length", s.lengths);
        ps.query("stringing_tension", s.stringing_tension);
        ps.get("diameter", s.diameter);
        ps.get("mass_per_length", s.mass_per_length);
        ps.get("axial_stiffness", s.axial_stiffness);
        ps.query("drag_coefficient", s.drag_coefficient);
        ps.query("damping_ratio", s.damping_ratio);
        ps.query("segments", s.segments);
        ps.query("insulator_length", s.insulator_length);
        ps.query("insulator_mass", s.insulator_mass);
        ps.query("insulator_diameter", s.insulator_diameter);
        s.output_root = in.diagnostics_dir + "/" + name;
        ps.query("output_root", s.output_root);
        // the slack is checked once the ends stand on the terrain: their heights here are above it
        const std::string err = validate_span(s, false);
        if (!err.empty()) { Abort(err); }
        in.spans.push_back(s);
    }
    for (const std::string& name : tnames) {
        // a transformer's block shares erf.conductors.<name> with the lines'
        if (!seen.insert(name).second) { Abort("erf.conductors: '" + name + "' names two lines or transformers"); }
        TransformerInputs t;
        t.name = name;
        ParmParse pt("erf.conductors." + name);
        std::vector<Real> p, sz;
        pt.getarr("position", p);
        pt.getarr("size", sz);
        if (p.size() != 2) { Abort("erf.conductors." + name + ".position needs two components, x and y (m)"); }
        if (sz.size() != 3) { Abort("erf.conductors." + name + ".size needs three components, length, width and height (m)"); }
        for (int d = 0; d < 2; ++d) { t.position[d] = p[d]; }
        for (int d = 0; d < 3; ++d) { t.size[d] = sz[d]; }
        pt.query("allowable_force", t.allowable_force);
        pt.query("allowable_moment", t.allowable_moment);
        const std::string terr = validate_transformer(t);
        if (!terr.empty()) { Abort(terr); }
        in.transformers.push_back(t);
    }
    const std::string err = validate_settings(in);
    if (!err.empty()) { Abort(err); }
    return in;
}

} // namespace erf_conductors
