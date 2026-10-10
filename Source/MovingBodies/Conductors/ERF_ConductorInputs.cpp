#include "ERF_ConductorInputs.H"
#include "ERF_ASCE74.H"
#include "ERF_Gusts.H"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <set>
#include <utility>

#include <AMReX.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

namespace erf_conductors {

namespace {
bool finite3 (const std::array<Real,3>& p) { return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]); }
}

const std::array<Real,3>& LineInputs::point (int k) const
{
    if (k == 0) { return end_a; }
    if (k == num_spans()) { return end_b; }
    return towers[static_cast<std::size_t>(k - 1)];
}

std::array<Real,3>& LineInputs::point (int k)
{
    if (k == 0) { return end_a; }
    if (k == num_spans()) { return end_b; }
    return towers[static_cast<std::size_t>(k - 1)];
}

Real LineInputs::chord (int k) const
{
    const auto& a = point(k);
    const auto& b = point(k + 1);
    Real c2 = 0.0;
    for (int d = 0; d < 3; ++d) { c2 += (b[d] - a[d]) * (b[d] - a[d]); }
    return std::sqrt(c2);
}

Real LineInputs::conductor_chord (int k) const
{
    const auto a = conductor_point(k);
    const auto b = conductor_point(k + 1);
    Real c2 = 0.0;
    for (int d = 0; d < 3; ++d) { c2 += (b[d] - a[d]) * (b[d] - a[d]); }
    return std::sqrt(c2);
}

std::array<Real,3> LineInputs::conductor_point (int k) const
{
    std::array<Real,3> p = point(k);
    if (has_insulators() && k > 0 && k < num_spans()) { p[2] -= insulator_length; }
    return p;
}

void LineInputs::lengths_from_stringing_tension (Real w)
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

Real LineInputs::catenary_sag (int k) const
{
    const Real c = conductor_chord(k);
    const Real L = lengths[static_cast<std::size_t>(k)];
    return (L > c) ? std::sqrt(Real(3.0) * c * (L - c) / Real(8.0)) : Real(0.0);
}

int LineInputs::num_line_nodes () const
{
    const int strings = has_insulators() ? static_cast<int>(towers.size()) : 0;
    return num_spans() * (segments + 1) + strings * (insulator_segments + 1);
}

std::string LineInputs::span_root (int k) const
{
    return (num_spans() == 1) ? output_root : output_root + "_span" + std::to_string(k + 1);
}

std::string LineInputs::span_name (int k) const
{
    return (num_spans() == 1) ? name : name + "_span" + std::to_string(k + 1);
}

GustType ConductorInputs::gust () const
{
    GustType g = GustType::None;
    return parse_gust_type(gust_type, g) ? g : GustType::None;
}

const LineInputs& ConductorInputs::gust_owner (const LineInputs& s) const
{
    // a line sharing towers takes the gusts of the towers' owner, and so those of the line the owner names in
    // gust_with; validate_settings refuses longer chains
    const LineInputs& t = tower_owner(s);
    const std::string& o = t.gust_with;
    if (o.empty()) { return t; }
    for (const auto& l : lines) { if (l.name == o) { return l; } }
    return t;
}

const LineInputs& ConductorInputs::tower_owner (const LineInputs& s) const
{
    if (s.share_towers.empty()) { return s; }
    for (const auto& o : lines) { if (o.name == s.share_towers) { return o; } }
    return s;   // checked when the inputs were read
}

const erf_towers::TowerType* ConductorInputs::tower_type (const LineInputs& s) const
{
    const LineInputs& owner = tower_owner(s);
    if (owner.tower_type.empty()) { return nullptr; }
    for (const auto& t : tower_types) { if (t.name == owner.tower_type) { return &t; } }
    return nullptr;
}

std::string ConductorInputs::validate_shared_towers (const std::vector<LineInputs>& lines)
{
    for (const auto& s : lines) {
        if (s.share_towers.empty()) { continue; }
        const std::string key = "erf.conductors." + s.name + ".share_towers";
        const LineInputs* owner = nullptr;
        for (const auto& o : lines) { if (o.name == s.share_towers) { owner = &o; } }
        if (owner == nullptr || owner == &s) {
            return key + " = " + s.share_towers + " is not another line of erf.conductors.lines";
        }
        if (!owner->share_towers.empty()) {
            return key + " = " + s.share_towers + ", which shares the towers of " + owner->share_towers +
                   "; name the line the towers belong to";
        }
        if (owner->tower_type.empty()) {
            return key + " = " + s.share_towers + ", which has no tower_type: only lattice towers are shared";
        }
        if (!s.tower_type.empty()) {
            return key + ": the line hangs from " + s.share_towers + "'s towers and takes their type; drop its own tower_type";
        }
        if (s.towers.size() != owner->towers.size()) {
            return key + ": the line has " + std::to_string(s.towers.size()) + " tower(s) and " + s.share_towers + " has " +
                   std::to_string(owner->towers.size()) + "; a line hangs from every tower of the line it shares them with";
        }
    }
    return std::string();
}

bool along_plus_x (const std::array<Real,3>& a, const std::array<Real,3>& b)
{
    const Real ax = b[0] - a[0], ay = b[1] - a[1];
    return ax > Real(0.0) && std::abs(ay) <= Real(1.0e-6) * ax;
}

Catenary elastic_catenary (Real chord, Real length, Real w, Real EA)
{
    Catenary cat;
    if (!(chord > 0.0 && length > 0.0 && w > 0.0 && EA > 0.0)) { return cat; }
    // for a horizontal tension H the catenary parameter is a = H / w, the stretched length
    // L_s = 2 a sinh(c / 2a), and the unstretched length L_s - (H / EA) (c/2 + (a/2) sinh(c/a)).
    // Where the line could hang, its end tension H cosh(c / 2a) is far below EA, and there the
    // unstretched length falls as H grows; on the slack side it turns back down once the stretch
    // term outgrows the arc (at end tensions from about EA to tens of EA), so every slack H whose
    // end tension reaches EA counts as too slack. The H that gives the line's length is then found by bisection (on
    // log H, in double precision whatever Real is)
    const double c = chord, L = length;
    auto unstretched = [&](double H) {
        const double a = H / static_cast<double>(w);
        const double x = c / (2.0 * a);
        if (x > 300.0) { return std::numeric_limits<double>::max(); }   // so slack that sinh overflows
        if (x > 1.0 && H * std::cosh(x) > static_cast<double>(EA)) { return std::numeric_limits<double>::max(); }
        return 2.0 * a * std::sinh(x) - H / static_cast<double>(EA) * (0.5 * c + 0.5 * a * std::sinh(c / a));
    };
    double lo = std::log(1.0e-6 * static_cast<double>(w) * c), hi = std::log(1.0e3 * static_cast<double>(EA));
    for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (unstretched(std::exp(mid)) > L) { lo = mid; } else { hi = mid; }
    }
    const double H = std::exp(0.5 * (lo + hi));
    const double a = H / static_cast<double>(w);
    // an answer on a bracket's edge or on the guard's seam is no answer: the model's stretch has no root
    // there (strains near 50 % and above), and the caller must not take it for a sag
    cat.solved = std::abs(unstretched(H) - L) <= 1.0e-6 * L;
    cat.horizontal_tension = static_cast<Real>(H);
    cat.sag = static_cast<Real>(a * (std::cosh(c / (2.0 * a)) - 1.0));
    cat.end_tension = static_cast<Real>(H * std::cosh(c / (2.0 * a)));
    cat.stretched_length = static_cast<Real>(2.0 * a * std::sinh(c / (2.0 * a)));
    return cat;
}

std::string ConductorInputs::validate_slack (const LineInputs& s, bool on_terrain)
{
    if (s.stringing_tension > 0.0) { return std::string(); }
    for (int k = 0; k < s.num_spans(); ++k) {
        // the span hangs between the points the conductor hangs from: the bottoms of the strings at the towers
        const auto a = s.conductor_point(k);
        const auto b = s.conductor_point(k + 1);
        const Real c = std::sqrt((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]) + (b[2] - a[2]) * (b[2] - a[2]));
        const std::string which = (s.num_spans() == 1) ? "" : " of span " + std::to_string(k + 1);
        const Real L = s.lengths[static_cast<std::size_t>(k)];
        if (!(L > c)) {
            return "erf.conductors." + s.name + ".length" + which + " (" + std::to_string(L) +
                   " m) must exceed the distance between the points the conductor hangs from (" + std::to_string(c) + " m" +
                   (on_terrain ? std::string(" where they stand on the terrain") : std::string()) +
                   (s.has_insulators() ? std::string("; the bottoms of the insulator strings at the towers") : std::string()) +
                   "): a span hangs with slack";
        }
    }
    return std::string();
}

std::string ConductorInputs::validate_output_roots (const std::vector<LineInputs>& lines)
{
    for (std::size_t i = 0; i < lines.size(); ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            if (lines[i].output_root == lines[j].output_root) {
                return "erf.conductors." + lines[i].name + ".output_root = " + lines[i].output_root + " is also " +
                       lines[j].name + "'s: two lines would write the same diagnostics files";
            }
        }
    }
    return std::string();
}

std::string ConductorInputs::validate_line (const LineInputs& s, bool check_slack)
{
    const std::string key = "erf.conductors." + s.name + ".";
    if (!finite3(s.end_a)) { return key + "end_a must be finite"; }
    if (!finite3(s.end_b)) { return key + "end_b must be finite"; }
    for (const auto& t : s.towers) { if (!finite3(t)) { return key + "towers must be finite"; } }
    for (const Real L : s.lengths) { if (!std::isfinite(L)) { return key + "length must be finite"; } }
    const std::pair<const char*, Real> scalars[] = {
        {"stringing_tension", s.stringing_tension}, {"diameter", s.diameter}, {"mass_per_length", s.mass_per_length},
        {"axial_stiffness", s.axial_stiffness}, {"drag_coefficient", s.drag_coefficient}, {"damping_ratio", s.damping_ratio},
        {"insulator_length", s.insulator_length}, {"insulator_mass", s.insulator_mass}, {"insulator_diameter", s.insulator_diameter}};
    for (const auto& kv : scalars) {
        if (!std::isfinite(kv.second)) { return key + kv.first + " must be finite"; }
    }
    if (s.stringing_tension < 0.0) { return key + "stringing_tension must be positive (N), or 0 with length"; }
    if (s.stringing_tension > 0.0 && !s.lengths.empty()) { return key + "length and stringing_tension both given: give one of them"; }
    if (s.stringing_tension > 0.0) {
        // the lengths come from the chords once the attachment points stand on the terrain
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
    if (!(s.diameter > 0.0)) { return key + "diameter must be positive (m)"; }
    if (!(s.mass_per_length > 0.0)) { return key + "mass_per_length must be positive (kg/m)"; }
    if (!(s.axial_stiffness > 0.0)) { return key + "axial_stiffness must be positive (N)"; }
    if (s.drag_coefficient < 0.0) { return key + "drag_coefficient must be >= 0"; }
    if (!(s.damping_ratio > 0.0 && s.damping_ratio <= 1.0)) { return key + "damping_ratio must be in (0, 1] (fraction of critical)"; }
    if (s.segments < 2) { return key + "segments must be >= 2"; }
    if (s.insulator_length < 0.0) { return key + "insulator_length must be >= 0 (m; 0: the conductor is clamped at the towers)"; }
    if (!(s.insulator_length > 0.0)) {
        if (s.insulator_mass_given) {
            return key + "insulator_mass is given but insulator_length is 0: the conductor is clamped at the towers and "
                         "the strings' mass is unused";
        }
        if (s.insulator_diameter_given) {
            return key + "insulator_diameter is given but insulator_length is 0: the conductor is clamped at the towers "
                         "and the strings' diameter is unused";
        }
    }
    if (s.insulator_length > 0.0) {
        if (s.towers.empty()) {
            return key + "insulator_length needs towers: a line is dead-ended at end_a and end_b and hangs from "
                         "insulator strings only at the towers between them";
        }
        if (!(s.insulator_mass > 0.0)) { return key + "insulator_mass must be positive (kg per string) with insulator_length"; }
        if (!(s.insulator_diameter > 0.0)) { return key + "insulator_diameter must be positive (m)"; }
        // whether each string's bottom clears the ground is checked once the towers stand on it (set_ground)
        // a string swings across the line, normal to the direction between the attachment points either side of its tower
        for (int j = 0; j + 2 <= s.num_spans(); ++j) {
            const auto& a = s.point(j);
            const auto& b = s.point(j + 2);
            if (!(std::hypot(b[0] - a[0], b[1] - a[1]) > 0.0)) {
                return key + "towers: the attachment points either side of tower " + std::to_string(j + 1) +
                       " stand at the same x, y, so the line turns back on itself there and the insulator string's "
                       "across-line direction is undefined";
            }
        }
    }
    // the slack last: it is measured to the bottoms of the insulator strings, so it needs them valid
    if (check_slack) {
        const std::string err = validate_slack(s);
        if (!err.empty()) { return err; }
    }
    return std::string();
}

std::string ConductorInputs::validate_transformer (const TransformerInputs& t)
{
    const std::string key = "erf.conductors." + t.name + ".";
    if (!(std::isfinite(t.position[0]) && std::isfinite(t.position[1]))) { return key + "position must be finite"; }
    if (!finite3(t.size)) { return key + "size must be finite"; }
    if (!std::isfinite(t.allowable_force)) { return key + "allowable_force must be finite"; }
    if (!std::isfinite(t.allowable_moment)) { return key + "allowable_moment must be finite"; }
    if (!(t.size[0] > 0.0 && t.size[1] > 0.0 && t.size[2] > 0.0)) { return key + "size needs a positive length, width and height (m)"; }
    if (t.allowable_force < 0.0) { return key + "allowable_force must be >= 0 (N; 0: not checked)"; }
    if (t.allowable_moment < 0.0) { return key + "allowable_moment must be >= 0 (N m; 0: not checked)"; }
    return std::string();
}

std::string ConductorInputs::validate_tower_type (const LineInputs& s, const std::vector<erf_towers::TowerType>& types)
{
    if (s.tower_type.empty()) { return std::string(); }
    const std::string key = "erf.conductors." + s.name + ".tower_type";
    if (s.towers.empty()) { return key + " needs towers: a single span between two dead ends has no tower"; }
    for (const auto& t : types) { if (t.name == s.tower_type) { return std::string(); } }
    return key + " = " + s.tower_type + " is not one of erf.conductors.tower_types";
}

std::string ConductorInputs::validate_settings (const ConductorInputs& in)
{
    const std::pair<const char*, Real> scalars[] = {
        {"air_density", in.air_density}, {"moordyn_dt", in.moordyn_dt}, {"moordyn_cfl", in.moordyn_cfl},
        {"surface_offset", in.surface_offset}, {"stats_start", in.stats_start}, {"epsilon", in.epsilon},
        {"flashover_distance", in.flashover_distance}, {"asce74_wind", in.asce74_wind},
        {"gust_sigma_factor", in.gust_sigma_factor}, {"gust_peak_factor", in.gust_peak_factor},
        {"gust_span_length_scale", in.gust_span_length_scale}, {"gust_event_time", in.gust_event_time},
        {"gust_event_speed", in.gust_event_speed}, {"gust_event_direction", in.gust_event_direction},
        {"gust_event_duration", in.gust_event_duration}, {"gust_integral_length", in.gust_integral_length},
        {"gust_event_origin", in.gust_event_origin[0]}, {"gust_event_origin", in.gust_event_origin[1]}};
    for (const auto& kv : scalars) {
        if (!std::isfinite(kv.second)) { return std::string("erf.conductors.") + kv.first + " must be finite"; }
    }
    if (in.has_prescribed_velocity && !finite3(in.prescribed_velocity)) {
        return "erf.conductors.prescribed_velocity must be finite";
    }
    if (in.diagnostics_int < 1) { return "erf.conductors.diagnostics_int must be >= 1"; }
    if (in.anchor_level < -1) { return "erf.conductors.anchor_level must be a level (0 .. amr.max_level) or -1 for amr.max_level"; }
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
    if (in.asce74_wind < 0.0) { return "erf.conductors.asce74_wind must be >= 0 (m/s; 0: no ASCE 74 table)"; }
    if (!in.asce74_exposure.empty()) {
        Exposure e = Exposure::C;
        if (!parse_exposure(in.asce74_exposure, e)) {
            return "erf.conductors.asce74_exposure must be B or C (ASCE 74's suburban or open-country exposure), not '" +
                   in.asce74_exposure + "'";
        }
        if (!(in.asce74_wind > 0.0)) { return "erf.conductors.asce74_exposure needs erf.conductors.asce74_wind, the gust it applies to"; }
    }
    GustType gust = GustType::None;
    if (!parse_gust_type(in.gust_type, gust)) {
        return "erf.conductors.gust_type must be none, factor, event or random, not '" + in.gust_type + "'";
    }
    // each gust key with the gust types that read it
    const bool event = (gust == GustType::Event), random = (gust == GustType::Random);
    const struct { const char* key; bool given; bool read; const char* needs; } gust_keys[] = {
        {"gust_sigma_factor", in.has_gust_sigma_factor, gust != GustType::None, "an erf.conductors.gust_type"},
        {"gust_peak_factor", in.has_gust_peak_factor, gust != GustType::None, "an erf.conductors.gust_type"},
        {"gust_span_length_scale", in.has_gust_span_length_scale, gust != GustType::None, "an erf.conductors.gust_type"},
        {"gust_event_time", in.has_gust_event_time, event, "erf.conductors.gust_type = event"},
        {"gust_event_speed", in.has_gust_event_speed, event, "erf.conductors.gust_type = event"},
        {"gust_event_direction", in.has_gust_event_direction, event, "erf.conductors.gust_type = event"},
        {"gust_event_origin", in.has_gust_event_origin, event, "erf.conductors.gust_type = event"},
        {"gust_event_duration", in.has_gust_event_duration, event, "erf.conductors.gust_type = event"},
        {"gust_seed", in.has_gust_seed, random, "erf.conductors.gust_type = random"},
        {"gust_integral_length", in.has_gust_integral_length, random, "erf.conductors.gust_type = random"}};
    for (const auto& kv : gust_keys) {
        if (kv.given && !kv.read) { return std::string("erf.conductors.") + kv.key + " needs " + kv.needs; }
    }
    if (in.has_gust_sigma_factor && !(in.gust_sigma_factor > 0.0)) { return "erf.conductors.gust_sigma_factor must be positive"; }
    if (!(in.gust_peak_factor > 0.0)) { return "erf.conductors.gust_peak_factor must be positive"; }
    if (!(in.gust_span_length_scale > 0.0)) { return "erf.conductors.gust_span_length_scale must be positive (m)"; }
    if (event) {
        if (!in.has_gust_event_time) {
            return "erf.conductors.gust_type = event needs erf.conductors.gust_event_time, when the gust's front crosses "
                   "gust_event_origin (s)";
        }
        if (!in.has_gust_event_speed) {
            return "erf.conductors.gust_type = event needs erf.conductors.gust_event_speed, how fast its front moves (m/s)";
        }
        if (!(in.gust_event_speed > 0.0)) { return "erf.conductors.gust_event_speed must be positive (m/s)"; }
        if (!(in.gust_event_duration > 0.0)) { return "erf.conductors.gust_event_duration must be positive (s)"; }
    }
    if (in.gust_seed < 0) { return "erf.conductors.gust_seed must be >= 0"; }
    if (in.has_gust_integral_length && !(in.gust_integral_length > 0.0)) {
        return "erf.conductors.gust_integral_length must be positive (m)";
    }
    if (gust != GustType::None && in.has_prescribed_velocity) {
        return "erf.conductors.gust_type = " + in.gust_type + " takes k from the flow, so it cannot be used with "
               "erf.conductors.prescribed_velocity";
    }
    if ((event || random) && in.drag_on_flow) {
        return "erf.conductors.gust_type = " + in.gust_type + " adds its gust to the wind of the lines and towers only; "
               "it cannot be used with erf.conductors.drag_on_flow, which would put the gust's drag into the RANS flow";
    }
    // a line taking another's random gusts: random only, another line of as many spans that takes no other's
    for (const LineInputs& s : in.lines) {
        if (s.gust_with.empty()) { continue; }
        const std::string key = "erf.conductors." + s.name + ".gust_with";
        if (!random) { return key + " needs erf.conductors.gust_type = random"; }
        if (!s.share_towers.empty()) {
            return key + ": the line shares " + s.share_towers + "'s towers and so already takes its gusts; drop gust_with";
        }
        const LineInputs* o = nullptr;
        for (const LineInputs& l : in.lines) { if (l.name == s.gust_with) { o = &l; } }
        if (o == nullptr || o == &s) { return key + " = " + s.gust_with + " is not another line of erf.conductors.lines"; }
        if (!o->gust_with.empty() || !o->share_towers.empty()) {
            return key + " = " + s.gust_with + ", which takes the gusts of " + (o->gust_with.empty() ? o->share_towers : o->gust_with) +
                   "; name that line";
        }
        if (o->num_spans() != s.num_spans()) {
            return key + ": the line has " + std::to_string(s.num_spans()) + " span(s) and " + s.gust_with + " has " +
                   std::to_string(o->num_spans()) + "; it takes that line's gusts span by span";
        }
    }
    if (gust != GustType::None) {
        // a line's gust statistics are <name>_gusts in the checkpoint and <output_root>_gusts_stats.csv: no other
        // line or span may carry those names
        for (const LineInputs& s : in.lines) {
            for (const LineInputs& o : in.lines) {
                bool clash = (o.name == s.name + "_gusts" || o.output_root == s.output_root + "_gusts");
                for (int k = 0; k < o.num_spans(); ++k) {
                    clash = clash || o.span_name(k) == s.name + "_gusts" || o.span_root(k) == s.output_root + "_gusts";
                }
                if (clash) {
                    return "erf.conductors." + o.name + ": its name or output_root clashes with line " + s.name +
                           "'s gust statistics (" + s.name + "_gusts); rename it";
                }
            }
        }
        // the gusts need the wind normal to each span, so a span needs a horizontal extent
        for (const LineInputs& s : in.lines) {
            for (int k = 0; k < s.num_spans(); ++k) {
                const auto a = s.conductor_point(k);
                const auto b = s.conductor_point(k + 1);
                if (!(std::hypot(b[0] - a[0], b[1] - a[1]) > Real(1.0e-6))) {
                    return "erf.conductors." + s.name + ": span " + std::to_string(k + 1) +
                           " has no horizontal extent, so the gusts have no wind normal to it";
                }
            }
        }
    }
    // the logs the run writes itself in diagnostics_dir, the towers' frame logs among them: no line's log may take
    // their names (the paths compared after lexical normalisation, so that conductors/./towers is conductors/towers)
    std::vector<std::string> own{"total_load", "separation", "transformers", "ground"};
    bool towers = false, moving = false;
    for (const LineInputs& s : in.lines) {
        const erf_towers::TowerType* type = in.tower_type(s);
        towers = towers || type != nullptr;
        moving = moving || (type != nullptr && type->moves());
        if (type == nullptr || !type->has_frame() || !s.share_towers.empty()) { continue; }
        for (std::size_t j = 1; j <= s.towers.size(); ++j) {
            const std::string t = s.name + "_t" + std::to_string(j);
            own.insert(own.end(), {"frame_" + t, "frame_" + t + "_members"});
            if (in.node_output_int > 0) { own.push_back("tower_" + t + "_frame"); }
        }
    }
    if (towers) { own.push_back("towers"); }
    if (moving) { own.push_back("coupling"); }
    if (in.gusts_in_wind()) { own.push_back("gust_series"); }
    auto normal = [] (const std::string& f) { return std::filesystem::path(f).lexically_normal().string(); };
    for (const LineInputs& s : in.lines) {
        std::vector<std::string> logs{normal(s.output_root + "_nodes.dat"), normal(s.output_root + "_insulators.dat")};
        for (int k = 0; k < s.num_spans(); ++k) { logs.push_back(normal(s.span_root(k) + ".dat")); }
        for (const std::string& o : own) {
            const std::string f = normal(in.diagnostics_dir + "/" + o + ".dat");
            if (std::find(logs.begin(), logs.end(), f) != logs.end()) {
                return "erf.conductors." + s.name + ": a log of the line would be " + f + ", which the run writes itself; "
                       "rename the line or set its output_root";
            }
        }
    }
    // the statistics the run keeps itself (transformers, pairs of lines, towers and their members), named in the
    // checkpoint and written to <diagnostics_dir>/<name>_stats.csv: no line's statistics may take either
    std::vector<std::string> own_stats;
    for (const auto& t : in.transformers) { own_stats.push_back("transformer_" + t.name); }
    for (std::size_t i = 0; i < in.lines.size(); ++i) {
        for (std::size_t j = i + 1; j < in.lines.size(); ++j) {
            own_stats.push_back("separation_" + in.lines[i].name + "-" + in.lines[j].name);
        }
    }
    for (const LineInputs& s : in.lines) {
        if (in.tower_type(s) == nullptr || !s.share_towers.empty()) { continue; }
        for (std::size_t j = 1; j <= s.towers.size(); ++j) {
            const std::string t = "tower_" + s.name + "_t" + std::to_string(j);
            own_stats.insert(own_stats.end(), {t, t + "_members"});
        }
    }
    for (const LineInputs& s : in.lines) {
        std::vector<std::pair<std::string, std::string>> mine{{s.name + "_insulators", s.output_root + "_insulators"},
                                                              {s.name + "_gusts", s.output_root + "_gusts"}};
        for (int k = 0; k < s.num_spans(); ++k) { mine.emplace_back(s.span_name(k), s.span_root(k)); }
        for (const auto& [name, root] : mine) {
            for (const std::string& o : own_stats) {
                if (name == o || normal(root + "_stats.csv") == normal(in.diagnostics_dir + "/" + o + "_stats.csv")) {
                    return "erf.conductors." + s.name + ": its statistics " + name + " would take the name or the file of "
                           "the run's own " + o + " statistics; rename the line or set its output_root";
                }
            }
        }
    }
    // a conductor lighter than the air it displaces has no still-air shape (and no elastic catenary)
    for (const LineInputs& s : in.lines) {
        const Real displaced = in.air_density * Real(0.25) * Real(3.14159265358979323846) * s.diameter * s.diameter;
        if (!(s.mass_per_length > displaced)) {
            return "erf.conductors." + s.name + ".mass_per_length (" + std::to_string(s.mass_per_length) +
                   " kg/m) must exceed the mass of the air the conductor displaces, air_density * pi * diameter^2 / 4 (" +
                   std::to_string(displaced) + " kg/m)";
        }
    }
    return std::string();
}

std::string ConductorInputs::validate_frame (Real surface_offset, Real prob_lo_z, Real prob_hi_z)
{
    if (!(surface_offset > prob_hi_z)) {
        return "erf.conductors.surface_offset (" + std::to_string(surface_offset) + " m) must exceed the domain top, "
               "geometry.prob_hi[2] (" + std::to_string(prob_hi_z) + " m): MoorDyn applies no fluid load above its free "
               "surface at ERF z = surface_offset";
    }
    if (!(-surface_offset < prob_lo_z)) {
        return "erf.conductors.surface_offset (" + std::to_string(surface_offset) + " m): -surface_offset must lie below "
               "the domain bottom, geometry.prob_lo[2] (" + std::to_string(prob_lo_z) + " m), so that no node reaches "
               "MoorDyn's flat bottom at ERF z = -surface_offset";
    }
    return std::string();
}

std::string ConductorInputs::validate_solver (int max_level, int anchor_level, bool fpe_traps, bool drag_on_flow)
{
    if (anchor_level < 0 || anchor_level > max_level) {
        return "erf.conductors.anchor_level = " + std::to_string(anchor_level) + " is not a level of this run (0 .. " +
               std::to_string(max_level) + ")";
    }
    if (drag_on_flow && anchor_level < max_level) {
        // the drag is spread on the anchor level only: a finer level never feels it, and its flow,
        // averaged down, replaces the coarse flow under it
        return "erf.conductors.drag_on_flow puts the lines' drag into the flow of the anchor level only, level " +
               std::to_string(anchor_level) + ", and amr.max_level = " + std::to_string(max_level) +
               " allows a finer level whose flow never feels it; step the lines on the finest level "
               "(erf.conductors.anchor_level = -1) or switch drag_on_flow off";
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

    if (pp.contains("spans")) {
        Abort("erf.conductors.spans is not an input: list the conductor lines in erf.conductors.lines");
    }
    std::vector<std::string> names, tnames, ttypes;
    pp.queryarr("lines", names);
    pp.queryarr("transformers", tnames);
    pp.queryarr("tower_types", ttypes);
    in.active = !names.empty();
    if (!in.active) {
        if (!tnames.empty()) { Abort("erf.conductors.transformers needs lines ending on them (erf.conductors.lines)"); }
        if (!ttypes.empty()) { Abort("erf.conductors.tower_types needs lines hanging from them (erf.conductors.lines)"); }
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
    const bool drag_given = pp.query("drag_on_flow", in.drag_on_flow) != 0;
    if (pp.query("epsilon", in.epsilon) != 0 && !in.drag_on_flow) {
        // the width the drag is spread over: nothing to act on without drag_on_flow
        if (drag_given) {
            Print() << "erf.conductors: WARNING: epsilon is not used: drag_on_flow is false\n";
        } else {
            Abort("erf.conductors.epsilon needs erf.conductors.drag_on_flow = true (the width the drag is spread over)");
        }
    }
    pp.query("flashover_distance", in.flashover_distance);
    pp.query("asce74_wind", in.asce74_wind);
    pp.query("asce74_exposure", in.asce74_exposure);
    pp.query("gust_type", in.gust_type);
    in.has_gust_sigma_factor = pp.query("gust_sigma_factor", in.gust_sigma_factor) != 0;
    in.has_gust_peak_factor = pp.query("gust_peak_factor", in.gust_peak_factor) != 0;
    in.has_gust_span_length_scale = pp.query("gust_span_length_scale", in.gust_span_length_scale) != 0;
    if (!in.has_gust_span_length_scale && !in.asce74_exposure.empty()) {
        // without an L_s of their own the gusts take the exposure of the run's ASCE 74 check, as asce74.csv does
        Exposure e = Exposure::C;
        if (parse_exposure(in.asce74_exposure, e)) { in.gust_span_length_scale = static_cast<Real>(exposure_constants(e).Ls); }
    }
    in.has_gust_event_time = pp.query("gust_event_time", in.gust_event_time) != 0;
    in.has_gust_event_speed = pp.query("gust_event_speed", in.gust_event_speed) != 0;
    in.has_gust_event_direction = pp.query("gust_event_direction", in.gust_event_direction) != 0;
    in.has_gust_event_duration = pp.query("gust_event_duration", in.gust_event_duration) != 0;
    in.has_gust_seed = pp.query("gust_seed", in.gust_seed) != 0;
    in.has_gust_integral_length = pp.query("gust_integral_length", in.gust_integral_length) != 0;
    {
        std::vector<Real> origin;
        if (pp.queryarr("gust_event_origin", origin)) {
            if (origin.size() != 2) { Abort("erf.conductors.gust_event_origin needs two coordinates, x y (m)"); }
            in.has_gust_event_origin = true;
            in.gust_event_origin = {{origin[0], origin[1]}};
        }
    }
    std::vector<Real> vel;
    if (pp.queryarr("prescribed_velocity", vel)) {
        if (vel.size() != 3) { Abort("erf.conductors.prescribed_velocity needs three components (m/s)"); }
        in.has_prescribed_velocity = true;
        for (int d = 0; d < 3; ++d) { in.prescribed_velocity[d] = vel[d]; }
    }

    std::set<std::string> seen;
    for (const std::string& name : names) {
        if (!seen.insert(name).second) { Abort("erf.conductors.lines lists '" + name + "' twice"); }
        LineInputs s;
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
        ps.query("tower_type", s.tower_type);
        ps.query("share_towers", s.share_towers);
        ps.query("gust_with", s.gust_with);
        ps.get("diameter", s.diameter);
        ps.get("mass_per_length", s.mass_per_length);
        ps.get("axial_stiffness", s.axial_stiffness);
        ps.query("drag_coefficient", s.drag_coefficient);
        ps.query("damping_ratio", s.damping_ratio);
        ps.query("segments", s.segments);
        ps.query("insulator_length", s.insulator_length);
        s.insulator_mass_given = ps.query("insulator_mass", s.insulator_mass) != 0;
        s.insulator_diameter_given = ps.query("insulator_diameter", s.insulator_diameter) != 0;
        s.output_root = in.diagnostics_dir + "/" + name;
        ps.query("output_root", s.output_root);
        // the slack is checked in Conductors::set_ground once the attachment points stand on the
        // terrain: here their z is still the height above the terrain
        const std::string err = validate_line(s, false);
        if (!err.empty()) { Abort(err); }
        in.lines.push_back(s);
    }
    {
        const std::string err = validate_output_roots(in.lines);
        if (!err.empty()) { Abort(err); }
    }
    for (const std::string& name : ttypes) {
        // a tower type's block shares erf.conductors.<name> with the lines' and the transformers'
        if (!seen.insert(name).second) { Abort("erf.conductors: '" + name + "' names two lines, transformers or tower types"); }
        erf_towers::TowerType t;
        t.name = name;
        ParmParse pt("erf.conductors." + name);
        pt.get("base_width", t.base_width);
        pt.get("top_width", t.top_width);
        pt.get("solidity", t.solidity);
        pt.get("arm_length", t.arm_length);
        pt.query("arm_depth", t.arm_depth);
        pt.query("peak", t.peak);
        pt.query("drag_coefficient", t.drag_coefficient);
        pt.query("segments", t.segments);
        pt.query("weight", t.weight);
        pt.query("leg_spacing", t.leg_spacing);
        pt.query("allowable_uplift", t.allowable_uplift);
        pt.query("allowable_compression", t.allowable_compression);
        t.frequency_given = pt.query("frequency", t.frequency) != 0;
        t.damping_given = pt.query("damping_ratio", t.damping_ratio) != 0;
        pt.query("foundation_rotational_stiffness", t.foundation_rotational_stiffness);
        pt.query("foundation_lateral_stiffness", t.foundation_lateral_stiffness);
        pt.query("frame_file", t.frame_file);
        pt.query("member_file", t.member_file);
        pt.query("frame_panels", t.frame_panels);
        pt.queryarr("leg_angle", t.leg_angle);
        pt.queryarr("brace_angle", t.brace_angle);
        pt.query("bracing", t.bracing);
        pt.query("yield_strength", t.yield_strength);
        pt.query("steel_temperature", t.steel_temperature);
        const std::string terr = t.validate();
        if (!terr.empty()) { Abort(terr); }
        const std::string unused = t.unused_on_a_still_tower();
        if (!unused.empty()) { Print() << "WARNING: " << unused << "\n"; }
        in.tower_types.push_back(t);
    }
    for (const LineInputs& s : in.lines) {
        const std::string err = validate_tower_type(s, in.tower_types);
        if (!err.empty()) { Abort(err); }
    }
    {
        const std::string err = validate_shared_towers(in.lines);
        if (!err.empty()) { Abort(err); }
    }
    for (const std::string& name : tnames) {
        // a transformer's block shares erf.conductors.<name> with the lines'
        if (!seen.insert(name).second) { Abort("erf.conductors: '" + name + "' names two lines, transformers or tower types"); }
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
