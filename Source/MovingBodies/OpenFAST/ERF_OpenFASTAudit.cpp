#include "ERF_OpenFASTAudit.H"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

#include <AMReX.H>
#include <AMReX_Print.H>

#include "ERF_OpenFASTRotor.H"

using namespace amrex;

namespace erf_openfast {

std::string
openfast_value (const std::string& fname, const std::string& key)
{
    std::ifstream in(fname);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string value, name;
        if (!(ls >> value >> name)) { continue; }
        if (name == key) {
            if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
                value = value.substr(1, value.size() - 2);
            }
            return value;
        }
    }
    return {};
}

std::string
openfast_module_path (const std::string& fst_file, const std::string& key)
{
    const std::string f = openfast_value(fst_file, key);
    if (f.empty()) { return {}; }
    const auto slash = fst_file.find_last_of('/');
    if (slash != std::string::npos && f.front() != '/') { return fst_file.substr(0, slash + 1) + f; }
    return f;
}

bool
is_openfast_model (const std::string& fst_file)
{
    return !openfast_value(fst_file, "CompAero").empty() || !openfast_value(fst_file, "AirDens").empty();
}

namespace {
bool to_real (const std::string& s, Real& v)
{
    if (s.empty()) { return false; }
    char* end = nullptr;
    const double d = std::strtod(s.c_str(), &end);
    if (end == s.c_str()) { return false; }
    v = static_cast<Real>(d);
    return true;
}
std::string fmt (Real v)
{
    std::ostringstream os;
    os.precision(6);
    os << v;
    return os.str();
}
} // namespace

bool
openfast_air_density (const std::string& fst_file, Real& rho)
{
    if (to_real(openfast_value(fst_file, "AirDens"), rho)) { return true; }
    const std::string aero = openfast_module_path(fst_file, "AeroFile");
    if (aero.empty()) { return false; }
    return to_real(openfast_value(aero, "AirDens"), rho);
}

bool
openfast_gravity (const std::string& fst_file, Real& g)
{
    return to_real(openfast_value(fst_file, "Gravity"), g);
}

std::vector<AuditFinding>
audit_model (const MovingBodyInputs& b, Real erf_gravity)
{
    std::vector<AuditFinding> f;
    if (!is_openfast_model(b.fst_file)) {
        f.push_back({false, "'" + b.fst_file + "' is not an OpenFAST primary input file (a stub deck): the model-file checks are skipped"});
        return f;
    }
    Real rho = 0.0;
    if (openfast_air_density(b.fst_file, rho)) {
        if (std::abs(rho - b.air_density) > Real(1.0e-4) * rho) {
            f.push_back({true, "the model's AirDens is " + fmt(rho) + " kg/m^3 but erf.moving_bodies." + b.name + ".air_density is " +
                               fmt(b.air_density) + "; set air_density to the model's value (it scales the nacelle drag and the lifting-line correction)"});
        }
    } else {
        f.push_back({false, "the model gives no numeric AirDens in '" + b.fst_file + "' or its AeroDyn file; the density checks are skipped"});
    }
    const std::string comp_aero = openfast_value(b.fst_file, "CompAero");
    if (b.mode != "none" && comp_aero == "0") {
        f.push_back({true, "CompAero = 0 in '" + b.fst_file + "': the model computes no aerodynamic loads, yet mode = " + b.mode +
                           " puts its loads into the flow; set CompAero = 2 (AeroDyn) or mode = none"});
    }
    Real g = 0.0;
    if (openfast_gravity(b.fst_file, g) && std::abs(g - erf_gravity) > Real(0.01) * erf_gravity) {
        f.push_back({false, "the model's Gravity is " + fmt(g) + " m/s^2 but ERF uses " + fmt(erf_gravity)});
    }
    if (b.num_force_points_tower > 0 && b.mode != "none") {
        const std::string aero = openfast_module_path(b.fst_file, "AeroFile");
        const std::string twr = aero.empty() ? std::string() : openfast_value(aero, "TwrAero");
        if (!twr.empty() && (twr == "False" || twr == "FALSE" || twr == "false" || twr == "F")) {
            f.push_back({false, "num_force_points_tower = " + std::to_string(b.num_force_points_tower) + " but the AeroDyn file sets TwrAero = " + twr +
                                ": the tower force nodes will carry no load"});
        }
    }
    return f;
}

std::vector<AuditFinding>
audit_geometry (const TurbineState& t, const MovingBodyInputs& b, const KernelWidths& eps,
                const std::array<Real,3>& prob_lo, const std::array<Real,3>& prob_hi,
                const std::array<Real,3>& dx, const std::array<int,3>& periodic, Real ground_z)
{
    std::vector<AuditFinding> f;
    const std::string key = "erf.moving_bodies." + b.name + ".";   // the prefix of the body's keys
    const Real epsilon = eps.rotor;
    const Real R = erf_actuator::tip_radius(t);
    const Real D = Real(2.0) * R;
    const Real dmin = std::min(dx[0], std::min(dx[1], dx[2]));
    const char* axes[3] = {"x", "y", "z"};
    for (int d = 0; d < 2; ++d) {
        if (b.base_pos[d] < prob_lo[d] || b.base_pos[d] > prob_hi[d]) {
            f.push_back({true, "base_pos lies outside the domain in " + std::string(axes[d]) + " (" + fmt(b.base_pos[d]) + " not in [" +
                               fmt(prob_lo[d]) + ", " + fmt(prob_hi[d]) + "])"});
        }
    }
    if (R > Real(0.0)) {
        // the swept disc: the hub +- R in every direction (a bound that ignores the tilt)
        if (t.hub_pos[2] - R < ground_z) {
            f.push_back({true, "the rotor reaches below the ground: hub height " + fmt(t.hub_pos[2]) + " m minus the tip radius " + fmt(R) +
                               " m is under the terrain surface at z = " + fmt(ground_z) + " m beneath the hub"});
        }
        if (t.hub_pos[2] + R > prob_hi[2]) {
            f.push_back({true, "the rotor reaches above the domain top: hub height " + fmt(t.hub_pos[2]) + " m plus the tip radius " + fmt(R) +
                               " m exceeds prob_hi z = " + fmt(prob_hi[2])});
        }
        for (int d = 0; d < 2; ++d) {
            if (periodic[d]) { continue; }
            if (t.hub_pos[d] - R < prob_lo[d] || t.hub_pos[d] + R > prob_hi[d]) {
                f.push_back({true, "the rotor crosses the non-periodic " + std::string(axes[d]) + " boundary (hub " + fmt(t.hub_pos[d]) +
                                   " m, tip radius " + fmt(R) + " m)"});
            } else if (t.hub_pos[d] - R - Real(3.0) * epsilon < prob_lo[d] || t.hub_pos[d] + R + Real(3.0) * epsilon > prob_hi[d]) {
                f.push_back({false, "the spreading kernel (3 epsilon = " + fmt(Real(3.0) * epsilon) + " m beyond the tips) is cut by the non-periodic " +
                                    std::string(axes[d]) + " boundary; the force is renormalised but its shape is not the kernel's there"});
            }
        }
        if (D / dmin < Real(8.0)) {
            f.push_back({false, "only " + fmt(D / dmin) + " cells across the rotor diameter (" + fmt(D) + " m over " + fmt(dmin) +
                                " m cells); the rotor is under-resolved below about 8"});
        }
        // the actuator-disk rings: their points at the tip must overlap, or the disk becomes a ring of blobs
        if (b.mode == "adm" && b.num_points_t > 0) {
            const Real ring_spacing = Real(2.0) * Real(3.14159265358979323846) * R / static_cast<Real>(b.num_points_t);
            if (ring_spacing > Real(2.0) * epsilon) {
                f.push_back({false, "actuator-disk ring points " + fmt(ring_spacing) + " m apart at the tip are more than two kernel "
                                    "widths (" + fmt(epsilon) + " m) apart, so the disk's force is lumpy there; raise " + key + "num_points_t to "
                                    "about " + std::to_string(static_cast<int>(std::ceil(Real(3.14159265358979323846) * R / epsilon))) +
                                    " so the disk is smooth"});
            }
        }
        if (b.mode == "alm" && t.num_force_pts_blade > 0) {
            const Real spacing = R / static_cast<Real>(t.num_force_pts_blade);
            if (spacing > epsilon) {
                f.push_back({false, "actuator-line points " + fmt(spacing) + " m apart along the blade are farther apart than the kernel width " +
                                    fmt(epsilon) + " m; raise " + key + "num_force_points_blade (or " + key + "epsilon) so the line is continuous"});
            }
        }
    }
    // every kernel the body spreads with: the rotor's, and the tower's and the nacelle's when they are forced
    // (the tower's only when the model has tower force nodes to spread)
    const bool forces = (b.mode != "none");
    const bool tower = forces && b.num_force_points_tower > 0 && t.num_force_pts_tower > 0;
    const bool nacelle = forces && b.nacelle_cd > Real(0.0);
    std::vector<std::pair<std::string,Real>> widths{{"epsilon", eps.rotor}};
    if (tower) { widths.push_back({"epsilon_tower", eps.tower}); }
    if (nacelle) { widths.push_back({"epsilon_nacelle", eps.nacelle}); }
    for (const auto& kw : widths) {
        if (kw.second < dmin) {
            // an actuator line with the lifting-line correction is run that narrow on purpose (about the chord)
            const bool line = (kw.first == "epsilon" && b.mode == "alm");
            f.push_back({false, "the kernel width " + kw.first + " = " + fmt(kw.second) + " m is narrower than the smallest cell (" +
                                fmt(dmin) + " m), so the mesh does not resolve the spread force" +
                                (line ? std::string("; an actuator line with the lifting-line correction may be run so (a width "
                                                    "about the chord), otherwise use ") + key + kw.first + " >= 1 (in cells)"
                                      : "; use " + key + kw.first + " >= 1 (in cells)")});
        }
    }
    if (forces && b.num_force_points_tower > 0 && t.num_force_pts_tower == 0) {
        f.push_back({false, key + "num_force_points_tower = " + std::to_string(b.num_force_points_tower) + " but the model has no "
                            "tower force nodes, so no tower force is put into the flow and " + key + "epsilon_tower is not used"});
    }
    // the tower's force nodes, after the hub's and the blades', must overlap like the line's
    if (tower) {
        const int first = 1 + t.num_blades * t.num_force_pts_blade;
        const int last = std::min(t.num_force_nodes, first + t.num_force_pts_tower);
        Real gap = 0.0;
        for (int nd = first + 1; nd < last && 3 * nd + 2 < static_cast<int>(t.force_pos.size()); ++nd) {
            Real d2 = 0.0;
            for (int c = 0; c < 3; ++c) { const Real d = t.force_pos[3*nd+c] - t.force_pos[3*(nd-1)+c]; d2 += d * d; }
            gap = std::max(gap, std::sqrt(d2));
        }
        if (gap > eps.tower) {
            f.push_back({false, "tower force points up to " + fmt(gap) + " m apart are farther apart than the tower's kernel width " +
                                fmt(eps.tower) + " m; raise " + key + "num_force_points_tower (or " + key + "epsilon_tower) so the tower "
                                "is continuous"});
        }
    }
    // the nacelle's drag point lowers the velocity it samples by the factor 1 - cd area / (4 pi eps^2), which the
    // drag divides out. That is a linear estimate: along the axis Bernoulli gives u^2 = U^2 (1 - 2 cd area / (4 pi eps^2)),
    // so the centre stagnates when the factor reaches 0.5, and the drag fed back through the corrected velocity is
    // then no longer a small change: the factor must stay at 0.5 or above
    if (nacelle) {
        const Real cda = b.nacelle_cd * b.nacelle_area;
        const Real lim = Real(4.0) * Real(3.14159265358979323846) * eps.nacelle * eps.nacelle;
        if (!(Real(2.0) * cda <= lim)) {
            const std::string given = (b.epsilon_nacelle > Real(0.0)) ? std::string("epsilon_nacelle") : std::string("epsilon, as epsilon_nacelle is not given");
            f.push_back({true, key + "nacelle_cd * " + key + "nacelle_area = " + fmt(cda) + " m^2 exceeds 2 pi epsilon_nacelle^2 = " +
                               fmt(Real(0.5) * lim) + " m^2 (the nacelle's kernel is " + fmt(eps.nacelle) + " m wide, from " + given +
                               "): the kernel correction of the hub velocity, 1 - cd area / (4 pi epsilon_nacelle^2) = " +
                               fmt(Real(1.0) - cda / lim) + ", is a linear estimate valid only at 0.5 or above; widen " + key +
                               "epsilon_nacelle (a factor of 0.8 or more on two cells or more is accurate)"});
        }
    }
    return f;
}

std::vector<AuditFinding>
audit_overlap (const std::vector<TurbineState>& turbs)
{
    std::vector<RotorDisc> discs;
    for (const auto& t : turbs) { discs.push_back({t.name, t.hub_pos, t.hub_axis, erf_actuator::tip_radius(t)}); }
    return audit_overlap(discs);
}

std::vector<AuditFinding>
audit_overlap (const std::vector<RotorDisc>& discs)
{
    std::vector<AuditFinding> f;
    for (std::size_t i = 0; i < discs.size(); ++i) {
        const Real Ri = discs[i].tip_radius;
        for (std::size_t j = i + 1; j < discs.size(); ++j) {
            const Real Rj = discs[j].tip_radius;
            Real d2 = 0.0, along = 0.0;
            for (int c = 0; c < 3; ++c) {
                const Real dd = discs[j].hub[c] - discs[i].hub[c];
                d2 += dd * dd;
                along += dd * discs[i].axis[c];
            }
            // the discs intersect when the hubs are closer across the axis than the two radii and
            // closer along it than half of them (a rotor right behind another counts too)
            const Real lateral = std::sqrt(std::max(d2 - along * along, Real(0.0)));
            if (lateral < Ri + Rj && std::abs(along) < Real(0.5) * (Ri + Rj)) {
                f.push_back({true, "the swept discs of " + discs[i].name + " and " + discs[j].name + " overlap: hubs " + fmt(lateral) +
                                   " m apart across the axis and " + fmt(std::abs(along)) + " m along it, tip radii " + fmt(Ri) + " and " + fmt(Rj) + " m"});
            }
        }
    }
    return f;
}

std::vector<AuditFinding>
audit_density (const std::string& name, Real rho_model, Real rho_hub, Real tolerance)
{
    std::vector<AuditFinding> f;
    const Real rel = std::abs(rho_hub - rho_model) / rho_model;
    if (rel > tolerance) {
        f.push_back({true, "ERF's density at the hub of " + name + " is " + fmt(rho_hub) + " kg/m^3 but the OpenFAST model uses " + fmt(rho_model) +
                           " (" + fmt(Real(100.0) * rel) + " % apart, tolerance " + fmt(Real(100.0) * tolerance) +
                           " %); the loads OpenFAST computes with its density go into a flow of ERF's: match them (prob.rho_0, the sounding, or AirDens), "
                           "or widen erf.moving_bodies.density_tolerance"});
    }
    return f;
}

std::vector<AuditFinding>
audit_facing (const TurbineState& t, const std::array<Real,3>& u_hub)
{
    std::vector<AuditFinding> f;
    const Real speed = std::sqrt(u_hub[0]*u_hub[0] + u_hub[1]*u_hub[1] + u_hub[2]*u_hub[2]);
    if (speed < Real(0.1)) { return f; }
    const Real c = (t.hub_axis[0]*u_hub[0] + t.hub_axis[1]*u_hub[1] + t.hub_axis[2]*u_hub[2]) / speed;
    const Real deg = std::acos(std::max(Real(-1.0), std::min(Real(1.0), c))) * Real(180.0) / Real(3.14159265358979323846);
    if (c < Real(0.0)) {
        f.push_back({false, "the rotor of " + t.name + " faces away from the wind at its hub (shaft axis against the flow, " + fmt(deg) +
                            " degrees); check the yaw and the base position"});
    } else if (deg > Real(30.0)) {
        f.push_back({false, "the rotor of " + t.name + " is yawed " + fmt(deg) + " degrees from the wind at its hub"});
    }
    return f;
}

std::string
format_audit (const std::string& name, const std::vector<AuditFinding>& findings, bool& any_fatal)
{
    std::string text = "OpenFAST input audit for " + name + ": " + std::to_string(findings.size()) + " finding(s)\n";
    for (const auto& f : findings) {
        text += std::string("  ") + (f.fatal ? "ERROR: " : "note: ") + f.message + "\n";
        if (f.fatal) { any_fatal = true; }
    }
    return text;
}

void
report_audit (const std::string& name, const std::vector<AuditFinding>& findings, bool& any_fatal)
{
    Print() << format_audit(name, findings, any_fatal);
}

Real
model_rotor_radius (const std::string& fst_file)
{
    Real r = -1.0;
    const std::string ed = openfast_module_path(fst_file, "EDFile");
    if (!ed.empty() && to_real(openfast_value(ed, "TipRad"), r) && r > Real(0.0)) { return r; }
    // the bundled stub's deck: "rotor_radius = R"
    std::ifstream in(fst_file);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string key, eq, value;
        if ((ls >> key >> eq >> value) && key == "rotor_radius" && eq == "=" && to_real(value, r) && r > Real(0.0)) { return r; }
    }
    return Real(-1.0);
}

Real
model_time_step (const std::string& fst_file)
{
    Real dt = -1.0;
    if (to_real(openfast_value(fst_file, "DT"), dt) && dt > Real(0.0)) { return dt; }
    // the bundled stub's deck: "dt = DT"
    std::ifstream in(fst_file);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string key, eq, value;
        if ((ls >> key >> eq >> value) && key == "dt" && eq == "=" && to_real(value, dt) && dt > Real(0.0)) { return dt; }
    }
    return Real(-1.0);
}

} // namespace erf_openfast
