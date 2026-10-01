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
audit_geometry (const TurbineState& t, const MovingBodyInputs& b, Real epsilon,
                const std::array<Real,3>& prob_lo, const std::array<Real,3>& prob_hi,
                const std::array<Real,3>& dx, const std::array<int,3>& periodic, Real ground_z)
{
    std::vector<AuditFinding> f;
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
        if (b.mode == "alm" && t.num_force_pts_blade > 0) {
            const Real spacing = R / static_cast<Real>(t.num_force_pts_blade);
            if (spacing > epsilon) {
                f.push_back({false, "actuator-line points " + fmt(spacing) + " m apart along the blade are farther apart than the kernel width " +
                                    fmt(epsilon) + " m; raise num_force_points_blade (or epsilon) so the line is continuous"});
            }
        }
    }
    if (epsilon < dmin) {
        f.push_back({false, "the kernel width epsilon = " + fmt(epsilon) + " m is narrower than the smallest cell (" + fmt(dmin) +
                            " m); the spread force is not resolved, use epsilon >= 1 (in cells), 2 is usual"});
    }
    return f;
}

std::vector<AuditFinding>
audit_overlap (const std::vector<TurbineState>& turbs)
{
    std::vector<AuditFinding> f;
    for (std::size_t i = 0; i < turbs.size(); ++i) {
        const Real Ri = erf_actuator::tip_radius(turbs[i]);
        for (std::size_t j = i + 1; j < turbs.size(); ++j) {
            const Real Rj = erf_actuator::tip_radius(turbs[j]);
            Real d2 = 0.0, along = 0.0;
            for (int c = 0; c < 3; ++c) {
                const Real dd = turbs[j].hub_pos[c] - turbs[i].hub_pos[c];
                d2 += dd * dd;
                along += dd * turbs[i].hub_axis[c];
            }
            // the discs intersect when the hubs are closer across the axis than the two radii and
            // closer along it than half of them (a rotor right behind another counts too)
            const Real lateral = std::sqrt(std::max(d2 - along * along, Real(0.0)));
            if (lateral < Ri + Rj && std::abs(along) < Real(0.5) * (Ri + Rj)) {
                f.push_back({true, "the swept discs of " + turbs[i].name + " and " + turbs[j].name + " overlap: hubs " + fmt(lateral) +
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

void
report_audit (const std::string& name, const std::vector<AuditFinding>& findings, bool& any_fatal)
{
    Print() << "OpenFAST input audit for " << name << ": " << findings.size() << " finding(s)\n";
    for (const auto& f : findings) {
        Print() << "  " << (f.fatal ? "ERROR: " : "note: ") << f.message << "\n";
        if (f.fatal) { any_fatal = true; }
    }
}

} // namespace erf_openfast
