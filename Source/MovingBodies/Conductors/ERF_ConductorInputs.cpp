#include "ERF_ConductorInputs.H"

#include <cmath>
#include <set>

#include <AMReX.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

namespace erf_conductors {

Real SpanInputs::chord () const
{
    Real c2 = 0.0;
    for (int d = 0; d < 3; ++d) { c2 += (end_b[d] - end_a[d]) * (end_b[d] - end_a[d]); }
    return std::sqrt(c2);
}

Real SpanInputs::catenary_sag () const
{
    const Real c = chord();
    return (length > c) ? std::sqrt(Real(3.0) * c * (length - c) / Real(8.0)) : Real(0.0);
}

std::string ConductorInputs::validate_span (const SpanInputs& s)
{
    const std::string key = "erf.conductors." + s.name + ".";
    const Real c = s.chord();
    if (!(c > 0.0)) { return key + "end_a and end_b must be distinct points"; }
    if (!(s.length > c)) {
        return key + "length (" + std::to_string(s.length) + " m) must exceed the distance between the ends (" +
               std::to_string(c) + " m): a span hangs with slack";
    }
    if (!(s.diameter > 0.0)) { return key + "diameter must be positive (m)"; }
    if (!(s.mass_per_length > 0.0)) { return key + "mass_per_length must be positive (kg/m)"; }
    if (!(s.axial_stiffness > 0.0)) { return key + "axial_stiffness must be positive (N)"; }
    if (s.drag_coefficient < 0.0) { return key + "drag_coefficient must be >= 0"; }
    if (!(s.damping_ratio > 0.0 && s.damping_ratio <= 1.0)) { return key + "damping_ratio must be in (0, 1] (fraction of critical)"; }
    if (s.segments < 2) { return key + "segments must be >= 2"; }
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
    for (const SpanInputs& s : in.spans) {
        if (s.end_a[2] >= in.surface_offset || s.end_b[2] >= in.surface_offset) {
            return "erf.conductors." + s.name + ": the attachment heights must stay below surface_offset (" +
                   std::to_string(in.surface_offset) + " m), MoorDyn's free surface";
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

    std::vector<std::string> names;
    pp.queryarr("spans", names);
    in.active = !names.empty();
    if (!in.active) { return in; }

    pp.query("diagnostics_dir", in.diagnostics_dir);
    pp.query("diagnostics_int", in.diagnostics_int);
    pp.query("anchor_level", in.anchor_level);
    pp.query("air_density", in.air_density);
    pp.query("substeps", in.substeps);
    pp.query("moordyn_dt", in.moordyn_dt);
    pp.query("moordyn_cfl", in.moordyn_cfl);
    pp.query("moordyn_log_level", in.moordyn_log_level);
    pp.query("surface_offset", in.surface_offset);
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
        ps.get("length", s.length);
        ps.get("diameter", s.diameter);
        ps.get("mass_per_length", s.mass_per_length);
        ps.get("axial_stiffness", s.axial_stiffness);
        ps.query("drag_coefficient", s.drag_coefficient);
        ps.query("damping_ratio", s.damping_ratio);
        ps.query("segments", s.segments);
        s.output_root = in.diagnostics_dir + "/" + name;
        ps.query("output_root", s.output_root);
        const std::string err = validate_span(s);
        if (!err.empty()) { Abort(err); }
        in.spans.push_back(s);
    }
    const std::string err = validate_settings(in);
    if (!err.empty()) { Abort(err); }
    return in;
}

} // namespace erf_conductors
