#include "ERF_MovingBodiesInputs.H"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <set>
#include <sstream>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

using namespace amrex;

namespace {

// a real input must be finite: "nan" and "inf" parse, and every check below is written to fail on them
void need_finite (const std::string& key, Real v)
{
    if (!std::isfinite(v)) { Abort(key + " must be a finite number"); }
}

// a key that is given but has no effect in this configuration: say so rather than ignore it
void warn_inactive (const ParmParse& pp, const std::string& prefix, const std::string& key, const std::string& why)
{
    if (pp.contains(key)) {
        Print() << "Warning: " << prefix << "." << key << " is given but not used: " << why << "\n";
    }
}

// a kernel width (cells) given for one kind of point: positive and finite
void read_width (const ParmParse& pp, const std::string& prefix, const std::string& key, Real& v)
{
    if (pp.query(key.c_str(), v) && (!(v > 0.0) || !std::isfinite(v))) {
        Abort(prefix + "." + key + " must be positive and finite (in cells of the anchor level, its x spacing)");
    }
}

// a component's width given while the component is off: a warning when its switch was set off
// explicitly, an abort when the switch was never set (the width then most likely meant the switch)
void check_inactive_width (const ParmParse& pp, const std::string& prefix, const std::string& key,
                           const std::string& switch_key, const std::string& need)
{
    if (!pp.contains(key)) { return; }
    if (pp.contains(switch_key)) {
        warn_inactive(pp, prefix, key, prefix + "." + switch_key + " = 0 puts no " + need + " in the flow");
    } else {
        Abort(prefix + "." + key + " is given but " + prefix + "." + switch_key + " is not; set " + prefix + "." + switch_key +
              " > 0 for the " + need + " it spreads, or leave " + prefix + "." + key + " out");
    }
}

// two record values agree: equal strings, or numbers equal to single precision (a checkpoint
// written by a double build may be continued by a single one)
bool same_record_value (const std::string& a, const std::string& b)
{
    if (a == b) { return true; }
    char* ea = nullptr;
    char* eb = nullptr;
    const double x = std::strtod(a.c_str(), &ea);
    const double y = std::strtod(b.c_str(), &eb);
    if (ea == a.c_str() || eb == b.c_str() || *ea != '\0' || *eb != '\0') { return false; }
    return std::abs(x - y) <= 1.0e-6 * std::max(1.0, std::max(std::abs(x), std::abs(y)));
}

} // namespace

MovingBodiesInputs
MovingBodiesInputs::read ()
{
    MovingBodiesInputs in;
    ParmParse pp("erf.moving_bodies");

    std::vector<std::string> names;
    pp.queryarr("bodies", names);
    in.active = !names.empty();
    if (!in.active) { return in; }

    pp.query("diagnostics_int", in.diagnostics_int);
    if (in.diagnostics_int < 1) {
        Abort("erf.moving_bodies.diagnostics_int must be >= 1");
    }
    pp.query("diagnostics_dir", in.diagnostics_dir);
    pp.query("diagnostics_format", in.diagnostics_format);
    if (!valid_diagnostics_format(in.diagnostics_format)) {
        Abort("erf.moving_bodies.diagnostics_format = " + in.diagnostics_format + ": use per_turbine, per_rank or farm");
    }

    {
        ParmParse pw("erf.moving_bodies.wake");
        pw.queryarr("lines_xD", in.wake.lines_xD);
        for (const Real xD : in.wake.lines_xD) {
            if (!(xD > 0.0) || !std::isfinite(xD)) { Abort("erf.moving_bodies.wake.lines_xD must be positive distances in rotor diameters"); }
        }
        pw.query("half_width", in.wake.half_width);
        if (!(in.wake.half_width > 0.0) || !std::isfinite(in.wake.half_width)) {
            Abort("erf.moving_bodies.wake.half_width must be > 0 (rotor diameters)");
        }
        pw.query("num_points", in.wake.num_points);
        if (in.wake.num_points < 2) { Abort("erf.moving_bodies.wake.num_points must be >= 2"); }
        pw.query("int", in.wake.interval);
        if (in.wake.interval < 0) { Abort("erf.moving_bodies.wake.int must be >= 0 (0 uses diagnostics_int)"); }
        if (in.wake.interval == 0) { in.wake.interval = in.diagnostics_int; }
        if (pw.contains("avg_start")) {
            Abort("erf.moving_bodies.wake.avg_start was renamed erf.moving_bodies.avg_start (it also starts the body statistics)");
        }
        if (in.wake.lines_xD.empty()) {
            for (const char* k : {"half_width", "num_points", "int"}) {
                warn_inactive(pw, "erf.moving_bodies.wake", k, "no wake lines (erf.moving_bodies.wake.lines_xD) are given");
            }
        }
    }
    pp.query("avg_start", in.avg_start);
    if (!(in.avg_start >= 0.0) || !std::isfinite(in.avg_start)) { Abort("erf.moving_bodies.avg_start must be a time >= 0 (s)"); }
    pp.query("anchor_level", in.anchor_level);
    if (in.anchor_level < -1) { Abort("erf.moving_bodies.anchor_level must be a level (0 .. amr.max_level) or -1 for the finest"); }
    pp.query("density_tolerance", in.density_tolerance);
    if (!(in.density_tolerance >= 0.0) || !std::isfinite(in.density_tolerance)) {
        Abort("erf.moving_bodies.density_tolerance must be >= 0 (relative)");
    }
    pp.query("envelope_check_interval", in.envelope_check_interval);
    if (in.envelope_check_interval < 1) {
        Abort("erf.moving_bodies.envelope_check_interval must be >= 1 (steps between two checks of the exchange envelopes)");
    }
    pp.query("alm_max_tip_cells", in.alm_max_tip_cells);
    if (!(in.alm_max_tip_cells > 0.0) || !std::isfinite(in.alm_max_tip_cells)) {
        Abort("erf.moving_bodies.alm_max_tip_cells must be > 0 (cells swept by a blade tip per step)");
    }
    pp.query("verify_tip_cells", in.verify_tip_cells);

    std::vector<Real> vel;
    if (pp.queryarr("prescribed_velocity", vel)) {
        if (vel.size() != 3) {
            Abort("erf.moving_bodies.prescribed_velocity needs three components");
        }
        in.has_prescribed_velocity = true;
        for (int d = 0; d < 3; ++d) { need_finite("erf.moving_bodies.prescribed_velocity", vel[d]); in.prescribed_velocity[d] = vel[d]; }
        if (!in.wake.lines_xD.empty()) {
            Print() << "Warning: erf.moving_bodies.wake.lines_xD is given but not used: with a prescribed velocity the "
                    << "flow is not sampled, so no wake lines are written\n";
        }
    }

    std::set<std::string> seen, roots;
    bool any_alm = false;
    for (const std::string& name : names) {
        if (!seen.insert(name).second) {
            Abort("erf.moving_bodies.bodies lists '" + name + "' twice");
        }
        MovingBodyInputs b;
        b.name = name;
        const std::string pre = "erf.moving_bodies." + name;
        ParmParse ppb(pre);

        ppb.get("type", b.type);
        if (b.type != "openfast_turbine" && b.type != "ct_disk") {
            Abort(pre + ".type = '" + b.type + "' is not known; this version supports openfast_turbine and ct_disk");
        }

        std::vector<Real> pos;
        ppb.getarr("base_pos", pos);
        if (pos.size() != 3) {
            Abort(pre + ".base_pos needs three components");
        }
        for (int d = 0; d < 3; ++d) { need_finite(pre + ".base_pos", pos[d]); b.base_pos[d] = pos[d]; }

        ppb.query("epsilon", b.epsilon);
        if (!(b.epsilon > 0.0) || !std::isfinite(b.epsilon)) {
            Abort(pre + ".epsilon must be positive and finite (in cells of the anchor level, its x spacing)");
        }

        b.output_root = in.diagnostics_dir + "/" + name;
        ppb.query("output_root", b.output_root);
        if (!roots.insert(b.output_root).second) {
            Abort(pre + ".output_root = '" + b.output_root + "' is the output_root of another body; their files would mix");
        }

        ppb.query("num_points_t", b.num_points_t);
        if (b.num_points_t < 2) {
            Abort(pre + ".num_points_t must be >= 2 (points per ring: one point cannot average the in-plane force away)");
        }

        if (b.type == "openfast_turbine") {
            ppb.get("fst_file", b.fst_file);

            ppb.query("mode", b.mode);
            if (b.mode != "adm" && b.mode != "alm" && b.mode != "none") {
                Abort(pre + ".mode must be adm, alm or none, not '" + b.mode + "'");
            }
            any_alm = any_alm || (b.mode == "alm");

            ppb.query("sampling", b.sampling);
            if (b.sampling.empty()) { b.sampling = (b.mode == "adm") ? "disk_corrected" : "disk"; }
            if (b.sampling != "disk" && b.sampling != "upstream" && b.sampling != "disk_corrected") {
                Abort(pre + ".sampling must be disk, upstream or disk_corrected, not '" + b.sampling + "'");
            }
            if (b.sampling != "disk" && b.mode != "adm") {
                Abort(pre + ".sampling = " + b.sampling + " needs mode = adm: an actuator line resolves its own induction");
            }
            if (b.mode != "adm") { warn_inactive(ppb, pre, "num_points_t", "the rings are for mode = adm"); }
            b.sample_diameters_upstream = 2.0;   // turbines: two diameters, where the rotor's upstream induction is below 1 %
            ppb.query("sample_diameters_upstream", b.sample_diameters_upstream);
            if (!(b.sample_diameters_upstream > 0.0) || !std::isfinite(b.sample_diameters_upstream)) {
                Abort(pre + ".sample_diameters_upstream must be positive (diameters ahead of the hub)");
            }
            if (b.sampling != "upstream") { warn_inactive(ppb, pre, "sample_diameters_upstream", "it is for sampling = upstream"); }
            ppb.query("correction_relax", b.correction_relax);
            if (!(b.correction_relax == -1.0 || (b.correction_relax > 0.0 && b.correction_relax <= 1.0))) {
                Abort(pre + ".correction_relax must be -1 (from the update's gain) or lie in (0, 1]");
            }
            if (b.sampling != "disk_corrected") { warn_inactive(ppb, pre, "correction_relax", "it is for sampling = disk_corrected"); }
            ppb.query("correction_time", b.correction_time);
            if (!(b.correction_time == -1.0 || (b.correction_time >= 0.0 && std::isfinite(b.correction_time)))) {
                Abort(pre + ".correction_time must be -1 (the rotor radius over the free stream), 0 (no time scale) or a time > 0 (s)");
            }
            if (b.sampling != "disk_corrected") {
                warn_inactive(ppb, pre, "correction_time", "it is for sampling = disk_corrected");
            } else if (b.correction_relax > 0.0) {
                warn_inactive(ppb, pre, "correction_time", "a fixed correction_relax is used as given");
            }
            ppb.query("num_force_points_blade", b.num_force_points_blade);
            if (b.num_force_points_blade < 1) {
                Abort(pre + ".num_force_points_blade must be >= 1");
            }
            ppb.query("num_force_points_tower", b.num_force_points_tower);
            if (b.num_force_points_tower < 0) {
                Abort(pre + ".num_force_points_tower must be >= 0");
            }
            ppb.query("nacelle_cd", b.nacelle_cd);
            if (!(b.nacelle_cd >= 0.0) || !std::isfinite(b.nacelle_cd)) {
                Abort(pre + ".nacelle_cd must be >= 0");
            }
            ppb.query("nacelle_area", b.nacelle_area);
            if (!(b.nacelle_area >= 0.0) || !std::isfinite(b.nacelle_area)) {
                Abort(pre + ".nacelle_area must be >= 0 (m^2)");
            }
            if (b.nacelle_cd > 0.0 && !(b.nacelle_area > 0.0)) {
                Abort(pre + ".nacelle_cd > 0 needs nacelle_area > 0 (m^2)");
            }
            read_width(ppb, pre, "epsilon_tower", b.epsilon_tower);
            read_width(ppb, pre, "epsilon_nacelle", b.epsilon_nacelle);
            if (b.mode == "none") {
                for (const char* k : {"nacelle_cd", "nacelle_area", "num_force_points_tower", "epsilon_tower", "epsilon_nacelle"}) {
                    warn_inactive(ppb, pre, k, "mode = none puts no force in the flow");
                }
            } else {
                if (b.num_force_points_tower == 0) {
                    check_inactive_width(ppb, pre, "epsilon_tower", "num_force_points_tower", "tower force");
                }
                if (!(b.nacelle_cd > 0.0)) {
                    check_inactive_width(ppb, pre, "epsilon_nacelle", "nacelle_cd", "nacelle drag");
                }
            }
            ppb.query("air_density", b.air_density);
            if (!(b.air_density > 0.0) || !std::isfinite(b.air_density)) {
                Abort(pre + ".air_density must be positive");
            }
            if (!ppb.query("fllc", b.fllc)) { b.fllc = (b.mode == "alm"); }   // an actuator line runs with the correction unless told not to
            if (b.fllc && b.mode != "alm") {
                Abort(pre + ".fllc = true needs mode = alm; the correction is for an actuator line");
            }
            if (b.fllc && b.num_force_points_blade < 2) {
                Abort(pre + ".num_force_points_blade must be >= 2 with fllc (the correction integrates along the blade)");
            }
            if (!b.fllc) {
                for (const char* k : {"fllc_relax", "fllc_start_time", "fllc_eps_chord", "fllc_eps_dr"}) {
                    warn_inactive(ppb, pre, k, "the lifting-line correction is off (fllc = false)");
                }
            }
            ppb.query("fllc_relax", b.fllc_relax);
            if (!(b.fllc_relax > 0.0 && b.fllc_relax <= 1.0)) {
                Abort(pre + ".fllc_relax must lie in (0, 1]");
            }
            ppb.query("fllc_start_time", b.fllc_start_time);
            if (!(b.fllc_start_time >= 0.0) || !std::isfinite(b.fllc_start_time)) {
                Abort(pre + ".fllc_start_time must be a time >= 0 (s, the simulation time)");
            }
            ppb.query("fllc_eps_chord", b.fllc_eps_chord);
            if (!(b.fllc_eps_chord > 0.0) || !std::isfinite(b.fllc_eps_chord)) {
                Abort(pre + ".fllc_eps_chord must be positive (chords)");
            }
            ppb.query("fllc_eps_dr", b.fllc_eps_dr);
            if (!(b.fllc_eps_dr > 0.0) || !std::isfinite(b.fllc_eps_dr)) {
                Abort(pre + ".fllc_eps_dr must be positive");
            }
        } else {
            // the keys of an openfast_turbine: a ct_disk has no model, rotor nodes, tower or nacelle
            for (const char* k : {"fst_file", "mode", "sampling", "num_force_points_blade", "num_force_points_tower", "nacelle_cd",
                                  "nacelle_area", "epsilon_tower", "epsilon_nacelle", "fllc", "fllc_relax", "fllc_start_time",
                                  "fllc_eps_chord", "fllc_eps_dr", "correction_relax", "correction_time"}) {
                warn_inactive(ppb, pre, k, "it is for an openfast_turbine; " + pre + " is a ct_disk, spread with " + pre + ".epsilon only");
            }
            ppb.get("rotor_radius", b.rotor_radius);
            if (!(b.rotor_radius > 0.0) || !std::isfinite(b.rotor_radius)) {
                Abort(pre + ".rotor_radius must be positive");
            }
            ppb.get("hub_height", b.hub_height);
            if (!(b.hub_height > b.rotor_radius) || !std::isfinite(b.hub_height)) {
                Abort(pre + ".hub_height must exceed rotor_radius so the disk clears the base");
            }
            ppb.get("ct", b.ct);
            if (!(b.ct > 0.0 && b.ct < 1.0)) {
                Abort(pre + ".ct must lie in (0, 1)");
            }
            ppb.query("yaw", b.yaw_deg);
            need_finite(pre + ".yaw", b.yaw_deg);
            ppb.query("num_points_r", b.num_points_r);
            if (b.num_points_r < 1) {
                Abort(pre + ".num_points_r must be >= 1");
            }
            ppb.query("sample_diameters_upstream", b.sample_diameters_upstream);
            if (!(b.sample_diameters_upstream > 0.0) || !std::isfinite(b.sample_diameters_upstream)) {
                Abort(pre + ".sample_diameters_upstream must be positive");
            }
            ppb.query("air_density", b.air_density);
            if (!(b.air_density > 0.0) || !std::isfinite(b.air_density)) {
                Abort(pre + ".air_density must be positive");
            }
        }

        in.bodies.push_back(b);
    }
    if (!any_alm) {
        warn_inactive(pp, "erf.moving_bodies", "alm_max_tip_cells", "no body has mode = alm");
        warn_inactive(pp, "erf.moving_bodies", "verify_tip_cells", "no body has mode = alm");
    }
    return in;
}

std::string
MovingBodiesInputs::validate_solver (bool all_levels_anelastic, bool fixed_dt, int max_level,
                                     int anchor_level, bool fpe_traps)
{
    if (!all_levels_anelastic) {
        return "erf.moving_bodies: this version supports the anelastic solver only; set erf.anelastic = 1";
    }
    if (!fixed_dt) {
        return "erf.moving_bodies: OpenFAST needs a fixed time step; set erf.fixed_dt to a whole multiple of the OpenFAST dt";
    }
    if (anchor_level < 0 || anchor_level > max_level) {
        return "erf.moving_bodies.anchor_level = " + std::to_string(anchor_level) + " is not a level of this run (amr.max_level = " +
               std::to_string(max_level) + "); leave it out for the finest level";
    }
    if (fpe_traps) {
        // OpenFAST's initialisation raises floating-point exceptions of its own (seen in
        // FAST_ProgStart with OpenFAST 4.2.1), so a trapped run dies inside the library
        return "erf.moving_bodies: OpenFAST is not floating-point-exception clean; turn off "
               "amrex.fpe_trap_invalid, amrex.fpe_trap_zero and amrex.fpe_trap_overflow";
    }
    return std::string();
}

std::vector<std::pair<std::string,std::string>>
MovingBodiesInputs::restart_record (const MovingBodyInputs& b)
{
    auto num = [] (double v) { std::ostringstream o; o << std::setprecision(17) << v; return o.str(); };
    std::vector<std::pair<std::string,std::string>> r{
        {"type", b.type}, {"base_x", num(b.base_pos[0])}, {"base_y", num(b.base_pos[1])}, {"base_z", num(b.base_pos[2])},
        {"epsilon", num(b.epsilon)}, {"air_density", num(b.air_density)}};
    // the keys the body's options use: one it ignores may change freely
    if (b.type != "openfast_turbine" || b.mode == "adm") { r.push_back({"num_points_t", std::to_string(b.num_points_t)}); }
    if (b.type == "openfast_turbine") {
        r.insert(r.end(), {{"fst_file", b.fst_file}, {"mode", b.mode}, {"sampling", b.sampling},
                           {"num_force_points_blade", std::to_string(b.num_force_points_blade)},
                           {"num_force_points_tower", std::to_string(b.num_force_points_tower)},
                           {"fllc", b.fllc ? "1" : "0"}});
        if (b.mode != "none") { r.insert(r.end(), {{"nacelle_cd", num(b.nacelle_cd)}, {"nacelle_area", num(b.nacelle_area)}}); }
        // the widths in cells, as given or taken from epsilon, after the switches that turn their points on
        const KernelWidths w = b.kernel_widths(1.0);
        if (b.mode != "none" && b.num_force_points_tower > 0) { r.push_back({"epsilon_tower", num(w.tower)}); }
        if (b.mode != "none" && b.nacelle_cd > 0.0) { r.push_back({"epsilon_nacelle", num(w.nacelle)}); }
        if (b.sampling == "upstream") { r.push_back({"sample_diameters_upstream", num(b.sample_diameters_upstream)}); }
        if (b.sampling == "disk_corrected") {
            r.insert(r.end(), {{"correction_relax", num(b.correction_relax)}, {"correction_time", num(b.correction_time)}});
        }
        if (b.fllc) {
            r.insert(r.end(), {{"fllc_relax", num(b.fllc_relax)}, {"fllc_start_time", num(b.fllc_start_time)},
                               {"fllc_eps_chord", num(b.fllc_eps_chord)}, {"fllc_eps_dr", num(b.fllc_eps_dr)}});
        }
    } else {
        r.insert(r.end(), {{"rotor_radius", num(b.rotor_radius)}, {"hub_height", num(b.hub_height)}, {"ct", num(b.ct)},
                           {"yaw", num(b.yaw_deg)}, {"num_points_r", std::to_string(b.num_points_r)},
                           {"sample_diameters_upstream", num(b.sample_diameters_upstream)}});
    }
    return r;
}

bool
MovingBodiesInputs::changed_record_entry (const std::map<std::string,std::string>& chk,
                                          const std::vector<std::pair<std::string,std::string>>& now,
                                          std::string& key, std::string& chk_value, std::string& now_value)
{
    for (const auto& kv : now) {
        auto f = chk.find(kv.first);
        if (f == chk.end() && (kv.first == "epsilon_tower" || kv.first == "epsilon_nacelle")) { f = chk.find("epsilon"); }
        if (f != chk.end() && !same_record_value(f->second, kv.second)) {
            key = kv.first;
            chk_value = f->second;
            now_value = kv.second;
            return true;
        }
    }
    return false;
}
