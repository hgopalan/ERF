#include "ERF_MovingBodiesInputs.H"

#include <set>

#include <AMReX.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

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

    {
        ParmParse pw("erf.moving_bodies.wake");
        pw.queryarr("lines_xD", in.wake.lines_xD);
        for (const Real xD : in.wake.lines_xD) {
            if (!(xD > 0.0)) { Abort("erf.moving_bodies.wake.lines_xD must be positive distances in rotor diameters"); }
        }
        pw.query("half_width", in.wake.half_width);
        if (!(in.wake.half_width > 0.0)) { Abort("erf.moving_bodies.wake.half_width must be > 0 (rotor diameters)"); }
        pw.query("num_points", in.wake.num_points);
        if (in.wake.num_points < 2) { Abort("erf.moving_bodies.wake.num_points must be >= 2"); }
        pw.query("int", in.wake.interval);
        if (in.wake.interval < 0) { Abort("erf.moving_bodies.wake.int must be >= 1 (0 uses diagnostics_int)"); }
        if (in.wake.interval == 0) { in.wake.interval = in.diagnostics_int; }
        if (pw.contains("avg_start")) {
            Abort("erf.moving_bodies.wake.avg_start was renamed erf.moving_bodies.avg_start (it also starts the body statistics)");
        }
    }
    pp.query("avg_start", in.avg_start);
    if (in.avg_start < 0.0) { Abort("erf.moving_bodies.avg_start must be >= 0"); }
    pp.query("alm_max_tip_cells", in.alm_max_tip_cells);
    if (!(in.alm_max_tip_cells > 0.0)) { Abort("erf.moving_bodies.alm_max_tip_cells must be > 0 (cells swept by a blade tip per step)"); }

    std::vector<Real> vel;
    if (pp.queryarr("prescribed_velocity", vel)) {
        if (vel.size() != 3) {
            Abort("erf.moving_bodies.prescribed_velocity needs three components");
        }
        in.has_prescribed_velocity = true;
        for (int d = 0; d < 3; ++d) { in.prescribed_velocity[d] = vel[d]; }
    }

    std::set<std::string> seen;
    for (const std::string& name : names) {
        if (!seen.insert(name).second) {
            Abort("erf.moving_bodies.bodies lists '" + name + "' twice");
        }
        MovingBodyInputs b;
        b.name = name;
        ParmParse ppb("erf.moving_bodies." + name);

        ppb.get("type", b.type);
        if (b.type != "openfast_turbine" && b.type != "ct_disk") {
            Abort("erf.moving_bodies." + name + ".type = '" + b.type +
                  "' is not known; this version supports openfast_turbine and ct_disk");
        }

        std::vector<Real> pos;
        ppb.getarr("base_pos", pos);
        if (pos.size() != 3) {
            Abort("erf.moving_bodies." + name + ".base_pos needs three components");
        }
        for (int d = 0; d < 3; ++d) { b.base_pos[d] = pos[d]; }

        ppb.query("epsilon", b.epsilon);
        if (!(b.epsilon > 0.0)) {
            Abort("erf.moving_bodies." + name + ".epsilon must be positive (in units of dx)");
        }

        b.output_root = in.diagnostics_dir + "/" + name;
        ppb.query("output_root", b.output_root);

        ppb.query("num_points_t", b.num_points_t);
        if (b.num_points_t < 1) {
            Abort("erf.moving_bodies." + name + ".num_points_t must be >= 1");
        }

        if (b.type == "openfast_turbine") {
            ppb.get("fst_file", b.fst_file);

            ppb.query("mode", b.mode);
            if (b.mode != "adm" && b.mode != "alm" && b.mode != "none") {
                Abort("erf.moving_bodies." + name + ".mode must be adm, alm or none, not '" + b.mode + "'");
            }

            ppb.query("num_force_points_blade", b.num_force_points_blade);
            if (b.num_force_points_blade < 1) {
                Abort("erf.moving_bodies." + name + ".num_force_points_blade must be >= 1");
            }
            ppb.query("num_force_points_tower", b.num_force_points_tower);
            if (b.num_force_points_tower < 0) {
                Abort("erf.moving_bodies." + name + ".num_force_points_tower must be >= 0");
            }
            ppb.query("nacelle_cd", b.nacelle_cd);
            if (b.nacelle_cd < 0.0) {
                Abort("erf.moving_bodies." + name + ".nacelle_cd must be >= 0");
            }
            ppb.query("nacelle_area", b.nacelle_area);
            if (b.nacelle_area < 0.0) {
                Abort("erf.moving_bodies." + name + ".nacelle_area must be >= 0 (m^2)");
            }
            if (b.nacelle_cd > 0.0 && !(b.nacelle_area > 0.0)) {
                Abort("erf.moving_bodies." + name + ".nacelle_cd > 0 needs nacelle_area > 0 (m^2)");
            }
            ppb.query("air_density", b.air_density);
            if (!(b.air_density > 0.0)) {
                Abort("erf.moving_bodies." + name + ".air_density must be positive");
            }
            ppb.query("fllc", b.fllc);
            if (b.fllc && b.mode != "alm") {
                Abort("erf.moving_bodies." + name + ".fllc = true needs mode = alm; the correction is for an actuator line");
            }
            ppb.query("fllc_relax", b.fllc_relax);
            if (!(b.fllc_relax > 0.0 && b.fllc_relax <= 1.0)) {
                Abort("erf.moving_bodies." + name + ".fllc_relax must lie in (0, 1]");
            }
            ppb.query("fllc_start_time", b.fllc_start_time);
            if (b.fllc_start_time < 0.0) {
                Abort("erf.moving_bodies." + name + ".fllc_start_time must be >= 0");
            }
            ppb.query("fllc_eps_chord", b.fllc_eps_chord);
            if (!(b.fllc_eps_chord > 0.0)) {
                Abort("erf.moving_bodies." + name + ".fllc_eps_chord must be positive (chords)");
            }
            ppb.query("fllc_eps_dr", b.fllc_eps_dr);
            if (!(b.fllc_eps_dr > 0.0)) {
                Abort("erf.moving_bodies." + name + ".fllc_eps_dr must be positive");
            }
        } else {
            ppb.get("rotor_radius", b.rotor_radius);
            if (!(b.rotor_radius > 0.0)) {
                Abort("erf.moving_bodies." + name + ".rotor_radius must be positive");
            }
            ppb.get("hub_height", b.hub_height);
            if (!(b.hub_height > b.rotor_radius)) {
                Abort("erf.moving_bodies." + name + ".hub_height must exceed rotor_radius so the disk clears the base");
            }
            ppb.get("ct", b.ct);
            if (!(b.ct > 0.0 && b.ct < 1.0)) {
                Abort("erf.moving_bodies." + name + ".ct must lie in (0, 1)");
            }
            ppb.query("yaw", b.yaw_deg);
            ppb.query("num_points_r", b.num_points_r);
            if (b.num_points_r < 1) {
                Abort("erf.moving_bodies." + name + ".num_points_r must be >= 1");
            }
            ppb.query("sample_diameters_upstream", b.sample_diameters_upstream);
            if (!(b.sample_diameters_upstream > 0.0)) {
                Abort("erf.moving_bodies." + name + ".sample_diameters_upstream must be positive");
            }
            ppb.query("air_density", b.air_density);
            if (!(b.air_density > 0.0)) {
                Abort("erf.moving_bodies." + name + ".air_density must be positive");
            }
        }

        in.bodies.push_back(b);
    }
    return in;
}

std::string
MovingBodiesInputs::validate_solver (bool all_levels_anelastic, bool fixed_dt, int max_level,
                                     bool fpe_traps)
{
    if (!all_levels_anelastic) {
        return "erf.moving_bodies: this version supports the anelastic solver only; set erf.anelastic = 1";
    }
    if (!fixed_dt) {
        return "erf.moving_bodies: OpenFAST needs a fixed time step; set erf.fixed_dt to a whole multiple of the OpenFAST dt";
    }
    if (max_level != 0) {
        return "erf.moving_bodies: this version supports a single level; set amr.max_level = 0";
    }
    if (fpe_traps) {
        // OpenFAST's initialisation raises floating-point exceptions of its own (seen in
        // FAST_ProgStart with OpenFAST 4.2.1), so a trapped run dies inside the library
        return "erf.moving_bodies: OpenFAST is not floating-point-exception clean; turn off "
               "amrex.fpe_trap_invalid, amrex.fpe_trap_zero and amrex.fpe_trap_overflow";
    }
    return std::string();
}
