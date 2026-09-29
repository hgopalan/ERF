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
        if (b.type != "openfast_turbine") {
            Abort("erf.moving_bodies." + name + ".type = '" + b.type +
                  "' is not known; this version supports openfast_turbine");
        }

        ppb.get("fst_file", b.fst_file);

        std::vector<Real> pos;
        ppb.getarr("base_pos", pos);
        if (pos.size() != 3) {
            Abort("erf.moving_bodies." + name + ".base_pos needs three components");
        }
        for (int d = 0; d < 3; ++d) { b.base_pos[d] = pos[d]; }

        ppb.query("mode", b.mode);
        if (b.mode != "adm" && b.mode != "alm") {
            Abort("erf.moving_bodies." + name + ".mode must be adm or alm, not '" + b.mode + "'");
        }

        ppb.query("num_force_points_blade", b.num_force_points_blade);
        if (b.num_force_points_blade < 1) {
            Abort("erf.moving_bodies." + name + ".num_force_points_blade must be >= 1");
        }
        ppb.query("num_force_points_tower", b.num_force_points_tower);
        if (b.num_force_points_tower < 0) {
            Abort("erf.moving_bodies." + name + ".num_force_points_tower must be >= 0");
        }

        b.output_root = in.diagnostics_dir + "/" + name;
        ppb.query("output_root", b.output_root);

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
