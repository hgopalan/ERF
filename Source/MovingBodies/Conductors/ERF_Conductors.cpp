#include "ERF_Conductors.H"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>

#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>

#include "ERF_ActuatorSampling.H"
#include "ERF_ActuatorSpreading.H"
#include "ERF_DiagnosticsLog.H"
#include "ERF_Constants.H"
#include "ERF_ConductorGeometry.H"
#include "ERF_MoorDynSystem.H"

using namespace amrex;
using erf_conductors::ConductorInputs;
using erf_conductors::ConductorSpan;
using erf_conductors::SpanInputs;
using erf_conductors::Transformer;

namespace {
constexpr Real rad2deg = Real(180.0 / 3.14159265358979323846);
}

std::unique_ptr<Conductors>
Conductors::create (int max_level)
{
    ConductorInputs in = ConductorInputs::read();
    if (!in.active) { return nullptr; }
    const int anchor = ConductorInputs::resolve_anchor_level(in.anchor_level, max_level);
    const std::string err = ConductorInputs::validate_solver(max_level, anchor, erf_moordyn::fpe_traps_requested());
    if (!err.empty()) { Abort(err); }
    Print() << "erf.conductors: " << in.spans.size() << " line(s) on MoorDyn-C " << erf_moordyn::library_version()
            << (erf_moordyn::is_stub() ? " (the bundled stub stands in for MoorDyn)" : "") << ", anchor level " << anchor
            << ", wind " << (in.has_prescribed_velocity ? "prescribed" : "sampled from the flow at the line nodes")
            << ", drag " << (in.drag_on_flow ? "put back into the flow (epsilon " + std::to_string(in.epsilon) + " cells)" : "not put into the flow") << "\n";
    return std::unique_ptr<Conductors>(new Conductors(std::move(in), anchor));
}

Conductors::Conductors (ConductorInputs_t in, int anchor)
    : m_in(std::move(in)), m_anchor(anchor) {}

void
Conductors::set_ground (const MultiFab* z_phys_nd, const Geometry& geom, const std::string& restart_chkdir)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!m_ground_set, "Conductors::set_ground: called twice");
    m_ground_set = true;

    // the attachments must lie inside the domain horizontally: the terrain height is read there
    std::vector<Real> pos;
    for (const SpanInputs& s : m_in.spans) {
        for (int k = 0; k <= s.num_spans(); ++k) {
            const auto& e = s.point(k);
            for (int d = 0; d < 2; ++d) {
                if (e[d] < geom.ProbLo(d) || e[d] > geom.ProbHi(d)) {
                    Abort("erf.conductors." + s.name + ": an attachment at (" + std::to_string(e[0]) + ", " +
                          std::to_string(e[1]) + ") lies outside the domain");
                }
            }
            pos.insert(pos.end(), {e[0], e[1], e[2]});
        }
    }
    erf_actuator::terrain_heights(z_phys_nd, geom, pos, m_ground);
    const Real floor_z = static_cast<Real>(geom.ProbLo(2));
    m_placed = m_in.spans;
    std::size_t ip = 0;
    // a misplaced line is reported after ground.dat is written, so that the placement can be looked at
    std::string misplaced;
    for (SpanInputs& s : m_placed) {
        for (int k = 0; k <= s.num_spans(); ++k, ++ip) {
            auto& e = s.point(k);
            e[2] += m_ground[ip] - floor_z;
            if (e[2] < geom.ProbLo(2) || e[2] > geom.ProbHi(2)) {
                Abort("erf.conductors." + s.name + ": an attachment at height " + std::to_string(e[2]) + " m lies outside the domain");
            }
        }
        s.lengths_from_stringing_tension((s.mass_per_length - m_in.air_density * Real(0.25) * Real(3.14159265358979323846) *
                                          s.diameter * s.diameter) * CONST_GRAV);
        const std::string slack = ConductorInputs::validate_slack(s, true);
        if (misplaced.empty()) { misplaced = slack; }
    }
    const std::string on_transformers = place_transformers(z_phys_nd, geom);
    if (misplaced.empty()) { misplaced = on_transformers; }
    if (ParallelDescriptor::IOProcessor()) {
        UtilCreateDirectory(m_in.diagnostics_dir, 0755);
        std::ofstream out(m_in.diagnostics_dir + "/ground.dat", std::ios::trunc);
        out << "span point x y ground z\n" << std::setprecision(10);
        ip = 0;
        for (const SpanInputs& s : m_placed) {
            for (int k = 0; k <= s.num_spans(); ++k, ++ip) {
                const auto& e = s.point(k);
                const std::string label = (k == 0) ? "a" : (k == s.num_spans() ? "b" : "t" + std::to_string(k));
                out << s.name << " " << label << " " << e[0] << " " << e[1] << " " << m_ground[ip] << " " << e[2] << "\n";
            }
        }
        // a transformer's row: the centre of its base on the terrain and the height of its top
        for (const Transformer& t : m_transformers) {
            const auto b = t.base();
            out << t.name() << " transformer " << b[0] << " " << b[1] << " " << b[2] << " " << t.box_hi()[2] << "\n";
        }
    }
    ParallelDescriptor::Barrier();
    if (!misplaced.empty()) { Abort(misplaced + " (the placement is in " + m_in.diagnostics_dir + "/ground.dat)"); }
    build_towers();

    if (!restart_chkdir.empty()) {
        const std::string dir = restart_chkdir + "/conductors";
        if (FileExists(dir + "/state")) {
            restore(dir);
            update_ground_under_nodes(z_phys_nd, geom);
            measure_separation();
            measure_transformers();
            return;
        }
        // a checkpoint written without conductors (a precursor, say): the spans start afresh here
        Print() << "erf.conductors: the checkpoint " << restart_chkdir << " holds no conductor state; the spans start now\n";
    }
    for (const SpanInputs& s : m_placed) {
        const std::string file = m_in.diagnostics_dir + "/" + s.name + ".moordyn.txt";
        m_spans.push_back(std::make_unique<ConductorSpan>(s, m_in, CONST_GRAV, file));
        const ConductorSpan& c = *m_spans.back();
        Print() << "erf.conductors." << s.name << ": " << s.num_spans() << " span(s)";
        if (s.has_insulators()) { Print() << ", hanging from insulator strings of " << s.insulator_length << " m at the towers"; }
        else if (!s.towers.empty()) { Print() << ", clamped at the towers"; }
        Print() << ", " << c.num_nodes() << " nodes, MoorDyn input " << file << "\n";
        for (int k = 0; k < s.num_spans(); ++k) {
            const std::string which = (s.num_spans() == 1) ? "" : " span " + std::to_string(k + 1);
            Print() << "  " << s.name << which << ": chord " << s.chord(k) << " m, length " << s.lengths[static_cast<std::size_t>(k)]
                    << " m, initial sag " << c.mid_sag(k) << " m, end tensions " << c.tension_a(k) << " and " << c.tension_b(k) << " N\n";
            if (!s.has_insulators() && std::abs(s.point(k + 1)[2] - s.point(k)[2]) < Real(1.0e-6) * s.chord(k)) {
                // a level span between fixed points: compare MoorDyn's still-air shape with the elastic catenary
                const Real w = (s.mass_per_length - m_in.air_density * Real(0.25) * Real(3.14159265358979323846) * s.diameter * s.diameter) * CONST_GRAV;
                const erf_conductors::Catenary cat =
                    erf_conductors::elastic_catenary(s.chord(k), s.lengths[static_cast<std::size_t>(k)], w, s.axial_stiffness);
                Print() << "  " << s.name << which << ": elastic catenary sag " << cat.sag << " m, end tension "
                        << cat.end_tension << " N (MoorDyn's differ by " << 100.0 * (c.mid_sag(k) / cat.sag - 1.0) << " % and "
                        << 100.0 * (c.tension_a(k) / cat.end_tension - 1.0) << " %)\n";
            }
        }
        add_stats(s);
    }
    add_pair_stats();
    add_transformer_stats();
    add_tower_stats();
    update_ground_under_nodes(z_phys_nd, geom);
    measure_separation();
    measure_transformers();
    for (std::size_t p = 0; p < m_pairs.size(); ++p) {
        const auto& c = m_sep[p];
        Print() << "erf.conductors: " << m_spans[m_pairs[p].first]->name() << " and " << m_spans[m_pairs[p].second]->name()
                << " hang " << c.distance << " m apart at their closest"
                << (c.distance < m_in.flashover_distance ? ", already inside the flashover distance" : "") << "\n";
    }
    for (std::size_t t = 0; t < m_transformers.size(); ++t) {
        const Transformer& tr = m_transformers[t];
        const auto& L = m_tload[t];
        Print() << "erf.conductors." << tr.name() << ": base at " << tr.base()[2] << " m, ends of";
        for (const auto& e : tr.ends()) { Print() << " " << m_spans[e.line]->name() << (e.end == 0 ? ".end_a" : ".end_b"); }
        Print() << "; still-air pull " << L.horizontal_force << " N horizontal, " << L.force[2] << " N vertical, overturning moment "
                << L.overturning_moment << " N m" << (L.over_allowable ? ", already over its allowable" : "")
                << "; closest conductor " << m_tclear[t].distance << " m from the box"
                << (m_tclear[t].distance < m_in.flashover_distance ? ", inside the flashover distance" : "") << "\n";
    }
    if (!m_towers.empty()) {
        Print() << "erf.conductors: " << m_towers.size() << " lattice tower(s) loaded by the wind:";
        for (const auto& t : m_towers) { Print() << " " << t.name() << " (" << t.type().name << ", " << t.arm_height() << " m)"; }
        Print() << "\n";
    }
}

void
Conductors::build_towers ()
{
    m_towers.clear();
    m_aero = std::make_unique<erf_towers::MemberDrag>(m_in.air_density);
    std::size_t ip = 0;
    for (const SpanInputs& s : m_placed) {
        const int npoints = s.num_spans() + 1;
        if (!s.tower_type.empty()) {
            const erf_towers::TowerType* type = nullptr;
            for (const auto& t : m_in.tower_types) { if (t.name == s.tower_type) { type = &t; } }
            AMREX_ALWAYS_ASSERT(type != nullptr);   // checked when the inputs were read
            for (int k = 1; k < s.num_spans(); ++k) {
                // the cross-arm runs across the line: normal to the mean horizontal direction of the spans either side
                const auto& a = s.point(k - 1);
                const auto& p = s.point(k);
                const auto& b = s.point(k + 1);
                Real back[2] = {p[0] - a[0], p[1] - a[1]}, ahead[2] = {b[0] - p[0], b[1] - p[1]};
                const Real nback = std::hypot(back[0], back[1]), nahead = std::hypot(ahead[0], ahead[1]);
                Real along[2] = {back[0] / nback + ahead[0] / nahead, back[1] / nback + ahead[1] / nahead};
                const Real na = std::hypot(along[0], along[1]);
                if (!(na > Real(1.0e-6))) {
                    Abort("erf.conductors." + s.name + ": the line turns back on itself at tower " + std::to_string(k) +
                          "; a cross-arm across it is undefined");
                }
                const std::array<Real,3> across{{-along[1] / na, along[0] / na, 0.0}};
                const Real ground = m_ground[ip + static_cast<std::size_t>(k)];
                m_towers.emplace_back(s.name + "_t" + std::to_string(k), *type, std::array<Real,3>{{p[0], p[1], ground}},
                                      p[2] - ground, across);
            }
        }
        ip += static_cast<std::size_t>(npoints);
    }
}

void
Conductors::add_tower_stats ()
{
    m_tower_stats.clear();
    for (const auto& t : m_towers) {
        const std::string name = "tower_" + t.name();
        m_tower_stats.emplace_back(name, m_in.diagnostics_dir + "/" + name, std::vector<std::string>{"drag_h", "moment_h"});
    }
}

std::vector<Real>
Conductors::tower_wind (const MultiFab& U, const MultiFab& V, const MultiFab& W, const MultiFab* z_phys_nd, const Geometry& geom) const
{
    std::vector<Real> pos;
    for (const auto& t : m_towers) { for (const auto& n : t.nodes()) { pos.insert(pos.end(), {n.pos[0], n.pos[1], n.pos[2]}); } }
    std::vector<Real> uvw(pos.size(), 0.0);
    if (m_in.has_prescribed_velocity) {
        for (std::size_t p = 0; p < uvw.size() / 3; ++p) { for (int d = 0; d < 3; ++d) { uvw[3*p+d] = m_in.prescribed_velocity[d]; } }
        return uvw;
    }
    const Real reach = std::max(geom.CellSize(0), geom.CellSize(1));
    std::string outside;
    if (!erf_actuator::points_covered_by(U.boxArray(), geom, pos, reach, outside)) {
        Abort("erf.conductors: the tower node " + outside + " is not covered, with the cells around it, by the grids of the "
              "anchor level " + std::to_string(m_anchor) + "; refine around the whole tower or lower anchor_level");
    }
    erf_actuator::sample_velocity(U, V, W, z_phys_nd, geom, pos, uvw);
    return uvw;
}

void
Conductors::load_towers (const std::vector<Real>& wind)
{
    std::size_t off = 0;
    for (auto& t : m_towers) {
        const std::size_t n = 3 * t.nodes().size();
        const std::vector<Real> u(wind.begin() + static_cast<std::ptrdiff_t>(off), wind.begin() + static_cast<std::ptrdiff_t>(off + n));
        const std::vector<Real> still(n, 0.0);   // rigid towers: the members do not move
        std::vector<Real> f;
        m_aero->loads(t.nodes(), u, still, f);
        t.set_loads(f);
        off += n;
    }
}

void
Conductors::write_towers (double time, bool first) const
{
    if (m_towers.empty() || !ParallelDescriptor::IOProcessor()) { return; }
    std::ofstream out;
    const bool header = erf_actuator::open_log(out, m_in.diagnostics_dir + "/towers.dat", first);
    if (header) {
        out << "time";
        for (const auto& t : m_towers) {
            const std::string& n = t.name();
            out << " " << n << "_Fx " << n << "_Fy " << n << "_Fz " << n << "_Mx " << n << "_My";
        }
        out << "\n";
    }
    out << std::setprecision(10) << time;
    for (const auto& t : m_towers) {
        const auto F = t.total_force();
        const auto M = t.base_moment();
        out << " " << F[0] << " " << F[1] << " " << F[2] << " " << M[0] << " " << M[1];
    }
    out << "\n";
}

std::string
Conductors::place_transformers (const MultiFab* z_phys_nd, const Geometry& geom)
{
    if (m_in.transformers.empty()) { return std::string(); }
    std::vector<Real> pos;
    for (const auto& t : m_in.transformers) {
        for (int d = 0; d < 2; ++d) {
            if (t.position[d] < geom.ProbLo(d) || t.position[d] > geom.ProbHi(d)) {
                Abort("erf.conductors." + t.name + ".position (" + std::to_string(t.position[0]) + ", " +
                      std::to_string(t.position[1]) + ") lies outside the domain");
            }
        }
        pos.insert(pos.end(), {t.position[0], t.position[1], Real(0.0)});
    }
    std::vector<Real> h;
    erf_actuator::terrain_heights(z_phys_nd, geom, pos, h);
    m_transformers.clear();
    for (std::size_t t = 0; t < m_in.transformers.size(); ++t) { m_transformers.emplace_back(m_in.transformers[t], h[t]); }
    return erf_conductors::attach_line_ends(m_transformers, m_placed);
}

void
Conductors::add_transformer_stats ()
{
    m_tstats.clear();
    for (const Transformer& t : m_transformers) {
        const std::string name = "transformer_" + t.name();
        m_tstats.emplace_back(name, m_in.diagnostics_dir + "/" + name,
                              std::vector<std::string>{"horizontal_force", "overturning_moment", "over_allowable", "clearance", "clash"});
    }
}

void
Conductors::measure_transformers ()
{
    m_tload.resize(m_transformers.size());
    m_tclear.resize(m_transformers.size());
    std::vector<std::vector<Real>> paths;
    for (const auto& span : m_spans) { paths.push_back(span->conductor_path()); }
    for (std::size_t t = 0; t < m_transformers.size(); ++t) {
        const Transformer& tr = m_transformers[t];
        std::vector<std::array<Real,3>> at, force;
        for (const auto& e : tr.ends()) {
            const SpanInputs& s = m_placed[e.line];
            at.push_back(e.end == 0 ? s.end_a : s.end_b);
            force.push_back(m_spans[e.line]->end_force(e.end));
        }
        m_tload[t] = tr.load(at, force);
        // every conductor, the ones ending on it as well: their ends clear the top by their standoff
        erf_conductors::Closest best;
        for (const auto& P : paths) {
            const erf_conductors::Closest c = erf_conductors::closest_polyline_box(P, tr.box_lo(), tr.box_hi());
            if (c.distance < best.distance) { best = c; }
        }
        m_tclear[t] = best;
    }
}

void
Conductors::write_transformers (double time, bool first) const
{
    if (m_transformers.empty() || !ParallelDescriptor::IOProcessor()) { return; }
    std::ofstream out;
    const bool header = erf_actuator::open_log(out, m_in.diagnostics_dir + "/transformers.dat", first);
    if (header) {
        out << "time";
        for (const Transformer& t : m_transformers) {
            const std::string& n = t.name();
            out << " " << n << "_Fx " << n << "_Fy " << n << "_Fz " << n << "_Fh " << n << "_Mx " << n << "_My " << n << "_Mh "
                << n << "_over " << n << "_clearance " << n << "_clash";
        }
        out << "\n";
    }
    out << std::setprecision(10) << time;
    for (std::size_t t = 0; t < m_transformers.size(); ++t) {
        const auto& L = m_tload[t];
        out << " " << L.force[0] << " " << L.force[1] << " " << L.force[2] << " " << L.horizontal_force << " " << L.moment[0]
            << " " << L.moment[1] << " " << L.overturning_moment << " " << (L.over_allowable ? 1 : 0) << " " << m_tclear[t].distance
            << " " << (m_tclear[t].distance < m_in.flashover_distance ? 1 : 0);
    }
    out << "\n";
}

void
Conductors::add_stats (const SpanInputs& s)
{
    std::vector<erf_actuator::RunningStats> st;
    for (int k = 0; k < s.num_spans(); ++k) {
        st.emplace_back(s.span_name(k), s.span_root(k),
                        std::vector<std::string>{"swing_deg", "mid_offset", "tension_a", "tension_b", "max_tension",
                                                 "min_clearance", "drag_y"});
    }
    if (s.has_insulators()) {
        std::vector<std::string> q;
        for (std::size_t j = 0; j < s.towers.size(); ++j) {
            const std::string t = "t" + std::to_string(j + 1);
            q.insert(q.end(), {t + "_swing_deg", t + "_across_deg", t + "_tension"});
        }
        st.emplace_back(s.name + "_insulators", s.output_root + "_insulators", q);
    }
    m_stats.push_back(std::move(st));
}

void
Conductors::add_pair_stats ()
{
    m_pairs.clear();
    m_pair_stats.clear();
    for (std::size_t i = 0; i < m_spans.size(); ++i) {
        for (std::size_t j = i + 1; j < m_spans.size(); ++j) {
            m_pairs.emplace_back(i, j);
            const std::string name = "separation_" + m_spans[i]->name() + "-" + m_spans[j]->name();
            m_pair_stats.emplace_back(name, m_in.diagnostics_dir + "/" + name, std::vector<std::string>{"distance", "clash"});
        }
    }
}

void
Conductors::measure_separation ()
{
    m_sep.resize(m_pairs.size());
    std::vector<std::vector<Real>> paths;
    for (const auto& span : m_spans) { paths.push_back(span->conductor_path()); }
    for (std::size_t p = 0; p < m_pairs.size(); ++p) {
        m_sep[p] = erf_conductors::closest_polylines(paths[m_pairs[p].first], paths[m_pairs[p].second]);
    }
}

void
Conductors::write_separation (double time, bool first) const
{
    if (m_pairs.empty() || !ParallelDescriptor::IOProcessor()) { return; }
    std::ofstream out;
    const bool header = erf_actuator::open_log(out, m_in.diagnostics_dir + "/separation.dat", first);
    if (header) {
        out << "time";
        for (const auto& pr : m_pairs) {
            const std::string n = m_spans[pr.first]->name() + "-" + m_spans[pr.second]->name();
            out << " " << n << "_distance " << n << "_x " << n << "_y " << n << "_z " << n << "_clash";
        }
        out << "\n";
    }
    out << std::setprecision(10) << time;
    for (const auto& c : m_sep) {
        out << " " << c.distance;
        for (int d = 0; d < 3; ++d) { out << " " << Real(0.5) * (c.a[d] + c.b[d]); }
        out << " " << (c.distance < m_in.flashover_distance ? 1 : 0);
    }
    out << "\n";
}

std::vector<std::string>
Conductors::log_files () const
{
    std::vector<std::string> f;
    for (const SpanInputs& s : m_placed) {
        for (int k = 0; k < s.num_spans(); ++k) { f.push_back(s.span_root(k) + ".dat"); }
        f.push_back(s.output_root + "_nodes.dat");
        if (s.has_insulators()) { f.push_back(s.output_root + "_insulators.dat"); }
    }
    f.push_back(m_in.diagnostics_dir + "/total_load.dat");
    f.push_back(m_in.diagnostics_dir + "/separation.dat");
    f.push_back(m_in.diagnostics_dir + "/transformers.dat");
    return f;
}

void
Conductors::restore (const std::string& dir)
{
    // the state file: the step count, the time, and the lines with their node counts, in order
    Vector<char> chars;
    ParallelDescriptor::ReadAndBcastFile(dir + "/state", chars);
    std::istringstream in(std::string(chars.dataPtr(), chars.size()));
    std::string line;
    bool have_step = false, have_time = false, have_t0 = false;
    std::vector<std::pair<std::string,unsigned>> saved;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string key, eq;
        if (!(ls >> key)) { continue; }
        if (key == "step") {
            if (!(ls >> eq >> m_step) || eq != "=") { Abort("malformed step line in '" + dir + "/state'"); }
            have_step = true;
        } else if (key == "time") {
            if (!(ls >> eq >> m_time) || eq != "=") { Abort("malformed time line in '" + dir + "/state'"); }
            have_time = true;
        } else if (key == "clock_offset") {
            if (!(ls >> eq >> m_t0) || eq != "=") { Abort("malformed clock_offset line in '" + dir + "/state'"); }
            have_t0 = true;
        } else if (key == "span") {
            std::string name;
            unsigned nodes = 0;
            if (!(ls >> name >> nodes)) { Abort("malformed span line in '" + dir + "/state'"); }
            saved.emplace_back(name, nodes);
        }
    }
    if (!have_step || !have_time || !have_t0) {
        Abort("no step count, time or clock offset in the conductor checkpoint '" + dir + "/state'");
    }
    if (saved.size() != m_placed.size()) {
        Abort("the conductor checkpoint '" + dir + "' holds " + std::to_string(saved.size()) + " line(s) but erf.conductors.spans lists " +
              std::to_string(m_placed.size()) + "; the erf.conductors block must match the run being restarted");
    }
    for (std::size_t i = 0; i < m_placed.size(); ++i) {
        const SpanInputs& s = m_placed[i];
        if (saved[i].first != s.name) {
            Abort("the conductor checkpoint '" + dir + "' holds line " + saved[i].first + " where erf.conductors.spans lists " +
                  s.name + "; the erf.conductors block must match the run being restarted");
        }
        if (saved[i].second != static_cast<unsigned>(s.num_line_nodes())) {
            Abort("erf.conductors." + s.name + ": the checkpoint '" + dir + "' holds a line of " + std::to_string(saved[i].second) +
                  " nodes but the inputs give " + std::to_string(s.num_line_nodes()) +
                  " (spans, towers, segments and insulator strings must match the run being restarted)");
        }
        const std::string file = m_in.diagnostics_dir + "/" + s.name + ".moordyn.txt";
        m_spans.push_back(std::make_unique<ConductorSpan>(s, m_in, CONST_GRAV, file, dir + "/" + s.name + ".moordyn"));
        m_spans.back()->set_clock_offset(m_t0);
        add_stats(s);
        for (auto& st : m_stats.back()) {
            if (!st.read_state(dir)) {
                Abort("erf.conductors." + s.name + ": the checkpoint '" + dir + "' holds no statistics " + st.name());
            }
        }
        const ConductorSpan& c = *m_spans.back();
        Print() << "erf.conductors." << s.name << ": continued from " << dir << " at t = " << m_time << " s (step " << m_step
                << "), mid-span offset " << c.mid_offset() << " m, sag " << c.mid_sag() << " m\n";
    }
    add_pair_stats();
    for (auto& st : m_pair_stats) {
        if (!st.read_state(dir)) { Abort("the conductor checkpoint '" + dir + "' holds no statistics " + st.name()); }
    }
    add_tower_stats();
    for (auto& st : m_tower_stats) {
        if (!st.read_state(dir)) {
            Abort("the conductor checkpoint '" + dir + "' holds no statistics " + st.name() +
                  "; the lines' tower_type and erf.conductors.tower_types must match the run being restarted");
        }
    }
    if (!m_towers.empty()) {
        // the members' last drag, so that the restored drag on the flow is the checkpointed step's
        Vector<char> chars;
        ParallelDescriptor::ReadAndBcastFile(dir + "/tower_loads", chars);
        std::istringstream tl(std::string(chars.dataPtr(), chars.size()));
        for (auto& t : m_towers) {
            std::vector<Real> f(3 * t.nodes().size());
            for (auto& v : f) {
                double x = 0.0;
                if (!(tl >> x)) { Abort("the conductor checkpoint '" + dir + "/tower_loads' holds too few tower node forces"); }
                v = static_cast<Real>(x);
            }
            t.set_loads(f);
        }
    }
    add_transformer_stats();
    for (auto& st : m_tstats) {
        if (!st.read_state(dir)) {
            Abort("the conductor checkpoint '" + dir + "' holds no statistics " + st.name() +
                  "; the erf.conductors.transformers must match the run being restarted");
        }
    }
    // the logs continue from the checkpoint: rows a run wrote after it are dropped
    if (ParallelDescriptor::IOProcessor()) {
        for (const std::string& f : log_files()) { erf_conductors::trim_log_after(f, m_time); }
        // the towers' rows carry the time their step starts at: the restarted run writes the row at m_time
        erf_conductors::trim_log_after(m_in.diagnostics_dir + "/towers.dat", m_time, true);
    }
    ParallelDescriptor::Barrier();
    m_restored = true;
}

void
Conductors::write_checkpoint (const std::string& chkdir) const
{
    if (!m_ground_set) { return; }
    const std::string dir = chkdir + "/conductors";
    if (ParallelDescriptor::IOProcessor()) {
        UtilCreateDirectory(dir, 0755);
        std::ofstream out(dir + "/state", std::ios::trunc);
        if (!out) { Abort("cannot write the conductor checkpoint state '" + dir + "/state'"); }
        out << std::setprecision(std::numeric_limits<double>::max_digits10)
            << "step = " << m_step << "\ntime = " << m_time << "\nclock_offset = " << m_t0 << "\n";
        for (const auto& span : m_spans) { out << "span " << span->name() << " " << span->num_nodes() << "\n"; }
        if (!out) { Abort("cannot write the conductor checkpoint state '" + dir + "/state'"); }
        // every rank holds the same line; one copy of each is saved
        for (const auto& span : m_spans) { span->save(dir + "/" + span->name() + ".moordyn"); }
    }
    ParallelDescriptor::Barrier();
    for (const auto& line : m_stats) { for (const auto& st : line) { st.write_state(dir); } }
    for (const auto& st : m_pair_stats) { st.write_state(dir); }
    for (const auto& st : m_tstats) { st.write_state(dir); }
    for (const auto& st : m_tower_stats) { st.write_state(dir); }
    if (!m_towers.empty() && ParallelDescriptor::IOProcessor()) {
        std::ofstream out(dir + "/tower_loads", std::ios::trunc);
        out << std::setprecision(std::numeric_limits<double>::max_digits10);
        for (const auto& t : m_towers) { for (const Real v : t.loads()) { out << static_cast<double>(v) << "\n"; } }
        if (!out) { Abort("cannot write the conductor checkpoint '" + dir + "/tower_loads'"); }
    }
    ParallelDescriptor::Barrier();
    ParallelDescriptor::Barrier();
}

void
Conductors::restore_sources (int lev, const MultiFab& U, const MultiFab* z_phys_nd, const MultiFab* detJ_cc, const Geometry& geom)
{
    if (lev != m_anchor || !m_restored || !m_in.drag_on_flow) { return; }
    spread_drag(U, z_phys_nd, detJ_cc, geom);
}

void
Conductors::check_nodes_in_domain (const ConductorSpan& span, const std::vector<Real>& pos, const Geometry& geom) const
{
    for (std::size_t p = 0; p < pos.size() / 3; ++p) {
        for (int d = 0; d < 3; ++d) {
            const bool inside = geom.isPeriodic(d) ||
                                (pos[3*p+d] >= geom.ProbLo(d) && pos[3*p+d] <= geom.ProbHi(d));
            if (!inside) {
                Abort("erf.conductors." + span.name() + ": node " + std::to_string(p) + " at (" + std::to_string(pos[3*p]) + ", " +
                      std::to_string(pos[3*p+1]) + ", " + std::to_string(pos[3*p+2]) + ") m has left the domain");
            }
        }
    }
}

void
Conductors::update_ground_under_nodes (const MultiFab* z_phys_nd, const Geometry& geom)
{
    for (auto& span : m_spans) {
        const std::vector<Real> pos = span->node_positions();
        check_nodes_in_domain(*span, pos, geom);
        std::vector<Real> h;
        erf_actuator::terrain_heights(z_phys_nd, geom, pos, h);
        span->set_ground_under_nodes(h);
    }
}

std::vector<Real>
Conductors::wind_at (const ConductorSpan& span,
                     const MultiFab& U, const MultiFab& V, const MultiFab& W,
                     const MultiFab* z_phys_nd, const Geometry& geom) const
{
    std::vector<Real> uvw(3 * static_cast<std::size_t>(span.num_kinematics_points()), 0.0);
    if (m_in.has_prescribed_velocity) {
        for (std::size_t p = 0; p < uvw.size() / 3; ++p) {
            for (int d = 0; d < 3; ++d) { uvw[3*p+d] = m_in.prescribed_velocity[d]; }
        }
        return uvw;
    }
    // where the line is now: a blown-out span samples the wind metres away from where it hung. MoorDyn
    // lists the line nodes first, then fixed entries (the attachment points and one entry at its own
    // origin, far outside ERF's domain); only the line nodes carry a fluid load here, so the flow is
    // sampled at the nodes and the fixed entries get no wind
    const std::vector<Real> kin = span.kinematics_points();
    const std::vector<Real> pos(kin.begin(), kin.begin() + 3 * static_cast<std::ptrdiff_t>(span.num_nodes()));
    check_nodes_in_domain(span, pos, geom);
    // the sampler reads the cells around each point: on a refined anchor level they must be on its grids
    const Real reach = std::max(geom.CellSize(0), geom.CellSize(1));
    std::string outside;
    if (!erf_actuator::points_covered_by(U.boxArray(), geom, pos, reach, outside)) {
        Abort("erf.conductors." + span.name() + ": the point " + outside + " is not covered, with the cells around it, by the "
              "grids of the anchor level " + std::to_string(m_anchor) + "; refine around the whole span or lower anchor_level");
    }
    std::vector<Real> at_nodes;
    erf_actuator::sample_velocity(U, V, W, z_phys_nd, geom, pos, at_nodes);
    std::copy(at_nodes.begin(), at_nodes.end(), uvw.begin());
    return uvw;
}

void
Conductors::advance (int lev, double time, double dt,
                     const MultiFab& U, const MultiFab& V, const MultiFab& W,
                     const MultiFab* z_phys_nd, const MultiFab* detJ_cc, const Geometry& geom)
{
    if (lev != m_anchor) { return; }
    if (!m_ground_set) { set_ground(z_phys_nd, geom); }
    if (m_step == 0 && !m_restored) {
        // the spans' MoorDyn clocks start at zero now: at ERF's time zero in a fresh run, at the
        // restart time when a restart creates them afresh (a checkpoint without conductor state)
        m_t0 = time;
        m_time = time;
        for (auto& span : m_spans) { span->set_clock_offset(m_t0); }
    }
    ++m_step;
    const bool first = (m_step == 1);
    const bool write = first || (m_step % m_in.diagnostics_int == 0);
    if (first) { write_separation(time, true); write_transformers(time, true); }
    if (!m_towers.empty()) {
        // the members' drag from the flow at the start of the step, which the step holds
        load_towers(tower_wind(U, V, W, z_phys_nd, geom));
        if (first || m_step % m_in.diagnostics_int == 0) { write_towers(time, first); }
        if (time >= m_in.stats_start) {
            for (std::size_t t = 0; t < m_towers.size(); ++t) {
                const auto F = m_towers[t].total_force();
                const auto M = m_towers[t].base_moment();
                m_tower_stats[t].accumulate(time, {std::hypot(F[0], F[1]), std::hypot(M[0], M[1])});
                if (write) { m_tower_stats[t].write(); }
            }
        }
    }
    for (auto& span : m_spans) {
        // the wind of the flow at the start of the step, where the line is, held over the step
        span->set_wind(wind_at(*span, U, V, W, z_phys_nd, geom), time + 0.5 * dt);
        if (first) {
            span->write_diagnostics(time, true);
            if (m_in.node_output_int > 0) { span->write_nodes(time, true); }
        }
        span->step(time, dt);
    }
    m_time = time + dt;
    // where the lines are now: their clearance to the terrain, then the outputs and statistics
    update_ground_under_nodes(z_phys_nd, geom);
    // the air's drag on the lines and on the towers' members
    m_drag_total = {{0.0, 0.0, 0.0}};
    for (const auto& t : m_towers) {
        const auto F = t.total_force();
        for (int d = 0; d < 3; ++d) { m_drag_total[d] += F[d]; }
    }
    const bool sample = (time + dt >= m_in.stats_start);
    for (std::size_t i = 0; i < m_spans.size(); ++i) {
        const ConductorSpan& span = *m_spans[i];
        const auto f = span.total_drag();
        for (int d = 0; d < 3; ++d) { m_drag_total[d] += f[d]; }
        if (m_step % m_in.diagnostics_int == 0) { span.write_diagnostics(time + dt, false); }
        if (m_in.node_output_int > 0 && m_step % m_in.node_output_int == 0) { span.write_nodes(time + dt, false); }
        if (!sample) { continue; }
        for (int k = 0; k < span.num_spans(); ++k) {
            unsigned low = 0;
            const Real cmin = span.min_clearance(low, k);
            m_stats[i][static_cast<std::size_t>(k)].accumulate(time + dt,
                {span.swing_angle(k) * rad2deg, span.mid_offset(k), span.tension_a(k), span.tension_b(k), span.max_tension(k),
                 cmin, span.span_drag(k)[1]});
        }
        if (span.num_insulators() > 0) {
            std::vector<Real> q;
            for (int j = 0; j < span.num_insulators(); ++j) {
                q.insert(q.end(), {span.insulator_swing(j) * rad2deg, span.insulator_swing_across(j) * rad2deg, span.insulator_tension(j)});
            }
            m_stats[i].back().accumulate(time + dt, q);
        }
        if (write) { for (const auto& st : m_stats[i]) { st.write(); } }
    }
    // how close the lines come to each other
    measure_separation();
    if (m_step % m_in.diagnostics_int == 0) { write_separation(time + dt, false); }
    if (sample) {
        for (std::size_t p = 0; p < m_pairs.size(); ++p) {
            m_pair_stats[p].accumulate(time + dt, {m_sep[p].distance, m_sep[p].distance < m_in.flashover_distance ? Real(1.0) : Real(0.0)});
            if (write) { m_pair_stats[p].write(); }
        }
    }
    // the lines' load on the transformers they end on, and how close any conductor comes to each
    measure_transformers();
    if (m_step % m_in.diagnostics_int == 0) { write_transformers(time + dt, false); }
    if (sample) {
        for (std::size_t t = 0; t < m_transformers.size(); ++t) {
            const auto& L = m_tload[t];
            const Real d = m_tclear[t].distance;
            m_tstats[t].accumulate(time + dt, {L.horizontal_force, L.overturning_moment, L.over_allowable ? Real(1.0) : Real(0.0),
                                               d, d < m_in.flashover_distance ? Real(1.0) : Real(0.0)});
            if (write) { m_tstats[t].write(); }
        }
    }
    // the lines' drag on the air, spread into the momentum sources ERF adds over the next step
    if (m_in.drag_on_flow) { spread_drag(U, z_phys_nd, detJ_cc, geom); }
    if (write) { write_total_load(time + dt, first); }
}

void
Conductors::spread_drag (const MultiFab& U, const MultiFab* z_phys_nd, const MultiFab* detJ_cc, const Geometry& geom)
{
    const BoxArray ba = amrex::convert(U.boxArray(), IntVect(0,0,0));
    if (!m_src_defined || m_src_x.boxArray() != amrex::convert(ba, IntVect(1,0,0))) {
        const DistributionMapping& dm = U.DistributionMap();
        m_src_x.define(amrex::convert(ba, IntVect(1,0,0)), dm, 1, 0);
        m_src_y.define(amrex::convert(ba, IntVect(0,1,0)), dm, 1, 0);
        m_src_z.define(amrex::convert(ba, IntVect(0,0,1)), dm, 1, 0);
        m_src_defined = true;
    }
    // the force each node exerts on the air is minus the air's drag on it
    std::vector<Real> pos, force;
    for (const auto& span : m_spans) {
        const std::vector<Real> p = span->node_positions();
        pos.insert(pos.end(), p.begin(), p.end());
        for (unsigned n = 0; n < span->num_nodes(); ++n) {
            const auto f = span->node_drag(n);
            force.insert(force.end(), {-f[0], -f[1], -f[2]});
        }
    }
    for (const auto& t : m_towers) {
        for (std::size_t n = 0; n < t.nodes().size(); ++n) {
            const auto& p = t.nodes()[n].pos;
            pos.insert(pos.end(), {p[0], p[1], p[2]});
            force.insert(force.end(), {-t.loads()[3*n], -t.loads()[3*n+1], -t.loads()[3*n+2]});
        }
    }
    const Real eps = m_in.epsilon * geom.CellSize(0);
    erf_actuator::spread_forces(pos, force, eps, z_phys_nd, detJ_cc, geom, m_src_x, m_src_y, m_src_z);
    m_source_integral = {{erf_actuator::integrate_source(0, m_src_x, detJ_cc, geom),
                          erf_actuator::integrate_source(1, m_src_y, detJ_cc, geom),
                          erf_actuator::integrate_source(2, m_src_z, detJ_cc, geom)}};
}

void
Conductors::add_momentum_sources (int lev, MultiFab& xmom_src, MultiFab& ymom_src, MultiFab& zmom_src) const
{
    if (lev != m_anchor || !m_src_defined) { return; }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(xmom_src.boxArray() == m_src_x.boxArray(),
                                     "Conductors: the momentum sources were built on a different grid than ERF's");
    MultiFab::Add(xmom_src, m_src_x, 0, 0, 1, 0);
    MultiFab::Add(ymom_src, m_src_y, 0, 0, 1, 0);
    MultiFab::Add(zmom_src, m_src_z, 0, 0, 1, 0);
}

void
Conductors::cell_sources (int lev, MultiFab& dst, int comp) const
{
    dst.setVal(0.0, comp, 3, 0);
    if (lev != m_anchor || !m_src_defined) { return; }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(amrex::convert(m_src_x.boxArray(), IntVect(0,0,0)) == dst.boxArray() &&
                                     m_src_x.DistributionMap() == dst.DistributionMap(),
                                     "Conductors::cell_sources: the plot grid differs from the grid the sources were spread on");
    for (MFIter mfi(dst, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        const auto sx = m_src_x.const_array(mfi);
        const auto sy = m_src_y.const_array(mfi);
        const auto sz = m_src_z.const_array(mfi);
        const auto d = dst.array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            d(i,j,k,comp)   = Real(0.5) * (sx(i,j,k) + sx(i+1,j,k));
            d(i,j,k,comp+1) = Real(0.5) * (sy(i,j,k) + sy(i,j+1,k));
            d(i,j,k,comp+2) = Real(0.5) * (sz(i,j,k) + sz(i,j,k+1));
        });
    }
}

void
Conductors::write_total_load (double time, bool first) const
{
    if (!ParallelDescriptor::IOProcessor()) { return; }
    std::ofstream out;
    const bool header = erf_actuator::open_log(out, m_in.diagnostics_dir + "/total_load.dat", first);
    if (header) { out << "time drag_x drag_y drag_z force_on_air_x force_on_air_y force_on_air_z source_x source_y source_z\n"; }
    // the force the lines put into the air is minus the air's drag on them, and only with drag_on_flow
    const Real on = m_in.drag_on_flow ? Real(-1.0) : Real(0.0);
    out << std::setprecision(10) << time;
    for (int d = 0; d < 3; ++d) { out << " " << m_drag_total[d]; }
    for (int d = 0; d < 3; ++d) { out << " " << on * m_drag_total[d]; }
    for (int d = 0; d < 3; ++d) { out << " " << m_source_integral[d]; }
    out << "\n";
}

namespace erf_conductors {

void trim_log_after (const std::string& fname, double t, bool at_too)
{
    std::ifstream in(fname);
    if (!in) { return; }
    std::vector<std::string> kept;
    std::string line;
    bool dropped = false;
    // the logs print the time to ten significant digits
    const double tol = 1.0e-9 * std::max(1.0, std::abs(t));
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        double row_t = 0.0;
        if ((ls >> row_t) && (row_t > t + tol || (at_too && row_t > t - tol))) { dropped = true; continue; }
        kept.push_back(line);
    }
    in.close();
    if (!dropped) { return; }
    std::ofstream out(fname, std::ios::trunc);
    if (!out) { Abort("cannot rewrite the conductor log '" + fname + "'"); }
    for (const auto& l : kept) { out << l << "\n"; }
}

} // namespace erf_conductors
