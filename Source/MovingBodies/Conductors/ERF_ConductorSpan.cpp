#include "ERF_ConductorSpan.H"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>

#include "ERF_DiagnosticsLog.H"
#include "ERF_MoorDynInputWriter.H"

using namespace amrex;

namespace erf_conductors {

namespace {
constexpr Real rad2deg = Real(180.0 / 3.14159265358979323846);
}

ConductorSpan::ConductorSpan (const SpanInputs& s, const ConductorInputs& in, Real gravity, const std::string& input_file,
                              const std::string& saved_state)
    : m_in(s), m_offset(in.surface_offset), m_substeps(in.substeps), m_file(input_file)
{
    write_moordyn_input(m_file, s, in, gravity);
    ParallelDescriptor::Barrier();   // every rank reads the file the I/O rank wrote
    std::string err;
    m_sys = erf_moordyn::MoorDynSystem::create(m_file, "", in.moordyn_log_level, err);
    if (!m_sys) { Abort("erf.conductors." + s.name + ": " + err); }
    if (m_sys->num_coupled_dof() != 0) {
        Abort("erf.conductors." + s.name + ": the MoorDyn system has " + std::to_string(m_sys->num_coupled_dof()) +
              " coupled degrees of freedom; this version supports fixed attachments only");
    }
    // a restored line takes its state from the file below, not from the initial-shape solve
    const bool restoring = !saved_state.empty();
    err = m_sys->init({}, {}, !restoring);
    if (!err.empty()) { Abort("erf.conductors." + s.name + ": " + err); }
    const unsigned nlines = static_cast<unsigned>(s.num_spans() + num_insulators());
    if (m_sys->num_lines() != nlines) {
        Abort("erf.conductors." + s.name + ": the MoorDyn system holds " + std::to_string(m_sys->num_lines()) + " lines, " +
              std::to_string(nlines) + " were written");
    }
    m_first.assign(1, 0);
    for (unsigned l = 1; l <= nlines; ++l) { m_first.push_back(m_first.back() + m_sys->line_num_nodes(l)); }
    m_nkin = m_sys->external_kinematics_init(err);
    if (m_nkin == 0) { Abort("erf.conductors." + s.name + ": " + err); }
    if (m_nkin < num_nodes()) {
        Abort("erf.conductors." + s.name + ": MoorDyn asks for the wind at " + std::to_string(m_nkin) +
              " points but the line has " + std::to_string(num_nodes()) + " nodes");
    }
    // after the external kinematics are set up: MoorDyn's state holds the line, not the wind points
    if (restoring) { m_sys->load(saved_state); }
}

void ConductorSpan::save (const std::string& path) const
{
    m_sys->save(path);
}

void ConductorSpan::locate (unsigned node, unsigned& line, unsigned& local) const
{
    const auto it = std::upper_bound(m_first.begin(), m_first.end(), node);
    const auto l = static_cast<unsigned>(it - m_first.begin()) - 1;
    line = l + 1;
    local = node - m_first[l];
}

std::vector<Real> ConductorSpan::kinematics_points () const
{
    const std::vector<double> r = m_sys->kinematics_points();
    std::vector<Real> out(r.size());
    for (std::size_t p = 0; p < r.size() / 3; ++p) {
        out[3*p]   = static_cast<Real>(r[3*p]);
        out[3*p+1] = static_cast<Real>(r[3*p+1]);
        out[3*p+2] = static_cast<Real>(r[3*p+2]) + m_offset;
    }
    return out;
}

void ConductorSpan::set_wind (const std::vector<Real>& uvw, double t)
{
    if (uvw.size() != 3 * static_cast<std::size_t>(m_nkin)) {
        Abort("erf.conductors." + m_in.name + ": " + std::to_string(3 * m_nkin) + " wind components are needed, " +
              std::to_string(uvw.size()) + " were given");
    }
    // the fluid acceleration enters MoorDyn's added-mass (Froude-Krylov) load, which in air is
    // smaller than the line's own inertia by the density ratio (about 1e-3): it is left at zero
    std::vector<double> U(uvw.begin(), uvw.end()), Ud(uvw.size(), 0.0);
    m_sys->set_kinematics(U, Ud, t);
    m_wind = uvw;
}

void ConductorSpan::set_ground_under_nodes (const std::vector<Real>& h)
{
    if (h.size() != num_nodes()) {
        Abort("erf.conductors." + m_in.name + ": " + std::to_string(num_nodes()) + " ground heights are needed, " +
              std::to_string(h.size()) + " were given");
    }
    m_ground = h;
}

Real ConductorSpan::clearance (unsigned node) const
{
    const Real ground = m_ground.empty() ? Real(0.0) : m_ground[node];
    return node_position(node)[2] - ground;
}

Real ConductorSpan::min_clearance (unsigned& node, int k) const
{
    const unsigned first = span_first_node(k), n = span_num_nodes(k);
    node = first;
    Real best = clearance(first);
    for (unsigned i = first + 1; i < first + n; ++i) {
        const Real c = clearance(i);
        if (c < best) { best = c; node = i; }
    }
    return best;
}

std::array<Real,3> ConductorSpan::node_drag (unsigned node) const
{
    unsigned line, local;
    locate(node, line, local);
    const auto f = m_sys->line_node_drag(line, local);
    return {{static_cast<Real>(f[0]), static_cast<Real>(f[1]), static_cast<Real>(f[2])}};
}

std::array<Real,3> ConductorSpan::total_drag () const
{
    std::array<Real,3> sum{{0.0, 0.0, 0.0}};
    for (unsigned i = 0; i < num_nodes(); ++i) {
        const auto f = node_drag(i);
        for (int d = 0; d < 3; ++d) { sum[d] += f[d]; }
    }
    return sum;
}

std::array<Real,3> ConductorSpan::span_drag (int k) const
{
    std::array<Real,3> sum{{0.0, 0.0, 0.0}};
    for (unsigned i = span_first_node(k); i < span_first_node(k) + span_num_nodes(k); ++i) {
        const auto f = node_drag(i);
        for (int d = 0; d < 3; ++d) { sum[d] += f[d]; }
    }
    return sum;
}

Real ConductorSpan::node_tension (unsigned node) const
{
    unsigned line, local;
    locate(node, line, local);
    const auto t = m_sys->line_node_tension(line, local);
    return static_cast<Real>(std::sqrt(t[0]*t[0] + t[1]*t[1] + t[2]*t[2]));
}

std::array<Real,3> ConductorSpan::wind_at_point (unsigned point) const
{
    if (m_wind.empty() || point >= m_nkin) { return {{0.0, 0.0, 0.0}}; }
    return {{m_wind[3*point], m_wind[3*point+1], m_wind[3*point+2]}};
}

void ConductorSpan::step (double time, double dt)
{
    std::vector<double> f;
    double t = time;
    const double sub = dt / m_substeps;
    for (int i = 0; i < m_substeps; ++i) {
        m_sys->step({}, {}, f, t, sub);
    }
    if (std::abs(m_t0 + t - (time + dt)) > 1.0e-8 * std::max(1.0, std::abs(time + dt))) {
        Abort("erf.conductors." + m_in.name + ": MoorDyn's clock (" + std::to_string(t) + " s since ERF's " +
              std::to_string(m_t0) + " s) left ERF's (" + std::to_string(time + dt) + ")");
    }
}

std::vector<Real> ConductorSpan::node_positions () const
{
    std::vector<Real> out(3 * static_cast<std::size_t>(num_nodes()));
    for (unsigned i = 0; i < num_nodes(); ++i) {
        const auto p = node_position(i);
        for (int d = 0; d < 3; ++d) { out[3*i+d] = p[static_cast<std::size_t>(d)]; }
    }
    return out;
}

std::array<Real,3> ConductorSpan::node_position (unsigned node) const
{
    unsigned line, local;
    locate(node, line, local);
    return to_erf_frame(m_sys->line_node_position(line, local), m_offset);
}

std::vector<Real> ConductorSpan::conductor_path () const
{
    std::vector<Real> out;
    out.reserve(3 * static_cast<std::size_t>(m_first[static_cast<std::size_t>(num_spans())]));
    for (unsigned i = 0; i < m_first[static_cast<std::size_t>(num_spans())]; ++i) {
        const auto p = node_position(i);
        out.insert(out.end(), {p[0], p[1], p[2]});
    }
    return out;
}

Real ConductorSpan::tension_a (int k) const
{
    const auto t = m_sys->line_node_tension(static_cast<unsigned>(k) + 1, 0);
    return static_cast<Real>(std::sqrt(t[0]*t[0] + t[1]*t[1] + t[2]*t[2]));
}

Real ConductorSpan::tension_b (int k) const { return static_cast<Real>(m_sys->line_end_tension(static_cast<unsigned>(k) + 1)); }

Real ConductorSpan::max_tension (int k) const { return static_cast<Real>(m_sys->line_max_tension(static_cast<unsigned>(k) + 1)); }

void ConductorSpan::chord_frame_offsets (unsigned node, int k, Real& along, Real& down, Real& side) const
{
    // the chord's unit vector, the "down" direction normal to it in the vertical plane, and the
    // horizontal normal to that plane; a node's offset from the chord is split on the latter two
    const auto& a = m_in.point(k);
    const auto& b = m_in.point(k + 1);
    const Real c = m_in.chord(k);
    std::array<Real,3> ec{{0.0, 0.0, 0.0}}, ed{{0.0, 0.0, -1.0}}, es{{0.0, 0.0, 0.0}};
    for (int d = 0; d < 3; ++d) { ec[d] = (b[d] - a[d]) / c; }
    const Real ddotc = -ec[2];
    for (int d = 0; d < 3; ++d) { ed[d] -= ddotc * ec[d]; }
    const Real dn = std::sqrt(ed[0]*ed[0] + ed[1]*ed[1] + ed[2]*ed[2]);
    if (dn > 1.0e-12) { for (int d = 0; d < 3; ++d) { ed[d] /= dn; } } else { ed = {{0.0, 1.0, 0.0}}; }
    // es = ec x ed
    es[0] = ec[1]*ed[2] - ec[2]*ed[1];
    es[1] = ec[2]*ed[0] - ec[0]*ed[2];
    es[2] = ec[0]*ed[1] - ec[1]*ed[0];
    const auto p = node_position(node);
    std::array<Real,3> r{{p[0] - a[0], p[1] - a[1], p[2] - a[2]}};
    along = r[0]*ec[0] + r[1]*ec[1] + r[2]*ec[2];
    down  = r[0]*ed[0] + r[1]*ed[1] + r[2]*ed[2];
    side  = r[0]*es[0] + r[1]*es[1] + r[2]*es[2];
}

Real ConductorSpan::mid_sag (int k) const
{
    Real along, down, side;
    chord_frame_offsets(span_first_node(k) + (span_num_nodes(k) - 1) / 2, k, along, down, side);
    return down;
}

Real ConductorSpan::mid_offset (int k) const
{
    Real along, down, side;
    chord_frame_offsets(span_first_node(k) + (span_num_nodes(k) - 1) / 2, k, along, down, side);
    return side;
}

Real ConductorSpan::swing_angle (int k) const
{
    Real along, down, side;
    chord_frame_offsets(span_first_node(k) + (span_num_nodes(k) - 1) / 2, k, along, down, side);
    return std::atan2(side, down);
}

std::array<Real,3> ConductorSpan::across_direction (int j) const
{
    // the line's horizontal direction at tower j+1, from the attachment before it to the one after
    const auto& a = m_in.point(j);
    const auto& b = m_in.point(j + 2);
    Real ex = b[0] - a[0], ey = b[1] - a[1];
    const Real n = std::sqrt(ex*ex + ey*ey);
    ex /= n; ey /= n;
    return {{-ey, ex, 0.0}};   // z x e: to the left of the line looking down
}

Real ConductorSpan::insulator_swing (int j) const
{
    const unsigned line = static_cast<unsigned>(num_spans() + j) + 1;
    const auto top = m_sys->line_node_position(line, 0);
    const auto bot = m_sys->line_node_position(line, m_sys->line_num_nodes(line) - 1);
    const double h = std::hypot(bot[0] - top[0], bot[1] - top[1]);
    return static_cast<Real>(std::atan2(h, top[2] - bot[2]));
}

Real ConductorSpan::insulator_swing_across (int j) const
{
    const unsigned line = static_cast<unsigned>(num_spans() + j) + 1;
    const auto top = m_sys->line_node_position(line, 0);
    const auto bot = m_sys->line_node_position(line, m_sys->line_num_nodes(line) - 1);
    const auto es = across_direction(j);
    const double side = (bot[0] - top[0]) * es[0] + (bot[1] - top[1]) * es[1];
    return static_cast<Real>(std::atan2(side, top[2] - bot[2]));
}

Real ConductorSpan::insulator_tension (int j) const
{
    const auto t = m_sys->line_node_tension(static_cast<unsigned>(num_spans() + j) + 1, 0);
    return static_cast<Real>(std::sqrt(t[0]*t[0] + t[1]*t[1] + t[2]*t[2]));
}

void ConductorSpan::write_diagnostics (double time, bool first) const
{
    if (!ParallelDescriptor::IOProcessor()) { return; }
    for (int k = 0; k < num_spans(); ++k) {
        std::ofstream out;
        const bool header = erf_actuator::open_log(out, m_in.span_root(k) + ".dat", first);
        if (header) {
            out << "time mid_x mid_y mid_z mid_sag mid_offset swing_deg tension_a tension_b max_tension mid_u mid_v mid_w"
                   " min_clearance min_clearance_x min_clearance_y drag_x drag_y drag_z\n";
        }
        const unsigned mid = span_first_node(k) + (span_num_nodes(k) - 1) / 2;
        const auto m = node_position(mid);
        const auto u = wind_at_point(mid);
        out << std::setprecision(10) << time << " " << m[0] << " " << m[1] << " " << m[2] << " " << mid_sag(k) << " "
            << mid_offset(k) << " " << swing_angle(k) * rad2deg << " " << tension_a(k) << " "
            << tension_b(k) << " " << max_tension(k) << " " << u[0] << " " << u[1] << " " << u[2];
        unsigned low = 0;
        const Real cmin = min_clearance(low, k);
        const auto pl = node_position(low);
        const auto f = span_drag(k);
        out << " " << cmin << " " << pl[0] << " " << pl[1] << " " << f[0] << " " << f[1] << " " << f[2] << "\n";
    }
    if (num_insulators() > 0) {
        std::ofstream out;
        const bool header = erf_actuator::open_log(out, m_in.output_root + "_insulators.dat", first);
        if (header) {
            out << "time";
            for (int j = 0; j < num_insulators(); ++j) {
                const std::string t = "t" + std::to_string(j + 1);
                out << " " << t << "_swing_deg " << t << "_across_deg " << t << "_tension";
            }
            out << "\n";
        }
        out << std::setprecision(10) << time;
        for (int j = 0; j < num_insulators(); ++j) {
            out << " " << insulator_swing(j) * rad2deg << " " << insulator_swing_across(j) * rad2deg << " " << insulator_tension(j);
        }
        out << "\n";
    }
}

void ConductorSpan::write_nodes (double time, bool first) const
{
    if (!ParallelDescriptor::IOProcessor()) { return; }
    std::ofstream out;
    const bool header = erf_actuator::open_log(out, m_in.output_root + "_nodes.dat", first);
    if (header) { out << "time node x y z clearance tension u v w drag_x drag_y drag_z\n"; }
    out << std::setprecision(10);
    for (unsigned i = 0; i < num_nodes(); ++i) {
        const auto p = node_position(i);
        const auto u = wind_at_point(i);
        const auto f = node_drag(i);
        out << time << " " << i << " " << p[0] << " " << p[1] << " " << p[2] << " " << clearance(i) << " " << node_tension(i)
            << " " << u[0] << " " << u[1] << " " << u[2] << " " << f[0] << " " << f[1] << " " << f[2] << "\n";
    }
}

} // namespace erf_conductors
