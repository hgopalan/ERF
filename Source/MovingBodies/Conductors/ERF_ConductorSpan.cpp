#include "ERF_ConductorSpan.H"

#include <cmath>
#include <fstream>
#include <iomanip>

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>

#include "ERF_DiagnosticsLog.H"
#include "ERF_MoorDynInputWriter.H"

using namespace amrex;

namespace erf_conductors {

ConductorSpan::ConductorSpan (const SpanInputs& s, const ConductorInputs& in, Real gravity, const std::string& input_file)
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
    err = m_sys->init({}, {});
    if (!err.empty()) { Abort("erf.conductors." + s.name + ": " + err); }
    if (m_sys->num_lines() != 1) {
        Abort("erf.conductors." + s.name + ": the MoorDyn system holds " + std::to_string(m_sys->num_lines()) + " lines, one was written");
    }
    m_nodes = m_sys->line_num_nodes(1);
    m_nkin = m_sys->external_kinematics_init(err);
    if (m_nkin == 0) { Abort("erf.conductors." + s.name + ": " + err); }
    if (m_nkin < m_nodes) {
        Abort("erf.conductors." + s.name + ": MoorDyn asks for the wind at " + std::to_string(m_nkin) +
              " points but the line has " + std::to_string(m_nodes) + " nodes");
    }
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
    std::vector<double> U(uvw.begin(), uvw.end()), Ud(uvw.size(), 0.0);
    m_sys->set_kinematics(U, Ud, t);
}

void ConductorSpan::step (double time, double dt)
{
    std::vector<double> f;
    double t = time;
    const double sub = dt / m_substeps;
    for (int i = 0; i < m_substeps; ++i) {
        m_sys->step({}, {}, f, t, sub);
    }
    if (std::abs(t - (time + dt)) > 1.0e-8 * std::max(1.0, std::abs(time + dt))) {
        Abort("erf.conductors." + m_in.name + ": MoorDyn's clock (" + std::to_string(t) + ") left ERF's (" +
              std::to_string(time + dt) + ")");
    }
}

std::vector<Real> ConductorSpan::node_positions () const
{
    std::vector<Real> out(3 * static_cast<std::size_t>(m_nodes));
    for (unsigned i = 0; i < m_nodes; ++i) {
        const auto p = node_position(i);
        for (int d = 0; d < 3; ++d) { out[3*i+d] = p[static_cast<std::size_t>(d)]; }
    }
    return out;
}

std::array<Real,3> ConductorSpan::node_position (unsigned node) const
{
    return to_erf_frame(m_sys->line_node_position(1, node), m_offset);
}

Real ConductorSpan::tension_a () const
{
    const auto t = m_sys->line_node_tension(1, 0);
    return static_cast<Real>(std::sqrt(t[0]*t[0] + t[1]*t[1] + t[2]*t[2]));
}

Real ConductorSpan::tension_b () const { return static_cast<Real>(m_sys->line_end_tension(1)); }

Real ConductorSpan::max_tension () const { return static_cast<Real>(m_sys->line_max_tension(1)); }

std::array<Real,3> ConductorSpan::chord_frame_offsets (unsigned node, Real& along, Real& down, Real& side) const
{
    // the chord's unit vector, the "down" direction normal to it in the vertical plane, and the
    // horizontal normal to that plane; a node's offset from the chord is split on the latter two
    const Real c = m_in.chord();
    std::array<Real,3> ec{{0.0, 0.0, 0.0}}, ed{{0.0, 0.0, -1.0}}, es{{0.0, 0.0, 0.0}};
    for (int d = 0; d < 3; ++d) { ec[d] = (m_in.end_b[d] - m_in.end_a[d]) / c; }
    const Real ddotc = -ec[2];
    for (int d = 0; d < 3; ++d) { ed[d] -= ddotc * ec[d]; }
    const Real dn = std::sqrt(ed[0]*ed[0] + ed[1]*ed[1] + ed[2]*ed[2]);
    if (dn > 1.0e-12) { for (int d = 0; d < 3; ++d) { ed[d] /= dn; } } else { ed = {{0.0, 1.0, 0.0}}; }
    // es = ec x ed
    es[0] = ec[1]*ed[2] - ec[2]*ed[1];
    es[1] = ec[2]*ed[0] - ec[0]*ed[2];
    es[2] = ec[0]*ed[1] - ec[1]*ed[0];
    const auto p = node_position(node);
    std::array<Real,3> r{{p[0] - m_in.end_a[0], p[1] - m_in.end_a[1], p[2] - m_in.end_a[2]}};
    along = r[0]*ec[0] + r[1]*ec[1] + r[2]*ec[2];
    down  = r[0]*ed[0] + r[1]*ed[1] + r[2]*ed[2];
    side  = r[0]*es[0] + r[1]*es[1] + r[2]*es[2];
    return p;
}

Real ConductorSpan::mid_sag () const
{
    Real along, down, side;
    chord_frame_offsets((m_nodes - 1) / 2, along, down, side);
    return down;
}

Real ConductorSpan::mid_offset () const
{
    Real along, down, side;
    chord_frame_offsets((m_nodes - 1) / 2, along, down, side);
    return side;
}

Real ConductorSpan::swing_angle () const
{
    Real along, down, side;
    chord_frame_offsets((m_nodes - 1) / 2, along, down, side);
    return std::atan2(side, down);
}

void ConductorSpan::write_diagnostics (double time, bool first) const
{
    if (!ParallelDescriptor::IOProcessor()) { return; }
    std::ofstream out;
    const bool header = erf_actuator::open_log(out, m_in.output_root + ".dat", first);
    if (header) {
        out << "time mid_x mid_y mid_z mid_sag mid_offset swing_deg tension_a tension_b max_tension\n";
    }
    const auto m = node_position((m_nodes - 1) / 2);
    out << std::setprecision(10) << time << " " << m[0] << " " << m[1] << " " << m[2] << " " << mid_sag() << " "
        << mid_offset() << " " << swing_angle() * 180.0 / 3.14159265358979323846 << " " << tension_a() << " "
        << tension_b() << " " << max_tension() << "\n";
}

} // namespace erf_conductors
