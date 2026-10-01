#include "ERF_Conductors.H"

#include <algorithm>
#include <fstream>
#include <iomanip>

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>

#include "ERF_ActuatorSampling.H"
#include "ERF_Constants.H"
#include "ERF_MoorDynSystem.H"

using namespace amrex;
using erf_conductors::ConductorInputs;
using erf_conductors::ConductorSpan;
using erf_conductors::SpanInputs;

std::unique_ptr<Conductors>
Conductors::create (int max_level)
{
    ConductorInputs in = ConductorInputs::read();
    if (!in.active) { return nullptr; }
    const int anchor = ConductorInputs::resolve_anchor_level(in.anchor_level, max_level);
    const std::string err = ConductorInputs::validate_solver(max_level, anchor, erf_moordyn::fpe_traps_requested());
    if (!err.empty()) { Abort(err); }
    Print() << "erf.conductors: " << in.spans.size() << " span(s) on MoorDyn-C " << erf_moordyn::library_version()
            << (erf_moordyn::is_stub() ? " (the bundled stub stands in for MoorDyn)" : "") << ", anchor level " << anchor
            << ", wind " << (in.has_prescribed_velocity ? "prescribed" : "sampled from the flow at the line nodes") << "\n";
    return std::unique_ptr<Conductors>(new Conductors(std::move(in), anchor));
}

Conductors::Conductors (ConductorInputs_t in, int anchor)
    : m_in(std::move(in)), m_anchor(anchor) {}

void
Conductors::set_ground (const MultiFab* z_phys_nd, const Geometry& geom)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!m_ground_set, "Conductors::set_ground: called twice");
    m_ground_set = true;

    // the attachments must lie inside the domain horizontally: the terrain height is read there
    for (const SpanInputs& s : m_in.spans) {
        for (const auto* e : {&s.end_a, &s.end_b}) {
            for (int d = 0; d < 2; ++d) {
                if ((*e)[d] < geom.ProbLo(d) || (*e)[d] > geom.ProbHi(d)) {
                    Abort("erf.conductors." + s.name + ": an attachment at (" + std::to_string((*e)[0]) + ", " +
                          std::to_string((*e)[1]) + ") lies outside the domain");
                }
            }
        }
    }
    std::vector<Real> pos;
    for (const SpanInputs& s : m_in.spans) {
        pos.insert(pos.end(), {s.end_a[0], s.end_a[1], s.end_a[2], s.end_b[0], s.end_b[1], s.end_b[2]});
    }
    erf_actuator::terrain_heights(z_phys_nd, geom, pos, m_ground);
    const Real floor_z = static_cast<Real>(geom.ProbLo(2));
    m_placed = m_in.spans;
    for (std::size_t i = 0; i < m_placed.size(); ++i) {
        m_placed[i].end_a[2] += m_ground[2*i] - floor_z;
        m_placed[i].end_b[2] += m_ground[2*i+1] - floor_z;
        for (const auto* e : {&m_placed[i].end_a, &m_placed[i].end_b}) {
            if ((*e)[2] < geom.ProbLo(2) || (*e)[2] > geom.ProbHi(2)) {
                Abort("erf.conductors." + m_placed[i].name + ": an attachment at height " + std::to_string((*e)[2]) +
                      " m lies outside the domain");
            }
        }
    }
    if (ParallelDescriptor::IOProcessor()) {
        UtilCreateDirectory(m_in.diagnostics_dir, 0755);
        std::ofstream out(m_in.diagnostics_dir + "/ground.dat", std::ios::trunc);
        out << "span end x y ground z\n" << std::setprecision(10);
        for (std::size_t i = 0; i < m_placed.size(); ++i) {
            out << m_placed[i].name << " a " << m_placed[i].end_a[0] << " " << m_placed[i].end_a[1] << " " << m_ground[2*i] << " " << m_placed[i].end_a[2] << "\n"
                << m_placed[i].name << " b " << m_placed[i].end_b[0] << " " << m_placed[i].end_b[1] << " " << m_ground[2*i+1] << " " << m_placed[i].end_b[2] << "\n";
        }
    }
    ParallelDescriptor::Barrier();

    for (const SpanInputs& s : m_placed) {
        const std::string file = m_in.diagnostics_dir + "/" + s.name + ".moordyn.txt";
        m_spans.push_back(std::make_unique<ConductorSpan>(s, m_in, CONST_GRAV, file));
        const ConductorSpan& c = *m_spans.back();
        Print() << "erf.conductors." << s.name << ": chord " << s.chord() << " m, length " << s.length << " m, "
                << c.num_nodes() << " nodes, initial sag " << c.mid_sag() << " m (catenary estimate " << s.catenary_sag()
                << " m), end tensions " << c.tension_a() << " and " << c.tension_b() << " N, MoorDyn input " << file << "\n";
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
                     const MultiFab* z_phys_nd, const Geometry& geom)
{
    if (lev != m_anchor) { return; }
    if (!m_ground_set) { set_ground(z_phys_nd, geom); }
    ++m_step;
    const bool first = (m_step == 1);
    for (auto& span : m_spans) {
        // the wind of the flow at the start of the step, where the line is, held over the step
        span->set_wind(wind_at(*span, U, V, W, z_phys_nd, geom), time + 0.5 * dt);
        if (first) { span->write_diagnostics(time, true); }
        span->step(time, dt);
        if (m_step % m_in.diagnostics_int == 0) { span->write_diagnostics(time + dt, false); }
    }
}
