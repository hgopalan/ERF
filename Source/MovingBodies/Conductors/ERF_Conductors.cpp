#include "ERF_Conductors.H"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>

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
            << ", wind " << (in.has_prescribed_velocity ? "prescribed" : "sampled from the flow at the line nodes")
            << ", drag " << (in.drag_on_flow ? "put back into the flow (epsilon " + std::to_string(in.epsilon) + " cells)" : "not put into the flow") << "\n";
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
                << c.num_nodes() << " nodes, initial sag " << c.mid_sag() << " m, end tensions " << c.tension_a() << " and "
                << c.tension_b() << " N, MoorDyn input " << file << "\n";
        if (std::abs(s.end_b[2] - s.end_a[2]) < Real(1.0e-6) * s.chord()) {
            // a level span: compare MoorDyn's still-air shape with the elastic catenary
            const Real w = (s.mass_per_length - m_in.air_density * Real(0.25) * Real(3.14159265358979323846) * s.diameter * s.diameter) * CONST_GRAV;
            const erf_conductors::Catenary cat = erf_conductors::elastic_catenary(s.chord(), s.length, w, s.axial_stiffness);
            Print() << "erf.conductors." << s.name << ": elastic catenary sag " << cat.sag << " m, end tension "
                    << cat.end_tension << " N (MoorDyn's differ by " << 100.0 * (c.mid_sag() / cat.sag - 1.0) << " % and "
                    << 100.0 * (c.tension_a() / cat.end_tension - 1.0) << " %)\n";
        }
        m_stats.emplace_back(s.name, s.output_root,
                             std::vector<std::string>{"swing_deg", "mid_offset", "tension_a", "tension_b", "max_tension",
                                                      "min_clearance", "drag_y"});
    }
    update_ground_under_nodes(z_phys_nd, geom);
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
    ++m_step;
    const bool first = (m_step == 1);
    const bool write = first || (m_step % m_in.diagnostics_int == 0);
    for (auto& span : m_spans) {
        // the wind of the flow at the start of the step, where the line is, held over the step
        span->set_wind(wind_at(*span, U, V, W, z_phys_nd, geom), time + 0.5 * dt);
        if (first) {
            span->write_diagnostics(time, true);
            if (m_in.node_output_int > 0) { span->write_nodes(time, true); }
        }
        span->step(time, dt);
    }
    // where the lines are now: their clearance to the terrain, then the outputs and statistics
    update_ground_under_nodes(z_phys_nd, geom);
    m_drag_total = {{0.0, 0.0, 0.0}};
    for (std::size_t i = 0; i < m_spans.size(); ++i) {
        const ConductorSpan& span = *m_spans[i];
        const auto f = span.total_drag();
        for (int d = 0; d < 3; ++d) { m_drag_total[d] += f[d]; }
        if (m_step % m_in.diagnostics_int == 0) { span.write_diagnostics(time + dt, false); }
        if (m_in.node_output_int > 0 && m_step % m_in.node_output_int == 0) { span.write_nodes(time + dt, false); }
        if (time + dt >= m_in.stats_start) {
            unsigned low = 0;
            const Real cmin = span.min_clearance(low);
            m_stats[i].accumulate(time + dt, {span.swing_angle() * Real(180.0 / 3.14159265358979323846), span.mid_offset(),
                                              span.tension_a(), span.tension_b(), span.max_tension(), cmin, f[1]});
            if (write) { m_stats[i].write(); }
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
