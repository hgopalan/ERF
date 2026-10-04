// A lattice tower that bends as its frame model: rigid links between the tower's points and the
// frame's nodes, the Newmark step, and the footings' loads from the support reactions.

#include "ERF_FrameTower.H"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

#include <AMReX.H>
#include <AMReX_BLassert.H>

using amrex::Real;

namespace erf_towers {

namespace {

std::array<double,3> cross (const std::array<double,3>& a, const std::array<double,3>& b)
{
    return {{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}};
}

/** The pseudo-inverse of a symmetric positive semi-definite 3 x 3 matrix (row-major), by Jacobi rotations; eigenvalues below 1e-9 of the largest are dropped. */
std::array<double,9> pseudo_inverse (std::array<double,9> a)
{
    std::array<double,9> v{{1, 0, 0, 0, 1, 0, 0, 0, 1}};
    for (int sweep = 0; sweep < 50; ++sweep) {
        const double off = a[1] * a[1] + a[2] * a[2] + a[5] * a[5];
        if (off <= 1.0e-30 * (a[0] * a[0] + a[4] * a[4] + a[8] * a[8] + 1.0e-300)) { break; }
        for (int p = 0; p < 3; ++p) {
            for (int q = p + 1; q < 3; ++q) {
                const double apq = a[static_cast<std::size_t>(3 * p + q)];
                if (apq == 0.0) { continue; }
                const double theta = (a[static_cast<std::size_t>(4 * q)] - a[static_cast<std::size_t>(4 * p)]) / (2.0 * apq);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < 3; ++k) {
                    const double akp = a[static_cast<std::size_t>(3 * k + p)], akq = a[static_cast<std::size_t>(3 * k + q)];
                    a[static_cast<std::size_t>(3 * k + p)] = c * akp - s * akq;
                    a[static_cast<std::size_t>(3 * k + q)] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; ++k) {
                    const double apk = a[static_cast<std::size_t>(3 * p + k)], aqk = a[static_cast<std::size_t>(3 * q + k)];
                    a[static_cast<std::size_t>(3 * p + k)] = c * apk - s * aqk;
                    a[static_cast<std::size_t>(3 * q + k)] = s * apk + c * aqk;
                }
                for (int k = 0; k < 3; ++k) {
                    const double vkp = v[static_cast<std::size_t>(3 * k + p)], vkq = v[static_cast<std::size_t>(3 * k + q)];
                    v[static_cast<std::size_t>(3 * k + p)] = c * vkp - s * vkq;
                    v[static_cast<std::size_t>(3 * k + q)] = s * vkp + c * vkq;
                }
            }
        }
    }
    const double lmax = std::max({a[0], a[4], a[8], 0.0});
    std::array<double,9> r{};
    for (int k = 0; k < 3; ++k) {
        const double l = a[static_cast<std::size_t>(4 * k)];
        if (!(l > 1.0e-9 * lmax)) { continue; }
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                r[static_cast<std::size_t>(3 * i + j)] += v[static_cast<std::size_t>(3 * i + k)] * v[static_cast<std::size_t>(3 * j + k)] / l;
            }
        }
    }
    return r;
}

std::array<double,3> times (const std::array<double,9>& m, const std::array<double,3>& x)
{
    return {{m[0] * x[0] + m[1] * x[1] + m[2] * x[2], m[3] * x[0] + m[4] * x[1] + m[5] * x[2], m[6] * x[0] + m[7] * x[1] + m[8] * x[2]}};
}

} // namespace

RigidLink::RigidLink (const Frame& frame, const std::array<double,3>& p, std::size_t count)
{
    const std::size_t n = frame.num_nodes();
    std::vector<double> dist(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& x = frame.node_position(i);
        dist[i] = std::sqrt((x[0] - p[0]) * (x[0] - p[0]) + (x[1] - p[1]) * (x[1] - p[1]) + (x[2] - p[2]) * (x[2] - p[2]));
    }
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t(0));
    std::stable_sort(order.begin(), order.end(), [&] (std::size_t a, std::size_t b) { return dist[a] < dist[b]; });
    m_distance = dist[order[0]];
    const std::size_t k = (m_distance <= 1.0e-6) ? 1 : std::min(count, n);
    m_nodes.assign(order.begin(), order.begin() + static_cast<long>(k));
    std::array<double,3> c{{0.0, 0.0, 0.0}};
    for (const std::size_t i : m_nodes) { for (int d = 0; d < 3; ++d) { c[static_cast<std::size_t>(d)] += frame.node_position(i)[static_cast<std::size_t>(d)] / static_cast<double>(k); } }
    std::array<double,9> inertia{};
    for (const std::size_t i : m_nodes) {
        const auto& x = frame.node_position(i);
        const std::array<double,3> d{{x[0] - c[0], x[1] - c[1], x[2] - c[2]}};
        m_d.push_back(d);
        const double d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                inertia[static_cast<std::size_t>(3 * a + b)] += (a == b ? d2 : 0.0) - d[static_cast<std::size_t>(a)] * d[static_cast<std::size_t>(b)];
            }
        }
    }
    m_r = {{p[0] - c[0], p[1] - c[1], p[2] - c[2]}};
    m_iinv = pseudo_inverse(inertia);
}

void RigidLink::add_load (const std::array<double,3>& force, std::vector<double>& loads) const
{
    const std::size_t k = m_nodes.size();
    const std::array<double,3> w = times(m_iinv, cross(m_r, force));
    for (std::size_t j = 0; j < k; ++j) {
        const std::array<double,3> extra = cross(w, m_d[j]);
        for (std::size_t d = 0; d < 3; ++d) { loads[6 * m_nodes[j] + d] += force[d] / static_cast<double>(k) + extra[d]; }
    }
}

std::array<double,3> RigidLink::motion (const std::vector<double>& u) const
{
    const std::size_t k = m_nodes.size();
    std::array<double,3> mean{{0.0, 0.0, 0.0}};
    for (const std::size_t i : m_nodes) { for (std::size_t d = 0; d < 3; ++d) { mean[d] += u[6 * i + d] / static_cast<double>(k); } }
    std::array<double,3> s{{0.0, 0.0, 0.0}};
    for (std::size_t j = 0; j < k; ++j) {
        const std::array<double,3> du{{u[6 * m_nodes[j]] - mean[0], u[6 * m_nodes[j] + 1] - mean[1], u[6 * m_nodes[j] + 2] - mean[2]}};
        const std::array<double,3> t = cross(m_d[j], du);
        for (std::size_t d = 0; d < 3; ++d) { s[d] += t[d]; }
    }
    const std::array<double,3> theta = times(m_iinv, s);
    const std::array<double,3> turn = cross(theta, m_r);
    return {{mean[0] + turn[0], mean[1] + turn[1], mean[2] + turn[2]}};
}

FrameTower::FrameTower (const Tower& tower, std::shared_ptr<const Frame> frame, double gravity)
    : m_frame(std::move(frame)), m_name(tower.name())
{
    const TowerType& type = tower.type();
    const std::string key = "erf.conductors." + type.name + ".frame_file = " + type.frame_file;
    const auto& ac = tower.across();
    m_across = {{static_cast<double>(ac[0]), static_cast<double>(ac[1]), 0.0}};
    m_along = {{m_across[1], -m_across[0], 0.0}};
    const auto& base = tower.base();
    auto local_point = [&] (const std::array<Real,3>& p) {
        return to_local({{p[0] - base[0], p[1] - base[1], p[2] - base[2]}});
    };
    const double reach = static_cast<double>(type.base_width);
    for (std::size_t i = 0; i < tower.nodes().size(); ++i) {
        m_drag.emplace_back(*m_frame, local_point(tower.nodes()[i].pos));
        if (!(m_drag.back().distance() <= reach)) {
            amrex::Abort("tower " + m_name + ": drag node " + std::to_string(i) + " lies " + std::to_string(m_drag.back().distance()) +
                         " m from the nearest node of " + key + ", more than the base width; the frame's axes must be "
                         "tower-local (origin at the base centre, x along the line, y along the cross-arm, z up)");
        }
    }
    for (std::size_t a = 0; a < tower.attachments().size(); ++a) {
        m_attach.emplace_back(*m_frame, local_point(tower.attachments()[a]));
        if (!(m_attach.back().distance() <= reach)) {
            amrex::Abort("tower " + m_name + ": the line attachment " + std::to_string(a) + " lies " +
                         std::to_string(m_attach.back().distance()) + " m from the nearest node of " + key +
                         ", more than the base width; the cross-arm of the frame must be where the lines hang");
        }
    }
    // the four supports, one per quadrant of the base, at z = 0
    const auto& sup = m_frame->inputs().supports;
    if (sup.size() != 4) { amrex::Abort("tower " + m_name + ": " + key + " has " + std::to_string(sup.size()) + " supports; a lattice tower stands on four legs"); }
    std::array<int,4> seen{{-1, -1, -1, -1}};
    for (std::size_t s = 0; s < 4; ++s) {
        const auto& x = m_frame->node_position(m_frame->node_of_joint(sup[s].joint));
        if (!(std::abs(x[2]) <= 1.0e-6 && x[0] != 0.0 && x[1] != 0.0)) {
            amrex::Abort("tower " + m_name + ": support joint " + std::to_string(sup[s].joint) + " of " + key +
                         " must stand at z = 0 off both axes (a leg at a corner of the base)");
        }
        // Tower::foundation()'s order: (+along, +across), (+along, -across), (-along, +across), (-along, -across)
        const std::size_t leg = (x[0] > 0.0 ? 0 : 2) + (x[1] > 0.0 ? 0 : 1);
        if (seen[leg] >= 0) { amrex::Abort("tower " + m_name + ": " + key + " has two supports in one quadrant of the base"); }
        seen[leg] = static_cast<int>(s);
        m_leg_support[leg] = s;
    }
    // the static equilibrium under the frame's weight, about which it moves
    const FrameSolution s0 = m_frame->solve({}, gravity);
    m_static = s0.reaction;
    FrameModes modes;
    const std::string err = frame_modes(*m_frame, 1, modes);
    if (!err.empty()) { amrex::Abort("tower " + m_name + ": " + key + ": " + err); }
    m_f1 = modes.frequency[0];
    double a0 = 0.0, a1 = 0.0;
    const double zeta = static_cast<double>(type.damping_ratio);
    if (zeta > 0.0) { rayleigh_coefficients(m_f1, zeta, 10.0 * m_f1, zeta, a0, a1); }
    m_a1 = a1;
    m_dyn = std::make_unique<FrameDynamics>(*m_frame, a0, a1);
    m_dyn->start_static({}, 0.0);
    m_load.assign(m_frame->num_dofs(), 0.0);
}

std::array<double,3> FrameTower::to_local (const std::array<Real,3>& v) const
{
    const double x = v[0], y = v[1], z = v[2];
    return {{x * m_along[0] + y * m_along[1], x * m_across[0] + y * m_across[1], z}};
}

std::array<Real,3> FrameTower::to_erf (const std::array<double,3>& v) const
{
    return {{static_cast<Real>(v[0] * m_along[0] + v[1] * m_across[0]), static_cast<Real>(v[0] * m_along[1] + v[1] * m_across[1]),
             static_cast<Real>(v[2])}};
}

void FrameTower::step (Real dt, const std::vector<Real>& node_force, const std::vector<std::array<Real,3>>& line_force)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(node_force.size() == 3 * m_drag.size(), "FrameTower::step: 3 forces per drag node are needed");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(line_force.size() == m_attach.size(), "FrameTower::step: one line pull per attachment is needed");
    std::vector<double> load(m_frame->num_dofs(), 0.0);
    for (std::size_t i = 0; i < m_drag.size(); ++i) {
        m_drag[i].add_load(to_local({{node_force[3 * i], node_force[3 * i + 1], node_force[3 * i + 2]}}), load);
    }
    for (std::size_t a = 0; a < m_attach.size(); ++a) { m_attach[a].add_load(to_local(line_force[a]), load); }
    for (const double f : load) {
        if (!std::isfinite(f)) { amrex::Abort("tower " + m_name + ": a load on its frame is not finite (the wind's drag or a line's pull)"); }
    }
    m_dyn->step(static_cast<double>(dt), load, 0.0);
    m_load = load;
    m_stepped = true;
}

std::array<Real,3> FrameTower::displacement (std::size_t node) const
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(node < m_drag.size(), "FrameTower: no such drag node");
    return to_erf(m_drag[node].motion(m_dyn->displacement()));
}

std::array<Real,3> FrameTower::velocity (std::size_t node) const
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(node < m_drag.size(), "FrameTower: no such drag node");
    return to_erf(m_drag[node].motion(m_dyn->velocity()));
}

std::array<Real,3> FrameTower::inertial_force (std::size_t node) const
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(node < m_drag.size(), "FrameTower: no such drag node");
    return {{0.0, 0.0, 0.0}};
}

std::array<Real,3> FrameTower::attachment_displacement (std::size_t a) const
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(a < m_attach.size(), "FrameTower: no such attachment");
    return to_erf(m_attach[a].motion(m_dyn->displacement()));
}

std::array<Real,3> FrameTower::attachment_velocity (std::size_t a) const
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(a < m_attach.size(), "FrameTower: no such attachment");
    return to_erf(m_attach[a].motion(m_dyn->velocity()));
}

std::vector<double> FrameTower::state () const
{
    std::vector<double> s = m_dyn->state();
    s.insert(s.end(), m_load.begin(), m_load.end());
    return s;
}

bool FrameTower::set_state (const std::vector<double>& s)
{
    const std::size_t n = m_frame->num_dofs();
    if (s.size() != 1 + 4 * n) { return false; }
    for (const double x : s) { if (!std::isfinite(x)) { return false; } }
    if (!m_dyn->set_state(std::vector<double>(s.begin(), s.begin() + static_cast<long>(1 + 3 * n)))) { return false; }
    m_load.assign(s.begin() + static_cast<long>(1 + 3 * n), s.end());
    m_stepped = true;
    return true;
}

bool FrameTower::foundation (const Tower& tower, FoundationLoad& L) const
{
    const std::size_t n = m_frame->num_dofs();
    const auto& sup = m_frame->inputs().supports;
    // per support, the reaction beyond the one under the frame's weight (frame axes)
    std::vector<std::array<double,6>> reaction(sup.size());
    if (m_stepped) {
        // the dynamic reactions K u + a1 K v + M a - f at the supports
        const auto& u = m_dyn->displacement();
        const auto& v = m_dyn->velocity();
        std::vector<double> w(n);
        for (std::size_t i = 0; i < n; ++i) { w[i] = u[i] + m_a1 * v[i]; }
        const std::vector<double> kw = m_frame->apply_stiffness(w);
        const std::vector<double> ma = m_frame->apply_mass(m_dyn->acceleration());
        for (std::size_t s = 0; s < sup.size(); ++s) {
            const std::size_t node = m_frame->node_of_joint(sup[s].joint);
            for (std::size_t d = 0; d < 6; ++d) { reaction[s][d] = kw[6 * node + d] + ma[6 * node + d] - m_load[6 * node + d]; }
        }
    } else {
        // at rest before the first step: the static reactions under the tower's present loads
        std::vector<double> load(n, 0.0);
        const auto& drag = tower.loads();
        for (std::size_t i = 0; i < m_drag.size() && 3 * i + 2 < drag.size(); ++i) {
            m_drag[i].add_load(to_local({{drag[3 * i], drag[3 * i + 1], drag[3 * i + 2]}}), load);
        }
        const auto& pulls = tower.line_forces();
        for (std::size_t a = 0; a < m_attach.size() && a < pulls.size(); ++a) { m_attach[a].add_load(to_local(pulls[a]), load); }
        reaction = m_frame->solve(load, 0.0).reaction;
    }
    // their resultant about the base centre
    std::array<double,3> force{{0.0, 0.0, 0.0}}, moment{{0.0, 0.0, 0.0}};
    std::array<double,4> up{};
    for (std::size_t s = 0; s < sup.size(); ++s) {
        const std::size_t node = m_frame->node_of_joint(sup[s].joint);
        const std::array<double,6>& r = reaction[s];
        const auto& x = m_frame->node_position(node);
        const std::array<double,3> f{{r[0], r[1], r[2]}};
        const std::array<double,3> m = cross(x, f);
        for (std::size_t d = 0; d < 3; ++d) { force[d] += f[d]; moment[d] += m[d] + r[3 + d]; }
        for (std::size_t leg = 0; leg < 4; ++leg) { if (m_leg_support[leg] == s) { up[leg] = r[2] + m_static[s][2]; } }
    }
    // the loads on the tower but its weight balance the dynamic reactions
    const auto F = to_erf({{-force[0], -force[1], -force[2]}});
    const auto M = to_erf({{-moment[0], -moment[1], -moment[2]}});
    L = FoundationLoad();
    L.force = F;
    L.moment = M;
    L.shear = std::hypot(L.force[0], L.force[1]);
    L.overturning = std::hypot(L.moment[0], L.moment[1]);
    double vertical = 0.0;
    for (std::size_t leg = 0; leg < 4; ++leg) { L.legs[leg] = static_cast<Real>(up[leg]); vertical += up[leg]; }
    L.vertical = static_cast<Real>(vertical);
    L.max_compression = std::max({L.legs[0], L.legs[1], L.legs[2], L.legs[3], Real(0.0)});
    L.max_uplift = std::max(Real(0.0), -std::min({L.legs[0], L.legs[1], L.legs[2], L.legs[3]}));
    const TowerType& type = tower.type();
    L.over_allowable = (type.allowable_uplift > 0.0 && L.max_uplift > type.allowable_uplift) ||
                       (type.allowable_compression > 0.0 && L.max_compression > type.allowable_compression);
    return true;
}

} // namespace erf_towers
