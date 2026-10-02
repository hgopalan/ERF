#include "ERF_Tower.H"

#include <algorithm>
#include <cmath>
#include <utility>

#include <AMReX.H>
#include <AMReX_BLassert.H>

using amrex::Real;

namespace erf_towers {

Tower::Tower (std::string name, const TowerType& type, const std::array<Real,3>& base, Real arm_height,
              const std::array<Real,3>& across)
    : m_name(std::move(name)), m_type(type), m_base(base), m_arm_height(arm_height), m_across(across)
{
    const Real phi = m_type.solidity;
    const Real cf = m_type.force_coefficient();
    // the body up to the cross-arm, tapering, then the peak at the top width
    const int n = m_type.segments;
    const Real dz = m_arm_height / n;
    for (int i = 0; i < n; ++i) {
        const Real z = (Real(i) + Real(0.5)) * dz;
        const Real w = m_type.base_width + (m_type.top_width - m_type.base_width) * z / m_arm_height;
        m_nodes.push_back(MemberNode{{{m_base[0], m_base[1], m_base[2] + z}}, {{0.0, 0.0, 1.0}}, dz, phi * w, cf});
    }
    if (m_type.peak > 0.0) {
        const int np = std::max(1, static_cast<int>(std::ceil(m_type.peak / dz)));
        const Real dp = m_type.peak / np;
        for (int i = 0; i < np; ++i) {
            const Real z = m_arm_height + (Real(i) + Real(0.5)) * dp;
            m_nodes.push_back(MemberNode{{{m_base[0], m_base[1], m_base[2] + z}}, {{0.0, 0.0, 1.0}}, dp, phi * m_type.top_width, cf});
        }
    }
    m_nbody = static_cast<int>(m_nodes.size());
    // the cross-arm, centred on the body
    const int na = TowerType::arm_segments;
    const Real da = m_type.arm_length / na;
    for (int i = 0; i < na; ++i) {
        const Real s = -Real(0.5) * m_type.arm_length + (Real(i) + Real(0.5)) * da;
        m_nodes.push_back(MemberNode{{{m_base[0] + s * m_across[0], m_base[1] + s * m_across[1], m_base[2] + m_arm_height}},
                                     m_across, da, phi * m_type.arm_face(), cf});
    }
    m_force.assign(3 * m_nodes.size(), 0.0);
    m_disp.assign(3 * m_nodes.size(), 0.0);
    m_vel.assign(3 * m_nodes.size(), 0.0);
    m_inertia.assign(3 * m_nodes.size(), 0.0);
}

void Tower::set_motion (const std::vector<Real>& displacement, const std::vector<Real>& velocity, const std::vector<Real>& inertia)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(displacement.size() == 3 * m_nodes.size() && velocity.size() == 3 * m_nodes.size() &&
                                     inertia.size() == 3 * m_nodes.size(), "Tower::set_motion: 3 values per node are needed");
    m_disp = displacement;
    m_vel = velocity;
    m_inertia = inertia;
}

std::vector<MemberNode> Tower::current_nodes () const
{
    std::vector<MemberNode> nodes = m_nodes;
    for (std::size_t i = 0; i < nodes.size(); ++i) { for (int d = 0; d < 3; ++d) { nodes[i].pos[d] += m_disp[3*i+d]; } }
    return nodes;
}

std::array<Real,3> Tower::arm_displacement () const
{
    // the cross-arm's nodes follow the body's nodes
    const std::size_t i = static_cast<std::size_t>(m_nbody);
    return {{m_disp[3*i], m_disp[3*i+1], m_disp[3*i+2]}};
}

std::array<Real,3> Tower::total_force () const
{
    std::array<Real,3> F{{0.0, 0.0, 0.0}};
    for (std::size_t i = 0; i < m_nodes.size(); ++i) { for (int d = 0; d < 3; ++d) { F[d] += m_force[3*i+d]; } }
    return F;
}

std::array<Real,3> Tower::base_moment () const
{
    std::array<Real,3> M{{0.0, 0.0, 0.0}};
    for (std::size_t i = 0; i < m_nodes.size(); ++i) {
        const Real rx = m_nodes[i].pos[0] - m_base[0], ry = m_nodes[i].pos[1] - m_base[1], rz = m_nodes[i].pos[2] - m_base[2];
        const Real fx = m_force[3*i], fy = m_force[3*i+1], fz = m_force[3*i+2];
        M[0] += ry * fz - rz * fy;
        M[1] += rz * fx - rx * fz;
        M[2] += rx * fy - ry * fx;
    }
    return M;
}

FoundationLoad Tower::foundation () const
{
    FoundationLoad L;
    auto Fd = total_force();
    auto Md = base_moment();
    // a moving tower's nodes also carry their inertial forces to the foundation
    for (std::size_t i = 0; i < m_nodes.size(); ++i) {
        const Real rx = m_nodes[i].pos[0] - m_base[0], ry = m_nodes[i].pos[1] - m_base[1], rz = m_nodes[i].pos[2] - m_base[2];
        const Real fx = m_inertia[3*i], fy = m_inertia[3*i+1], fz = m_inertia[3*i+2];
        Fd[0] += fx; Fd[1] += fy; Fd[2] += fz;
        Md[0] += ry * fz - rz * fy;
        Md[1] += rz * fx - rx * fz;
        Md[2] += rx * fy - ry * fx;
    }
    const Real rx = m_line_at[0] - m_base[0], ry = m_line_at[1] - m_base[1], rz = m_line_at[2] - m_base[2];
    const auto& F = m_line_force;
    for (int d = 0; d < 3; ++d) { L.force[d] = Fd[d] + F[d]; }
    L.moment = {{Md[0] + ry * F[2] - rz * F[1], Md[1] + rz * F[0] - rx * F[2], Md[2] + rx * F[1] - ry * F[0]}};
    L.shear = std::hypot(L.force[0], L.force[1]);
    L.overturning = std::hypot(L.moment[0], L.moment[1]);
    L.vertical = m_type.weight - L.force[2];
    // the legs in the frame of the line (along) and the cross-arm (across), a = s / 2 from the centre
    const Real a = Real(0.5) * m_type.legs();
    const std::array<Real,2> across{{m_across[0], m_across[1]}};
    const std::array<Real,2> along{{m_across[1], -m_across[0]}};
    int i = 0;
    for (const Real su : {Real(1.0), Real(-1.0)}) {
        for (const Real sv : {Real(1.0), Real(-1.0)}) {
            const Real x = a * (su * along[0] + sv * across[0]);
            const Real y = a * (su * along[1] + sv * across[1]);
            // the reactions R_i balance the load: sum R_i = P, sum x_i R_i = M_y, sum y_i R_i = -M_x
            L.legs[static_cast<std::size_t>(i++)] = Real(0.25) * L.vertical + (L.moment[1] * x - L.moment[0] * y) / (Real(4.0) * a * a);
        }
    }
    L.max_compression = std::max({L.legs[0], L.legs[1], L.legs[2], L.legs[3], Real(0.0)});
    L.max_uplift = std::max(Real(0.0), -std::min({L.legs[0], L.legs[1], L.legs[2], L.legs[3]}));
    L.over_allowable = (m_type.allowable_uplift > 0.0 && L.max_uplift > m_type.allowable_uplift) ||
                       (m_type.allowable_compression > 0.0 && L.max_compression > m_type.allowable_compression);
    return L;
}

} // namespace erf_towers
