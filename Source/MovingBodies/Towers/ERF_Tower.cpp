#include "ERF_Tower.H"

#include <algorithm>
#include <cmath>
#include <utility>

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

} // namespace erf_towers
