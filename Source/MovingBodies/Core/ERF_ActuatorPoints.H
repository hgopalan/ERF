#ifndef ERF_ACTUATOR_POINTS_H
#define ERF_ACTUATOR_POINTS_H

#include <vector>

#include <AMReX.H>
#include <AMReX_REAL.H>

namespace erf_actuator {

/**
 * The actuator points of every body, flattened into one list so that sampling (and later
 * spreading) sees one set of points whatever the bodies are. Body b owns the points
 * [offset(b), offset(b+1)). Positions are refreshed every step from the bodies; the velocities
 * are what the flow sampled at them.
 */
class ActuatorPoints
{
public:
    /** Start a new list. */
    void clear () { m_offsets.assign(1, 0); m_pos.clear(); m_vel.clear(); }

    /** Append a body's points (3*n coordinates) and return its index. */
    int add_body (const std::vector<amrex::Real>& pos)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(pos.size() % 3 == 0, "ActuatorPoints: positions come as x,y,z triples");
        if (m_offsets.empty()) { m_offsets.assign(1, 0); }
        m_pos.insert(m_pos.end(), pos.begin(), pos.end());
        m_offsets.push_back(static_cast<int>(m_pos.size() / 3));
        m_vel.assign(m_pos.size(), amrex::Real(0.0));
        return static_cast<int>(m_offsets.size()) - 2;
    }

    int num_bodies () const { return m_offsets.empty() ? 0 : static_cast<int>(m_offsets.size()) - 1; }
    int num_points () const { return static_cast<int>(m_pos.size() / 3); }
    int begin (int body) const { return m_offsets[body]; }
    int end (int body) const { return m_offsets[body + 1]; }

    const std::vector<amrex::Real>& positions () const { return m_pos; }
    std::vector<amrex::Real>& velocities () { return m_vel; }
    const std::vector<amrex::Real>& velocities () const { return m_vel; }

    /** The velocities of one body's points, 3*n values. */
    std::vector<amrex::Real> body_velocities (int body) const
    {
        return std::vector<amrex::Real>(m_vel.begin() + 3 * begin(body), m_vel.begin() + 3 * end(body));
    }

private:
    std::vector<int> m_offsets{0};
    std::vector<amrex::Real> m_pos;
    std::vector<amrex::Real> m_vel;
};

} // namespace erf_actuator

#endif
