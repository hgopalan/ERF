#ifndef ERF_TOWER_H
#define ERF_TOWER_H

#include <array>
#include <string>
#include <vector>

#include <AMReX_REAL.H>

#include "ERF_MemberDrag.H"
#include "ERF_TowerInputs.H"

namespace erf_towers {

/**
 * One lattice tower of a TowerType standing on the terrain: a vertical body from its base up to
 * the cross-arm (and the peak above it), tapering linearly from base_width to top_width, and a
 * horizontal cross-arm of arm_length centred on the body at arm_height above the base, along
 * the horizontal unit vector `across`. Each is cut into drag nodes at the middle of equal
 * segments: the body's width there times the solidity is its drag width; the cross-arm's is its
 * face depth times the solidity. The body's nodes come first, then the arm's.
 */
class Tower
{
public:
    Tower (std::string name, const TowerType& type, const std::array<amrex::Real,3>& base,
           amrex::Real arm_height, const std::array<amrex::Real,3>& across);

    const std::string& name () const { return m_name; }
    const TowerType& type () const { return m_type; }
    /** The centre of the base, on the terrain (m). */
    const std::array<amrex::Real,3>& base () const { return m_base; }
    amrex::Real arm_height () const { return m_arm_height; }
    const std::array<amrex::Real,3>& across () const { return m_across; }
    const std::vector<MemberNode>& nodes () const { return m_nodes; }
    int num_body_nodes () const { return m_nbody; }

    /** Hand over the force on every node (3 per node, N) from the members' aerodynamics. */
    void set_loads (const std::vector<amrex::Real>& force) { m_force = force; }
    const std::vector<amrex::Real>& loads () const { return m_force; }
    /** The sum of the node forces (N), and their moment about the centre of the base (N m). */
    std::array<amrex::Real,3> total_force () const;
    std::array<amrex::Real,3> base_moment () const;

private:
    std::string m_name;
    TowerType m_type;
    std::array<amrex::Real,3> m_base;
    amrex::Real m_arm_height{0.0};
    std::array<amrex::Real,3> m_across;
    std::vector<MemberNode> m_nodes;
    int m_nbody{0};
    std::vector<amrex::Real> m_force;
};

} // namespace erf_towers

#endif
