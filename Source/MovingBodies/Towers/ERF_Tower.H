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
 * What a tower's foundation takes: the loads on the tower about the centre of its base, and the
 * axial force in each of its four legs, a rigid square of footings sharing the vertical load and
 * resisting the overturning moment (compression positive).
 */
struct FoundationLoad
{
    std::array<amrex::Real,3> force{{0.0, 0.0, 0.0}};   //!< all loads on the tower but its weight (N)
    std::array<amrex::Real,3> moment{{0.0, 0.0, 0.0}};  //!< their moment about the base centre (N m)
    amrex::Real shear{0.0};                             //!< the horizontal force, the base shear (N)
    amrex::Real overturning{0.0};                       //!< the horizontal moment (N m)
    amrex::Real vertical{0.0};                          //!< the downward load on the footings, the weight included (N)
    std::array<amrex::Real,4> legs{{0.0, 0.0, 0.0, 0.0}};  //!< each leg's axial force, compression positive (N)
    amrex::Real max_compression{0.0};                   //!< the largest leg compression (N)
    amrex::Real max_uplift{0.0};                        //!< the largest leg tension, 0 when none pulls (N)
    bool over_allowable{false};                         //!< either exceeds its allowable (where one is given)
};

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

    /**
     * Hand over how the tower moves (3 per node each): every node's displacement from where it stands
     * unloaded (m), its velocity (m/s) and its inertial force, minus its mass times its acceleration
     * (N). A tower that stands still keeps them zero.
     */
    void set_motion (const std::vector<amrex::Real>& displacement, const std::vector<amrex::Real>& velocity,
                     const std::vector<amrex::Real>& inertia);
    /** The nodes where they are now, moved by their displacement. */
    std::vector<MemberNode> current_nodes () const;
    /** The nodes' velocities (3 per node, m/s), zero for a tower that stands still. */
    const std::vector<amrex::Real>& node_velocities () const { return m_vel; }
    /** The cross-arm's displacement (m): where the lines hang from has moved by it. */
    std::array<amrex::Real,3> arm_displacement () const;

    /** Hand over the force on every node (3 per node, N) from the members' aerodynamics. */
    void set_loads (const std::vector<amrex::Real>& force) { m_force = force; }
    const std::vector<amrex::Real>& loads () const { return m_force; }
    /** The sum of the node forces (N), and their moment about the centre of the base (N m): the wind's drag. */
    std::array<amrex::Real,3> total_force () const;
    std::array<amrex::Real,3> base_moment () const;

    /** Hand over the pull of the lines on the tower (N), applied where they hang from it. */
    void set_line_load (const std::array<amrex::Real,3>& force, const std::array<amrex::Real,3>& at)
    {
        m_line_force = force;
        m_line_at = at;
    }
    const std::array<amrex::Real,3>& line_force () const { return m_line_force; }
    /**
     * The foundation's load from the wind's drag and the lines' pull together, with the tower's
     * weight and, for a tower that moves, the inertial forces of its nodes (the reactions balance
     * the loads less the mass times the acceleration of every node): the legs stand at
     * (+-s/2, +-s/2) in the frame of the line and the cross-arm, s the leg spacing, and share
     * the downward load P equally and the overturning moment M linearly,
     * N_i = P/4 + (M_y x_i - M_x y_i) / (4 (s/2)^2) in that frame.
     */
    FoundationLoad foundation () const;

private:
    std::string m_name;
    TowerType m_type;
    std::array<amrex::Real,3> m_base;
    amrex::Real m_arm_height{0.0};
    std::array<amrex::Real,3> m_across;
    std::vector<MemberNode> m_nodes;
    int m_nbody{0};
    std::vector<amrex::Real> m_force;
    std::vector<amrex::Real> m_disp;      //!< 3 per node: the displacement from where it stands unloaded
    std::vector<amrex::Real> m_vel;       //!< 3 per node
    std::vector<amrex::Real> m_inertia;   //!< 3 per node: minus mass times acceleration
    std::array<amrex::Real,3> m_line_force{{0.0, 0.0, 0.0}};
    std::array<amrex::Real,3> m_line_at{{0.0, 0.0, 0.0}};
};

} // namespace erf_towers

#endif
