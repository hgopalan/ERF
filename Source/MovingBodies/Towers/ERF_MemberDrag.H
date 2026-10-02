#ifndef ERF_MEMBER_DRAG_H
#define ERF_MEMBER_DRAG_H

#include <array>
#include <vector>

#include <AMReX_REAL.H>

namespace erf_towers {

/**
 * One node of a structural member, for its aerodynamic load: where it is, the member's axis there
 * (a unit vector), the length of member it stands for, and the member's drag per unit length: the
 * width the wind acts on (a diameter, or a lattice face's width times its solidity) and the drag
 * coefficient on that width.
 */
struct MemberNode
{
    std::array<amrex::Real,3> pos{{0.0, 0.0, 0.0}};
    std::array<amrex::Real,3> axis{{0.0, 0.0, 1.0}};
    amrex::Real length{0.0};             //!< m of member
    amrex::Real drag_width{0.0};         //!< m: the area per unit length the wind acts on
    amrex::Real drag_coefficient{1.0};
};

/**
 * The drag on one node (N): 1/2 rho Cd w L |U_n| U_n, where U_n is the wind relative to the node
 * with its component along the member's axis removed (only the normal flow loads a slender
 * member), w the drag width and L the node's length of member.
 */
std::array<amrex::Real,3> member_drag (const MemberNode& node, const std::array<amrex::Real,3>& wind,
                                       const std::array<amrex::Real,3>& velocity, amrex::Real rho);

/**
 * The force coefficient of a square lattice tower of flat-sided members on the members' projected
 * area of one face, from the face's solidity phi (the members' area over the face's outline):
 * Cf = 4 phi^2 - 5.9 phi + 4 (ASCE 7, trussed towers of square cross-section). It counts the
 * windward and leeward faces together, so the drag of a lattice body is Cf q phi times its face
 * area.
 */
amrex::Real lattice_force_coefficient (amrex::Real solidity);

/**
 * The aerodynamics of a structure's members behind a narrow interface: node positions and axes
 * (the MemberNodes), the wind at the nodes and the nodes' own velocities in (3 per node), the force
 * on each node out (3 per node). The ERF member drag implements it; another aerodynamic model,
 * such as an external tower-drag library, can stand in its place without touching the callers.
 */
class MemberAero
{
public:
    virtual ~MemberAero () = default;
    virtual void loads (const std::vector<MemberNode>& nodes, const std::vector<amrex::Real>& wind,
                        const std::vector<amrex::Real>& velocity, std::vector<amrex::Real>& force) const = 0;
};

/** ERF's member drag: member_drag() at every node with the air density it is given. */
class MemberDrag final : public MemberAero
{
public:
    explicit MemberDrag (amrex::Real rho) : m_rho(rho) {}
    void loads (const std::vector<MemberNode>& nodes, const std::vector<amrex::Real>& wind,
                const std::vector<amrex::Real>& velocity, std::vector<amrex::Real>& force) const override;
    amrex::Real air_density () const { return m_rho; }
private:
    amrex::Real m_rho{1.225};
};

} // namespace erf_towers

#endif
