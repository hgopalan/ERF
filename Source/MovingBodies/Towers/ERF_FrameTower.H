#ifndef ERF_FRAME_TOWER_H
#define ERF_FRAME_TOWER_H

/**
 * A lattice tower that bends as its frame model (ERF_Frame.H, read from a SubDyn input file): the
 * TowerModel the conductor coupling drives when a tower type gives erf.conductors.<type>.frame_file.
 * The wind's drag on the tower's drag nodes and the lines' pull at its attachments are carried onto
 * the frame's nodes by rigid links, the frame advances by Newmark's method (ERF_FrameDynamics.H), and
 * the nodes' motion comes back to the drag nodes and attachments through the same links. The frame
 * moves about its static equilibrium under its own weight; its support reactions give the footings'
 * loads.
 *
 * Axes: the frame file's are tower-local, origin at the base centre, x along the line (along =
 * (across_y, -across_x, 0)), y along the cross-arm (Tower::across()), z up. Everything this class takes
 * and returns is in ERF's axes.
 */

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "ERF_Frame.H"
#include "ERF_FrameDynamics.H"
#include "ERF_Tower.H"
#include "ERF_TowerDynamics.H"

namespace erf_towers {

/**
 * A point tied rigidly to a few frame nodes. A force at the point is shared among the nodes so that
 * the force and its moment are kept: F_i = F/n + w x d_i, with d_i the node's offset from the nodes'
 * centroid c and w = I^+ ((p - c) x F), I = sum (|d_i|^2 1 - d_i d_i^T) and ^+ the pseudo-inverse. The
 * point's motion is the adjoint, so the links do no work: u_p = mean(u_i) + theta x (p - c), theta =
 * I^+ sum d_i x (u_i - mean(u)). A moment about a line through collinear nodes is not carried. A node
 * within 1e-6 m of the point takes the force alone and gives its motion.
 */
class RigidLink
{
public:
    RigidLink () = default;
    /** Tie point p (m, frame axes) to the count frame nodes nearest to it (all of them when there are fewer). */
    RigidLink (const Frame& frame, const std::array<double,3>& p, std::size_t count = 4);
    /** The distance (m) from the point to its nearest node. */
    double distance () const { return m_distance; }
    /** The frame nodes the point is tied to (0-based). */
    const std::vector<std::size_t>& nodes () const { return m_nodes; }
    /** Add a force at the point (N, frame axes) to loads, 6 values per frame node (the translations only). */
    void add_load (const std::array<double,3>& force, std::vector<double>& loads) const;
    /** The point's translation (m, frame axes) from u, 6 values per frame node. */
    std::array<double,3> motion (const std::vector<double>& u) const;

private:
    std::vector<std::size_t> m_nodes;
    std::vector<std::array<double,3>> m_d;      //!< each node's offset from the centroid (m)
    std::array<double,3> m_r{{0.0, 0.0, 0.0}};  //!< the point's offset from the centroid (m)
    std::array<double,9> m_iinv{};              //!< I^+ (1/m^2), row-major
    double m_distance{0.0};
};

/** One tower bending as its frame model. */
class FrameTower final : public TowerModel
{
public:
    /**
     * Stand the frame on the tower: tie the tower's drag nodes and attachments to the frame's nodes
     * (each must lie within the type's base_width of a frame node), match the frame's four supports to
     * the tower's legs (one per quadrant of the base, all at z = 0), solve the static equilibrium under
     * the frame's weight (gravity in m/s^2), find the first natural frequency and set the Rayleigh
     * damping of the type's damping_ratio at that frequency and ten times it. Aborts, naming the tower
     * and erf.conductors.<type>.frame_file, when the frame does not fit the tower.
     */
    FrameTower (const Tower& tower, std::shared_ptr<const Frame> frame, double gravity);

    using TowerModel::step;
    std::size_t num_attachments () const override { return m_attach.size(); }
    void step (amrex::Real dt, const std::vector<amrex::Real>& node_force,
               const std::vector<std::array<amrex::Real,3>>& line_force) override;
    /** A drag node's displacement from where it stands under the frame's weight (m, ERF axes). */
    std::array<amrex::Real,3> displacement (std::size_t node) const override;
    std::array<amrex::Real,3> velocity (std::size_t node) const override;
    /** Zero: the footings' loads come from the frame's support reactions (foundation()), not from the drag nodes' inertia. */
    std::array<amrex::Real,3> inertial_force (std::size_t node) const override;
    std::array<amrex::Real,3> attachment_displacement (std::size_t a = 0) const override;
    std::array<amrex::Real,3> attachment_velocity (std::size_t a = 0) const override;
    /** The frame's first natural frequency (Hz), which the coupling resolves; Newmark's method integrates the higher modes stably. */
    amrex::Real frequency () const override { return static_cast<amrex::Real>(m_f1); }
    /** The Newmark state (time, u, v, a of every frame degree of freedom) and the frame loads of the last step. */
    std::vector<double> state () const override;
    bool set_state (const std::vector<double>& s) override;
    /**
     * The footings' load from the frame's support reactions: the static reactions under its weight plus
     * the dynamic ones, K u + a1 K v + M a - f at the supports (the mass-proportional damping excluded).
     * Before the frame's first step, the static reactions under the tower's present drag (Tower::loads())
     * and line pulls (Tower::line_forces()), as Tower::foundation() reckons a tower at rest.
     * force and moment (about the base centre) are the loads on the tower but its weight, as
     * Tower::foundation() defines them; legs[i] is the upward reaction at the support of leg i
     * (compression positive), in Tower::foundation()'s leg order.
     */
    bool foundation (const Tower& tower, FoundationLoad& load) const override;

    const Frame& frame () const { return *m_frame; }
    double mass () const { return m_frame->total_mass(); }

private:
    std::array<double,3> to_local (const std::array<amrex::Real,3>& v) const;
    std::array<amrex::Real,3> to_erf (const std::array<double,3>& v) const;

    std::shared_ptr<const Frame> m_frame;
    std::string m_name;                          //!< the tower's name, for messages
    std::array<double,3> m_along{}, m_across{};  //!< the tower-local x and y axes in ERF's axes
    std::vector<RigidLink> m_drag;               //!< per drag node of the tower
    std::vector<RigidLink> m_attach;             //!< per attachment of the tower
    std::array<std::size_t,4> m_leg_support{};   //!< per leg (Tower::foundation()'s order): the index into the frame's supports
    std::vector<std::array<double,6>> m_static;  //!< per support: the reaction under the frame's weight (N, N m; frame axes)
    double m_f1{0.0};                            //!< first natural frequency (Hz)
    double m_a1{0.0};                            //!< the stiffness-proportional Rayleigh coefficient (s)
    std::unique_ptr<FrameDynamics> m_dyn;
    std::vector<double> m_load;                  //!< the frame loads of the last step (6 per node, frame axes)
    bool m_stepped{false};                       //!< whether the frame has taken a step (or continues a saved state)
};

} // namespace erf_towers

#endif
