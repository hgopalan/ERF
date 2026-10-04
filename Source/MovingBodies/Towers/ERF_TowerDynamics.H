#ifndef ERF_TOWER_DYNAMICS_H
#define ERF_TOWER_DYNAMICS_H

#include <array>
#include <cstddef>
#include <vector>

#include <AMReX_REAL.H>

#include "ERF_Tower.H"

namespace erf_towers {

/**
 * The structural model of a tower behind a narrow interface: the loads on its nodes (the members'
 * drag, 3 per node) and each line's pull where it hangs from the tower (one per attachment) go in,
 * held over a step; the motion of every node and of every attachment comes out, with each node's
 * inertial force for the foundation's reactions. The one-mode tower implements it; a frame model of the
 * lattice can stand in its place without touching the callers.
 */
class TowerModel
{
public:
    virtual ~TowerModel () = default;
    /** The attachments the lines hang from. */
    virtual std::size_t num_attachments () const = 0;
    /** Advance by dt under the node forces (3 per node, N) and each line's pull at its attachment (N), held over the step. */
    virtual void step (amrex::Real dt, const std::vector<amrex::Real>& node_force,
                       const std::vector<std::array<amrex::Real,3>>& line_force) = 0;
    /** The same for a tower with a single attachment. */
    void step (amrex::Real dt, const std::vector<amrex::Real>& node_force, const std::array<amrex::Real,3>& line_force)
    {
        step(dt, node_force, std::vector<std::array<amrex::Real,3>>{line_force});
    }
    /** A node's displacement from where it stands unloaded (m) and its velocity (m/s). */
    virtual std::array<amrex::Real,3> displacement (std::size_t node) const = 0;
    virtual std::array<amrex::Real,3> velocity (std::size_t node) const = 0;
    /** A node's inertial force, minus its mass times its acceleration (N), under the loads of the last step. */
    virtual std::array<amrex::Real,3> inertial_force (std::size_t node) const = 0;
    /** The displacement (m) and velocity (m/s) of attachment a (the first by default). */
    virtual std::array<amrex::Real,3> attachment_displacement (std::size_t a = 0) const = 0;
    virtual std::array<amrex::Real,3> attachment_velocity (std::size_t a = 0) const = 0;
    /** The natural frequency of the slowest mode (Hz), which sets how finely the coupling must step. */
    virtual amrex::Real frequency () const = 0;
    /** The state to checkpoint, and its restoration: false when the values do not fit the model. */
    virtual std::vector<double> state () const = 0;
    virtual bool set_state (const std::vector<double>& s) = 0;
};

/**
 * A tower bending in its first mode, the same way in x and y (a square tower), on a foundation
 * that can tilt and slide. Its mass, the type's weight over gravity, is spread over the drag nodes
 * by the length of member each stands for. A node at height z above the base moves horizontally
 * by phi(z) times the cross-arm's displacement q, where
 *
 *   phi(z) = a_b (z/H)^2 + a_r z/H + a_l,   H the cross-arm's height,
 *
 * the bending shape of a cantilever, the tilt of the footing and its slide, each in proportion to
 * its compliance under a load at the cross-arm: a_b = (1/K_b)/c, a_r = (H^2/k_r)/c, a_l = (1/k_l)/c,
 * c = 1/K_b + H^2/k_r + 1/k_l, a rigid foundation (stiffness 0) adding nothing. The generalized
 * mass is M = sum m_i phi_i^2 and the stiffness K = 1/c, the strain energy of that shape exactly;
 * K_b is the bending stiffness that gives the type's frequency on a rigid foundation,
 * K_b = sum m_i (z_i/H)^4 (2 pi f)^2. In each direction
 *
 *   M q'' + 2 zeta omega M q' + K q = Q,   Q = sum phi_i F_i + F_line,   omega^2 = K / M,
 *
 * with the load Q held over a step integrated exactly, so any step is stable. Each line pulls at
 * its attachment (the tower's attachments, or the centre of the cross-arm when it has none), which
 * moves by phi at the attachment's height times q and weights the line's pull in Q by the same:
 * phi = 1 at the cross-arm, more on a peak above it. The tower does not twist, rise or sink.
 */
class OneModeTower final : public TowerModel
{
public:
    OneModeTower (const Tower& tower, amrex::Real gravity);

    using TowerModel::step;
    std::size_t num_attachments () const override { return m_phi_att.size(); }
    void step (amrex::Real dt, const std::vector<amrex::Real>& node_force,
               const std::vector<std::array<amrex::Real,3>>& line_force) override;
    std::array<amrex::Real,3> displacement (std::size_t node) const override;
    std::array<amrex::Real,3> velocity (std::size_t node) const override;
    std::array<amrex::Real,3> inertial_force (std::size_t node) const override;
    std::array<amrex::Real,3> attachment_displacement (std::size_t a = 0) const override;
    std::array<amrex::Real,3> attachment_velocity (std::size_t a = 0) const override;
    amrex::Real frequency () const override;
    std::vector<double> state () const override;
    bool set_state (const std::vector<double>& s) override;

    /** The node's mass (kg) and its value of the mode shape. */
    double node_mass (std::size_t node) const { return m_mass[node]; }
    double mode_shape (std::size_t node) const { return m_phi[node]; }
    /** The mode shape at attachment a. */
    double attachment_shape (std::size_t a) const { return m_phi_att[a]; }
    /** The generalized mass (kg) and stiffness (N/m) of the mode, the cross-arm's displacement its coordinate. */
    double generalized_mass () const { return m_M; }
    double stiffness () const { return m_K; }
    /** The stiffness of the bending alone, on a rigid foundation (N/m). */
    double bending_stiffness () const { return m_Kb; }
    /** The cross-arm's horizontal displacement (m) and velocity (m/s), x and y. */
    const std::array<double,2>& q () const { return m_q; }
    const std::array<double,2>& v () const { return m_v; }

private:
    double acceleration (int d) const { return (m_Q[d] - 2.0 * m_zeta * m_omega * m_M * m_v[d] - m_K * m_q[d]) / m_M; }

    std::vector<double> m_mass;
    std::vector<double> m_phi;
    std::vector<double> m_phi_att;   //!< the mode shape at each attachment
    double m_M{0.0}, m_K{0.0}, m_Kb{0.0}, m_omega{0.0}, m_zeta{0.0};
    std::array<double,2> m_q{{0.0, 0.0}}, m_v{{0.0, 0.0}}, m_Q{{0.0, 0.0}};   //!< the generalized force of the last step
};

} // namespace erf_towers

#endif
