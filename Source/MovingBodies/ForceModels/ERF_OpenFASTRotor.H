#ifndef ERF_OPENFAST_ROTOR_H
#define ERF_OPENFAST_ROTOR_H

#include <array>
#include <vector>

#include <AMReX_REAL.H>

#include "ERF_OpenFASTDriver.H"

namespace erf_actuator {

/**
 * Actuator-line representation of an OpenFAST rotor's loads: one point per force node (the hub
 * and every blade node, at the positions OpenFAST reports for the rotating, yawed and deflected
 * blades after its step) carrying the node's force on the fluid, the negative of OpenFAST's.
 * Tower nodes are not handled here.
 *
 * @param[in] t the turbine after a step (positions and forces)
 * @param[out] pos 3*n point positions (m), appended
 * @param[out] force 3*n forces on the fluid (N), appended
 */
void alm_points (const erf_openfast::TurbineState& t,
                 std::vector<amrex::Real>& pos, std::vector<amrex::Real>& force);

/**
 * The tower force nodes as actuator points: the nodes after the blades (num_force_pts_tower of
 * them, base to top, at the positions OpenFAST reports) carrying minus the tower loads OpenFAST
 * computed from the velocities sampled at its tower nodes (AeroDyn's tower drag, or the stub's
 * cylinder drag). Nothing is appended for a model without tower force nodes.
 *
 * @param[in] t the turbine after a step
 * @param[out] pos 3*n point positions (m), appended
 * @param[out] force 3*n forces on the fluid (N), appended
 */
void tower_points (const erf_openfast::TurbineState& t,
                   std::vector<amrex::Real>& pos, std::vector<amrex::Real>& force);

/**
 * Drag of the nacelle on the fluid: -1/2 rho cd area |u| u for the velocity u at the hub, a
 * single point force at the hub node. cd and area are the user's.
 */
std::array<amrex::Real,3> nacelle_drag_force (const std::array<amrex::Real,3>& u_hub,
                                              amrex::Real rho, amrex::Real cd, amrex::Real area);

/**
 * The velocity the nacelle sees, corrected for the drag point's own kernel: a Gaussian body
 * force of width epsilon induces -F / (4 pi rho epsilon^2 ... ) at its own centre, so the
 * sampled hub velocity understates the free velocity by the factor 1 - cd area / (4 pi eps^2)
 * (Kynema's correction, here with the run's kernel width). Returns u_sampled divided by that
 * factor; aborts when the factor is not positive, which means the kernel is far too narrow
 * for the nacelle's drag area.
 */
std::array<amrex::Real,3> nacelle_corrected_velocity (const std::array<amrex::Real,3>& u_sampled,
                                                      amrex::Real cd, amrex::Real area, amrex::Real epsilon);

/**
 * The rotor's tip radius: the largest distance of a blade force node from the hub axis (m).
 * Zero for a rotor without blade nodes.
 */
amrex::Real tip_radius (const erf_openfast::TurbineState& t);

/**
 * How many cells the blade tip sweeps in one ERF step: rotor_speed * tip_radius * dt / dx.
 * The actuator line's force must not jump over cells between two steps, so the caller aborts
 * when this exceeds its limit (one cell by default).
 */
amrex::Real tip_cells_per_step (const erf_openfast::TurbineState& t, amrex::Real dt, amrex::Real dx);

/**
 * Actuator-disk representation of an OpenFAST rotor's loads.
 *
 * Each blade force node, at radius r from the hub on the shaft axis, is replaced by a ring of
 * num_points_t points at that radius carrying its force divided over the ring: the axial
 * component is kept, the radial and tangential components are rotated with the ring azimuth.
 * The force along the shaft and the torque about it are therefore preserved exactly; the net
 * in-plane force (the blades' in-plane loads not cancelling at an instant) is averaged away
 * with the azimuth, which is what an actuator disk does. The hub node's force is kept as one
 * point; tower nodes are not handled here.
 *
 * The forces produced are the forces ON THE FLUID: the negative of OpenFAST's, which are on
 * the structure.
 *
 * @param[in] t the turbine after a step (positions, forces, hub position and axis)
 * @param[in] num_points_t points per ring
 * @param[out] pos 3*n ring point positions (m), appended
 * @param[out] force 3*n forces on the fluid (N), appended
 */
void adm_rings (const erf_openfast::TurbineState& t, int num_points_t,
                std::vector<amrex::Real>& pos, std::vector<amrex::Real>& force);

/**
 * Check that the blade force nodes lie close to the rotor plane of the hub axis: the largest
 * |(p - hub) . axis| / |p - hub| over the blade nodes. A large value means the axis convention
 * does not match this OpenFAST build; precone and shaft tilt give a few degrees at most.
 */
amrex::Real max_out_of_plane_sine (const erf_openfast::TurbineState& t);

} // namespace erf_actuator

#endif
