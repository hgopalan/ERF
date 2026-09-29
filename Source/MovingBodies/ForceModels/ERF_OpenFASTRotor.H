#ifndef ERF_OPENFAST_ROTOR_H
#define ERF_OPENFAST_ROTOR_H

#include <array>
#include <vector>

#include <AMReX_REAL.H>

#include "ERF_OpenFASTDriver.H"

namespace erf_actuator {

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
