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
 * The velocity nodes shifted upstream by diameters * 2 * tip_radius along the horizontal projection
 * of the shaft axis, where the flow is the free stream: there OpenFAST's own induction model applies
 * (sampling = upstream). The shaft's tilt is left out, so the nodes sample the free stream at their
 * own heights (a 6 degree tilt over two diameters of the IEA 15 MW would lift them by about 50 m).
 * Every node moves by the same vector, so the blade nodes keep their radii and the hub node leads them.
 */
std::vector<amrex::Real> upstream_sampling_positions (const erf_openfast::TurbineState& t, amrex::Real diameters);

/**
 * Filtered-actuator-disk correction factor of Shapiro, Gayme and Meneveau (Wind Energy 22, 2019, eq. 25):
 * the velocity sampled at a Gaussian-smeared disk understates the thin-disk induction by the factor
 * M = 1 / (1 + Ct'/4 * (Delta/R) / sqrt(3 pi)), where Ct' = 4a/(1-a) is the thrust coefficient on the
 * disk velocity and Delta the width of the paper's filter G = (6/(pi Delta^2))^(3/2) exp(-6 r^2/Delta^2);
 * valid for Delta/R up to about 1.25. ERF's kernel exp(-r^2/eps^2) has Delta = sqrt(6) eps.
 */
amrex::Real filtered_disk_factor (amrex::Real ct_prime, amrex::Real delta_over_R);

//! the paper's filter width for ERF's kernel width eps: Delta = sqrt(6) eps
amrex::Real filter_width_from_eps (amrex::Real eps);

/** One update of the corrected free stream for sampling = disk_corrected. */
struct DiskCorrection
{
    amrex::Real u_disk{0.0};     //!< radius-weighted axial velocity sampled at the blade nodes (m/s)
    amrex::Real ct{0.0};         //!< thrust coefficient on the previous free stream
    amrex::Real a{0.0};          //!< axial induction factor from Ct (momentum theory)
    amrex::Real ct_prime{0.0};   //!< 4a/(1-a)
    amrex::Real M{1.0};          //!< filtered-disk factor
    amrex::Real u_inf{0.0};      //!< corrected free stream M u_disk / (1 - a)
    amrex::Real factor{1.0};     //!< u_inf / u_disk applied to the sampled node velocities
    amrex::Real gain{0.0};       //!< linear gain of the unrelaxed update (disk_correction_gain): unstable beyond -1
    amrex::Real relax{1.0};      //!< the relaxation factor the update used
    bool clamped{false};         //!< Ct was clamped to 0.96
};

/**
 * The linear gain of the disk_corrected update U_new = M u_disk / (1 - a) with Ct taken from the previous
 * step, for induction a and filter width Delta/R, with the flow induced as (1 - delta) of the thin disk
 * (delta = (Delta/R)/sqrt(3 pi)): G = -2 a (1 - a)(1 - delta) / ((1 - 2a)(1 - a + delta a)). |G| > 1 means the
 * unrelaxed update oscillates and grows: Ct above about 0.75 + 0.11 Delta/R.
 */
amrex::Real disk_correction_gain (amrex::Real a, amrex::Real delta_over_R);

/**
 * The relaxation factor that makes the linear error of the update vanish in one step,
 * 1 / (1 - G) for the (negative) gain G, held in [0.2, 1].
 */
amrex::Real disk_correction_relax (amrex::Real gain);

/**
 * The same factor, held at most dt / (tau (1 - G)) as well, so an error of the estimate decays on
 * the time scale tau (s; none when tau <= 0): the flow answers a change of thrust only after a
 * delay, and an update faster than that can ring at the delay's period however its one-step gain
 * is cancelled.
 */
amrex::Real disk_correction_relax (amrex::Real gain, amrex::Real dt, amrex::Real tau);

/**
 * The corrected free stream from the disk velocity u_disk, the rotor thrust of the previous step
 * (force on the structure along the shaft, N), the previous free stream (the Ct is evaluated on it),
 * the air density, the tip radius and ERF's kernel width eps (m). With no previous loads (thrust
 * <= 0 or u_inf_prev <= 0) the free stream is the disk velocity itself. Ct is clamped to [0, 0.96].
 */
DiskCorrection filtered_disk_correction (amrex::Real u_disk, amrex::Real thrust, amrex::Real u_inf_prev,
                                         amrex::Real rho, amrex::Real tip_radius, amrex::Real eps);

/**
 * Area-weighted mean of the axial (shaft-axis) velocity over the blade velocity nodes: each node
 * weighted by r dr, its distance r from the hub axis times its share dr of the blade (half the span
 * to its neighbours on the same blade; the full spacing to its one neighbour at either end). The hub
 * and tower nodes carry no weight. Zero for a rotor without blade nodes. This takes the nodes as the
 * centres of the blade's elements, as OpenFAST's actuator points are: for equal elements it is the
 * area average to 1/(4 n^2) of n nodes. Nodes placed at the root and the tip would count the end
 * nodes double (for 50 nodes, +0.4 to +1 % for a velocity rising towards the tip).
 */
amrex::Real disk_axial_velocity (const erf_openfast::TurbineState& t, const std::vector<amrex::Real>& uvw);

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
