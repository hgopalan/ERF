#ifndef ERF_ACTUATOR_SPREADING_H
#define ERF_ACTUATOR_SPREADING_H

/** Spreading of point forces onto the face-centred momentum sources with a normalised Gaussian kernel, and the integral check. */

#include <string>
#include <vector>

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include "ERF_ActuatorSampling.H"

namespace erf_actuator {

/**
 * Spread point forces onto the face-centred momentum sources with a 3-D Gaussian kernel.
 *
 * Each point p carries a force F_p on the fluid (N) and has its own kernel width epsilon_p
 * (the blades, tower and nacelle of a turbine, or two bodies, may differ). Its share of the source at a face is
 * F_p w_p(face) / S_p, where w_p = exp(-|x_face - x_p|^2 / epsilon_p^2) is cut off beyond
 * 3 epsilon_p and S_p = sum over the faces of that component of w_p dV is the discrete
 * normalisation with the face volumes (dx dy dz detJ on a terrain-following mesh). The sum
 * of src dV over the mesh therefore equals F_p exactly for every point and component,
 * whatever the resolution and however much of the kernel is cut off by the ground or the
 * domain top; the kernel's shape is Gaussian only where it is resolved.
 *
 * Every rank accumulates the faces its boxes own (each physical face once, however the boxes
 * share their boundary faces) and the normalisations are summed across ranks, so the result
 * does not depend on the domain decomposition; every rank must pass the same points. Every periodic image of a point within
 * 3 epsilon_p of the domain (in x and y; z is not imaged) adds its kernel, so a kernel wraps across
 * a periodic boundary, and overlaps itself when the period is under 6 epsilon_p. The sources are
 * set (not added to) on every valid face, both copies of a shared face alike; the faces on a
 * non-periodic domain boundary get no source.
 *
 * Aborts when a position or force is not finite. z_phys_nd and detJ_cc must share the boxes and
 * distribution of src_x; src_x, src_y and src_z are face-centred in x, y and z.
 *
 * @param[in] pos 3*n point coordinates (m, ERF domain coordinates)
 * @param[in] force 3*n forces ON the fluid BY the body (N, ERF frame): minus the fluid's force on the body
 * @param[in] epsilon n kernel widths, one per point (m, > 0 and finite)
 * @param[in] z_phys_nd nodal heights (m), or nullptr on a uniform-dz mesh
 * @param[in] detJ_cc cell Jacobians (1 ghost cell), or nullptr on a uniform-dz mesh
 * @param[in] geom the level's geometry
 * @param[out] src_x,src_y,src_z face-centred momentum sources (N/m^3), one component each
 */
void spread_forces (const std::vector<amrex::Real>& pos,
                    const std::vector<amrex::Real>& force,
                    const std::vector<amrex::Real>& epsilon,
                    const amrex::MultiFab* z_phys_nd,
                    const amrex::MultiFab* detJ_cc,
                    const amrex::Geometry& geom,
                    amrex::MultiFab& src_x,
                    amrex::MultiFab& src_y,
                    amrex::MultiFab& src_z);

/**
 * The same, but added to what the sources hold (spread_forces() sets them): for several sets of
 * points spread one after the other. Collective: every rank must pass the same points. (MovingBodies
 * spreads its turbines and disks through BodyExchange::spread_forces, which adds to the sources too.)
 */
void add_spread_forces (const std::vector<amrex::Real>& pos,
                        const std::vector<amrex::Real>& force,
                        const std::vector<amrex::Real>& epsilon,
                        const amrex::MultiFab* z_phys_nd,
                        const amrex::MultiFab* detJ_cc,
                        const amrex::Geometry& geom,
                        amrex::MultiFab& src_x,
                        amrex::MultiFab& src_y,
                        amrex::MultiFab& src_z);

/** The same with one kernel width epsilon (m, > 0 and finite) for every point. */
void spread_forces (const std::vector<amrex::Real>& pos,
                    const std::vector<amrex::Real>& force,
                    amrex::Real epsilon,
                    const amrex::MultiFab* z_phys_nd,
                    const amrex::MultiFab* detJ_cc,
                    const amrex::Geometry& geom,
                    amrex::MultiFab& src_x,
                    amrex::MultiFab& src_y,
                    amrex::MultiFab& src_z);

/**
 * The first of spread_forces()'s two passes, on this rank's boxes only: for each point and each
 * component, this rank's part of the normalisation S_p (the sum of w_p dV over the faces its boxes
 * own). No reduction: the caller sums the ranks' parts (BodyExchange sums those of the ranks a
 * point reaches, in rank order). The boxes are src_x's, which z_phys_nd and detJ_cc must share.
 * zb_local, when given, is this rank's mesh_z_bounds(z_phys_nd, geom, false), which the call then
 * need not compute; the same for add_scaled_kernels.
 * @param[out] norm 3*n partial normalisations (m^3), component fastest
 */
void spread_partial_norms (const std::vector<amrex::Real>& pos,
                           const std::vector<amrex::Real>& epsilon,
                           const amrex::MultiFab* z_phys_nd,
                           const amrex::MultiFab* detJ_cc,
                           const amrex::Geometry& geom,
                           const amrex::MultiFab& src_x,
                           std::vector<amrex::Real>& norm,
                           const ZBounds* zb_local = nullptr);

/**
 * The whole normalisation S_p of one kernel on a uniform mesh (no z_phys_nd, no detJ_cc), on any
 * rank, without communication: there a face's weight and volume depend only on the geometry, so
 * S_p is the sum of w_p dx dy dz over the faces of the domain within 3 epsilon of the point or one
 * of its periodic images (boundary faces excluded), which this rank can add up alone. The spherical
 * cut-off keeps the sum from factoring into three 1-D sums, so the faces are visited one by one,
 * as spread_partial_norms() visits them; only the order of the additions differs (a relative
 * difference of a few ulps). Returns false, leaving norm alone, when the boxes of ba (cell-centred,
 * the level's) do not cover every cell the kernel can reach: on a refined level the faces outside
 * them get no source, and spread_partial_norms() must be summed over the ranks instead.
 * @param[out] norm the 3 components of S_p (m^3)
 */
bool uniform_mesh_norm (amrex::Real px, amrex::Real py, amrex::Real pz, amrex::Real eps,
                        const amrex::Geometry& geom, const amrex::BoxArray& ba, amrex::Real norm[3]);

/**
 * The second pass, on this rank's boxes only: add (F_p / S_p) w_p to every face of every box (both
 * copies of a shared face alike), each point with its own width.
 * @param[in] f_over_s 3*n forces over their whole normalisations (N/m^3), component fastest; a zero skips the point's component
 */
void add_scaled_kernels (const std::vector<amrex::Real>& pos,
                         const std::vector<amrex::Real>& f_over_s,
                         const std::vector<amrex::Real>& epsilon,
                         const amrex::MultiFab* z_phys_nd,
                         const amrex::MultiFab* detJ_cc,
                         const amrex::Geometry& geom,
                         amrex::MultiFab& src_x,
                         amrex::MultiFab& src_y,
                         amrex::MultiFab& src_z,
                         const ZBounds* zb_local = nullptr);

/** What is wrong with the points, forces and widths for spread_forces (an empty string when nothing is). */
std::string spread_points_problem (const std::vector<amrex::Real>& pos, const std::vector<amrex::Real>& force,
                                   const std::vector<amrex::Real>& epsilon);

/** Abort unless the points, forces and widths match in number and are finite, and the widths positive (spread_forces' checks). */
void check_spread_points (const std::vector<amrex::Real>& pos, const std::vector<amrex::Real>& force,
                          const std::vector<amrex::Real>& epsilon);

/** Abort unless the sources are face-centred in x, y and z and share boxes and distribution with z_phys_nd and detJ_cc (one ghost cell). */
void check_spread_layout (const amrex::MultiFab* z_phys_nd, const amrex::MultiFab* detJ_cc, const amrex::MultiFab& src_x,
                          const amrex::MultiFab& src_y, const amrex::MultiFab& src_z);

/**
 * Return the integral of a face-centred source over the mesh, sum of src dV (the total force, N).
 *
 * @param dir 0, 1 or 2: src holds the x-, y- or z-face component
 */
amrex::Real integrate_source (int dir,
                              const amrex::MultiFab& src,
                              const amrex::MultiFab* detJ_cc,
                              const amrex::Geometry& geom);

} // namespace erf_actuator

#endif
