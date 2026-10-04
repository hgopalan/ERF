#ifndef ERF_ACTUATOR_SAMPLING_H
#define ERF_ACTUATOR_SAMPLING_H

/**
 * Interpolation of the flow and of the terrain height at arbitrary points, and the check that
 * points lie on a level's grids.
 */

#include <string>
#include <vector>

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

namespace erf_actuator {

/**
 * Sample the velocity at a set of points from the face-centred velocity fields.
 *
 * Each component is interpolated trilinearly from its own staggered grid: u from the x faces,
 * v from the y faces, w from the z faces, so a field linear in x, y and z is reproduced
 * exactly. The vertical weight uses the physical face heights (z_phys_nd on a stretched or
 * terrain-following mesh, prob_lo + k dz otherwise), so the same holds on those meshes as long
 * as the field is linear in the physical height.
 *
 * Every rank interpolates the points whose containing cell lies in one of its valid boxes,
 * reading the neighbours from the ghost cells, and the results are summed across ranks; the
 * velocity fields must therefore have at least one filled ghost cell, and z_phys_nd one ghost
 * node, on the velocities' boxes and distribution. A point outside the domain, or one found by no
 * rank or by more than one, aborts with its coordinates: the caller owns the points and must keep
 * them inside the domain.
 *
 * @param[in] U,V,W face-centred velocities on one level, ghost cells filled (m/s)
 * @param[in] z_phys_nd nodal physical heights (m), or nullptr on a uniform-dz mesh
 * @param[in] geom the level's geometry
 * @param[in] pos 3*n point coordinates (x0,y0,z0, x1,y1,z1, ...) (m, ERF domain coordinates)
 * @param[out] vel 3*n sampled velocities (m/s), the same on every rank on return
 */
void sample_velocity (const amrex::MultiFab& U,
                      const amrex::MultiFab& V,
                      const amrex::MultiFab& W,
                      const amrex::MultiFab* z_phys_nd,
                      const amrex::Geometry& geom,
                      const std::vector<amrex::Real>& pos,
                      std::vector<amrex::Real>& vel);

/**
 * Sample one component of a cell-centred field at a set of points: bilinear between the four
 * cell centres around the point in the horizontal and linear in the physical height between
 * the two cell centres that bracket it in the column (cell-centre heights from z_phys_nd on a
 * stretched or terrain-following mesh); below the first and above the last cell centre the
 * nearest pair extrapolates linearly. Same ownership rules and reduction as sample_velocity();
 * the field needs one filled ghost cell.
 *
 * @param[in] mf the cell-centred field
 * @param[in] comp the component of mf (0-based)
 * @param[in] pos 3*n point coordinates (m, ERF domain coordinates)
 * @param[out] val n sampled values, the same on every rank on return
 */
void sample_cell_scalar (const amrex::MultiFab& mf, int comp,
                         const amrex::MultiFab* z_phys_nd,
                         const amrex::Geometry& geom,
                         const std::vector<amrex::Real>& pos,
                         std::vector<amrex::Real>& val);

/**
 * Return the height of the terrain surface (the k = 0 node plane of z_phys_nd, bilinear between
 * the four mesh nodes around each point's (x, y)) under a set of points; prob_lo z everywhere on
 * a uniform-dz mesh (z_phys_nd null), where no point is checked. Otherwise the same ownership
 * rules and reduction as sample_velocity(): the points must lie inside the domain horizontally,
 * a point on the domain's upper x or y face belonging to the last cell; z is ignored.
 *
 * @param[in] pos 3*n positions (m)
 * @param[out] h n terrain heights (m), the same on every rank on return
 */
void terrain_heights (const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom,
                      const std::vector<amrex::Real>& pos, std::vector<amrex::Real>& h);

/**
 * Return whether every point, with the cells within `reach` metres around it, lies inside the
 * union of boxes of a level (the anchor level's grids): the sampler needs the neighbouring cells
 * and the spreader the kernel's reach, and on a refined level that is not the whole domain. In a
 * periodic direction the needed cells wrap across the seam to the domain's other side; in a
 * non-periodic one they are clamped to the domain. Returns false and names the first point
 * outside in `first_outside`.
 */
bool points_covered_by (const amrex::BoxArray& ba, const amrex::Geometry& geom,
                        const std::vector<amrex::Real>& pos, amrex::Real reach, std::string& first_outside);

} // namespace erf_actuator

#endif
