#ifndef ERF_ACTUATOR_SAMPLING_H
#define ERF_ACTUATOR_SAMPLING_H

/**
 * Interpolation of the flow and of the terrain height at arbitrary points, and the check that
 * points lie on a level's grids.
 */

#include <algorithm>
#include <cmath>
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
 * node, on the velocities' boxes and distribution. In a periodic direction a point is moved by
 * whole periods into the domain; a point on the domain's upper x or y face belongs to the last
 * cell. A point outside the domain otherwise, or one found by no rank or by more than one, aborts
 * with its coordinates: the caller owns the points and must keep them inside the domain.
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
 * The points (3 coordinates each, m) with their coordinates in every periodic direction moved by
 * whole periods into [prob_lo, prob_hi): the image the boxes hold, so that a point that has crossed
 * a periodic seam is sampled on the far side.
 */
std::vector<amrex::Real> wrap_periodic (const std::vector<amrex::Real>& pos, const amrex::Geometry& geom);

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
 * rules, periodic images and reduction as sample_velocity(): the points must lie inside the domain
 * horizontally, a point on the domain's upper x or y face belonging to the last cell; z is ignored.
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
 * outside in `first_outside`. On a terrain-following or stretched mesh a point's k is not its
 * height over dz, so z is checked as `z` says: Reach (the cells within reach of the height over dz,
 * for a flat, uniform mesh), Footprint (the reach across only, at any height: the sampler then finds
 * a point above or below the grids by itself), or Column (for the spreading, whose kernel would
 * otherwise be cut where the grids stop: the reach across over every cell the heights within reach
 * may lie in, from the mesh's bounds `zb` (mesh_z_bounds), or over the whole height without them).
 * Not with a periodic z.
 */
enum class CoverZ { Reach, Footprint, Column };

/** The lowest and highest ground and the smallest and largest spacing of the levels (m) of a mesh. */
struct ZBounds {
    amrex::Real ground_lo{0.0}, ground_hi{0.0}, dz_lo{0.0}, dz_hi{0.0};
    bool all_ground{true};   //!< every box of the level reaches the ground (else the bounds hold for those that do)

    /** Whether the level's grids reach the ground (a refined level aloft does not, and has no bounds). */
    bool has_ground () const { return ground_lo <= ground_hi && dz_lo > amrex::Real(0.0); }

    /**
     * The cells k that heights from zl to zh may lie in, whatever the column: [kb, kt], clamped to
     * [klo - 1, khi + 1] (so a range outside the domain stays outside it); [klo, khi] without bounds.
     */
    void k_range (amrex::Real zl, amrex::Real zh, int klo, int khi, int& kb, int& kt) const
    {
        kb = klo;
        kt = khi;
        if (!has_ground()) { return; }
        const double b = std::floor((static_cast<double>(zl) - ground_hi) / dz_hi);
        const double t = std::floor((static_cast<double>(zh) - ground_lo) / dz_lo);
        kb = static_cast<int>(std::min<double>(khi + 1, std::max<double>(klo - 1, b)));
        kt = static_cast<int>(std::min<double>(khi + 1, std::max<double>(klo - 1, t)));
    }
};

/**
 * The bounds of a terrain-following or stretched level's heights, from its z_phys_nd (nodal, one
 * component, on every rank alike): a height Z lies in a cell k with floor((Z - ground_hi) / dz_hi)
 * <= k <= floor((Z - ground_lo) / dz_lo).
 */
ZBounds mesh_z_bounds (const amrex::MultiFab& z_phys_nd, const amrex::Geometry& geom);

bool points_covered_by (const amrex::BoxArray& ba, const amrex::Geometry& geom,
                        const std::vector<amrex::Real>& pos, amrex::Real reach, std::string& first_outside,
                        CoverZ z = CoverZ::Reach, const ZBounds* zb = nullptr);

} // namespace erf_actuator

#endif
