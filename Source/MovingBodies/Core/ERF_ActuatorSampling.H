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

#include "ERF_ActuatorDeviceBuffers.H"

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
 * with its coordinates: the caller owns the points and must keep them inside the domain. Every
 * rank must pass the same points (BodyExchange samples the points of bodies each rank owns alone).
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
 * The part of sample_velocity() that a rank does on its own: the velocity at each of the points that
 * one of this rank's boxes holds (cnt 1), zero with cnt 0 at the others. No reduction and no check:
 * the caller combines the ranks' answers. The points must already be moved into the domain across
 * its periodic seams (wrap_periodic).
 * @param[in] wpos 3*n point coordinates inside the domain (m)
 * @param[out] vel 3*n velocities (m/s), zero at the points no box of this rank holds
 * @param[out] cnt n counts: how many of this rank's boxes hold each point (0 or 1)
 * @param buffers device memory kept from call to call (the call uses its own when nullptr); the points
 *        go to the device and the answers come back with asynchronous copies and one synchronisation
 */
void sample_velocity_local (const amrex::MultiFab& U,
                            const amrex::MultiFab& V,
                            const amrex::MultiFab& W,
                            const amrex::MultiFab* z_phys_nd,
                            const amrex::Geometry& geom,
                            const std::vector<amrex::Real>& wpos,
                            std::vector<amrex::Real>& vel,
                            std::vector<int>& cnt,
                            ActuatorDeviceBuffers* buffers = nullptr);

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

/**
 * The lowest and highest ground and the smallest and largest spacing of the levels (m) of a mesh,
 * and, over one rank's boxes (mesh_z_bounds with over_all_ranks false), a table of each node level's
 * lowest and highest height and of the cells those boxes hold, which bounds the cells a height may
 * lie in whether or not the boxes reach the ground (a column split in z, a level aloft).
 */
struct ZBounds {
    amrex::Real ground_lo{0.0}, ground_hi{0.0}, dz_lo{0.0}, dz_hi{0.0};
    bool all_ground{true};   //!< every box of the level reaches the ground (else the bounds hold for those that do)
    int k0{0};                                  //!< the node level of node_lo[0] and the cell of has_cell[0]
    std::vector<amrex::Real> node_lo, node_hi;  //!< node level k0 + n: the lowest and highest height (empty: no table)
    std::vector<char> has_cell;                 //!< cell k0 + n is in one of the boxes

    /** Whether the level's grids reach the ground (a refined level aloft does not, and has no bounds). */
    bool has_ground () const { return ground_lo <= ground_hi && dz_lo > amrex::Real(0.0); }
    /** Whether the bounds come with the table of node heights (the bounds of one rank's boxes). */
    bool has_table () const { return !has_cell.empty(); }

    /**
     * The cells k that heights from zl to zh may lie in, whatever the column: [kb, kt], clamped to
     * [klo - 1, khi + 1] (so a range outside the domain stays outside it); [klo, khi] without bounds.
     * With the table: exactly the boxes' cells k whose lowest bottom node is not above zh and whose
     * highest top node is not below zl, or kb > kt when there is none.
     */
    void k_range (amrex::Real zl, amrex::Real zh, int klo, int khi, int& kb, int& kt) const
    {
        kb = klo;
        kt = khi;
        if (has_table()) {
            kb = khi + 1;
            kt = klo - 1;
            // a relative 1e-5 more each way: the caller's zl and zh are rounded sums (a point's height and
            // its reach), and a face at a node level's own height h just past them still has weight
            const amrex::Real slack = amrex::Real(1.0e-5) * std::max(std::abs(zl), std::max(std::abs(zh), zh - zl));
            const amrex::Real lo = zl - slack, hi = zh + slack;
            for (std::size_t n = 0; n < has_cell.size(); ++n) {
                if (has_cell[n] == 0 || node_lo[n] > hi || node_hi[n+1] < lo) { continue; }
                const int k = k0 + static_cast<int>(n);
                kb = std::min(kb, k);
                kt = std::max(kt, k);
            }
            return;
        }
        if (!has_ground()) { return; }
        const double b = std::floor((static_cast<double>(zl) - ground_hi) / dz_hi);
        const double t = std::floor((static_cast<double>(zh) - ground_lo) / dz_lo);
        kb = static_cast<int>(std::min<double>(khi + 1, std::max<double>(klo - 1, b)));
        kt = static_cast<int>(std::min<double>(khi + 1, std::max<double>(klo - 1, t)));
    }
};

/**
 * The bounds of a terrain-following or stretched level's heights, from its z_phys_nd (nodal, one
 * component): a height Z lies in a cell k with floor((Z - ground_hi) / dz_hi) <= k <= floor((Z -
 * ground_lo) / dz_lo). Over every rank's boxes, the same on every rank (a collective), or with
 * over_all_ranks false over this rank's boxes only, which bounds the cells of those boxes alone.
 * With over_all_ranks false and a box of this rank that does not reach the ground (a column split
 * in z, a level aloft), where that formula does not hold, also the table of the boxes' node heights
 * (ZBounds::has_table). The table is this rank's: points_covered_by takes the global bounds only.
 */
ZBounds mesh_z_bounds (const amrex::MultiFab& z_phys_nd, const amrex::Geometry& geom, bool over_all_ranks = true);

/**
 * The smallest physical cell height (the spacing of z_phys_nd's nodes along a column) in each of a
 * set of regions: region r is the box regions[6r .. 6r+2] (x, y, z low) to regions[6r+3 .. 6r+5]
 * (high), in metres. Every node column whose cells touch the region across (with a periodic x or y
 * wrapping across the seam) counts, with the spacings whose span overlaps the region's heights. On a
 * flat uniform mesh (ERF fills z_phys_nd with k dz there) that is dz to rounding. A region with a low
 * end above its high end, a non-finite bound, or no positive spacing overlapping it gets the nominal dz
 * (a folded cell's non-positive spacing is skipped), as does every region when z_phys_nd is null
 * (outside ERF, which always allocates it). A collective: the same on every rank.
 *
 * @param[in] regions 6*n bounds (m)
 * @param[out] dz n heights (m)
 */
void min_cell_heights (const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom,
                       const std::vector<amrex::Real>& regions, std::vector<amrex::Real>& dz);

/**
 * The cell height that limits a rotating blade tip, per region: the smallest dz / q_s over the cells of
 * the region (as in min_cell_heights), with q_s the largest fraction of the tip speed that can cross the
 * cell's faces. The tip's velocity is omega n x r with r in the rotor plane, so its vertical part at a
 * height Z of the disk is at most omega sqrt(R^2 - r_z^2), and r_z differs from Z - hub_z by at most the
 * blades' off-plane reach times |n_z| (the slack): q = sqrt(R^2 - d^2) / R, with d the distance from
 * the hub height to the cell's nearest point less the slack. A face of slope G (the steepest of the
 * cell's two faces, to the neighbouring nodes) is also crossed by the tip moving sideways, so
 * q_s = min(sqrt(1 + G^2), q + G): the full speed normal to the face at most. On a flat mesh a cell at
 * hub height counts its full height; on a stretched mesh the short cells low in the disk, which the tip
 * crosses moving mostly sideways, count less; on terrain the slope makes every cell count somewhat
 * more. Reads z_phys_nd's ghost nodes next to the region inside the domain and across a periodic
 * seam (FillBoundary fills them), never those outside a non-periodic side.
 * Region r is rows[9r .. 9r+8]: the box low (x, y, z), the box high, the hub height, the tip radius R
 * and the slack, in metres. The nominal dz in the same cases as min_cell_heights, and for R <= 0.
 * A collective: the same on every rank.
 *
 * @param[in] rows 9*n values (m)
 * @param[out] dz n heights (m)
 */
void tip_cell_heights (const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom,
                       const std::vector<amrex::Real>& rows, std::vector<amrex::Real>& dz);

/**
 * tip_cell_heights for every region at once: the same heights, with one device launch over the
 * columns every region covers in this rank's boxes (each column's minimum folded into its region's
 * with an atomic minimum) and one reduction over the ranks, rather than a reduction, and a device
 * synchronisation, per region. A collective: the same on every rank.
 *
 * @param[in] rows 9*n values (m), as for tip_cell_heights
 * @param[out] dz n heights (m)
 */
void tip_cell_heights_batched (const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom,
                               const std::vector<amrex::Real>& rows, std::vector<amrex::Real>& dz);

bool points_covered_by (const amrex::BoxArray& ba, const amrex::Geometry& geom,
                        const std::vector<amrex::Real>& pos, amrex::Real reach, std::string& first_outside,
                        CoverZ z = CoverZ::Reach, const ZBounds* zb = nullptr);

/**
 * A level's boxes as points_covered_by tests them, built once for every test of a check: the boxes
 * cell-centred, and flattened onto the domain's bottom layer (for CoverZ::Footprint).
 */
struct CoverageGrids
{
    CoverageGrids (const amrex::BoxArray& ba, const amrex::Geometry& geom);
    amrex::BoxArray cells;        //!< the boxes, cell-centred
    amrex::BoxArray footprints;   //!< each box flattened onto the bottom layer (they may overlap)
};

/** points_covered_by on boxes built once (CoverageGrids of the same level and geometry). */
bool points_covered_by (const CoverageGrids& grids, const amrex::Geometry& geom,
                        const std::vector<amrex::Real>& pos, amrex::Real reach, std::string& first_outside,
                        CoverZ z = CoverZ::Reach, const ZBounds* zb = nullptr);

} // namespace erf_actuator

#endif
