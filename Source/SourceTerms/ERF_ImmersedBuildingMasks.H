#ifndef ERF_IMMERSED_BUILDING_MASKS_H_
#define ERF_IMMERSED_BUILDING_MASKS_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>

/**
 * Which forcing the buildings immersed forcing applies on a momentum face, shared by the source
 * kernels (ImmersedForcingBuildings_{X,Y,Z}mom) and the implicit projection
 * (ERF::make_if_projection_sigma), so the two agree on which faces carry the wall law and which
 * the linear drag toward zero. The arguments are the face blanking of the kernels (thresholded,
 * snapped with erf.if_snap_partial_cells) of the face and of its neighbours.
 */
namespace if_bld {

//! Masks of a horizontal-momentum face (x or y). lo/hi are the neighbouring faces across the
//! other horizontal direction: south/north for an x-face, west/east for a y-face.
struct HorizMasks {
    amrex::Real roof, lo, hi, wall, side, interior;
    //! wall-law (MOST) faces
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real most () const noexcept { return roof + lo + hi; }
    //! weight of the linear drag CdM |u| on this face
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real linear (amrex::Real t) const noexcept { return (wall + side) * t + interior; }
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HorizMasks horiz_masks (amrex::Real t, amrex::Real below, amrex::Real above,
                        amrex::Real lo, amrex::Real hi,
                        bool normal_face, bool use_most, bool snap) noexcept
{
    const amrex::Real zero = amrex::Real(0.0), one = amrex::Real(1.0);
    HorizMasks m{};
    // with the snap a roof face lies in the top solid row and a wall-law face is not also an
    // interior face
    m.roof = (!normal_face && t > zero && (snap ? t <= below : t < below) && above == zero && use_most) ? one : zero;
    m.lo   = (!normal_face && t > zero && t <= hi && lo == zero && use_most) ? one : zero;
    m.hi   = (!normal_face && t > zero && t <= lo && hi == zero && use_most) ? one : zero;
    m.wall = (t > zero && t < one && !use_most) ? one : zero;                       // all walls without MOST
    m.side = (t > zero && t < one && use_most && m.most() == zero) ? one : zero;    // partial faces not under MOST
    m.interior = (t == one && !(snap && m.most() > zero)) ? one : zero;
    return m;
}

//! Masks of a z-face. The roof face takes the linear drag (weighted by its blanking), not a wall law.
struct VertMasks {
    amrex::Real south, north, west, east, wall, roof, interior;
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real most () const noexcept { return south + north + west + east; }
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real linear (amrex::Real t) const noexcept { return (wall + roof) * t + interior; }
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
VertMasks vert_masks (amrex::Real t, amrex::Real above,
                      amrex::Real south, amrex::Real north, amrex::Real west, amrex::Real east,
                      bool above_floor, bool normal_face, bool use_most, bool snap) noexcept
{
    const amrex::Real zero = amrex::Real(0.0), one = amrex::Real(1.0);
    VertMasks m{};
    m.south = (!normal_face && t > zero && t <= north && south == zero && use_most && above_floor) ? one : zero;
    m.north = (!normal_face && t > zero && t <= south && north == zero && use_most && above_floor) ? one : zero;
    m.west  = (!normal_face && t > zero && t <= east  && west  == zero && use_most && above_floor) ? one : zero;
    m.east  = (!normal_face && t > zero && t <= west  && east  == zero && use_most && above_floor) ? one : zero;
    m.wall  = (t > zero && t < one && !use_most) ? one : zero;
    m.roof  = (!normal_face && t > zero && above == zero && use_most) ? one : zero;
    m.interior = (t == one && !(snap && (m.most() + m.roof) > zero)) ? one : zero;
    return m;
}

} // namespace if_bld

#endif
