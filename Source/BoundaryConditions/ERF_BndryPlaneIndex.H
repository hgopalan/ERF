#ifndef ERF_BNDRY_PLANE_INDEX_H
#define ERF_BNDRY_PLANE_INDEX_H

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>

/**
 * The index along a boundary plane that fills the ghost cell at index i: inside the domain, i
 * itself; outside it, the periodic image when the direction is periodic, so that a ghost cell
 * beside the inflow face holds what the plane holds at the periodic image (as the interior cells
 * across the seam do); else the nearest cell of the plane. A plane holds the domain's cells only
 * along its face, so its ghost cells in the other directions come from here.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int bndry_plane_index (int i, int lo, int hi, bool periodic) noexcept
{
    if (i >= lo && i <= hi) { return i; }
    if (periodic) {
        const int n = hi - lo + 1;
        const int r = (i - lo) % n;
        return lo + ((r < 0) ? r + n : r);
    }
    return (i < lo) ? lo : hi;
}

#endif
