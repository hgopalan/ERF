#include "ERF_ActuatorSampling.H"

#include <cmath>
#include <string>

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>

using namespace amrex;

namespace erf_actuator {

namespace {

// Height of the centre of face (i,j,k) of the staggered grid dir (0: x face, 1: y face,
// 2: z face): the mean of the four nodes of that face on a terrain-following mesh, or the
// nominal height on a uniform-dz mesh.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real face_height (int dir, int i, int j, int k,
                  Array4<Real const> const& znd, bool has_znd, Real zlo, Real dz)
{
    if (!has_znd) {
        return (dir == 2) ? zlo + k * dz : zlo + (k + Real(0.5)) * dz;
    }
    if (dir == 0) {
        return Real(0.25) * (znd(i,j,k) + znd(i,j+1,k) + znd(i,j,k+1) + znd(i,j+1,k+1));
    }
    if (dir == 1) {
        return Real(0.25) * (znd(i,j,k) + znd(i+1,j,k) + znd(i,j,k+1) + znd(i+1,j,k+1));
    }
    return Real(0.25) * (znd(i,j,k) + znd(i+1,j,k) + znd(i,j+1,k) + znd(i+1,j+1,k));
}

// Value of the face field f at height z in the column (i,j) of its staggered grid, linear in
// the physical height between the two faces that bracket z. The faces klo..khi are searched;
// outside them the nearest pair extrapolates, so a field linear in height stays exact down to
// the ground and up to the top without reading boundary ghost cells.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real column_value (Array4<Real const> const& f, int dir, int i, int j, int klo, int khi, Real z,
                   Array4<Real const> const& znd, bool has_znd, Real zlo, Real dz)
{
    int k0 = klo;
    for (int k = klo; k < khi; ++k) {
        if (z >= face_height(dir, i, j, k, znd, has_znd, zlo, dz)) { k0 = k; } else { break; }
    }
    const Real h0 = face_height(dir, i, j, k0,   znd, has_znd, zlo, dz);
    const Real h1 = face_height(dir, i, j, k0+1, znd, has_znd, zlo, dz);
    const Real w  = (z - h0) / (h1 - h0);
    return (Real(1.0) - w) * f(i,j,k0) + w * f(i,j,k0+1);
}

// One velocity component at (x,y,z): bilinear in the two horizontal directions between the
// four staggered columns around the point, each column evaluated at the physical height z.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real sample_component (int dir, Array4<Real const> const& f, Real x, Real y, Real z,
                       const GpuArray<Real,AMREX_SPACEDIM>& plo,
                       const GpuArray<Real,AMREX_SPACEDIM>& dxi,
                       int klo, int khi,
                       Array4<Real const> const& znd, bool has_znd, Real dz)
{
    // fractional index on the component's own grid: whole numbers on faces, half on centres
    const Real xi = (x - plo[0]) * dxi[0] - ((dir == 0) ? Real(0.0) : Real(0.5));
    const Real yi = (y - plo[1]) * dxi[1] - ((dir == 1) ? Real(0.0) : Real(0.5));
    const int i0 = static_cast<int>(std::floor(xi));
    const int j0 = static_cast<int>(std::floor(yi));
    const Real wx = xi - i0;
    const Real wy = yi - j0;
    const Real f00 = column_value(f, dir, i0,   j0,   klo, khi, z, znd, has_znd, plo[2], dz);
    const Real f10 = column_value(f, dir, i0+1, j0,   klo, khi, z, znd, has_znd, plo[2], dz);
    const Real f01 = column_value(f, dir, i0,   j0+1, klo, khi, z, znd, has_znd, plo[2], dz);
    const Real f11 = column_value(f, dir, i0+1, j0+1, klo, khi, z, znd, has_znd, plo[2], dz);
    return (Real(1.0) - wy) * ((Real(1.0) - wx) * f00 + wx * f10)
         +               wy  * ((Real(1.0) - wx) * f01 + wx * f11);
}

} // namespace

void
sample_velocity (const MultiFab& U, const MultiFab& V, const MultiFab& W,
                 const MultiFab* z_phys_nd, const Geometry& geom,
                 const std::vector<Real>& pos, std::vector<Real>& vel)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(pos.size() % 3 == 0, "sample_velocity: pos holds x,y,z triples");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(U.nGrow() >= 1 && V.nGrow() >= 1 && W.nGrow() >= 1,
                                     "sample_velocity needs one filled ghost cell on the velocities");
    const int npts = static_cast<int>(pos.size() / 3);
    vel.assign(3 * static_cast<std::size_t>(npts), Real(0.0));
    if (npts == 0) { return; }

    const Box& domain = geom.Domain();
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();
    const Real dz = geom.CellSize(2);
    const int klo = domain.smallEnd(2);
    const int khi = domain.bigEnd(2);
    const bool has_znd = (z_phys_nd != nullptr);

    Gpu::DeviceVector<Real> d_pos(pos.size());
    Gpu::DeviceVector<Real> d_vel(pos.size(), Real(0.0));
    Gpu::DeviceVector<int>  d_cnt(npts, 0);
    Gpu::copyAsync(Gpu::hostToDevice, pos.begin(), pos.end(), d_pos.begin());
    Real* p_pos = d_pos.data();
    Real* p_vel = d_vel.data();
    int*  p_cnt = d_cnt.data();

    for (MFIter mfi(U, false); mfi.isValid(); ++mfi) {
        // the cells this rank owns; a point is sampled by the box holding its containing cell
        const Box vbx = amrex::enclosedCells(mfi.validbox());
        Array4<Real const> const& u = U.const_array(mfi);
        Array4<Real const> const& v = V.const_array(mfi);
        Array4<Real const> const& w = W.const_array(mfi);
        Array4<Real const> znd = has_znd ? z_phys_nd->const_array(mfi) : Array4<Real const>{};

        ParallelFor(npts, [=] AMREX_GPU_DEVICE (int p) noexcept
        {
            const Real x = p_pos[3*p], y = p_pos[3*p+1], z = p_pos[3*p+2];
            const int ic = static_cast<int>(std::floor((x - plo[0]) * dxi[0]));
            const int jc = static_cast<int>(std::floor((y - plo[1]) * dxi[1]));
            if (ic < vbx.smallEnd(0) || ic > vbx.bigEnd(0) ||
                jc < vbx.smallEnd(1) || jc > vbx.bigEnd(1)) { return; }
            // the containing cell in z: between the z faces of column (ic,jc)
            int kc = -1;
            if (!has_znd) {
                const int k = static_cast<int>(std::floor((z - plo[2]) * dxi[2]));
                if (k >= klo && k <= khi) { kc = k; }
            } else {
                for (int k = klo; k <= khi; ++k) {
                    const Real zb = face_height(2, ic, jc, k,   znd, true, plo[2], dz);
                    const Real zt = face_height(2, ic, jc, k+1, znd, true, plo[2], dz);
                    if (z >= zb && z < zt) { kc = k; break; }
                }
            }
            if (kc < vbx.smallEnd(2) || kc > vbx.bigEnd(2)) { return; }

            p_vel[3*p]   = sample_component(0, u, x, y, z, plo, dxi, klo, khi,   znd, has_znd, dz);
            p_vel[3*p+1] = sample_component(1, v, x, y, z, plo, dxi, klo, khi,   znd, has_znd, dz);
            p_vel[3*p+2] = sample_component(2, w, x, y, z, plo, dxi, klo, khi+1, znd, has_znd, dz);
            p_cnt[p] = 1;
        });
    }
    Gpu::streamSynchronize();

    std::vector<int> cnt(npts, 0);
    Gpu::copy(Gpu::deviceToHost, d_vel.begin(), d_vel.end(), vel.begin());
    Gpu::copy(Gpu::deviceToHost, d_cnt.begin(), d_cnt.end(), cnt.begin());
    ParallelDescriptor::ReduceRealSum(vel.data(), static_cast<int>(vel.size()));
    ParallelDescriptor::ReduceIntSum(cnt.data(), npts);

    for (int p = 0; p < npts; ++p) {
        if (cnt[p] != 1) {
            Abort("sample_velocity: point (" + std::to_string(pos[3*p]) + ", " + std::to_string(pos[3*p+1]) +
                  ", " + std::to_string(pos[3*p+2]) + ") was sampled by " + std::to_string(cnt[p]) +
                  " boxes; actuator points must lie inside the domain");
        }
    }
}

} // namespace erf_actuator
