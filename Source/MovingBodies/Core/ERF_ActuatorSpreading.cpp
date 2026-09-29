#include "ERF_ActuatorSpreading.H"

#include <cmath>

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>

#include "ERF_ActuatorGeometry.H"

using namespace amrex;

namespace erf_actuator {

namespace {

// Squared distance from face (i,j,k) of grid dir to the point, with the minimum image in
// the periodic directions so that a kernel wraps across a periodic boundary and both copies
// of a periodic image face see the same distance, and the kernel's cut-off
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real kernel_weight (int dir, int i, int j, int k, Real px, Real py, Real pz,
                    const GpuArray<Real,AMREX_SPACEDIM>& plo,
                    const GpuArray<Real,AMREX_SPACEDIM>& dx,
                    const GpuArray<Real,AMREX_SPACEDIM>& period,   // 0 where not periodic
                    Array4<Real const> const& znd, bool has_znd,
                    Real inv_eps2, Real reach2)
{
    Real ddx = plo[0] + (i + ((dir == 0) ? Real(0.0) : Real(0.5))) * dx[0] - px;
    Real ddy = plo[1] + (j + ((dir == 1) ? Real(0.0) : Real(0.5))) * dx[1] - py;
    const Real ddz = face_height(dir, i, j, k, znd, has_znd, plo[2], dx[2]) - pz;
    if (period[0] > Real(0.0)) { ddx -= period[0] * std::round(ddx / period[0]); }
    if (period[1] > Real(0.0)) { ddy -= period[1] * std::round(ddy / period[1]); }
    const Real r2 = ddx*ddx + ddy*ddy + ddz*ddz;
    return (r2 > reach2) ? Real(0.0) : std::exp(-r2 * inv_eps2);
}

// The faces of grid dir that a box owns: its cells' low faces, so a face shared with the next
// box (or, at a periodic end, with the first box's image) is counted once. On a non-periodic
// boundary the low domain face carries no source; the high one is never owned.
Box owned_faces (int dir, const Box& valid_cells)
{
    Box fbx = amrex::convert(valid_cells, IntVect::TheDimensionVector(dir));
    fbx.growHi(dir, -1);
    return fbx;
}

// whether face index f of grid dir lies on a non-periodic domain boundary
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool boundary_face (int dir, int i, int j, int k, const GpuArray<int,AMREX_SPACEDIM>& per,
                    const GpuArray<int,AMREX_SPACEDIM>& dlo, const GpuArray<int,AMREX_SPACEDIM>& dhi)
{
    if (per[dir] != 0) { return false; }
    const int f = (dir == 0) ? i : ((dir == 1) ? j : k);
    return (f == dlo[dir]) || (f == dhi[dir] + 1);
}

} // namespace

void
spread_forces (const std::vector<Real>& pos, const std::vector<Real>& force, Real epsilon,
               const MultiFab* z_phys_nd, const MultiFab* detJ_cc, const Geometry& geom,
               MultiFab& src_x, MultiFab& src_y, MultiFab& src_z)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(pos.size() == force.size() && pos.size() % 3 == 0,
                                     "spread_forces: one x,y,z force per x,y,z point");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(epsilon > 0.0, "spread_forces: epsilon must be positive");
    const int npts = static_cast<int>(pos.size() / 3);
    src_x.setVal(0.0);
    src_y.setVal(0.0);
    src_z.setVal(0.0);
    if (npts == 0) { return; }

    const auto plo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();
    const Real dxdydz = dx[0] * dx[1] * dx[2];
    GpuArray<Real,AMREX_SPACEDIM> period{{0.0, 0.0, 0.0}};
    GpuArray<int,AMREX_SPACEDIM> per{{0, 0, 0}};
    GpuArray<int,AMREX_SPACEDIM> dlo{{0, 0, 0}}, dhi{{0, 0, 0}};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        per[d] = geom.isPeriodic(d) ? 1 : 0;
        period[d] = geom.isPeriodic(d) ? geom.ProbLength(d) : Real(0.0);
        dlo[d] = geom.Domain().smallEnd(d);
        dhi[d] = geom.Domain().bigEnd(d);
    }
    const Real inv_eps2 = Real(1.0) / (epsilon * epsilon);
    const Real reach2 = Real(9.0) * epsilon * epsilon;
    const bool has_znd = (z_phys_nd != nullptr);
    const bool has_detj = (detJ_cc != nullptr);

    Gpu::DeviceVector<Real> d_pos(pos.size());
    Gpu::DeviceVector<Real> d_force(force.size());
    Gpu::copyAsync(Gpu::hostToDevice, pos.begin(), pos.end(), d_pos.begin());
    Gpu::copyAsync(Gpu::hostToDevice, force.begin(), force.end(), d_force.begin());
    // normalisation of every point on each of the three face grids
    Gpu::DeviceVector<Real> d_norm(3 * npts, Real(0.0));
    const Real* p_pos = d_pos.data();
    const Real* p_force = d_force.data();
    Real* p_norm = d_norm.data();

    MultiFab* src[3] = {&src_x, &src_y, &src_z};

    // pass 1: S_p = sum over the faces of w_p dV, each physical face once (the owned faces of
    // this rank's boxes, boundary faces excluded)
    for (MFIter mfi(src_x, false); mfi.isValid(); ++mfi) {
        const Box vbx = amrex::enclosedCells(mfi.validbox());
        Array4<Real const> znd = has_znd ? z_phys_nd->const_array(mfi) : Array4<Real const>{};
        Array4<Real const> detj = has_detj ? detJ_cc->const_array(mfi) : Array4<Real const>{};
        for (int dir = 0; dir < 3; ++dir) {
            const Box fbx = owned_faces(dir, vbx);
            ParallelFor(fbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if (boundary_face(dir, i, j, k, per, dlo, dhi)) { return; }
                const Real dV = face_volume(dir, i, j, k, detj, has_detj, dxdydz);
                for (int p = 0; p < npts; ++p) {
                    const Real w = kernel_weight(dir, i, j, k, p_pos[3*p], p_pos[3*p+1], p_pos[3*p+2],
                                                 plo, dx, period, znd, has_znd, inv_eps2, reach2);
                    if (w > Real(0.0)) { Gpu::Atomic::AddNoRet(&p_norm[3*p+dir], w * dV); }
                }
            });
        }
    }
    Gpu::streamSynchronize();
    std::vector<Real> norm(3 * npts, 0.0);
    Gpu::copy(Gpu::deviceToHost, d_norm.begin(), d_norm.end(), norm.begin());
    ParallelDescriptor::ReduceRealSum(norm.data(), 3 * npts);
    for (int p = 0; p < npts; ++p) {
        for (int dir = 0; dir < 3; ++dir) {
            if (!(norm[3*p+dir] > Real(0.0))) {
                Abort("spread_forces: point (" + std::to_string(pos[3*p]) + ", " + std::to_string(pos[3*p+1]) +
                      ", " + std::to_string(pos[3*p+2]) + ") reaches no face of the mesh within 3 epsilon = " +
                      std::to_string(3.0 * epsilon) + " m; actuator points must lie inside the domain");
            }
        }
    }
    Gpu::copyAsync(Gpu::hostToDevice, norm.begin(), norm.end(), d_norm.begin());

    // pass 2: src(face) = sum over the points of F_p w_p / S_p, on every face of every box so
    // that both copies of a shared or periodic image face carry the same value
    for (MFIter mfi(src_x, false); mfi.isValid(); ++mfi) {
        Array4<Real const> znd = has_znd ? z_phys_nd->const_array(mfi) : Array4<Real const>{};
        for (int dir = 0; dir < 3; ++dir) {
            const Box fbx = mfi.validbox().convert(IntVect::TheDimensionVector(dir));
            Array4<Real> const& s = src[dir]->array(mfi);
            ParallelFor(fbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real acc = Real(0.0);
                if (!boundary_face(dir, i, j, k, per, dlo, dhi)) {
                    for (int p = 0; p < npts; ++p) {
                        const Real w = kernel_weight(dir, i, j, k, p_pos[3*p], p_pos[3*p+1], p_pos[3*p+2],
                                                     plo, dx, period, znd, has_znd, inv_eps2, reach2);
                        if (w > Real(0.0)) { acc += p_force[3*p+dir] * w / p_norm[3*p+dir]; }
                    }
                }
                s(i,j,k) = acc;
            });
        }
    }
    Gpu::streamSynchronize();
}

Real
integrate_source (int dir, const MultiFab& src, const MultiFab* detJ_cc, const Geometry& geom)
{
    const auto dx = geom.CellSizeArray();
    const Real dxdydz = dx[0] * dx[1] * dx[2];
    const bool has_detj = (detJ_cc != nullptr);
    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;
    for (MFIter mfi(src, false); mfi.isValid(); ++mfi) {
        // each physical face once: a box's cells' low faces
        const Box fbx = owned_faces(dir, amrex::enclosedCells(mfi.validbox()));
        Array4<Real const> const& s = src.const_array(mfi);
        Array4<Real const> detj = has_detj ? detJ_cc->const_array(mfi) : Array4<Real const>{};
        reduce_op.eval(fbx, reduce_data, [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            return { s(i,j,k) * face_volume(dir, i, j, k, detj, has_detj, dxdydz) };
        });
    }
    Real total = amrex::get<0>(reduce_data.value(reduce_op));
    ParallelDescriptor::ReduceRealSum(total);
    return total;
}

} // namespace erf_actuator
