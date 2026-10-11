// Gaussian force spreading with discrete normalisation, and the source integral.

#include "ERF_ActuatorSpreading.H"

#include <array>
#include <cmath>
#include <string>

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>

#include "ERF_ActuatorGeometry.H"
#include "ERF_ActuatorSampling.H"

using namespace amrex;

namespace erf_actuator {

namespace {

// The kernel weight exp(-r^2/eps^2) of face (i,j,k) of grid dir for the point (px,py,pz) (one
// periodic image), 0 beyond the cut-off radius sqrt(reach2) = 3 eps
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real kernel_weight (int dir, int i, int j, int k, Real px, Real py, Real pz,
                    const GpuArray<Real,AMREX_SPACEDIM>& plo,
                    const GpuArray<Real,AMREX_SPACEDIM>& dx,
                    Array4<Real const> const& znd, bool has_znd,
                    Real inv_eps2, Real reach2)
{
    const Real ddx = plo[0] + (i + ((dir == 0) ? Real(0.0) : Real(0.5))) * dx[0] - px;
    const Real ddy = plo[1] + (j + ((dir == 1) ? Real(0.0) : Real(0.5))) * dx[1] - py;
    const Real ddz = face_height(dir, i, j, k, znd, has_znd, plo[2], dx[2]) - pz;
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
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(src_x.ixType() == IndexType(IntVect(1,0,0)) && src_y.ixType() == IndexType(IntVect(0,1,0)) &&
                                     src_z.ixType() == IndexType(IntVect(0,0,1)),
                                     "spread_forces: src_x, src_y and src_z must be face-centred in x, y and z");
    for (const MultiFab* mf : {static_cast<const MultiFab*>(&src_y), static_cast<const MultiFab*>(&src_z), z_phys_nd, detJ_cc}) {
        if (mf == nullptr) { continue; }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mf->DistributionMap() == src_x.DistributionMap() && mf->boxArray().CellEqual(src_x.boxArray()),
                                         "spread_forces: the sources, z_phys_nd and detJ_cc must share boxes and distribution");
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(detJ_cc == nullptr || detJ_cc->nGrow() >= 1,
                                     "spread_forces: detJ_cc needs one ghost cell (the face volumes read the cell below each low face)");
    for (std::size_t q = 0; q < pos.size(); ++q) {
        if (!std::isfinite(pos[q]) || !std::isfinite(force[q])) {
            const std::size_t p = q / 3;
            Abort("spread_forces: point " + std::to_string(p) + " (0-based) has a non-finite position or force: (" +
                  std::to_string(pos[3*p]) + ", " + std::to_string(pos[3*p+1]) + ", " + std::to_string(pos[3*p+2]) + ") m, (" +
                  std::to_string(force[3*p]) + ", " + std::to_string(force[3*p+1]) + ", " + std::to_string(force[3*p+2]) + ") N");
        }
    }
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

    // Work per point over the faces within its reach, so the cost is npts x (6 eps / dx)^3
    // (npts x (6 eps / dx)^2 x nz with z_phys_nd) rather than nfaces x npts. A point near a periodic boundary is visited again as its
    // periodic image(s), which is how the kernel wraps.
    const int reach_cells[3] = {static_cast<int>(std::ceil(Real(3.0) * epsilon / dx[0])) + 1,
                                static_cast<int>(std::ceil(Real(3.0) * epsilon / dx[1])) + 1,
                                static_cast<int>(std::ceil(Real(3.0) * epsilon / dx[2])) + 1};
    // the index box of the faces of grid dir within reach of a point at (px,py,pz); with z_phys_nd
    // (terrain-following or stretched) a face's height is not its index times dz, so the box takes
    // every k a height within reach may lie in, from the mesh's bounds (all of them for a level
    // aloft), and kernel_weight's cut-off at 3 eps, measured in physical heights, picks the faces
    // (bounds from the boxes that reach the ground do not hold for a box aloft over higher ground)
    ZBounds zb = has_znd ? mesh_z_bounds(*z_phys_nd, geom) : ZBounds{};
    if (!zb.all_ground) { zb = ZBounds{}; }
    auto reach_box = [&](int dir, Real px, Real py, Real pz) {
        const Real pc[3] = {px, py, pz};
        IntVect lo, hi;
        for (int d = 0; d < 3; ++d) {
            const Real fi = (pc[d] - plo[d]) / dx[d] - ((d == dir) ? Real(0.0) : Real(0.5));
            lo[d] = static_cast<int>(std::floor(fi)) - reach_cells[d];
            hi[d] = static_cast<int>(std::floor(fi)) + reach_cells[d] + 1;
        }
        if (has_znd) {
            int kb = dlo[2], kt = dhi[2];
            zb.k_range(pz - Real(3.0) * epsilon, pz + Real(3.0) * epsilon, dlo[2], dhi[2], kb, kt);
            // the faces bounding those cells, one more each way for the faces of a cell's neighbours
            lo[2] = std::max(kb - 1, dlo[2]);
            hi[2] = std::min(kt + 2, dhi[2] + 1);
        }
        return Box(lo, hi, IntVect::TheDimensionVector(dir));
    };
    // the periodic images of a point that can reach the domain: shifts of -n L .. +n L per
    // periodic direction, with n the number of domain widths the kernel reaches (one unless
    // the domain is narrower than 3 epsilon, as in a unit test)
    int n_img[2] = {0, 0};
    for (int d = 0; d < 2; ++d) {
        if (per[d] != 0) { n_img[d] = static_cast<int>(std::ceil(Real(3.0) * epsilon / period[d])); }
    }
    std::vector<std::array<Real,3>> images;
    auto point_images = [&](Real px, Real py, Real pz) {
        images.clear();
        for (int sx = -n_img[0]; sx <= n_img[0]; ++sx) {
            for (int sy = -n_img[1]; sy <= n_img[1]; ++sy) {
                images.push_back({{px + sx * period[0], py + sy * period[1], pz}});
            }
        }
    };

    std::vector<Real> norm(3 * npts, 0.0);
    Gpu::DeviceVector<Real> d_norm(3 * npts, Real(0.0));
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
            for (int p = 0; p < npts; ++p) {
                point_images(pos[3*p], pos[3*p+1], pos[3*p+2]);
                for (const auto& q : images) {
                    const Box rb = reach_box(dir, q[0], q[1], q[2]) & fbx;
                    if (rb.isEmpty()) { continue; }
                    const Real qx = q[0], qy = q[1], qz = q[2];
                    Real* np = &p_norm[3*p+dir];
                    ParallelFor(rb, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        if (boundary_face(dir, i, j, k, per, dlo, dhi)) { return; }
                        const Real w = kernel_weight(dir, i, j, k, qx, qy, qz, plo, dx, znd, has_znd, inv_eps2, reach2);
                        if (w > Real(0.0)) {
                            Gpu::Atomic::AddNoRet(np, w * face_volume(dir, i, j, k, detj, has_detj, dxdydz));
                        }
                    });
                }
            }
        }
    }
    Gpu::streamSynchronize();
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

    // pass 2: src(face) += F_p w_p / S_p on every face of every box, so that both copies of a
    // shared or periodic image face carry the same value
    for (MFIter mfi(src_x, false); mfi.isValid(); ++mfi) {
        Array4<Real const> znd = has_znd ? z_phys_nd->const_array(mfi) : Array4<Real const>{};
        for (int dir = 0; dir < 3; ++dir) {
            const Box fbx = mfi.validbox().convert(IntVect::TheDimensionVector(dir));
            Array4<Real> const& s = src[dir]->array(mfi);
            for (int p = 0; p < npts; ++p) {
                const Real f_over_s = force[3*p+dir] / norm[3*p+dir];
                if (f_over_s == Real(0.0)) { continue; }
                point_images(pos[3*p], pos[3*p+1], pos[3*p+2]);
                for (const auto& q : images) {
                    const Box rb = reach_box(dir, q[0], q[1], q[2]) & fbx;
                    if (rb.isEmpty()) { continue; }
                    const Real qx = q[0], qy = q[1], qz = q[2];
                    ParallelFor(rb, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        if (boundary_face(dir, i, j, k, per, dlo, dhi)) { return; }
                        const Real w = kernel_weight(dir, i, j, k, qx, qy, qz, plo, dx, znd, has_znd, inv_eps2, reach2);
                        if (w > Real(0.0)) { Gpu::Atomic::AddNoRet(&s(i,j,k), f_over_s * w); }
                    });
                }
            }
        }
    }
    Gpu::streamSynchronize();
}

Real
integrate_source (int dir, const MultiFab& src, const MultiFab* detJ_cc, const Geometry& geom)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dir >= 0 && dir <= 2 && src.ixType() == IndexType(IntVect::TheDimensionVector(dir)),
                                     "integrate_source: src must be face-centred in direction dir (0, 1 or 2)");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(detJ_cc == nullptr || detJ_cc->nGrow() >= 1, "integrate_source: detJ_cc needs one ghost cell");
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
