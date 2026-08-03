#include <UrbanCanopy/ERF_UCMTreeDrag.H>

#include <AMReX_GpuLaunch.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>
#include <ERF_IndexDefines.H>
#include <cmath>

void apply_ucm_tree_drag_to_source(
    amrex::MultiFab&       xmom_src,
    amrex::MultiFab&       ymom_src,
    const amrex::MultiFab& S_cons,
    const amrex::MultiFab& S_xmom,
    const amrex::MultiFab& S_ymom,
    const amrex::iMultiFab& is_tree_atm,
    const amrex::MultiFab& H_tree_atm,
    const amrex::MultiFab& H_crown_base_atm,
    const amrex::MultiFab& LAD_atm,
    const amrex::MultiFab& crown_area_frac_atm,
    const amrex::MultiFab& Cd_leaf_atm,
    const amrex::MultiFab* z_phys_nd,
    const amrex::Geometry& geom_atm,
    TreeDragMode           drag_mode,
    int                    /*lev*/,
    bool                   ucm_debug)
{
    if (drag_mode != TreeDragMode::Explicit) {
        return;
    }

    using namespace amrex;

    const auto& dom_lo = geom_atm.Domain().loVect();
    const auto  dx = geom_atm.CellSizeArray();
    const Real dz = dx[2];
    const int klo = dom_lo[2];
    constexpr Real min_density = 1.0e-12;
    const bool use_terrain = (z_phys_nd != nullptr);

    ReduceOps<ReduceOpMin, ReduceOpMax, ReduceOpMin, ReduceOpMax> reduce_op;
    ReduceData<Real, Real, Real, Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (MFIter mfi(S_cons, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto xmom_a = xmom_src.array(mfi);
        auto ymom_a = ymom_src.array(mfi);
        auto const s_cons_a = S_cons.const_array(mfi);
        auto const s_xmom_a = S_xmom.const_array(mfi);
        auto const s_ymom_a = S_ymom.const_array(mfi);
        auto const is_tree_a = is_tree_atm.const_array(mfi);
        auto const H_tree_a = H_tree_atm.const_array(mfi);
        auto const H_crown_base_a = H_crown_base_atm.const_array(mfi);
        auto const LAD_a = LAD_atm.const_array(mfi);
        auto const crown_frac_a = crown_area_frac_atm.const_array(mfi);
        auto const Cd_leaf_a = Cd_leaf_atm.const_array(mfi);
        Array4<const Real> z_nd_a = use_terrain ? z_phys_nd->const_array(mfi)
                                                : Array4<const Real>{};

        const int klo_c = klo;
        const Real dz_c = dz;
        const bool use_terrain_c = use_terrain;

        reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            if (k == klo_c) {
                return {0.0, 0.0, 0.0, 0.0};
            }
            if (is_tree_a(i, j, klo_c) != 1) {
                return {0.0, 0.0, 0.0, 0.0};
            }

            const Real z_center = use_terrain_c
                ? 0.125 * (
                    z_nd_a(i  , j  , k  ) + z_nd_a(i+1, j  , k  ) +
                    z_nd_a(i  , j+1, k  ) + z_nd_a(i+1, j+1, k  ) +
                    z_nd_a(i  , j  , k+1) + z_nd_a(i+1, j  , k+1) +
                    z_nd_a(i  , j+1, k+1) + z_nd_a(i+1, j+1, k+1))
                : (static_cast<Real>(k - klo_c) + 0.5) * dz_c;

            const Real H_tree = H_tree_a(i, j, klo_c);
            const Real H_crown_base = H_crown_base_a(i, j, klo_c);
            if (z_center < H_crown_base || z_center > H_tree) {
                return {0.0, 0.0, 0.0, 0.0};
            }

            const Real rho = amrex::max(s_cons_a(i, j, k, Rho_comp), min_density);
            const Real u = s_xmom_a(i, j, k) / rho;
            const Real v = s_ymom_a(i, j, k) / rho;
            const Real Uh = std::sqrt(u*u + v*v);
            const Real drag_coeff = 0.5 * rho
                                  * amrex::max(0.0, LAD_a(i, j, klo_c))
                                  * amrex::max(0.0, crown_frac_a(i, j, klo_c))
                                  * amrex::max(0.0, Cd_leaf_a(i, j, klo_c));
            const Real Fx_tree = -drag_coeff * Uh * u;
            const Real Fy_tree = -drag_coeff * Uh * v;

            xmom_a(i, j, k) += Fx_tree;
            ymom_a(i, j, k) += Fy_tree;

            return {Fx_tree, Fx_tree, Fy_tree, Fy_tree};
        });
    }

    if (ucm_debug) {
        auto [Fx_min, Fx_max, Fy_min, Fy_max] = reduce_data.value();
        if (ParallelDescriptor::IOProcessor()) {
            Print() << "[UCM][6.1][apply_ucm_tree_drag_to_source]\n"
                    << "  Mode: explicit\n"
                    << "  Fx_tree: min=" << Fx_min << " max=" << Fx_max << " [N/m^3]\n"
                    << "  Fy_tree: min=" << Fy_min << " max=" << Fy_max << " [N/m^3]\n";
        }
    }
}
