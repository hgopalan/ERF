/**
 * @file ERF_UCMAtmCoupling.cpp
 * @brief Implementation of atmospheric coupling for UCM (coarsen and inject fluxes)
 *
 * **RK-stage safety contract (Phase 2.3 fix)**
 *
 * `apply_ucm_tendency_to_cc_source` OWNS `cc_source[RhoTheta_comp]` (and
 * optionally `cc_source[RhoQ1_comp]`) for the duration of each RK stage.
 * It zeros those components at the top of each call and writes with `=`
 * (not `+=`), so the result is the UCM tendency alone — independent of how
 * many times it is called per coarse step.
 *
 * Call-site contract:
 *  - `UCMLayer::advance` (SEB + flux computation) runs ONCE per coarse step,
 *    in `ERF::Advance`, before the MRI integrator starts.
 *  - `coarsen_ucm_flux_to_atm` also runs ONCE per coarse step to cache
 *    `m_ucm_H_atm[lev]` and `m_ucm_LE_atm[lev]`.
 *  - `apply_ucm_tendency_to_cc_source` is then called ONCE PER RK STAGE
 *    from `slow_rhs_fun_pre` in `ERF_TI_slow_rhs_pre.H`, after `make_sources`
 *    has reset `cc_src` to zero.
 *
 * This mirrors the ERF-Fire convention documented in
 * `Source/Fire/ERF_FireAtmCoupling.H` (ERF-Hazard branch).
 *
 * If any other physics module needs to write into `cc_source[RhoTheta_comp]`
 * per RK stage on the same cells, this function must be moved to a
 * pre-integrator path and semantics changed back to `+=`.
 *
 * References:
 *  - WRF Single-Layer Urban Canopy Model (Chen et al., 2011)
 *  - WRF module_sf_urban.F / module_fire_tendency.F
 *  - Mandel et al. (2011) "Coupled atmosphere-wildland fire modeling"
 *  - Source/Dust/ERF_DustAtmCoupling.H/.cpp
 */

#include <UrbanCanopy/ERF_UCMAtmCoupling.H>
#include <UrbanCanopy/ERF_UCMAtmAggregation.H>
#include <ERF_IndexDefines.H>
#include <ERF_Constants.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Reduce.H>
#include <AMReX_Print.H>

/**
 * @brief Refine an ATM-grid MultiFab onto the UCM 2D slab grid.
 *
 * Piecewise-constant injection: each ATM cell (i,j) in the slab is replicated
 * into a grid_ratio x grid_ratio block of UCM cells.
 *
 * Implementation:
 * 1. Extract 2D slab from Q_atm_in at k=klo_atm
 * 2. Build coarse MultiFab on UCM DM with coarsened UCM BA
 * 3. ParallelCopy from slab to coarse
 * 4. Inject coarse -> fine using division of cell indices
 */
void refine_atm_to_ucm(amrex::MultiFab&       Q_ucm_out,
                       const amrex::MultiFab& Q_atm_in,
                       int                    grid_ratio,
                       int                    klo_atm)
{
    using namespace amrex;

    // Step 1: build a 2D ATM slab at k = klo_atm covering Q_atm_in's BA
    BoxList bl;
    for (int i = 0; i < Q_atm_in.boxArray().size(); ++i) {
        Box b = Q_atm_in.boxArray()[i];
        b.setSmall(2, klo_atm);
        b.setBig(2,   klo_atm);
        bl.push_back(b);
    }
    BoxArray ba_atm_slab(std::move(bl));
    MultiFab atm_slab(ba_atm_slab, Q_atm_in.DistributionMap(), 1, 0);
    atm_slab.setVal(0.0);
    atm_slab.ParallelCopy(Q_atm_in, 0, 0, 1);

    if (grid_ratio == 1) {
        Q_ucm_out.ParallelCopy(atm_slab, 0, 0, 1);
        return;
    }

    // Step 2: align ranks: build a coarse MF on the UCM DM but with the
    // coarsened UCM BA, then ParallelCopy from ATM slab.
    BoxArray ba_ucm_coarsened = Q_ucm_out.boxArray();
    ba_ucm_coarsened.coarsen(IntVect(grid_ratio, grid_ratio, 1));
    MultiFab coarse_on_ucm_dm(ba_ucm_coarsened, Q_ucm_out.DistributionMap(), 1, 0);
    coarse_on_ucm_dm.setVal(0.0);
    coarse_on_ucm_dm.ParallelCopy(atm_slab, 0, 0, 1);

    // Step 3: piecewise-constant inject coarse -> fine on the UCM grid
    for (MFIter mfi(Q_ucm_out, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto       fine_a = Q_ucm_out.array(mfi);
        auto const crse_a = coarse_on_ucm_dm.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            fine_a(i, j, k) = crse_a(i / grid_ratio, j / grid_ratio, k);
        });
    }

    // Collective min/max on ALL ranks; print on IO rank only.
    Real qmin = Q_ucm_out.min(0, 0);
    Real qmax = Q_ucm_out.max(0, 0);
    if (ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][1.4][refine_atm_to_ucm] gr=" << grid_ratio
                << " klo_atm=" << klo_atm
                << " Q_ucm=[" << qmin << ", " << qmax << "]\n";
    }
}

/**
 * @brief Coarsen UCM fluxes to ATM grid.
 *
 * Implementation of the coarsen pattern. When grid_ratio==1, performs a direct copy.
 * When grid_ratio>1, uses amrex::average_down to spatially average the UCM flux to ATM.
 *
 * The UCM fluxes live on a 2D slab, while Q_atm_out is a 3D MultiFab.
 * We create a 2D slab at k=klo on the ATM x-y decomposition, coarsen into it,
 * then ParallelCopy into the k=klo plane of Q_atm_out.
 */
void coarsen_ucm_flux_to_atm(
    amrex::MultiFab&           Q_atm_out,
    const amrex::MultiFab&     Q_ucm,
    const amrex::iMultiFab&    is_urban_ucm,
    const amrex::Geometry&     /*geom_ucm*/,
    const amrex::Geometry&     geom_atm,
    int                        grid_ratio,
    int                        /*lev*/)
{
    using namespace amrex;

    // Phase 2.5 aggregation, convention B (conservation-preserving, no injection-side reweight):
    //
    //   Q_atm(I,J) = (1 / N_total) * sum over N=grid_ratio^2 UCM cells of (is_urban * Q_ucm)
    //
    // where N_total = grid_ratio^2 counts ALL UCM cells inside the ATM cell (urban + non-urban).
    //
    // Total column-integrated flux is preserved by construction. Proof:
    //   sum_over_ATM(Q_atm * dA_atm)
    //     = sum_over_ATM(Q_atm * N_total * dA_ucm)
    //     = sum_over_ATM(sum_over_N(is_urban * Q_ucm) * dA_ucm)
    //     = sum_over_UCM_urban(Q_ucm * dA_ucm)
    //     = total urban heat production.
    //
    // The injection kernel `apply_ucm_tendency_to_cc_source` reads Q_atm AS-IS.
    // It does NOT multiply by f_urb_atm. Mirrors `Source/Fire/ERF_FireAtmCoupling.H`
    // on branch ERF-Hazard.
    //
    // Verified by `Exec/CanonicalTests/SLUCM/UCMScaleAwareAggregation/check_conservation.py`.

    Q_atm_out.setVal(0.0);

    // Build a 2D ATM slab BoxArray at k = klo of the ATM domain.
    const int klo_atm = geom_atm.Domain().smallEnd(2);

    BoxList bl;
    for (int i = 0; i < Q_atm_out.boxArray().size(); ++i) {
       Box b = Q_atm_out.boxArray()[i];
       b.setSmall(2, klo_atm);
       b.setBig(2,   klo_atm);
       bl.push_back(b);
    }
    BoxArray ba_atm_slab(std::move(bl));

    MultiFab atm_slab(ba_atm_slab, Q_atm_out.DistributionMap(), 1, 0);
    atm_slab.setVal(0.0);

    if (grid_ratio == 1) {
       MultiFab::Copy(atm_slab, Q_ucm, 0, 0, 1, 0);
    } else {
       // Create masked flux: multiply UCM flux by is_urban to zero non-urban cells
       MultiFab Q_masked(Q_ucm.boxArray(), Q_ucm.DistributionMap(), 1, Q_ucm.nGrow());
       Q_masked.setVal(0.0);

       for (MFIter mfi(Q_masked, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
           const Box& bx = mfi.tilebox();
           auto q_masked_a = Q_masked.array(mfi);
           auto const q_a = Q_ucm.const_array(mfi);
           auto const is_urban_a = is_urban_ucm.const_array(mfi);

           ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
               q_masked_a(i, j, k) = q_a(i, j, k) * static_cast<Real>(is_urban_a(i, j, 0));
           });
       }


       // Coarsen masked flux via average_down
       average_down(Q_masked, atm_slab, 0, 1, IntVect(grid_ratio, grid_ratio, 1));
    }

    // ParallelCopy the slab into k=klo_atm of Q_atm_out (only that plane overlaps).
    Q_atm_out.ParallelCopy(atm_slab, 0, 0, 1);

    // Collectives on all ranks (do not put inside IOProcessor() guard).
    Real min_ucm = Q_ucm.min(0, 0);
    Real max_ucm = Q_ucm.max(0, 0);
    Real min_atm = Q_atm_out.min(0, 0);
    Real max_atm = Q_atm_out.max(0, 0);

    if (ParallelDescriptor::IOProcessor()) {
       Print() << "[UCM][2.5][coarsen_ucm_flux_to_atm]\n"
               << "  grid_ratio=" << grid_ratio << "\n"
               << "  before: Q_ucm  min=" << min_ucm << " max=" << max_ucm << "\n"
               << "  after:  Q_atm  min=" << min_atm << " max=" << max_atm
               << " (area-averaged, only k=" << klo_atm << " plane written)\n";
    }
}

/**
 * @brief Apply exponential-decay vertical tendency to cc_source (Phase 2.6: morphology-aware).
 *
 * Phase 2.6 rewrite: replaces single scalar alpha_ucm with per-cell e-folding depth,
 * and splits injection into surface (roads) + exponential (walls/roofs/AH) terms.
 */
void apply_ucm_tendency_to_cc_source(
    amrex::MultiFab&        cc_source,
    const amrex::MultiFab&  H_atm,
    const amrex::MultiFab&  H_road_atm,
    const amrex::MultiFab&  H_wallroof_atm,
    const amrex::MultiFab&  H_bldg_mean_atm,
    const amrex::MultiFab*  LE_atm,
    const amrex::MultiFab&  /*z_phys_cc*/,   // reserved for Phase 4 terrain support
    const amrex::MultiFab&  S_old,
    const amrex::Geometry&  geom_atm,
    const amrex::iMultiFab& is_urban_atm,
    amrex::Real             alpha_scale,
    amrex::Real             alpha_min,
    amrex::Real             alpha_max,
    bool                    use_morphology_injection,
    amrex::Real             alpha_ucm_fallback,
    amrex::Real             feedback,
    bool                    has_moisture,
    bool                    ucm_debug,
    int                     /*lev*/)
{
    // One-time warning if feedback is zero
    static bool warned_feedback_zero = false;
    if (feedback == 0.0 && !warned_feedback_zero && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "[UCM][1.4][apply_ucm_tendency_to_cc_source]\n";
        amrex::Print() << "  WARNING: atm_feedback = 0.0 (one-way coupling OFF)\n";
        amrex::Print() << "  Tendency IS computed but NOT injected. Set atm_feedback in (0,1] to enable.\n";
        warned_feedback_zero = true;
    }

    // One-time debug message for Phase 2.6
    static bool debug_injection_once = false;
    if (!debug_injection_once && ucm_debug && amrex::ParallelDescriptor::IOProcessor()) {
        debug_injection_once = true;
        if (use_morphology_injection) {
            amrex::Print() << "[UCM][2.6] injection: surface term (roads) + morphology-aware exponential (walls+roof+AH)\n";
        } else {
            amrex::Print() << "[UCM][2.5-compat] fallback: uniform alpha_ucm = " << alpha_ucm_fallback << " [m]\n";
        }
    }

    // Early return if coupling is off
    if (feedback == 0.0) {
        return;
    }

    // Get grid parameters
    const auto& dom_lo = geom_atm.Domain().loVect();
    const auto& dom_hi = geom_atm.Domain().hiVect();
    const auto  dx     = geom_atm.CellSizeArray();
    const amrex::Real dz = dx[2];
    const int klo = dom_lo[2];
    const int khi = dom_hi[2];

    // Physical constants
    const amrex::Real Cp = Cp_d;

    // Safety clamp for theta tendency (K/s absolute)
    constexpr amrex::Real theta_tend_cap = 0.05;
    static bool warned_clamp_exceeded = false;

    // Hoist LE availability out of the kernel
    const bool have_le = (has_moisture && LE_atm != nullptr);

    // -----------------------------------------------------------------------
    // RK-stage safety: ZERO the components we own at entry.
    // Then always use += to accumulate contributions from both surface and exp terms.
    // This ensures the result is correct even if called multiple times per RK stage
    // (though in practice it should be called once per stage after make_sources resets cc_src).
    // -----------------------------------------------------------------------
    cc_source.setVal(0.0, RhoTheta_comp, 1, cc_source.nGrowVect());
    if (have_le) {
        cc_source.setVal(0.0, RhoQ1_comp, 1, cc_source.nGrowVect());
    }

    // Local reduction: track min/max of alpha_ij, and sums for each injection term
    // Tuple: (alpha_min, alpha_max, sum_road_tend, sum_exp_tend)
    amrex::ReduceOps<amrex::ReduceOpMin, amrex::ReduceOpMax, amrex::ReduceOpSum, amrex::ReduceOpSum>
        reduce_op;
    amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real>
        reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    // Iteration over boxes with tiling
    for (amrex::MFIter mfi(cc_source, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();

        // Get Array4 references (by value for GPU)
        auto cc_src_a           = cc_source.array(mfi);
        auto const h_total_a    = H_atm.const_array(mfi);  // Phase 2.5 lumped (unused if use_morphology)
        auto const h_road_a     = H_road_atm.const_array(mfi);
        auto const h_wr_a       = H_wallroof_atm.const_array(mfi);
        auto const h_bldg_a     = H_bldg_mean_atm.const_array(mfi);
        auto const s_a          = S_old.const_array(mfi);
        auto const urban_a      = is_urban_atm.const_array(mfi);

        // Optional LE_atm; default-constructed Array4 is safe to capture when unused
        amrex::Array4<const amrex::Real> le_a = have_le
            ? LE_atm->const_array(mfi)
            : amrex::Array4<const amrex::Real>{};

        // Capture parameters for device lambda
        const int    klo_c = klo;
        const amrex::Real dz_c = dz;
        const amrex::Real Cp_c = Cp;
        const amrex::Real alpha_scale_c = alpha_scale;
        const amrex::Real alpha_min_c = alpha_min;
        const amrex::Real alpha_max_c = alpha_max;
        const amrex::Real alpha_ucm_fallback_c = alpha_ucm_fallback;
        const bool use_morphology_c = use_morphology_injection;
        const amrex::Real feedback_c = feedback;

        // Main injection kernel
        reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            // Guard: skip non-urban cells (mask is stored at k=klo plane)
            if (urban_a(i, j, klo_c) == 0) {
                return { alpha_min_c, alpha_min_c, 0.0, 0.0 };
            }

            // Compute per-cell alpha_ij for morphology-aware injection
            amrex::Real alpha_ij;
            if (use_morphology_c) {
                const amrex::Real H_mean = h_bldg_a(i, j, klo_c);
                // Clamp alpha_ij to [alpha_min, alpha_max]
                alpha_ij = amrex::max(alpha_min_c,
                                      amrex::min(alpha_max_c, alpha_scale_c * H_mean));
            } else {
                // Fallback to uniform alpha_ucm (Phase 2.5)
                alpha_ij = alpha_ucm_fallback_c;
            }

            // Height-above-surface (flat terrain)
            const amrex::Real z_k      = (k       - klo_c) * dz_c;
            const int         kp1      = amrex::min(k + 1, khi);
            const amrex::Real z_k_plus = (kp1     - klo_c) * dz_c;

            // Density at this level
            const amrex::Real rho_k = s_a(i, j, k, Rho_comp);

            amrex::Real theta_tend_total = 0.0;

            // ===================================================================
            // Surface term: road flux → k=klo only, no vertical decay
            // ===================================================================
            if (k == klo_c) {
                const amrex::Real H_road = h_road_a(i, j, klo_c);
                if (H_road > 0.0) {
                    // Road flux goes directly to surface, normalized by dz(klo)
                    theta_tend_total += (H_road / (Cp_c * dz_c));
                }
            }

            // ===================================================================
            // Exponential term: wall+roof+AH flux → distributed over column with per-cell alpha
            // ===================================================================
            {
                const amrex::Real H_wr = h_wr_a(i, j, klo_c);
                if (H_wr > 0.0 && alpha_ij > 0.0) {
                    const amrex::Real exp_factor      = std::exp(-z_k      / alpha_ij);
                    const amrex::Real exp_factor_plus = std::exp(-z_k_plus / alpha_ij);

                    // Height-tendency [K/s] at this level and the one above
                    const amrex::Real h_tend      = (H_wr / Cp_c) * exp_factor;
                    const amrex::Real h_tend_plus = (H_wr / Cp_c) * exp_factor_plus;

                    // Exponential term [K/s/m]
                    theta_tend_total += (h_tend - h_tend_plus) / dz_c;
                }
            }

            // Convert to per-unit-volume source: multiply by density
            const amrex::Real dtheta = feedback_c * rho_k * theta_tend_total;

            // Safety clamp: skip if tendency exceeds physical bounds
            if (std::abs(dtheta / rho_k) > theta_tend_cap) {
                #ifdef AMREX_PRAGMA_OMP_ATOMIC
                #pragma omp atomic write
                #endif
                warned_clamp_exceeded = true;
                return { alpha_ij, alpha_ij, 0.0, 0.0 };
            }

            // ACCUMULATE (+=): UCM owns this component and may write from multiple pathways
            cc_src_a(i, j, k, RhoTheta_comp) += dtheta;

            // Latent heat (optional)
            if (have_le) {
                const amrex::Real LE_sfc = le_a(i, j, klo_c);
                if (LE_sfc > 0.0) {
                    amrex::Real q_tend_total = 0.0;

                    // Surface term for LE (if needed; for now latent follows sensible split)
                    // This would require separate LE_road and LE_wallroof; for Phase 2.6 keep lumped
                    if (k == klo_c) {
                        q_tend_total += (LE_sfc / L_v / dz_c);
                    }

                    const amrex::Real dq = feedback_c * rho_k * q_tend_total;
                    cc_src_a(i, j, k, RhoQ1_comp) += dq;
                }
            }

            // Return for reduction: (alpha_ij_min, alpha_ij_max, sum_road, sum_exp)
            // For accounting, we compute the surface and exp contributions separately for diagnostics
            amrex::Real contrib_road = 0.0;
            amrex::Real contrib_exp = 0.0;

            if (k == klo_c && h_road_a(i, j, klo_c) > 0.0) {
                contrib_road = (feedback_c * rho_k * h_road_a(i, j, klo_c) / Cp_c);
            }
            if (h_wr_a(i, j, klo_c) > 0.0 && alpha_ij > 0.0) {
                const amrex::Real exp_factor = std::exp(-z_k / alpha_ij);
                const amrex::Real exp_factor_plus = std::exp(-z_k_plus / alpha_ij);
                const amrex::Real h_wr = h_wr_a(i, j, klo_c);
                const amrex::Real h_tend = (h_wr / Cp_c) * exp_factor;
                const amrex::Real h_tend_plus = (h_wr / Cp_c) * exp_factor_plus;
                contrib_exp = (feedback_c * rho_k * (h_tend - h_tend_plus));
            }

            return { alpha_ij, alpha_ij, contrib_road, contrib_exp };
        });
    }

    // Ensure all ParallelFor / reduce_op writes are visible to subsequent reductions
    amrex::Gpu::streamSynchronize();

    // Collect statistics across all ranks
    ReduceTuple hv = reduce_data.value(reduce_op);
    amrex::Real alpha_ij_min  = amrex::get<0>(hv);
    amrex::Real alpha_ij_max  = amrex::get<1>(hv);
    amrex::Real road_tend_sum = amrex::get<2>(hv);
    amrex::Real exp_tend_sum  = amrex::get<3>(hv);

    // Collectives OUTSIDE IOProcessor guard (per PR #209)
    amrex::ParallelDescriptor::ReduceRealMin(alpha_ij_min);
    amrex::ParallelDescriptor::ReduceRealMax(alpha_ij_max);
    amrex::ParallelDescriptor::ReduceRealSum(road_tend_sum);
    amrex::ParallelDescriptor::ReduceRealSum(exp_tend_sum);

    // Emit warning if clamp was exceeded
    if (warned_clamp_exceeded && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "[UCM][2.6][WARN] apply_ucm_tendency_to_cc_source: "
                       << "|theta_tend|/rho exceeded " << theta_tend_cap << " K/s, "
                       << "skipping affected cells.\n";
    }

    // -----------------------------------------------------------------------
    // Debug diagnostics: gated on ucm_debug to avoid unconditional per-step MPI collectives.
    // -----------------------------------------------------------------------
    if (ucm_debug) {
        amrex::Gpu::streamSynchronize();

        // Flux diagnostics (collective on all ranks, inside debug gate)
        amrex::Real min_h_road = H_road_atm.min(0, 0);
        amrex::Real max_h_road = H_road_atm.max(0, 0);
        amrex::Real min_h_wr = H_wallroof_atm.min(0, 0);
        amrex::Real max_h_wr = H_wallroof_atm.max(0, 0);

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.6][apply_ucm_tendency_to_cc_source]\n";
            amrex::Print() << "  Mode: " << (use_morphology_injection ? "per-cell alpha (2.6)" : "uniform alpha (2.5-compat)") << "\n";
            if (use_morphology_injection) {
                amrex::Print() << "  alpha_ij (per-cell)     min=" << alpha_ij_min << " max=" << alpha_ij_max << " [m]\n";
            } else {
                amrex::Print() << "  alpha_ucm (uniform)     " << alpha_ucm_fallback << " [m]\n";
            }
            amrex::Print() << "  H_road_atm              min=" << min_h_road << " max=" << max_h_road << " [W/m2]\n";
            amrex::Print() << "  H_wallroof_atm          min=" << min_h_wr << " max=" << max_h_wr << " [W/m2]\n";
            amrex::Print() << "  Surface term (road)     sum=" << road_tend_sum << " [K*kg/m3/s]\n";
            amrex::Print() << "  Exponential term (wr)   sum=" << exp_tend_sum << " [K*kg/m3/s]\n";
            amrex::Print() << "  atm_feedback=" << feedback << "\n";
        }
    }
}
