/**
 * @file ERF_UCMAtmCoupling.cpp
 * @brief Implementation of atmospheric coupling for UCM (coarsen and inject fluxes)
 *
 * References:
 *  - WRF Single-Layer Urban Canopy Model (Chen et al., 2011)
 *  - WRF module_sf_urban.F / module_fire_tendency.F
 *  - Mandel et al. (2011) "Coupled atmosphere-wildland fire modeling"
 *  - Source/Dust/ERF_DustAtmCoupling.H/.cpp
 */

#include <UrbanCanopy/ERF_UCMAtmCoupling.H>
#include <ERF_IndexDefines.H>
#include <ERF_Constants.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Print.H>

/**
 * @brief Coarsen UCM fluxes to ATM grid.
 *
 * Implementation of the coarsen pattern. When grid_ratio==1, performs a direct copy.
 * When grid_ratio>1, uses amrex::average_down to spatially average the UCM flux to ATM.
 */
void coarsen_ucm_flux_to_atm(
    amrex::MultiFab&       Q_atm_out,
    const amrex::MultiFab& Q_ucm,
    const amrex::Geometry& /*geom_ucm*/,
    const amrex::Geometry& /*geom_atm*/,
    int                    grid_ratio,
    int                    /*lev*/)
{
    if (grid_ratio == 1) {
        // Direct copy when grids are aligned
        amrex::MultiFab::Copy(Q_atm_out, Q_ucm, 0, 0, 1, 0);
    } else {
        // Use average_down to coarsen from UCM grid to ATM grid
        amrex::average_down(Q_ucm, Q_atm_out, 0, 1, grid_ratio);
    }

    // Debug trace: print min/max before and after
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Real min_ucm = Q_ucm.min(0);
        amrex::Real max_ucm = Q_ucm.max(0);
        amrex::Real min_atm = Q_atm_out.min(0);
        amrex::Real max_atm = Q_atm_out.max(0);

        amrex::Print() << "[UCM][1.4][coarsen_ucm_flux_to_atm]\n";
        amrex::Print() << "  grid_ratio=" << grid_ratio << "\n";
        amrex::Print() << "  before: Q_ucm   min=" << min_ucm << " max=" << max_ucm << "\n";
        amrex::Print() << "  after:  Q_atm   min=" << min_atm << " max=" << max_atm << "\n";
    }
}

/**
 * @brief Apply exponential-decay vertical tendency to cc_source.
 *
 * Exponential injection pattern after Mandel 2011 / WRF-SFIRE fire_tendency.
 * Distributes surface sensible heat flux (and optionally latent heat) vertically
 * with exponential decay over alpha_ucm depth scale.
 */
void apply_ucm_tendency_to_cc_source(
    amrex::MultiFab&       cc_source,
    const amrex::MultiFab& H_atm,
    const amrex::MultiFab* LE_atm,
    const amrex::MultiFab& z_phys_cc,
    const amrex::MultiFab& S_old,
    const amrex::Geometry& geom_atm,
    const amrex::iMultiFab& is_urban_atm,
    amrex::Real            alpha_ucm,
    amrex::Real            feedback,
    bool                   has_moisture,
    bool                   ucm_debug,
    int                    /*lev*/)
{
    // One-time warning if feedback is zero
    static bool warned_feedback_zero = false;
    if (feedback == 0.0 && !warned_feedback_zero && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "[UCM][1.4][apply_ucm_tendency_to_cc_source]\n";
        amrex::Print() << "  WARNING: atm_feedback = 0.0 (one-way coupling OFF)\n";
        amrex::Print() << "  Tendency IS computed but NOT injected. Set atm_feedback in (0,1] to enable.\n";
        warned_feedback_zero = true;
    }

    // Early return if coupling is off
    if (feedback == 0.0) {
        return;
    }

    // Get grid parameters
    const auto& dom_lo = geom_atm.Domain().loVect();
    const auto  dx     = geom_atm.CellSizeArray();
    const amrex::Real dz = dx[2];
    const int klo = dom_lo[2];

    // Physical constants
    const amrex::Real Cp = Cp_d;

    // Hoist LE availability out of the kernel
    const bool have_le = (has_moisture && LE_atm != nullptr);

    // Iteration over boxes with tiling
    for (amrex::MFIter mfi(cc_source, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();

        // Get Array4 references (by value for GPU)
        auto cc_src_a      = cc_source.array(mfi);
        auto const h_a     = H_atm.const_array(mfi);
        auto const z_a     = z_phys_cc.const_array(mfi);
        auto const s_a     = S_old.const_array(mfi);
        auto const urban_a = is_urban_atm.const_array(mfi);

        // Optional LE_atm; default-constructed Array4 is safe to capture when unused
        amrex::Array4<const amrex::Real> le_a = have_le
            ? LE_atm->const_array(mfi)
            : amrex::Array4<const amrex::Real>{};

        // Kernel: exponential injection
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // Guard: skip non-urban cells
            if (urban_a(i, j, 0) == 0) return;

            // Height-above-surface
            const amrex::Real z_sfc = z_a(i, j, klo);
            const amrex::Real z_k   = z_a(i, j, k) - z_sfc;

            // Exponential profile at this level
            const amrex::Real exp_factor = std::exp(-z_k / alpha_ucm);

            // Sensible heat tendency at k [K/s]
            const amrex::Real h_tend = (h_a(i, j, 0) / Cp) * exp_factor;

            // Density at this level
            const amrex::Real rho_k = s_a(i, j, k, Rho_comp);

            // Vertical divergence of exponential flux at k+1/2
            const amrex::Real z_k_plus       = z_a(i, j, k+1) - z_sfc;
            const amrex::Real exp_factor_plus = std::exp(-z_k_plus / alpha_ucm);
            const amrex::Real h_tend_plus    = (h_a(i, j, 0) / Cp) * exp_factor_plus;

            // Tendency for potential temperature equation (per unit volume)
            const amrex::Real theta_tend = rho_k * (h_tend - h_tend_plus) / dz;

            // Apply feedback coupling coefficient and accumulate
            cc_src_a(i, j, k, RhoTheta_comp) += feedback * theta_tend;

            // Latent heat (optional)
            if (have_le) {
                const amrex::Real le_tend      = (le_a(i, j, 0) / L_v) * exp_factor;
                const amrex::Real le_tend_plus = (le_a(i, j, 0) / L_v) * exp_factor_plus;
                const amrex::Real q_tend       = rho_k * (le_tend - le_tend_plus) / dz;
                cc_src_a(i, j, k, RhoQ1_comp) += feedback * q_tend;
            }
        });
    }

    // Debug trace
    if (ucm_debug && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Real min_h    = H_atm.min(0);
        amrex::Real max_h    = H_atm.max(0);
        amrex::Real min_tend = cc_source.min(0, RhoTheta_comp);
        amrex::Real max_tend = cc_source.max(0, RhoTheta_comp);

        // Estimate expected surface tendency magnitude
        const amrex::Real rho_0 = 1.2;  // approximate surface density
        const amrex::Real exp_factor_dz  = std::exp(-dz / alpha_ucm);
        const amrex::Real expected_scale = rho_0 * (max_h / Cp) * (1.0 - exp_factor_dz) / dz;

        amrex::Print() << "[UCM][1.4][apply_ucm_tendency_to_cc_source]\n";
        amrex::Print() << "  atm_feedback=" << feedback
                       << " (injection_active=" << (feedback > 0.0 ? "yes" : "no") << ")\n";
        amrex::Print() << "  k=0: rho≈1.2 dz=" << dz << "\n";
        amrex::Print() << "  alpha_ucm=" << alpha_ucm << " [m]\n";
        amrex::Print() << "  H_atm min=" << min_h << " max=" << max_h << " [W/m²]\n";
        amrex::Print() << "  RhoTheta_tend min=" << min_tend << " max=" << max_tend << "\n";
        amrex::Print() << "  expected surface tend magnitude ≈ " << expected_scale << " [K/s]\n";
    }
}