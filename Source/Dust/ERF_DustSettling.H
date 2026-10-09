/**
 * @file ERF_DustSettling.H
 * @brief Stokes settling velocity and vertical settling tendency for dust scalars.
 *
 * Computes per-bin Stokes settling velocity and applies the resulting
 * downward mass flux divergence as a tendency in cc_source.
 *
 * Settling velocity (Stokes law, valid for d < 100 um):
 *   v_s = (rho_p - rho_a) * g * d^2 / (18 * mu_a)
 *
 * Cunningham slip correction (Stokes-Cunningham):
 *   C_c = 1 + (2 * lambda / d) * (A1 + A2 * exp(-A3 * d / lambda))
 *   v_s_corrected = v_s * C_c
 * where lambda = 0.066 um (mean free path at STP),
 * A1=1.257, A2=0.400, A3=0.55: Davies (1945), the constants most references
 * tabulate (Allen & Raabe 1985 give 1.142, 0.558, 0.999).
 * For d > 1 um the correction is < 15% and may be omitted for coarse bins.
 *
 * Vertical tendency (first-order upwind flux divergence):
 *   d(RhoDust)/dt += (v_s(k+1)*RhoDust(k+1) - v_s(k)*RhoDust(k)) / h_k
 * with h_k = detJ(k) dz the cell thickness (until October 2026 the
 * centre-to-centre spacing above the cell was used, which creates mass on
 * stretched grids: (r - 1)/2 per transfer, 2.5 % at a 1.05 ratio, 5 % at 1.1).
 * Downward flux at k=0 is
 * the deposition kernel's.
 *
 * With the bins transported as one scalar the settling velocity is the mean of
 * the per-bin velocities weighted by the bins' shares of the mass emitted so
 * far (equal until anything is emitted):
 * erf.dust.lumped_settling = mean; bin0 is the form until October 2026, which
 * settled the 50 um third of the default bins at the 7 um velocity (residence
 * 5850 s in a 23 m cell instead of 117 s).
 *
 * Pattern reference: ERF_FireAtmCoupling.H in Source/Fire/:
 *   https://github.com/hgopalan/ERF/blob/7856affedd8bb3c878b62c308f39c36bf2228791/
 *   Source/Fire/ERF_FireAtmCoupling.H
 *
 * References:
 *   Stokes, G.G. (1851). Trans. Cambridge Phil. Soc., 9, 8-106.
 *   Davies, C.N. (1945). Proc. Phys. Soc., 57, 259-270.
 *     https://doi.org/10.1088/0959-5309/57/4/301
 *   Allen, M.D., & Raabe, O.G. (1985). Aerosol Sci. Tech., 4, 269-286.
 *     https://doi.org/10.1080/02786828508959055
 *   Seinfeld, J.H., & Pandis, S.N. (2006). Atmospheric Chemistry and Physics,
 *     2nd ed. Wiley. Chapter 9. ISBN 978-0-471-72018-8.
 *   Ginoux, P. et al. (2001). J. Geophys. Res., 106, 20255.
 *     https://doi.org/10.1029/2000JD901323
 */

#ifndef ERF_DUST_SETTLING_H
#define ERF_DUST_SETTLING_H

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_MFIter.H>
#include <AMReX_Print.H>
#include <cmath>

// Density component index (needed for S_old array access)
#include <ERF_IndexDefines.H>

namespace DustSettlingConst {
    constexpr amrex::Real G_ACCEL    = 9.81;     ///< Gravitational acceleration [m/s^2]
    constexpr amrex::Real MU_AIR_STD = 1.81e-5;  ///< Dynamic viscosity of air at 293 K [Pa s]
    constexpr amrex::Real LAMBDA_MFP = 6.6e-8;   ///< Mean free path at STP [m] (0.066 um)
    constexpr amrex::Real CC_A1      = 1.257;    ///< Cunningham coefficient A1
    constexpr amrex::Real CC_A2      = 0.400;    ///< Cunningham coefficient A2
    constexpr amrex::Real CC_A3      = 0.550;    ///< Cunningham coefficient A3
    constexpr amrex::Real V_S_MAX    = 1.0;      ///< Settling velocity cap [m/s]
    constexpr amrex::Real V_S_MIN    = 0.0;      ///< Settling velocity floor [m/s]
    constexpr int         MAX_BINS   = 8;        ///< Bins a kernel can carry by value (erf.dust.n_size_bins <= 8)
}

/// Diameters of the bins a settling or deposition kernel averages over.
using DustBinDiameters = amrex::GpuArray<amrex::Real, DustSettlingConst::MAX_BINS>;
/// Mass shares of the bins a transported scalar carries (sum 1 over the nb used).
using DustBinWeights   = amrex::GpuArray<amrex::Real, DustSettlingConst::MAX_BINS>;

/**
 * @brief Cunningham slip correction factor for sub-micron particles.
 *
 * C_c = 1 + (2*lambda/d) * (A1 + A2*exp(-A3*d/lambda))
 * For d >> lambda, C_c approaches 1.
 * Constants of Davies (1945); see the file header.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_cunningham_cc(amrex::Real d_m) noexcept
{
    if (d_m <= 0.0) return 1.0;
    amrex::Real ratio = 2.0 * DustSettlingConst::LAMBDA_MFP / d_m;
    return 1.0 + ratio * (DustSettlingConst::CC_A1
                        + DustSettlingConst::CC_A2
                          * std::exp(-DustSettlingConst::CC_A3 * d_m
                                     / DustSettlingConst::LAMBDA_MFP));
}

/**
 * @brief Stokes settling velocity with Cunningham correction [m/s].
 *
 * v_s = (rho_p - rho_a) * g * d^2 * C_c / (18 * mu_a)
 * Clamped to [V_S_MIN, V_S_MAX].
 *
 * @param d_m    Particle diameter [m].
 * @param rho_p  Particle density [kg/m^3].
 * @param rho_a  Local air density [kg/m^3].
 * @param mu_a   Dynamic viscosity [Pa s]. Uses MU_AIR_STD if <= 0.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_stokes_settling(amrex::Real d_m, amrex::Real rho_p,
                                     amrex::Real rho_a, amrex::Real mu_a) noexcept
{
    if (d_m <= 0.0 || rho_p <= 0.0) return 0.0;
    amrex::Real mu = (mu_a > 0.0) ? mu_a : DustSettlingConst::MU_AIR_STD;
    amrex::Real rho_diff = rho_p - amrex::max(static_cast<amrex::Real>(rho_a), static_cast<amrex::Real>(0.0));
    if (rho_diff <= 0.0) return 0.0;
    amrex::Real Cc  = compute_cunningham_cc(d_m);
    amrex::Real vs  = rho_diff * DustSettlingConst::G_ACCEL * d_m * d_m * Cc / (18.0 * mu);
    return amrex::min(amrex::max(vs, amrex::Real(DustSettlingConst::V_S_MIN)),
                      amrex::Real(DustSettlingConst::V_S_MAX));
}

/**
 * @brief Settling velocity of the transported scalar: the mass-weighted mean
 *        over nb bins of the Stokes velocity at the local air density (nb = 1:
 *        one bin). The weights are the bins' shares of the mass emitted so far
 *        (DustLayer::scalar_bin_diameters); equal shares until anything is
 *        emitted. Until October 2026 the shares were taken as equal whatever
 *        the sources had emitted, so haul-road dust (bin 0 only) settled at the
 *        three-bin mean, 17x its own velocity with the default bins.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_mean_settling(const DustBinDiameters& d_bins, const DustBinWeights& w_bins, int nb,
                                  amrex::Real rho_p, amrex::Real rho_a, amrex::Real mu_a) noexcept
{
    amrex::Real sum = 0.0, wsum = 0.0;
    const int n = amrex::max(1, amrex::min(nb, DustSettlingConst::MAX_BINS));
    for (int b = 0; b < n; ++b) {
        const amrex::Real w = amrex::max(w_bins[b], amrex::Real(0.0));
        sum  += w * compute_stokes_settling(d_bins[b], rho_p, rho_a, mu_a);
        wsum += w;
    }
    if (!(wsum > 0.0)) {   // no shares given: equal
        sum = 0.0;
        for (int b = 0; b < n; ++b) sum += compute_stokes_settling(d_bins[b], rho_p, rho_a, mu_a);
        return sum / amrex::Real(n);
    }
    return sum / wsum;
}

/**
 * @brief Apply Stokes settling tendency to cc_source for one dust scalar.
 *
 * First-order upwind flux divergence of the downward flux v_s RhoDust, read
 * from the state S_old (NOT from cc_source, which holds the tendency):
 *   F_top(k) = v_s(k+1) RhoDust(k+1)   (enters cell k through its top face; 0 at khi)
 *   F_bot(k) = v_s(k)   RhoDust(k)     (leaves cell k through its bottom face)
 *   tend(k) += (F_top(k) - F_bot(k)) / dz_k
 *
 * At k = klo the bottom-face flux is the deposition to the surface, which
 * apply_dust_deposition_to_cc_source removes with the deposition velocity,
 * so F_bot(klo) = 0 here (the column mass is conserved by this kernel alone).
 * Before 2026-10 the kernel multiplied v_s into the TENDENCY and took the
 * neighbour from below, so the dust never settled and the write raced with
 * the k-1 read; the Stokes column gtest pins the corrected form.
 *
 * v_s is computed per-cell from local air density S_old(Rho_comp), as the
 * mean over the transported bins. h_k = detJ(k) dz is the cell thickness.
 *
 * @param[in,out] cc_source     Cell-centred source MultiFab on atm grid.
 * @param[in]     S_old         Conserved state (air density and the dust density).
 * @param[in]     detJ          Cell volume factor on the atmosphere grid (1 on a flat grid).
 * @param[in]     geom_atm      Atmospheric geometry.
 * @param[in]     d_bins        Diameters of the bins this scalar carries [m].
 * @param[in]     nb            Number of those bins (1 for a single bin).
 * @param[in]     rho_p         Particle density [kg/m^3].
 * @param[in]     dust_comp     cc_source component index for this scalar.
 * @param[in]     dust_debug    Print diagnostics when true.
 */
inline void apply_dust_settling_to_cc_source(
    amrex::MultiFab&       cc_source,
    const amrex::MultiFab& S_old,
    const amrex::MultiFab& detJ,
    const amrex::Geometry& geom_atm,
    const DustBinDiameters& d_bins,
    const DustBinWeights&  w_bins,
    int                    nb,
    amrex::Real            rho_p,
    int                    dust_comp,
    bool                   dust_debug = false)
{
    using namespace amrex;
    const Box& domain = geom_atm.Domain();
    int klo = domain.smallEnd(2);
    int khi = domain.bigEnd(2);
    const Real dz_avg = geom_atm.CellSize(2);

    for (MFIter mfi(cc_source, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto src_arr = cc_source.array(mfi);
        auto s_arr   = S_old.const_array(mfi);
        auto dj_arr  = detJ.const_array(mfi);
        const int comp  = dust_comp;
        const Real rhop = rho_p;
        const DustBinDiameters bins = d_bins;
        const DustBinWeights   w    = w_bins;
        const int nbins = nb;

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real rho_a = s_arr(i, j, k, Rho_comp);
            if (rho_a <= 0.0) rho_a = 1.2;

            // the cell thickness (not the centre-to-centre spacing above it)
            Real dz = dj_arr(i,j,k) * dz_avg;
            if (dz <= 1.0e-10) dz = dz_avg;

            // Settling velocity at this level using local air density.
            Real vs_k = compute_mean_settling(bins, w, nbins, rhop, rho_a,
                                              DustSettlingConst::MU_AIR_STD);

            // Downward flux leaving cell k through its bottom face, from the
            // dust density of the STATE (the surface flux at klo is the
            // deposition kernel's).
            Real flux_bot = (k > klo) ? vs_k * amrex::max(s_arr(i, j, k, comp), Real(0.0)) : Real(0.0);

            // Downward flux entering cell k through its top face, from the
            // cell above (none above the domain top).
            Real flux_top = 0.0;
            if (k < khi) {
                Real rho_a_kp1 = s_arr(i, j, k+1, Rho_comp);
                if (rho_a_kp1 <= 0.0) rho_a_kp1 = 1.2;
                Real vs_kp1 = compute_mean_settling(bins, w, nbins, rhop, rho_a_kp1,
                                                    DustSettlingConst::MU_AIR_STD);
                flux_top = vs_kp1 * amrex::max(s_arr(i, j, k+1, comp), Real(0.0));
            }

            // First-order upwind: tend += (F_top - F_bot) / dz; every read is
            // from S_old, so the write never races with a neighbour's read
            src_arr(i, j, k, comp) += (flux_top - flux_bot) / dz;
        });
    }

    if (dust_debug) {
        Real vs_std = compute_mean_settling(d_bins, w_bins, nb, rho_p, 1.225,
                                            DustSettlingConst::MU_AIR_STD);
        Real tend_max = cc_source.max(dust_comp);
        amrex::Print() << "[DUST SETTLING] " << nb << " bin(s), mean v_s=" << vs_std << " m/s"
                       << "  tend_max=" << tend_max << " [kg/m^3/s]\n";
    }
}

#endif // ERF_DUST_SETTLING_H
