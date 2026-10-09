// One-way fire-to-atmosphere coupling: distribute fire heat flux through
// the atmospheric column and inject into cc_source.
//
// This mirrors the WRF-Fire approach in module_fr_sfire_atm.F
// (subroutine fire_tendency), which:
//   1. Builds a vertical flux profile with exponential decay:
//        hfx(z) = (Q_sfc / Cp) * exp(-z / alfg)
//      where z is the height above the lowest cell centre of the column (half
//      a cell above the terrain: the profile is evaluated at cell centres and
//      the first cell's share is e^{-dz/2a} - e^{-3dz/2a}, 0.100 for dz = 5 m
//      and a = 45 m, against 1 - e^{-dz/a} = 0.105 from the ground; the column
//      total is the same) and alfg is the e-folding scale [m].
//   2. Converts the flux profile to a theta tendency via flux divergence:
//        d(RhoTheta)/dt = -rho * d(hfx)/dz
//   3. Does the same for latent heat (grnqfx -> rqvfrten):
//        qfx(z) = (Q_lat / Lv) * exp(-z / alfg)
//        d(RhoQ1)/dt   = -rho * d(qfx)/dz
//
// The e-folding scale alfg controls how deep into the atmosphere the fire
// heating penetrates. With alfg = 45 m, the flux has decayed to 37% at 45 m,
// 5% at 135 m, and 1% at 225 m above the surface.
//
// Reference:
//   Mandel, J. et al. (2011). A wildland fire model with data assimilation.
//   Mathematics and Computers in Simulation, 79(3), 584-606.
//   WRF-SFIRE source: phys/module_fr_sfire_atm.F, subroutine fire_tendency.
//
// Coupling time lag:
//   ERF uses one-step explicit lag: the fire heat flux computed at the end of
//   step n is held in Q_atm_prev and injected into cc_source during step n+1,
//   from the slow right-hand side of every RK stage (ERF_TI_slow_rhs_pre.H),
//   after make_sources has rebuilt cc_source for that stage.
//   This is consistent with WRF-Fire's explicit coupling approach.
//
// RK-stage safety:
//   apply_fire_tendency_to_cc_source is invoked once per RK stage of the
//   slow-RHS integrator. Because the lagged Q_atm_prev is fixed for the whole
//   timestep, the fire tendency it produces must also be identical at every
//   RK stage. With erf.fire.source_mode = overwrite (the default) we OVERWRITE
//   cc_source[RhoTheta_comp] (and RhoQ1_comp when latent heat is on): the
//   values are cleared to zero at the top of the routine and then set via '='
//   inside the kernel, which discards every other source make_sources put
//   there (radiative heating, Rayleigh damping, nudging). source_mode = add
//   adds to the rebuilt source instead. Accumulating with '+=' across RK
//   stages without the per-stage rebuild would triple the injected heat.
//
// Latent heat:
//   WRF-Fire computes grnqfx (ground latent heat flux) alongside grnhfx and
//   passes both through fire_tendency. ERF-Fire follows the same approach:
//   Q_latent = Lv * (bmst + (1 - bmst) * f_water) * (dmass/dt)
//   with bmst = M_f / (1 + M_f) the fuel-moisture share and f_water = 0.56 the
//   water of combustion per unit dry mass (compute_fire_latent_flux). This is
//   stored as fire_latent_flux [W/m2] and distributed with the same
//   exponential profile.
//
// Assumptions:
//   - The tendency goes into the source of the level the fire grid refines
//     (erf.fire.anchor_level, the finest level by default); coarser levels see
//     the result through average-down with erf.coupling_type = TwoWay (the
//     default), not with OneWay; finer levels not at all.
//   - fire_heat_flux and fire_latent_flux are on the fire grid (C*nx x C*ny x 1
//     over the fire region).
//   - The coarsening from fire grid to atmospheric grid is the area average of
//     average_down(), with the region's lower corner added (FireGrid::atm_lo).
//   - z_phys_cc is the cell-centre terrain-following height MultiFab [m].
//   - The atmospheric BoxArray is not split in z (max_grid_size_z = nz enforced).
//   - Latent heat injection into RhoQ1_comp is skipped when moisture is inactive.
//   - fire_atm_feedback multiplier (equivalent to WRF-Fire's fire_atm_feedback)
//     scales both sensible and latent flux. Set to 0 for one-way coupling only
//     (fire responds to wind but atmosphere does not respond to fire heat).
//   - The RhoTheta / RhoQ1 slots of cc_source are OWNED by the fire coupling
//     for the duration of a timestep. If any other physics writes into those
//     same slots per stage, this routine must be moved out of the per-stage
//     slow-RHS path and called once before mri_integrator.advance() instead.

#ifndef ERF_FIRE_ATM_COUPLING_H
#define ERF_FIRE_ATM_COUPLING_H

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_Box.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFabUtil.H>
#include <cmath>
#include <limits>
#include <ERF_Constants.H>
#include <ERF_IndexDefines.H>
#include <ERF_EOS.H>
#include <ERF_FireGrid.H>

// Default e-folding height for ground fire heat distribution [m].
// WRF-Fire default: alfg ~ 45-75 m. At z = alfg: flux decays to 1/e ~ 37%.
// The crown-fire release is added to the same surface flux and injected with
// this one profile; there is no separate canopy e-folding height.
constexpr amrex::Real FIRE_ATM_ALFG_DEFAULT = 45.0;

// Coarsen a fire-grid field (C*nx x C*ny x 1) onto the atmospheric columns of
// the level the fire grid refines (nx x ny x 1) by area averaging.
//
// Q_atm_out lives on the k = 0 slabs of that level's boxes, in their order and
// with their owners; Q_fire on the fire grid, whose index space starts at the
// region's corner. On level 0 the corner is the origin and this is
// amrex::average_down() with ratio (C, C, 1), as it always was. Elsewhere the
// same average is taken with the corner added: fire cells (i - lo_x) C + r,
// (j - lo_y) C + q for r, q in [0, C) average into atmospheric cell (i, j).
// average_down() itself would place the result at the fire grid's own indices.
inline void coarsen_fire_flux_to_atm(
    amrex::MultiFab& Q_atm_out,
    const amrex::MultiFab& Q_fire,
    const amrex::Geometry& geom_atm,
    const FireGrid& fg)
{
    const int C = fg.C;
    if (fg.atm_lo == amrex::IntVect::TheZeroVector()) {
        amrex::average_down(Q_fire, Q_atm_out, fg.geom, geom_atm, 0, Q_fire.nComp(),
                            amrex::IntVect(C, C, 1));
        return;
    }

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        Q_fire.boxArray().size() == Q_atm_out.boxArray().size() &&
        Q_fire.DistributionMap() == Q_atm_out.DistributionMap(),
        "[FIRE] coarsen_fire_flux_to_atm: the fire grid and the atmospheric slabs must pair box for box");
    AMREX_ALWAYS_ASSERT(Q_atm_out.nComp() >= Q_fire.nComp());

    const int lo_x  = fg.atm_lo[0];
    const int lo_y  = fg.atm_lo[1];
    const int ncomp = Q_fire.nComp();
    const amrex::Real volfrac = amrex::Real(1.0) / amrex::Real(C * C);
    for (amrex::MFIter mfi(Q_atm_out); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto crse = Q_atm_out.array(mfi);
        auto fine = Q_fire.const_array(mfi);
        amrex::ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            const int ii = (i - lo_x) * C;
            const int jj = (j - lo_y) * C;
            amrex::Real c = 0.0;
            for (int q = 0; q < C; ++q) {
                for (int r = 0; r < C; ++r) {
                    c += fine(ii + r, jj + q, 0, n);
                }
            }
            crse(i, j, k, n) = volfrac * c;
        });
    }
}

// Compute fire latent heat flux [W/m2] from sensible heat flux and fuel moisture.
//
// WRF-SFIRE's bookkeeping (module_fr_sfire_atm.F): the burn rate Q/h is read as
// a wet-mass rate, of which the share b = M/(1 + M) is fuel water and the rest
// dry fuel yielding f_w = 0.56 kg of water of combustion per kg. The loads the
// fuel models carry are oven-dry, for which the water released per unit dry
// fuel is M + 0.56 and the factor would be (M + 0.56) instead of b + (1 - b)
// 0.56: 0.640 against 0.593 at M = 0.08 (7 %) and 0.860 against 0.662 at
// M = 0.30 (23 %). The SFIRE form is kept for parity with WRF-SFIRE and the
// Community Fire Behavior Model; fire_coupling.rst states the basis.
inline void compute_fire_latent_flux(
    amrex::MultiFab& Q_lat_out,
    const amrex::MultiFab& Q_sfc,
    amrex::Real M_f,
    amrex::Real h_fuel_Jkg)
{
    if (h_fuel_Jkg <= 0.0) {
        Q_lat_out.setVal(0.0);
        return;
    }

    constexpr amrex::Real L_V = 2.501e6;
    amrex::Real bmst = M_f / (1.0 + M_f);
    constexpr amrex::Real f_water = 0.56;
    amrex::Real ratio = L_V / h_fuel_Jkg;

    for (amrex::MFIter mfi(Q_sfc, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const& q_sfc_a = Q_sfc.array(mfi);
        auto const& q_lat_a = Q_lat_out.array(mfi);
        auto const& bx = mfi.tilebox();
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(const amrex::IntVect& iv) {
            amrex::Real factor = ratio * (bmst + (1.0 - bmst) * f_water);
            q_lat_a(iv) = factor * q_sfc_a(iv);
        });
    }
}

// Distribute fire surface heat flux through the atmospheric column and write
// the resulting tendency into cc_source.
//
// Behaviour: OVERWRITES cc_source[RhoTheta_comp] and (when latent heat is on)
// cc_source[RhoQ1_comp]. See file-level "RK-stage safety" note above.
//
// Optional fire_debug argument prints per-call tendency diagnostics to stdout:
//   [FIRE COUPLING] RhoTheta tendency sum = ...  max = ... [kg K m^-3 s^-1]
//   expected_surface_max = Q_max / (Cp * 1.2) * (1 - exp(-dz/alfg)) / dz
//   (the printed estimate uses rho = 1.2 and the default heat_tendency_density
//   = false form, d(rho theta)/dt = -(1/Cp) dQ/dz; with density on it would
//   carry an extra factor rho)
//
// With Q_max=14330 W/m2, Cp=1004.64, alfg=45 m, dz=5 m (the 5 m fire decks):
//   expected_surface_max ~ 0.25 K m^-3 s^-1 (kg K m^-3 s^-1 with density on)
inline void apply_fire_tendency_to_cc_source(
    amrex::MultiFab& cc_source,
    const amrex::MultiFab& Q_atm,
    const amrex::MultiFab* Q_lat_atm,
    const amrex::MultiFab& z_phys_cc,
    const amrex::MultiFab& S_old,
    const amrex::Geometry& geom_atm,
    amrex::Real alfg,
    amrex::Real feedback,
    bool has_moisture,
    bool fire_debug = false,
    bool additive = false,
    const amrex::MultiFab* open_frac_atm = nullptr,
    const amrex::MultiFab* roof_h_atm = nullptr,
    bool use_open_weights = false,
    bool density_factor = true,
    bool exner_factor = false)
{
    using namespace amrex;

    if (alfg <= 0.0) { return; }
    if (feedback <= 0.0) { return; }

    constexpr Real Cp_fire = 1004.64;
    constexpr Real L_V  = 2.501e6;
    // Exner function of the cell from its rho theta (erf.fire.heat_tendency_exner):
    // heating q [W/m3] gives d(theta)/dt = q / (rho cp Pi), so the rho theta
    // tendency carries 1/Pi; without it the enthalpy injected is Pi times the
    // flux, 4.5 % short at 850 hPa. WRF-SFIRE omits the factor; the default here
    // does too.
    const Real rdOcp = R_d / Cp_d;

    const Box& domain = geom_atm.Domain();
    int klo = domain.smallEnd(2);
    int khi = domain.bigEnd(2);
    const auto& dx = geom_atm.CellSize();
    Real dz_avg = dx[2];

    const bool has_latent_global = (Q_lat_atm != nullptr) && has_moisture;
    // The open fraction and roof height are handed in whenever structures are
    // configured, for the diagnostics; the weights only use them when asked.
    const bool have_open = (open_frac_atm != nullptr) && (roof_h_atm != nullptr);
    const bool use_open  = have_open && use_open_weights;

    // Debug only: the fire tendency alone, times dz (in additive mode
    // cc_source also holds the other sources), for the energy check below.
    MultiFab tend_dbg;
    if (fire_debug) {
        tend_dbg.define(cc_source.boxArray(), cc_source.DistributionMap(), 1, 0);
        tend_dbg.setVal(0.0);
    }

    // Legacy (overwrite) mode: zero the components we own before writing them,
    // so any state left over from a previous stage does not leak through. In
    // additive mode make_sources has just rebuilt the source for this stage,
    // so the fire tendency is accumulated once per stage on top of it.
    if (!additive) {
        cc_source.setVal(0.0, RhoTheta_comp, 1, cc_source.nGrowVect());
        if (has_latent_global) {
            cc_source.setVal(0.0, RhoQ1_comp, 1, cc_source.nGrowVect());
        }
    }

    for (MFIter mfi(cc_source, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const& cc_src_a = cc_source.array(mfi);
        auto const& q_atm_a  = Q_atm.array(mfi);
        auto const& z_cc_a   = z_phys_cc.array(mfi);
        auto const& s_old_a  = S_old.array(mfi);

        const Box& bx = mfi.tilebox();

        const bool has_latent = has_latent_global;
        Array4<Real const> q_lat_arr;
        if (has_latent) {
            q_lat_arr = Q_lat_atm->const_array(mfi);
        }
        Array4<Real const> open_arr, roof_arr;
        if (use_open) {
            open_arr = open_frac_atm->const_array(mfi);
            roof_arr = roof_h_atm->const_array(mfi);
        }
        Array4<Real> dbg_arr;
        if (fire_debug) { dbg_arr = tend_dbg.array(mfi); }

        // Column normalisation of the open-fraction weights, once per column
        // on the tile's k = klo slab (it was recomputed inside every cell of
        // the column, O(nz) per cell, three times a step).
        FArrayBox norm_fab;
        Elixir norm_eli;   // keeps the slab alive until the kernels below have read it
        Array4<Real> norm_arr;
        if (use_open) {
            const Box slab = makeSlab(bx, 2, klo);
            norm_fab.resize(slab, 1, The_Arena());
            norm_eli = norm_fab.elixir();
            norm_arr = norm_fab.array();
            amrex::ParallelFor(slab, [=] AMREX_GPU_DEVICE (int i, int j, int /*kslab*/) noexcept {
                const Real f_open = open_arr(i, j, 0);
                const Real h_roof = roof_arr(i, j, 0);
                const Real z_sfc  = z_cc_a(i, j, klo);
                Real norm = 0.0;
                for (int kk = klo; kk <= khi; ++kk) {
                    const Real zz    = z_cc_a(i, j, kk) - z_sfc;
                    const Real zzp1  = (kk < khi) ? (z_cc_a(i, j, kk + 1) - z_sfc) : (zz + dz_avg);
                    Real dzz = (kk < khi) ? (z_cc_a(i, j, kk + 1) - z_cc_a(i, j, kk)) : dz_avg;
                    if (dzz <= 1.0e-10) { dzz = dz_avg; }
                    Real s = 1.0;
                    if (h_roof > 0.0) {
                        const Real zlo2 = zz - 0.5 * dzz;
                        const Real zhi2 = zz + 0.5 * dzz;
                        if (zhi2 <= h_roof)     { s = f_open; }
                        else if (zlo2 < h_roof) { s = f_open + (1.0 - f_open) * (zhi2 - h_roof) / dzz; }
                    }
                    norm += (std::exp(-zz / alfg) - std::exp(-zzp1 / alfg)) * s;
                }
                norm_arr(i, j, klo) = norm;
            });
        }

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(const IntVect& iv) {
            int i = iv[0];
            int j = iv[1];
            int k = iv[2];

            Real q_sfc_sensible = feedback * q_atm_a(i, j, 0);
            Real q_sfc_latent = 0.0;
            if (has_latent) {
                q_sfc_latent = feedback * q_lat_arr(i, j, 0);
            }

            Real z_sfc  = z_cc_a(i, j, klo);
            Real z_k    = z_cc_a(i, j, k) - z_sfc;
            Real exp_k  = std::exp(-z_k / alfg);

            Real z_kp1  = (k < khi) ? (z_cc_a(i, j, k + 1) - z_sfc) : (z_k + dz_avg);
            Real exp_kp1 = std::exp(-z_kp1 / alfg);

            Real dz = (k < khi) ? (z_cc_a(i, j, k + 1) - z_cc_a(i, j, k)) : dz_avg;
            if (dz <= 1.0e-10) { dz = dz_avg; }

            Real rho = s_old_a(i, j, k, Rho_comp);
            if (rho <= 0.0) { rho = 1.2; }
            // Heating gives d(rho theta)/dt = -(1/Cp) dQ/dz, with no density
            // factor. The legacy form carries an extra rho, so its injected
            // energy is rho times the flux; density_factor = false drops it.
            // The Exner factor divides the tendency by Pi(p) so that the
            // enthalpy, not rho theta, integrates to the flux.
            Real rho_fac = density_factor ? rho : 1.0;
            if (exner_factor) {
                const Real rhotheta = amrex::max(s_old_a(i, j, k, RhoTheta_comp), Real(1.0e-10));
                // the moist pressure where there is vapour (0.3 % in Pi at 15 g/kg)
                const Real qv = has_moisture ? amrex::max(s_old_a(i, j, k, RhoQ1_comp), Real(0.0)) / rho : Real(0.0);
                const Real p = getPgivenRTh(rhotheta, qv);
                const Real exner = std::pow(p / p_0, rdOcp);
                rho_fac /= amrex::max(exner, Real(1.0e-6));
            }

            // Layer weight: the fraction of the column's exponential profile
            // that falls in this cell. Without structures this is the flux
            // difference form used since the coupling was written. With the
            // open-fraction option, cells below the sampled roof height carry
            // only their open share and the column is renormalised over the
            // cells that remain, so the injected energy per column is unchanged
            // and none of it lands inside a building.
            Real w_k = exp_k - exp_kp1;
            if (use_open) {
                const Real f_open = open_arr(i, j, 0);
                const Real h_roof = roof_arr(i, j, 0);
                // Open share of this cell: fully open above the roof, f_open
                // below it, linear across the cell that holds the roof.
                Real share = 1.0;
                if (h_roof > 0.0) {
                    const Real z_lo = z_k - 0.5 * dz;
                    const Real z_hi = z_k + 0.5 * dz;
                    if (z_hi <= h_roof)      { share = f_open; }
                    else if (z_lo < h_roof)  { share = f_open + (1.0 - f_open) * (z_hi - h_roof) / dz; }
                }
                // Column normalisation over the open shares of the same
                // profile, precomputed per column above.
                const Real norm = norm_arr(i, j, klo);
                // The legacy profile integrates to (1 - exp(-z_top/alfg)); keep
                // that total so the two modes inject the same energy per column.
                const Real z_top   = (z_cc_a(i, j, khi) - z_sfc) + dz_avg;
                const Real legacy  = 1.0 - std::exp(-z_top / alfg);
                w_k = (norm > 1.0e-30) ? w_k * share * legacy / amrex::max(norm, Real(1.0e-30)) : 0.0;
            }

            // Without the open-fraction option the flux-difference expression is
            // evaluated exactly as it always was, so existing results do not
            // move by round-off.
            Real theta_tend, q_tend = 0.0;
            if (use_open) {
                theta_tend = rho_fac * (q_sfc_sensible / Cp_fire) * w_k / dz;
                if (has_latent) { q_tend = rho_fac * (q_sfc_latent / L_V) * w_k / dz; }
            } else {
                Real hfx_k    = (q_sfc_sensible / Cp_fire) * exp_k;
                Real hfx_kp1  = (q_sfc_sensible / Cp_fire) * exp_kp1;
                theta_tend = -rho_fac * (hfx_kp1 - hfx_k) / dz;
                if (has_latent) {
                    Real qfx_k   = (q_sfc_latent / L_V) * exp_k;
                    Real qfx_kp1 = (q_sfc_latent / L_V) * exp_kp1;
                    q_tend  = -rho_fac * (qfx_kp1 - qfx_k) / dz;
                }
            }
            if (additive) {
                cc_src_a(i, j, k, RhoTheta_comp) += theta_tend;
            } else {
                // OVERWRITE (see file-level RK-stage safety note).
                cc_src_a(i, j, k, RhoTheta_comp) = theta_tend;
            }
            if (has_latent) {
                if (additive) {
                    cc_src_a(i, j, k, RhoQ1_comp) += q_tend;
                } else {
                    cc_src_a(i, j, k, RhoQ1_comp) = q_tend;
                }
            }
            if (dbg_arr) { dbg_arr(i, j, k) = theta_tend * dz; }
        });
    }

    // Tendency diagnostics: runs on host after GPU kernel completes.
    // Prints once per apply call (3x per timestep for RK3).
    // Expected rt_tend_max ~ rho * Q_max / Cp * (1 - exp(-dz/alfg)) / dz
    if (fire_debug) {
        Real Q_max        = Q_atm.max(0);
        Real rt_tend_sum  = cc_source.sum(RhoTheta_comp);
        Real rt_tend_max  = cc_source.max(RhoTheta_comp);
        Real expected_max = Q_max / (Cp_fire * 1.2)
                            * (1.0 - std::exp(-dz_avg / alfg)) / dz_avg;
        amrex::Print() << "[FIRE COUPLING] Q_atm_max=" << Q_max
                       << " W/m2  alfg=" << alfg << " m\n"
                       << "[FIRE COUPLING] RhoTheta tendency"
                       << "  sum="      << rt_tend_sum
                       << "  max="      << rt_tend_max
                       << "  expected_surface_max=" << expected_max
                       << " [kg K m^-3 s^-1]\n";

        // Energy check: Cp times the sum over cells of (rho theta tendency) dz
        // per unit area, against the flux handed in. The profile carries
        // 1 - exp(-z_top/alfg) of the flux, essentially all of it for any
        // realistic domain height, so the ratio should be one. The largest
        // fire tendency inside a fully blocked cell below the roof must be
        // zero with the open-fraction option.
        const Real area = dx[0] * dx[1];
        const Real e_out = Cp_fire * tend_dbg.sum(0) * area;
        const Real e_in  = feedback * Q_atm.sum(0) * area;
        // Columns that straddle a footprint (0 < open fraction < 1) are where
        // the placement matters: fully covered columns carry no flux once their
        // fire cells are masked. For those partial columns report the largest
        // share of the column's heating that lands below the roof (the plain
        // profile gives 1 - exp(-H/alfg); the open-fraction weights reduce it
        // by the open fraction), and the largest potential temperature of the
        // state in cells below a roof, which shows heat that earlier steps left
        // inside buildings.
        Real below_roof_max = -1.0, theta_blocked_max = -1.0;
        if (have_open) {
            MultiFab col(open_frac_atm->boxArray(), open_frac_atm->DistributionMap(), 1, 0);
            MultiFab thb(tend_dbg.boxArray(), tend_dbg.DistributionMap(), 1, 0);
            col.setVal(0.0); thb.setVal(0.0);
            for (MFIter mfi(col, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box& bx2 = mfi.tilebox();
                auto const& c   = col.array(mfi);
                auto const& t   = tend_dbg.const_array(mfi);
                auto const& z   = z_phys_cc.const_array(mfi);
                auto const& fo  = open_frac_atm->const_array(mfi);
                auto const& hr  = roof_h_atm->const_array(mfi);
                amrex::ParallelFor(bx2, [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) noexcept {
                    const Real f = fo(i, j, 0);
                    if (f <= 1.0e-6 || f >= 1.0 - 1.0e-6) { return; }
                    Real below = 0.0, total = 0.0;
                    for (int kk = klo; kk <= khi; ++kk) {
                        const Real zk = z(i, j, kk) - z(i, j, klo);
                        total += t(i, j, kk);
                        if (zk < hr(i, j, 0)) { below += t(i, j, kk); }
                    }
                    c(i, j, 0) = (total > 1.0e-30) ? below / amrex::max(total, Real(1.0e-30)) : 0.0;
                });
            }
            for (MFIter mfi(thb, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.tilebox();
                auto const& b   = thb.array(mfi);
                auto const& z   = z_phys_cc.const_array(mfi);
                auto const& so  = S_old.const_array(mfi);
                auto const& fo  = open_frac_atm->const_array(mfi);
                auto const& hr  = roof_h_atm->const_array(mfi);
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    const Real zk = z(i, j, k) - z(i, j, klo);
                    if (fo(i, j, 0) < 1.0 - 1.0e-6 && zk < hr(i, j, 0)) {
                        b(i, j, k) = so(i, j, k, RhoTheta_comp) / so(i, j, k, Rho_comp);
                    }
                });
            }
            below_roof_max    = col.max(0);
            theta_blocked_max = thb.max(0);
        }
        // The ratio divides only when e_in > 0, but the optimiser evaluates the division
        // speculatively for both arms of the selection, and 0/0 on the first lagged step
        // raised the invalid flag that aborts the run under amrex.fpe_trap_invalid; the
        // floored denominator (the smallest normal Real, so it holds in single
        // precision too) keeps the discarded value finite and the printed one unchanged.
        amrex::Print() << "[FIRE COUPLING] energy_in=" << e_in << " W  energy_out=" << e_out
                       << " W  ratio=" << ((e_in > 0.0) ? e_out / amrex::max(e_in, std::numeric_limits<Real>::min()) : 0.0)
                       << "  below_roof_share_max=" << below_roof_max
                       << "  theta_blocked_max=" << theta_blocked_max << "\n";
    }
}

#endif  // ERF_FIRE_ATM_COUPLING_H