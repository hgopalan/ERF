/**
 * @file ERF_DustPM.H
 * @brief Particulate matter size fractionation for EPA NAAQS compliance.
 *
 * Classifies dust size bins into PM2.5 and PM10 fractions based on
 * aerodynamic diameter. The classification follows US EPA convention:
 *
 *   PM2.5: aerodynamic diameter <= 2.5 µm
 *   PM10:  aerodynamic diameter <= 10 µm (includes PM2.5)
 *
 * A bin with diameter d [m] contributes its full mass to the PM class
 * if d <= threshold. No partial fractional attribution is applied
 * within a bin. With the bins transported as ONE scalar (the only layout
 * for n_size_bins > 1) the scalar's mass is apportioned by the bins' shares
 * of the emitted flux, which are equal: PM10 = C * n(d <= 10 um) / n_bins,
 * PM2.5 = C * n(d <= 2.5 um) / n_bins. Until October 2026 the bin-0 diameter
 * classified the whole lumped mass, so with the default bins {7, 2.5, 50} um
 * PM10 counted the 50 um third and PM2.5 was identically zero.
 *
 * The 24-hour averages are block means over a 24-slot ring of hourly means
 * (erf.dust.averaging = window, the 40 CFR 50 Appendix K/N form); the
 * exponential running mean used until October 2026 (averaging = exponential)
 * reports 0.632 C after 24 h of a constant C, so a 50 ug/m3 day never flagged
 * the 35 ug/m3 PM2.5 standard.
 *
 * References:
 *   US EPA (2024). NAAQS for Particulate Matter.
 *     https://www.epa.gov/pm-pollution/
 *     national-ambient-air-quality-standards-naaqs-particulate-matter-pm
 *   40 CFR Part 50.
 *     https://www.ecfr.gov/current/title-40/chapter-I/subchapter-C/part-50
 *   Watson, J.G. et al. (1994). Atmos. Environ., 28, 2493.
 *     https://doi.org/10.1016/1352-2310(94)90400-6
 */

#ifndef ERF_DUST_PM_H
#define ERF_DUST_PM_H

#include <vector>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>
#include <AMReX_MFIter.H>

namespace DustPMConst {
    constexpr amrex::Real PM25_THRESHOLD_M  = 2.5e-6; ///< 2.5 µm in meters
    constexpr amrex::Real PM10_THRESHOLD_M  = 10.0e-6; ///< 10 µm in meters
    constexpr amrex::Real UG_PER_KG         = 1.0e9;  ///< kg/m^3 -> µg/m^3
    constexpr amrex::Real PM25_24H_NAAQS    = 35.0;   ///< µg/m^3, 40 CFR 50.18
    constexpr amrex::Real PM10_24H_NAAQS    = 150.0;  ///< µg/m^3, 40 CFR 50.6
}

/**
 * @brief Returns true when bin diameter contributes to PM2.5.
 * @param d_m  Bin aerodynamic diameter [m].
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool is_pm25(amrex::Real d_m) noexcept {
    return d_m <= DustPMConst::PM25_THRESHOLD_M;
}

/**
 * @brief Returns true when bin diameter contributes to PM10.
 * @param d_m  Bin aerodynamic diameter [m].
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool is_pm10(amrex::Real d_m) noexcept {
    return d_m <= DustPMConst::PM10_THRESHOLD_M;
}

/**
 * @brief Accumulate PM2.5 and PM10 concentrations from per-bin dust_conc_sfc.
 *
 * Loops over active bins and sums mass into pm25 and pm10 MultiFabs [µg/m^3].
 * Both MultiFabs are zeroed on entry and filled by this function.
 *
 * @param[out] pm25         PM2.5 concentration [µg/m^3] on dust grid.
 * @param[out] pm10         PM10 concentration [µg/m^3] on dust grid.
 * @param[in]  dust_conc    Near-surface dust concentration [kg/m^3] on dust grid.
 *                          When transport_bins_separately=false this is a single
 *                          component representing the total dust (all bins).
 *                          When transport_bins_separately=true, comp b holds bin b.
 * @param[in]  bin_diameters  Vector of bin diameters [m].
 * @param[in]  n_active     Number of active bins (1 or n_size_bins).
 */
inline void compute_pm_concentrations(
    amrex::MultiFab&       pm25,
    amrex::MultiFab&       pm10,
    const amrex::MultiFab& dust_conc,
    const std::vector<amrex::Real>& bin_diameters,
    int n_active,
    bool lumped = false)
{
    pm25.setVal(0.0);
    pm10.setVal(0.0);
    const amrex::Real scale = DustPMConst::UG_PER_KG;

    if (lumped) {
        // one scalar carrying every bin with an equal share of the flux
        const int nb = amrex::max(1, (int)bin_diameters.size());
        int n25 = 0, n10 = 0;
        for (auto d : bin_diameters) { if (is_pm25(d)) ++n25; if (is_pm10(d)) ++n10; }
        const amrex::Real f25 = amrex::Real(n25) / amrex::Real(nb);
        const amrex::Real f10 = amrex::Real(n10) / amrex::Real(nb);
        for (amrex::MFIter mfi(dust_conc, amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            auto conc = dust_conc.const_array(mfi);
            auto p25  = pm25.array(mfi);
            auto p10  = pm10.array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                amrex::Real c = amrex::max(conc(i,j,k,0), amrex::Real(0.0)) * scale;
                p25(i,j,k) = f25 * c;
                p10(i,j,k) = f10 * c;
            });
        }
        return;
    }

    for (int b = 0; b < n_active; ++b) {
        int d_idx = (b < (int)bin_diameters.size())
                  ? b : (int)bin_diameters.size() - 1;
        amrex::Real d_m = bin_diameters[d_idx];

        bool contrib_pm25 = is_pm25(d_m);
        bool contrib_pm10 = is_pm10(d_m);
        if (!contrib_pm25 && !contrib_pm10) continue;

        for (amrex::MFIter mfi(dust_conc, amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            auto conc = dust_conc.const_array(mfi);
            auto p25  = pm25.array(mfi);
            auto p10  = pm10.array(mfi);
            const int src_comp = b;
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                amrex::Real c = amrex::max(static_cast<amrex::Real>(conc(i,j,k,src_comp)), static_cast<amrex::Real>(0.0)) * scale;
                if (contrib_pm25) p25(i,j,k) += c;
                if (contrib_pm10) p10(i,j,k) += c;
            });
        }
    }
}

/**
 * @brief Block mean over a window: a ring of n_slots slot means plus the
 *        running accumulation of the current slot.
 *
 * State: ring (n_slots components, the completed slot means), accum (one
 * component, integral of C over the open slot), slot_elapsed [s], slot_index
 * (next ring entry to overwrite), n_filled (completed slots, <= n_slots).
 * The mean is (sum of the filled slot means times the slot length + the open
 * accumulation) over the covered time, so it is exact for a constant and
 * equals the window mean once n_slots slots are filled. A step longer than a
 * slot closes that slot with the step's value.
 */
struct DustWindowMean
{
    amrex::MultiFab ring;      ///< integral of C over each completed slot [n_slots comps]
    amrex::MultiFab accum;     ///< integral of C over the open slot
    amrex::Vector<amrex::Real> slot_len;   ///< seconds each completed slot covered (slot_dur, or the step that closed it)
    amrex::Real slot_elapsed = 0.0;
    int         slot_index   = 0;
    int         n_filled     = 0;
    int         n_slots      = 0;
    amrex::Real slot_dur     = 0.0;

    void define (const amrex::BoxArray& ba, const amrex::DistributionMapping& dm,
                 int nslots, amrex::Real window_s)
    {
        n_slots = nslots; slot_dur = window_s / nslots;
        ring.define(ba, dm, nslots, 0); ring.setVal(0.0);
        accum.define(ba, dm, 1, 0); accum.setVal(0.0);
        slot_len.assign(nslots, 0.0);
        slot_elapsed = 0.0; slot_index = 0; n_filled = 0;
    }

    /// Seconds the completed slots and the open slot cover together.
    amrex::Real covered () const
    {
        amrex::Real c = slot_elapsed;
        for (int s = 0; s < n_filled && s < n_slots; ++s) c += slot_len[s];
        return c;
    }

    /// Add C_now over dt and write the current window mean into C_mean.
    void update (const amrex::MultiFab& C_now, amrex::Real dt, amrex::MultiFab& C_mean)
    {
        for (amrex::MFIter mfi(accum, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            auto a = accum.array(mfi);
            auto c = C_now.const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                a(i,j,k) += c(i,j,k) * dt;
            });
        }
        slot_elapsed += dt;
        if (slot_elapsed >= slot_dur * (1.0 - 1.0e-9)) {
            // close the slot with the time it covered (a step longer than a
            // slot closes it late; the slot keeps that longer span)
            const int idx = slot_index;
            for (amrex::MFIter mfi(accum, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.tilebox();
                auto a = accum.array(mfi);
                auto r = ring.array(mfi);
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    r(i,j,k,idx) = a(i,j,k);
                    a(i,j,k) = 0.0;
                });
            }
            slot_len[idx] = slot_elapsed;
            slot_index = (slot_index + 1) % n_slots;
            n_filled = amrex::min(n_filled + 1, n_slots);
            slot_elapsed = 0.0;
        }
        const amrex::Real cov = covered();
        const amrex::Real inv_cov = (cov > 0.0) ? 1.0 / cov : 0.0;
        const int nf = n_filled, ns = n_slots;
        for (amrex::MFIter mfi(C_mean, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            auto m = C_mean.array(mfi);
            auto r = ring.const_array(mfi);
            auto a = accum.const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                amrex::Real sum = a(i,j,k);
                for (int s = 0; s < nf && s < ns; ++s) sum += r(i,j,k,s);
                m(i,j,k) = sum * inv_cov;
            });
        }
    }
};

/**
 * @brief Update running time average using an exponential moving average
 *        (erf.dust.averaging = exponential, the form until October 2026).
 *
 * C_avg(t) = C_avg(t-dt) * (T - dt) / T  +  C_now * dt / T
 *
 * The average starts from zero (the first call is after the first step, so
 * the former "initialise to C_now at step 0" branch never ran). A dt above T
 * clamps the weight so the average stays in [0, max C].
 *
 * @param[in,out] C_avg   Running average MultiFab [µg/m^3].
 * @param[in]     C_now   Current concentration [µg/m^3].
 * @param[in]     dt      Timestep [s].
 * @param[in]     T_avg   Averaging period [s] (3600 or 86400).
 */
inline void update_running_average(
    amrex::MultiFab& C_avg,
    const amrex::MultiFab& C_now,
    amrex::Real dt, amrex::Real T_avg)
{
    amrex::Real alpha = amrex::min(dt / T_avg, amrex::Real(1.0));
    amrex::Real beta  = 1.0 - alpha;

    for (amrex::MFIter mfi(C_avg, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto avg = C_avg.array(mfi);
        auto now = C_now.const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            avg(i,j,k) = beta * avg(i,j,k) + alpha * now(i,j,k);
        });
    }
}

/**
 * @brief Compute exceedance flag: 1 where C_avg > threshold, 0 otherwise.
 *
 * @param[out] flag      Exceedance MultiFab [0/1].
 * @param[in]  C_avg     Running average [µg/m^3].
 * @param[in]  threshold NAAQS threshold [µg/m^3].
 */
inline void compute_exceedance_flag(
    amrex::MultiFab& flag,
    const amrex::MultiFab& C_avg,
    amrex::Real threshold)
{
    for (amrex::MFIter mfi(flag, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto fl  = flag.array(mfi);
        auto avg = C_avg.const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            fl(i,j,k) = (avg(i,j,k) > threshold) ? 1.0 : 0.0;
        });
    }
}

#endif  // ERF_DUST_PM_H