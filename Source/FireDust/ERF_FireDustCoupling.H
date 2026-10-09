#ifndef ERF_FIRE_DUST_COUPLING_H
#define ERF_FIRE_DUST_COUPLING_H

#if defined(ERF_ENABLE_FIRE) && defined(ERF_USE_DUST)

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>

/**
 * @brief Fire-Dust coupling interactions.
 *
 * Handles interactions between fire and dust modules, particularly
 * burned area effects on dust emission.
 *
 * The fire level-set phi lives on the fire sub-grid (grid_ratio times
 * finer than the atmosphere) and the dust crust index on the dust
 * sub-grid. The two grid ratios must be equal (asserted at start-up in
 * ERF::InitData_post), and both layers live on level 0 (the dust module's
 * prerequisite): the crust path maps the burned status by physical (x, y)
 * position, the fire-wind and fire-heat paths copy the fire fields onto
 * the dust BoxArray by index overlap (ERF.cpp) and read them cell for cell.
 */
struct FireDustCoupling {
    bool        enabled                   = false;
    amrex::Real post_fire_crust_reduction = 0.8;  // fraction of crust removed in burned cells
    bool        debug                     = false; // erf.dust.dust_debug: per-step coupling prints

    // Fire level set (phi < 0 => burned) copied onto the DUST BoxArray by ERF
    // before every dust step (ParallelCopy with the dust periodicity, the same
    // path as the fire wind and heat). The grid ratios are asserted equal at
    // start-up, so the lookup is cell for cell. Until October 2026 the whole
    // fire level set was gathered into a fire-domain-sized host vector,
    // all-reduced across the ranks and copied to the device every step
    // (134 MB per rank per step at 1024^2 x 4).
    const amrex::MultiFab* fire_phi_scratch = nullptr;

    /**
     * @brief Reduce crust_index in burned cells, lowering u*_t and raising dust emission.
     *
     * For each dust cell we compute its physical centre (x,y), find the
     * containing fire cell, and if fire_phi < 0 there we apply
     *   crust *= (1 - post_fire_crust_reduction),  crust = max(crust, 0).
     *
     * @param[in,out] dust_crust_index  Dust layer crust index MultiFab (dust grid)
     * @param[in]     geom_dust         Geometry of the dust sub-grid
     */
    void apply_burned_area_to_crust(amrex::MultiFab& dust_crust_index) const;

    // ── Phase 2: fire reference wind → dust u* inside the perimeter ──────────
    // When fire_wind_to_dust = true, the fire's wind at erf.fire.wind_ref_ht
    // (fire_wind_ref, the datum fire_wind_zref names; until October 2026 the
    // WAF-reduced midflame wind fire_wind_eff was handed over as if it were
    // the 6.1 m wind, 0.36x for grass, so the coupling never won) is copied
    // onto the dust grid and converted to a friction velocity via log-law.
    // Inside the fire perimeter (level set below zero) the maximum of the
    // surface layer's u* and the fire-derived u* is kept; outside it the
    // surface layer's u* stands (the fire value overrode it domain-wide until
    // October 2026: the reference wind is the same atmospheric wind there and
    // the neutral log law ignores the surface layer's roughness and stability).
    bool        fire_wind_to_dust  = true;
    amrex::Real fire_wind_z0       = 0.1;  // surface roughness for log-law [m]
    amrex::Real fire_wind_zref     = 6.1;  // reference height matching fire wind_ref_ht [m]


    /**
     * @brief Interaction 2: the fire's reference wind gives a log-law u* inside the perimeter.
     *
     * For each dust cell inside the fire perimeter (fire_phi_scratch < 0),
     * reads the fire's reference wind (fire_wind_ref on the dust grid, the
     * grid ratios are equal), derives u* via log-law
     * (u* = U * kappa / ln(zref/z0)) and writes the larger of it and the
     * field's current value into dust_ustar_in. ERF hands the zeroed
     * fire-u* scratch (DustLayer::get_ustar_fire_mut), which advance() maxes
     * into the dust u* after the surface-layer fill.
     *
     * MPI safety: fire_wind_ref_mf must already be copied into a scratch
     * MultiFab on the dust BoxArray/DM by the caller (ERF.cpp) before this
     * function is called — same pattern as LNG_MPI_SKILLS Rule B3. This
     * function receives the pre-copied scratch MultiFab as fire_wind_scratch.
     *
     * @param[in,out] dust_ustar_in      u* on dust grid [m/s] — modified in place
     * @param[in]     fire_wind_scratch  fire_wind_ref already ParallelCopied onto
     *                                   the dust BoxArray/DM (2-comp: u, v) [m/s]
     * @param[in]     geom_dust          Geometry of the dust sub-grid
     * @param[in]     z0                 Surface roughness [m]
     * @param[in]     zref               Reference height [m]
     * @param[in]     C                  Fire-to-dust grid ratio (fire cells per dust cell)
     */
    void apply_fire_wind_to_dust_ustar(
       amrex::MultiFab&       dust_ustar_in,
       const amrex::MultiFab& fire_wind_scratch,
       const amrex::MultiFab& fire_phi_scratch,
       const amrex::Geometry& geom_dust,
       amrex::Real            z0,
       amrex::Real            zref,
       int                    C) const;

    // ── Phase 3: fire heat flux → convective dust lofting ────────────────────
    // Fire heat flux drives a convective updraft above the fire front that lofts
    // dust particles. Enhancement applied to emission flux as:
    //   flux *= (1 + k_loft * max(Q_fire - Q_threshold, 0) / Q_ref)
    // capped at (1 + k_loft). No effect when Q_fire <= Q_threshold.
    bool        fire_lofting_enabled = false;
    amrex::Real lofting_k_loft       = 2.0;
    amrex::Real lofting_Q_threshold  = 50.0;
    amrex::Real lofting_Q_ref        = 500.0;
};

#endif
#endif
