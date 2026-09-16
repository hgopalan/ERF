// Structure ignition and house-to-house spread on the exposure accumulators.
//
// With erf.fire.structures.ignition.enable each structure of the exposure
// diagnostics (a connected footprint of the heightmap, see
// build_structure_ids()) carries a state: unignited, burning or burned out.
// A structure ignites when one of three thresholds on its exposure
// accumulators is met: the largest accumulated heat load in its wall band,
// the number of embers that landed on its footprint, or the time during
// which the largest fireline intensity in its band stayed above a threshold.
//
// A burning structure releases heat on its footprint cells following the
// EN 1991-1-2 Annex E.4 heat-release curve per unit floor area (t-squared
// growth to a peak flux, a plateau until 70 % of the fuel load is consumed,
// a linear decay to zero), so it enters fire_heat_flux and reaches the
// atmosphere through the coupling like the fire's own heat, launches brands
// through the Albini spotting model with an equivalent fireline intensity,
// and radiates onto the cells around it: its radiant fraction is spread from
// each footprint cell as a hemispherical point source, and the incident flux
// adds to the heat-load accumulator of every cell within a cutoff radius.
// House-to-house spread needs no rule of its own: the radiant flux and the
// brands of a burning house load its neighbours' accumulators, and the same
// thresholds fire there.
//
// The per-structure state lives on the fire grid as a four-component field
// (state, ignition time, time above the intensity threshold, cause), which
// the plotfile and the checkpoint carry; the host-side vectors that drive
// the rule are rebuilt from it on a restart.
//
// References:
//   EN 1991-1-2 (2002). Eurocode 1: Actions on structures, Part 1-2: General
//     actions, actions on structures exposed to fire. Annex E (fire load
//     densities, rate of heat release per unit area, t-squared growth).
//   Cohen, J. D. (2004). Relating flame radiation to home ignition using
//     modeling and experimental crown fires. Can. J. For. Res. 34, 1616-1626.
//   Babrauskas, V. (2003). Ignition Handbook. Fire Science Publishers.
//   Masoudvaziri, N., et al. (2021). Streamlined wildland-urban interface fire
//     tracing (SWUIFT). Environmental Modelling & Software 143, 105097.

#ifndef ERF_STRUCTURE_IGNITION_H
#define ERF_STRUCTURE_IGNITION_H

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>
#include <ERF_FireParams.H>
#include <ERF_FireGrid.H>
#include <string>
#include <vector>

namespace erf_structure_ignition {

/// Structure state codes carried in component 0 of the state field.
enum State : int { Unignited = 0, Burning = 1, BurnedOut = 2 };

/// Ignition cause codes carried in component 3 of the state field.
enum Cause : int { None = 0, HeatLoad = 1, Embers = 2, Intensity = 3 };

/// Components of the state field.
enum Comp : int { StateComp = 0, IgnitionTimeComp = 1, TimeAboveComp = 2, CauseComp = 3, NComp = 4 };

inline const char* cause_name (int c) noexcept
{
    switch (c) {
    case HeatLoad:  return "heat";
    case Embers:    return "ember";
    case Intensity: return "intensity";
    default:        return "none";
    }
}

inline const char* state_name (int s) noexcept
{
    switch (s) {
    case Burning:   return "burning";
    case BurnedOut: return "burned_out";
    default:        return "unignited";
    }
}

/**
 * @brief EN 1991-1-2 Annex E.4 heat release per unit floor area [W/m2].
 *
 * Growth q = q_peak (tau / t_growth)^2 until the peak, a plateau at q_peak
 * until 70 % of the fuel load has been released, then a linear decay to zero
 * that releases the remaining 30 %. Zero before ignition and after burnout.
 */
struct BurnCurve
{
    amrex::Real q_peak    = 0.0;   ///< Peak flux [W/m2]
    amrex::Real t_growth  = 0.0;   ///< Time from ignition to the peak [s]
    amrex::Real t_plateau = 0.0;   ///< Duration at the peak [s]
    amrex::Real t_decay   = 0.0;   ///< Duration of the linear decay [s]

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real duration () const noexcept { return t_growth + t_plateau + t_decay; }

    /// Flux at time tau since ignition [W/m2]; non-negative.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real flux_at (amrex::Real tau) const noexcept
    {
        if (tau < 0.0) { return 0.0; }
        if (tau < t_growth) {
            const amrex::Real r = tau / t_growth;
            return q_peak * r * r;
        }
        tau -= t_growth;
        if (tau < t_plateau) { return q_peak; }
        tau -= t_plateau;
        if (tau < t_decay) { return q_peak * (1.0 - tau / t_decay); }
        return 0.0;
    }
};

/**
 * @brief Build the burn curve from the peak flux, the fuel load per unit area
 *        and the time to the peak.
 *
 * The growth phase releases q_peak t_growth / 3, the decay 0.3 of the load;
 * the plateau takes what is left. Returns false when the growth phase alone
 * would release more than 70 % of the load, when no such curve exists.
 */
inline bool make_burn_curve (amrex::Real q_peak, amrex::Real fuel_load_J_m2,
                             amrex::Real t_growth, BurnCurve& c)
{
    if (q_peak <= 0.0 || fuel_load_J_m2 <= 0.0 || t_growth < 0.0) { return false; }
    const amrex::Real e_growth = q_peak * t_growth / 3.0;
    const amrex::Real e_decay  = 0.3 * fuel_load_J_m2;
    if (e_growth > fuel_load_J_m2 - e_decay) { return false; }
    c.q_peak    = q_peak;
    c.t_growth  = t_growth;
    c.t_plateau = (fuel_load_J_m2 - e_decay - e_growth) / q_peak;
    c.t_decay   = 2.0 * e_decay / q_peak;
    return true;
}

/**
 * @brief Incident flux [W/m2] at squared horizontal distance r2 from a
 *        ground-level point source of radiant power P_rad [W]: P / (2 pi r^2)
 *        over the hemisphere above the ground, with r^2 floored at r2_min so a
 *        source's own cell stays finite.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real point_source_incident_flux (amrex::Real P_rad, amrex::Real r2, amrex::Real r2_min) noexcept
{
    const amrex::Real rr = amrex::max(r2, r2_min);
    return P_rad / (2.0 * amrex::Math::pi<amrex::Real>() * rr);
}

/**
 * @brief The ignition rule for one unignited structure.
 *
 * Returns the first criterion met, in the order heat load, embers,
 * intensity, or None. A threshold that is zero or negative is off.
 *
 * @param hl_band_max     Largest accumulated heat load in the wall band [J/m2]
 * @param heat_load_thr   erf.fire.structures.ignition.heat_load_J_m2
 * @param embers          Brands that landed on the footprint
 * @param ember_thr       erf.fire.structures.ignition.ember_count
 * @param t_above         Time the band intensity has spent above its threshold [s]
 * @param intensity_thr   erf.fire.structures.ignition.intensity_kW_m
 * @param residence_thr   erf.fire.structures.ignition.residence_s
 */
inline int ignition_cause (amrex::Real hl_band_max, amrex::Real heat_load_thr,
                           amrex::Real embers, int ember_thr,
                           amrex::Real t_above, amrex::Real intensity_thr, amrex::Real residence_thr) noexcept
{
    if (heat_load_thr > 0.0 && hl_band_max >= heat_load_thr) { return HeatLoad; }
    if (ember_thr > 0 && embers >= static_cast<amrex::Real>(ember_thr)) { return Embers; }
    if (intensity_thr > 0.0 && t_above >= residence_thr && t_above > 0.0) { return Intensity; }
    return None;
}

} // namespace erf_structure_ignition

/**
 * @brief Per-structure ignition state, heat and ember sources, and the
 *        incident radiant flux, owned by the fire layer.
 *
 * The fire layer owns one instance when erf.fire.structures.ignition.enable
 * is set, calls build_footprints() once the structure ids exist,
 * apply_heat_sources() after its own heat flux is computed on every step,
 * and update_state() at the end of the step; the checkpoint writes and reads
 * state_field() and calls restore_from_field() afterwards. All collective
 * calls (build, restore, update) must be made on every rank.
 *
 * Every method that launches a kernel is public: nvcc rejects an extended
 * device lambda inside a private or protected member function.
 */
class StructureIgnition
{
public:
    StructureIgnition (const FireGrid& fg,
                       const FireParams::StructureParams::IgnitionParams& params,
                       int n_structures, int ring, bool debug);

    /// Number of structures (ids run 1..N).
    int n_structures () const { return m_n; }

    /// The four-component state field on the fire grid (see erf_structure_ignition::Comp).
    const amrex::MultiFab* state_field () const { return m_state.get(); }
    amrex::MultiFab*       state_field_mut ()   { return m_state.get(); }

    /// Incident radiant flux from burning structures on every fire cell [W/m2].
    const amrex::MultiFab* rad_flux () const { return m_rad_flux.get(); }

    /// Equivalent fireline intensity of the burning footprint cells [kW/m]
    /// handed to the spotting model as extra launch sites; zero elsewhere.
    const amrex::MultiFab* launch_intensity () const { return m_launch_intensity.get(); }

    /// Heat release curve every structure follows once ignited.
    const erf_structure_ignition::BurnCurve& curve () const { return m_curve; }

    int         state_of      (int s) const { return m_h_state[s]; }
    amrex::Real ignition_time (int s) const { return m_h_t_ign[s]; }
    amrex::Real time_above    (int s) const { return m_h_t_above[s]; }
    int         cause_of      (int s) const { return m_h_cause[s]; }
    /// The structure's heat release at time t [W/m2] (zero unless burning).
    amrex::Real flux_of       (int s, amrex::Real time) const;

    int n_ignited () const;      ///< burning or burned out
    int n_burning () const;
    int n_burned_out () const;

    /// Collective. Gather the footprint cells of every structure from the id
    /// field (each rank owns disjoint boxes), so the radiation sources of a
    /// burning structure are known on every rank.
    void build_footprints (const amrex::MultiFab& structure_id);

    /// Collective. Rebuild the host-side state from the state field after a
    /// checkpoint restore (every footprint cell of a structure holds the same
    /// values; the maximum over the footprint is reduced across ranks).
    void restore_from_field (const amrex::MultiFab& structure_id);

    /// Add the burning structures' heat release to heat_flux on their
    /// footprint cells, fill launch_intensity(), and rebuild rad_flux() from
    /// the radiant fraction of that release as ground point sources.
    void apply_heat_sources (amrex::Real time, amrex::MultiFab& heat_flux,
                             const amrex::MultiFab& structure_id);

    /// Collective. Evaluate the ignition rule on the accumulators at the end
    /// of a step (time_new = time + dt), retire structures whose curve has
    /// ended, and write the state field.
    void update_state (amrex::Real time_new, amrex::Real dt,
                       const amrex::MultiFab& structure_id,
                       const amrex::MultiFab& nonburnable,
                       const amrex::MultiFab& heat_load,
                       const amrex::MultiFab& fireline_intensity,
                       const amrex::MultiFab& ember_landings);

    /// Write the host-side vectors into the state field on every footprint cell.
    void write_state_field (const amrex::MultiFab& structure_id);

private:
    const FireGrid&                                  m_fg;
    FireParams::StructureParams::IgnitionParams      m_p;
    int                                              m_n    = 0;
    int                                              m_ring = 1;
    bool                                             m_debug = false;
    erf_structure_ignition::BurnCurve                m_curve;

    std::unique_ptr<amrex::MultiFab>                 m_state;             // NComp comps, no ghosts
    std::unique_ptr<amrex::MultiFab>                 m_rad_flux;          // [W/m2]
    std::unique_ptr<amrex::MultiFab>                 m_launch_intensity;  // [kW/m]

    // Host-side per-structure state, index 0 unused; identical on every rank.
    std::vector<int>                                 m_h_state;
    std::vector<amrex::Real>                         m_h_t_ign;
    std::vector<amrex::Real>                         m_h_t_above;
    std::vector<int>                                 m_h_cause;

    // Footprint cells per structure (fire-grid indices), the same on every rank.
    std::vector<std::vector<int>>                    m_foot_i;
    std::vector<std::vector<int>>                    m_foot_j;
};

#endif
