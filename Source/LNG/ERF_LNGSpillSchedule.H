/**
 * @file ERF_LNGSpillSchedule.H
 * @brief Time-varying multi-source LNG spill schedule from CSV.
 * @note Phase 8
 *
 * CSV format (space or comma delimited, # = comment):
 *   event_name  start_time_s  end_time_s  cx_m  cy_m  radius_m  rate_kg_s
 *   spill_1     0.0           -1          1500  1500  10.0      50.0
 *   spill_2     30.0          90.0        1600  1400  5.0       25.0
 *   # end_time_s = -1 means active for entire simulation
 *
 * MPI pattern (LNG_MPI_SKILLS.md Rule B1):
 *   - Rank 0 reads CSV, broadcasts to all ranks
 *   - No IOProcessor guard around the broadcast (collective)
 *   - Pool source contributions summed across all active events per timestep
 *
 * References:
 *   ERF_DustBlastSchedule.H (CSV read + MPI broadcast pattern)
 *   ERF_DustRoadSchedule.H (time-window activation pattern)
 */
#ifndef ERF_LNG_SPILL_SCHEDULE_H
#define ERF_LNG_SPILL_SCHEDULE_H
#ifdef ERF_USE_LNG

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParallelDescriptor.H>
#include <string>
#include <vector>

/// Single spill event definition (POD struct for easy MPI broadcast)
struct LNGSpillEvent {
    char   name[64]     = {};      ///< Event label for diagnostics
    double start_time_s = 0.0;     ///< Activation start [s]
    double end_time_s   = -1.0;    ///< Activation end [s]; -1 = entire simulation
    double cx_m         = 0.0;     ///< Pool centre x [m]
    double cy_m         = 0.0;     ///< Pool centre y [m]
    double radius_m     = 10.0;    ///< Spill radius [m]
    double rate_kg_s    = 0.0;     ///< Release rate [kg/s]
};

/// Container for all events (loaded once at initialize, used every advance)
struct LNGSpillSchedule {
    std::vector<LNGSpillEvent> events;
    bool loaded = false;
};

/// Load spill schedule from CSV. Rank 0 reads, all ranks receive via broadcast.
/// No-op if filename is empty.
void load_lng_spill_schedule(const std::string& filename,
                              LNGSpillSchedule& schedule);

/// Apply all currently active spill events to lng_pool_depth.
/// Time-window check: active if t >= start_time_s AND (end_time_s<0 OR t<=end_time_s).
/// For each active event: apply_spill_source() on its pool region.
/// Additive: multiple events contribute to the same pool depth field.
/// Debug print per active event when lng_debug=true.
void apply_spill_schedule(amrex::MultiFab&       lng_pool_depth,
                           const amrex::Geometry& geom_lng,
                           const LNGSpillSchedule& schedule,
                           amrex::Real            cur_time,
                           amrex::Real            dt,
                           amrex::Real            rho_LNG,
                           bool                   lng_debug);

/// Returns total mass released so far [kg] from all events up to cur_time.
amrex::Real compute_total_released_mass(const LNGSpillSchedule& schedule,
                                         amrex::Real cur_time);

#endif  // ERF_USE_LNG
#endif  // ERF_LNG_SPILL_SCHEDULE_H
