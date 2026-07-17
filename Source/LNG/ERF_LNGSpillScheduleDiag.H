/**
 * @file ERF_LNGSpillScheduleDiag.H
 * @brief CSV diagnostics output for spill schedule (header-only).
 * @note Phase 8
 *
 * Writes per-timestep spill schedule diagnostics including:
 *   - Active event count
 *   - Total mass released so far
 *   - Per-event release rates (0 if inactive)
 *
 * MPI pattern (LNG_MPI_SKILLS.md Rule A4):
 *   - All data is scalar from schedule struct (no MultiFab reductions)
 *   - IOProcessor guard safe for file I/O
 *   - Always write rows (not gated on lng_debug)
 */
#ifndef ERF_LNG_SPILL_SCHEDULE_DIAG_H
#define ERF_LNG_SPILL_SCHEDULE_DIAG_H
#ifdef ERF_USE_LNG

#include "ERF_LNGSpillSchedule.H"
#include <AMReX_Print.H>
#include <AMReX_ParallelDescriptor.H>
#include <fstream>
#include <sstream>
#include <iomanip>

/// Write CSV header for spill schedule diagnostics.
/// Columns: step, time_s, n_active_events, total_released_mass_kg, event_1_rate_kg_s, event_2_rate_kg_s, ...
inline void write_spill_schedule_header(const std::string& filename,
                                         const LNGSpillSchedule& schedule)
{
    if (!amrex::ParallelDescriptor::IOProcessor() || filename.empty()) return;

    std::ofstream file(filename);
    if (!file.is_open()) {
        amrex::Print() << "[LNG] ERROR: cannot open spill schedule diagnostics file: "
                       << filename << "\n";
        return;
    }

    // Write header
    file << "step,time_s,n_active_events,total_released_mass_kg";
    for (int i = 0; i < static_cast<int>(schedule.events.size()); ++i) {
        file << ",event_" << i << "_rate_kg_s";
    }
    file << "\n";

    file.close();
}

/// Append one row of spill schedule diagnostics per timestep.
inline void append_spill_schedule_row(int step, double time_s,
                                      const std::string& filename,
                                      const LNGSpillSchedule& schedule,
                                      amrex::Real cur_time)
{
    if (!amrex::ParallelDescriptor::IOProcessor() || filename.empty()) return;

    // Count active events and compute total mass
    int n_active = 0;
    amrex::Real total_mass = 0.0;

    std::vector<amrex::Real> event_rates(schedule.events.size(), 0.0);

    for (int i = 0; i < static_cast<int>(schedule.events.size()); ++i) {
        const auto& event = schedule.events[i];

        // Check if event is active
        bool is_active = (cur_time >= event.start_time_s) &&
                        (event.end_time_s < 0.0 || cur_time <= event.end_time_s);

        if (is_active) {
            n_active++;
            event_rates[i] = event.rate_kg_s;
        }
    }

    // Total released mass (cumulative)
    total_mass = compute_total_released_mass(schedule, cur_time);

    // Append row to file
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        amrex::Print() << "[LNG] ERROR: cannot open spill schedule diagnostics file: "
                       << filename << "\n";
        return;
    }

    file << step << "," << std::fixed << std::setprecision(6) << time_s << ","
         << n_active << "," << total_mass;

    for (const auto& rate : event_rates) {
        file << "," << std::fixed << std::setprecision(6) << rate;
    }
    file << "\n";

    file.close();
}

#endif  // ERF_USE_LNG
#endif  // ERF_LNG_SPILL_SCHEDULE_DIAG_H
