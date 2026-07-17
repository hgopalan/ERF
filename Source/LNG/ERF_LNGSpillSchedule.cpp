#include "ERF_LNGSpillSchedule.H"
#include "ERF_LNGPool.H"
#include <AMReX_Print.H>
#include <AMReX_ParallelDescriptor.H>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstring>

void load_lng_spill_schedule(const std::string& filename,
                              LNGSpillSchedule& schedule)
{
    schedule.events.clear();
    schedule.loaded = false;

    if (filename.empty()) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[LNG] load_lng_spill_schedule: empty filename, schedule disabled\n";
        }
        return;
    }

    // Rank 0 reads the CSV file
    int n_events = 0;
    std::vector<LNGSpillEvent> events_vec;

    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            amrex::Print() << "[LNG] ERROR: cannot open spill schedule file: " << filename << "\n";
            return;
        }

        std::string line;
        int line_num = 0;
        while (std::getline(file, line)) {
            line_num++;

            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') continue;

            // Parse line
            std::istringstream iss(line);
            LNGSpillEvent event;
            std::string name_str;  // Use temporary string for safe parsing

            if (!(iss >> name_str >> event.start_time_s >> event.end_time_s
                     >> event.cx_m >> event.cy_m >> event.radius_m >> event.rate_kg_s)) {
                amrex::Print() << "[LNG] WARNING: parse error at line " << line_num
                              << " in " << filename << "\n";
                continue;
            }

            // Copy name with bounds checking (max 63 chars + null terminator)
            std::strncpy(event.name, name_str.c_str(), sizeof(event.name) - 1);
            event.name[sizeof(event.name) - 1] = '\0';
            }

            events_vec.push_back(event);
        }
        file.close();
        n_events = events_vec.size();
    }

    // Broadcast event count to all ranks
    amrex::ParallelDescriptor::Bcast(&n_events, 1, 0);

    // Resize schedule.events on all ranks
    schedule.events.resize(n_events);

    // Broadcast event data: convert to raw array for MPI_Bcast (POD struct)
    if (n_events > 0) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            // Rank 0: copy parsed events to schedule.events
            for (int i = 0; i < n_events; ++i) {
                schedule.events[i] = events_vec[i];
            }
        }

        // Broadcast entire event array to all ranks (POD, safe for MPI_Bcast)
        MPI_Bcast(schedule.events.data(), n_events * static_cast<int>(sizeof(LNGSpillEvent)),
                  MPI_BYTE, 0, amrex::ParallelDescriptor::Communicator());
    }

    schedule.loaded = true;

    // All ranks print debug info
    if (n_events > 0) {
        amrex::Print() << "[LNG DEBUG] Phase 8: spill schedule loaded, " << n_events
                       << " events from " << filename << "\n";
    }
}

void apply_spill_schedule(amrex::MultiFab&       lng_pool_depth,
                           const amrex::Geometry& geom_lng,
                           const LNGSpillSchedule& schedule,
                           amrex::Real            cur_time,
                           amrex::Real            dt,
                           amrex::Real            rho_LNG,
                           bool                   lng_debug)
{
    if (!schedule.loaded || schedule.events.empty() || dt <= 0.0) return;

    // Apply each active event
    for (int i = 0; i < static_cast<int>(schedule.events.size()); ++i) {
        const auto& event = schedule.events[i];

        // Time-window check: active if t >= start_time_s AND (end_time_s<0 OR t<=end_time_s)
        bool is_active = (cur_time >= event.start_time_s) &&
                        (event.end_time_s < 0.0 || cur_time <= event.end_time_s);

        if (!is_active) continue;

        // Event is active: apply spill source
        // Convert radius to area: pool_area = π * radius²
        amrex::Real pool_area_m2 = M_PI * event.radius_m * event.radius_m;

        apply_spill_source(lng_pool_depth, geom_lng,
                          event.rate_kg_s, rho_LNG,
                          pool_area_m2, event.cx_m, event.cy_m, dt);

        // Debug print (all ranks print independently, safe for collective calls above)
        if (lng_debug) {
            amrex::Print() << "[LNG DEBUG] Phase 8: spill event '" << event.name
                          << "' ACTIVE  rate=" << event.rate_kg_s << " kg/s"
                          << "  radius=" << event.radius_m << " m"
                          << "  at (" << event.cx_m << "," << event.cy_m << ") m\n";
        }
    }
}

amrex::Real compute_total_released_mass(const LNGSpillSchedule& schedule,
                                         amrex::Real cur_time)
{
    amrex::Real total_mass_kg = 0.0;

    for (const auto& event : schedule.events) {
        if (cur_time < event.start_time_s) continue;  // Event hasn't started

        amrex::Real time_active_s = 0.0;

        if (event.end_time_s < 0.0) {
            // Entire period from start to current time
            time_active_s = cur_time - event.start_time_s;
        } else {
            // Period from start to min(end, current time)
            amrex::Real end_time = amrex::min(event.end_time_s, cur_time);
            time_active_s = end_time - event.start_time_s;
        }

        if (time_active_s > 0.0) {
            total_mass_kg += event.rate_kg_s * time_active_s;
        }
    }

    return total_mass_kg;
}
