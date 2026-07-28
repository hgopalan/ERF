/**
 * @file ERF_UCMOccupancyReader.cpp
 * @brief Implementation of occupancy profile CSV reader (Phase 5.2)
 */

#include <UrbanCanopy/ERF_UCMOccupancyReader.H>
#include <AMReX_Print.H>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>

UCMOccupancyReader::UCMOccupancyReader(const std::string& csv_path)
{
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        amrex::Error("[UCM][5.2] UCMOccupancyReader: Cannot open file: " + csv_path);
    }

    // Temporary map to collect rows by profile_id
    std::map<int, std::vector<std::pair<int, amrex::Real>>> temp_rows;

    std::string line;
    int line_num = 0;
    bool header_seen = false;

    while (std::getline(file, line)) {
        ++line_num;

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Detect and skip header row
        if (!header_seen) {
            if (line.find("occupancy_profile_id") != std::string::npos) {
                header_seen = true;
                continue;
            }
        }

        // Parse comma-separated values
        std::stringstream ss(line);
        std::string profile_id_str, hour_str, frac_str;

        if (!std::getline(ss, profile_id_str, ',') ||
            !std::getline(ss, hour_str, ',') ||
            !std::getline(ss, frac_str, ',')) {
            amrex::Warning("[UCM][5.2] OccupancyReader: Line " + std::to_string(line_num) +
                          " format error, skipping");
            continue;
        }

        try {
            int profile_id = std::stoi(profile_id_str);
            int hour = std::stoi(hour_str);
            amrex::Real frac = std::stod(frac_str);

            // Validate hour range
            if (hour < 0 || hour > 23) {
                amrex::Warning("[UCM][5.2] OccupancyReader: Line " + std::to_string(line_num) +
                              " hour_of_day out of range [0, 23], skipping");
                continue;
            }

            // Clamp occupancy fraction to [0, 1]
            if (frac < 0.0 || frac > 1.0) {
                amrex::Warning("[UCM][5.2] OccupancyReader: Line " + std::to_string(line_num) +
                              " occupancy_fraction outside [0, 1], clamping");
                frac = amrex::max(0.0, amrex::min(1.0, frac));
            }

            temp_rows[profile_id].push_back({hour, frac});
        } catch (const std::exception& e) {
            amrex::Warning("[UCM][5.2] OccupancyReader: Line " + std::to_string(line_num) +
                          " conversion error (" + e.what() + "), skipping");
            continue;
        }
    }

    file.close();

    // Convert temporary map to profiles_ vector, validating completeness
    for (const auto& [profile_id, rows] : temp_rows) {
        if (rows.size() != 24) {
            amrex::Error("[UCM][5.2] OccupancyReader: Profile " + std::to_string(profile_id) +
                        " has " + std::to_string(rows.size()) + " hours; expected 24.");
        }

        // Build hourly array
        OccupancyProfile profile;
        profile.id = profile_id;
        profile.hourly_frac.fill(1.0);  // Default to fully occupied

        for (const auto& [hour, frac] : rows) {
            profile.hourly_frac[hour] = frac;
        }

        profiles_.push_back(profile);
    }

    if (profiles_.empty()) {
        amrex::Error("[UCM][5.2] OccupancyReader: No valid profiles in " + csv_path);
    }

    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "[UCM][5.2][occupancy-csv] Loaded " << profiles_.size()
                       << " occupancy profiles from " << csv_path << "\n";
    }
}

amrex::Real UCMOccupancyReader::get_occupancy(int profile_id, int hour_of_day) const
{
    if (hour_of_day < 0 || hour_of_day > 23) {
        return 1.0;  // Default: fully occupied if hour is invalid
    }

    for (const auto& p : profiles_) {
        if (p.id == profile_id) {
            return p.hourly_frac[hour_of_day];
        }
    }

    return 1.0;  // Default: fully occupied if profile not found
}

const OccupancyProfile* UCMOccupancyReader::get_profile(int profile_id) const
{
    for (const auto& p : profiles_) {
        if (p.id == profile_id) {
            return &p;
        }
    }
    return nullptr;
}
