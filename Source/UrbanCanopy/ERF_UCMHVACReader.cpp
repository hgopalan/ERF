/**
 * @file ERF_UCMHVACReader.cpp
 * @brief Implementation of HVAC profile CSV reader (Phase 5.2)
 */

#include <UrbanCanopy/ERF_UCMHVACReader.H>
#include <AMReX_Print.H>
#include <fstream>
#include <sstream>
#include <algorithm>

UCMHVACReader::UCMHVACReader(const std::string& csv_path)
{
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        amrex::Error("[UCM][5.2] UCMHVACReader: Cannot open file: " + csv_path);
    }

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
            if (line.find("hvac_profile_id") != std::string::npos) {
                header_seen = true;
                continue;
            }
        }

        // Parse comma-separated values
        std::stringstream ss(line);
        std::string id_str, cop_str, setpt_str, occ_id_str, desc_str;

        if (!std::getline(ss, id_str, ',') ||
            !std::getline(ss, cop_str, ',') ||
            !std::getline(ss, setpt_str, ',') ||
            !std::getline(ss, occ_id_str, ',') ||
            !std::getline(ss, desc_str, ',')) {
            amrex::Warning("[UCM][5.2] HVACReader: Line " + std::to_string(line_num) +
                          " format error, skipping");
            continue;
        }

        try {
            int id = std::stoi(id_str);
            amrex::Real cop = std::stod(cop_str);
            amrex::Real t_setpt = std::stod(setpt_str);
            int occ_id = std::stoi(occ_id_str);

            // Trim whitespace from description
            size_t start = desc_str.find_first_not_of(" \t\r\n");
            size_t end = desc_str.find_last_not_of(" \t\r\n");
            std::string desc = (start == std::string::npos) ? "" : desc_str.substr(start, end - start + 1);

            HVACProfile profile;
            profile.id = id;
            profile.cop = cop;
            profile.t_setpoint_K = t_setpt;
            profile.occupancy_profile_id = occ_id;
            profile.description = desc;

            // Sanity check: COP > 1.0
            if (profile.cop < 1.0) {
                amrex::Warning("[UCM][5.2] HVACReader: Line " + std::to_string(line_num) +
                              " COP < 1.0 is non-physical, clamping to 1.0");
                profile.cop = 1.0;
            }

            profiles_.push_back(profile);
        } catch (const std::exception& e) {
            amrex::Warning("[UCM][5.2] HVACReader: Line " + std::to_string(line_num) +
                          " conversion error (" + e.what() + "), skipping");
            continue;
        }
    }

    file.close();

    if (profiles_.empty()) {
        amrex::Error("[UCM][5.2] HVACReader: No valid data rows in " + csv_path);
    }

    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "[UCM][5.2][hvac-csv] Loaded " << profiles_.size()
                       << " HVAC profiles from " << csv_path << "\n";
    }
}

const HVACProfile* UCMHVACReader::get_profile(int profile_id) const
{
    for (const auto& p : profiles_) {
        if (p.id == profile_id) {
            return &p;
        }
    }
    return nullptr;
}
