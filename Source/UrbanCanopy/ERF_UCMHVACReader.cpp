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
    int sensible_frac_col_index = -1;  // Phase 5.5: track column indices
    int rejection_facet_col_index = -1;

    while (std::getline(file, line)) {
        ++line_num;

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Detect and skip header row
        if (!header_seen) {
            if (line.find("hvac_profile_id") != std::string::npos) {
                // Phase 5.5: Detect optional columns in header
                size_t pos_sf = line.find("sensible_fraction");
                if (pos_sf != std::string::npos) {
                    // Count commas before sensible_fraction to get column index
                    sensible_frac_col_index = std::count(line.begin(), line.begin() + pos_sf, ',');
                }
                size_t pos_rf = line.find("rejection_facet");
                if (pos_rf != std::string::npos) {
                    rejection_facet_col_index = std::count(line.begin(), line.begin() + pos_rf, ',');
                }
                header_seen = true;
                continue;
            }
        }

        // Parse comma-separated values
        std::stringstream ss(line);
        std::vector<std::string> fields;
        std::string field;
        
        while (std::getline(ss, field, ',')) {
            // Trim leading/trailing whitespace
            size_t start = field.find_first_not_of(" \t\r\n");
            size_t end = field.find_last_not_of(" \t\r\n");
            if (start == std::string::npos) {
                fields.push_back("");
            } else {
                fields.push_back(field.substr(start, end - start + 1));
            }
        }

        // Minimum 5 fields required (id, cop, setpt, occ_id, desc)
        if (fields.size() < 5) {
            amrex::Warning("[UCM][5.5] HVACReader: Line " + std::to_string(line_num) +
                          " has " + std::to_string(fields.size()) + " fields (need >= 5), skipping");
            continue;
        }

        try {
            int id = std::stoi(fields[0]);
            amrex::Real cop = std::stod(fields[1]);
            amrex::Real t_setpt = std::stod(fields[2]);
            int occ_id = std::stoi(fields[3]);
            std::string desc = fields[4];

            // Phase 5.5: Parse sensible_fraction with default 1.0
            amrex::Real sensible_frac = 1.0;
            if (sensible_frac_col_index >= 0 && sensible_frac_col_index < static_cast<int>(fields.size())) {
                try {
                    sensible_frac = std::stod(fields[sensible_frac_col_index]);
                } catch (...) {
                    amrex::Warning("[UCM][5.5] HVACReader: Line " + std::to_string(line_num) +
                                  " sensible_fraction parse error, using default 1.0");
                    sensible_frac = 1.0;
                }
            }

            // Phase 5.5: Parse rejection_facet with default 0 (roof)
            int rej_facet = 0;  // roof = default
            if (rejection_facet_col_index >= 0 && rejection_facet_col_index < static_cast<int>(fields.size())) {
                std::string facet_str = fields[rejection_facet_col_index];
                if (facet_str == "roof") {
                    rej_facet = 0;
                } else if (facet_str == "road") {
                    rej_facet = 1;
                } else if (facet_str == "distributed") {
                    rej_facet = 2;
                } else {
                    amrex::Warning("[UCM][5.5] HVACReader: Line " + std::to_string(line_num) +
                                  " unknown rejection_facet '" + facet_str + "', using default 'roof'");
                    rej_facet = 0;
                }
            }

            HVACProfile profile;
            profile.id = id;
            profile.cop = cop;
            profile.t_setpoint_K = t_setpt;
            profile.occupancy_profile_id = occ_id;
            profile.description = desc;
            profile.sensible_fraction = sensible_frac;      // Phase 5.5
            profile.rejection_facet = rej_facet;            // Phase 5.5

            // Sanity check: COP > 1.0
            if (profile.cop < 1.0) {
                amrex::Warning("[UCM][5.2] HVACReader: Line " + std::to_string(line_num) +
                              " COP < 1.0 is non-physical, clamping to 1.0");
                profile.cop = 1.0;
            }

            // Phase 5.5: Sanity check sensible_fraction in [0, 1]
            if (profile.sensible_fraction < 0.0 || profile.sensible_fraction > 1.0) {
                amrex::Warning("[UCM][5.5] HVACReader: Line " + std::to_string(line_num) +
                              " sensible_fraction out of range [0, 1], clamping");
                profile.sensible_fraction = amrex::max(0.0, amrex::min(1.0, profile.sensible_fraction));
            }

            profiles_.push_back(profile);
        } catch (const std::exception& e) {
            amrex::Warning("[UCM][5.5] HVACReader: Line " + std::to_string(line_num) +
                          " conversion error (" + e.what() + "), skipping");
            continue;
        }
    }

    file.close();

    if (profiles_.empty()) {
        amrex::Error("[UCM][5.2] HVACReader: No valid data rows in " + csv_path);
    }

    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "[UCM][5.5][hvac-csv] Loaded " << profiles_.size()
                       << " HVAC profiles from " << csv_path;
        if (sensible_frac_col_index >= 0) {
            amrex::Print() << " (with sensible_fraction column)";
        } else {
            amrex::Print() << " (sensible_fraction: using default 1.0)";
        }
        if (rejection_facet_col_index >= 0) {
            amrex::Print() << " (with rejection_facet column)";
        } else {
            amrex::Print() << " (rejection_facet: using default 'roof')";
        }
        amrex::Print() << "\n";
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
