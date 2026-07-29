/**
 * @file ERF_UCMBuildingLayoutReader.cpp
 * @brief Building-layout CSV reader implementation for the ERF-SLUCM module (Phase 2.1 → 3.7)
 *
 * Implements rank-0-read + MPI_Bcast of building-layout data following patterns from
 * `Source/LNG/ERF_LNGSpillSchedule.cpp`.
 *
 * ### Phase 3.7 extension: Physical-coordinate CSV support
 * - Detects CSV mode from header: "x_m,y_m,..." (physical) vs "i,j,..." (legacy)
 * - Physical mode: loads sparse rows, performs nearest-neighbor lookup per UCM cell
 * - Legacy mode: preserves existing exact-match (i,j) behavior with row-count validation
 */

#include <UrbanCanopy/ERF_UCMBuildingLayoutReader.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>
#include <set>
#include <iomanip>
#include <cmath>

// Verify POD struct is MPI_Bcast safe
static_assert(std::is_trivially_copyable_v<UCMBuildingRow>,
              "UCMBuildingRow must be trivially copyable for MPI_Bcast");

namespace {
    // Enum to distinguish CSV mode
    enum class CSVMode { LEGACY_INDEX, PHYSICAL_COORDS };

    // Helper to detect CSV mode from header
    CSVMode detect_csv_mode(const std::string& header_line) {
        auto remove_spaces = [](std::string s) {
            s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
            return s;
        };

        std::string header_clean = remove_spaces(header_line);

        // Check if it starts with x_m,y_m (physical mode)
        if (header_clean.find("x_m,y_m,") == 0) {
            return CSVMode::PHYSICAL_COORDS;
        }

        // Check if it starts with i,j (legacy mode)
        if (header_clean.find("i,j,") == 0) {
            return CSVMode::LEGACY_INDEX;
        }

        // If we get here, it's an unknown format
        return CSVMode::LEGACY_INDEX;  // Default to legacy
    }

    // Helper to compute Euclidean distance
    amrex::Real euclidean_distance(amrex::Real x1, amrex::Real y1,
                                    amrex::Real x2, amrex::Real y2) {
        amrex::Real dx = x1 - x2;
        amrex::Real dy = y1 - y2;
        return std::sqrt(dx * dx + dy * dy);
    }
}  // namespace

void UCMBuildingLayoutReader::read_and_broadcast(const std::string& path, 
                                                  int nx_ucm, int ny_ucm,
                                                  int lev, bool ucm_debug)
{
    // Clear any previous data
    m_rows.clear();

    int n_rows = 0;
    CSVMode csv_mode = CSVMode::LEGACY_INDEX;
    bool is_physical_mode = false;
    UCMBuildingRow min_vals{}, max_vals{};
    int count_is_urban_zero = 0;
    int count_AH_Wm2_populated = 0;  // Phase 2.9: track non-zero AH_Wm2
    amrex::Real min_AH_Wm2 = std::numeric_limits<amrex::Real>::infinity();
    amrex::Real max_AH_Wm2 = -std::numeric_limits<amrex::Real>::infinity();
    amrex::Real sum_AH_Wm2 = 0.0;
    bool has_duplicate = false;
    std::string duplicate_msg;

    // Phase 3.7: Physical-mode bounding box
    amrex::Real x_min = std::numeric_limits<amrex::Real>::infinity();
    amrex::Real x_max = -std::numeric_limits<amrex::Real>::infinity();
    amrex::Real y_min = std::numeric_limits<amrex::Real>::infinity();
    amrex::Real y_max = -std::numeric_limits<amrex::Real>::infinity();

    // =========================================================================
    // Rank 0: Read and parse CSV
    // =========================================================================
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        std::ifstream csv_file(path);
        if (!csv_file.is_open()) {
            amrex::Abort("[UCM][3.7][UCMBuildingLayoutReader::read_and_broadcast] "
                        "Cannot open file: " + path);
        }

        // Read and validate header
        std::string header_line;
        if (!std::getline(csv_file, header_line)) {
            amrex::Abort("[UCM][3.7][UCMBuildingLayoutReader::read_and_broadcast] "
                        "CSV file is empty: " + path);
        }

        // Phase 2.5-fix2: Task 6 — Strip UTF-8 BOM and leading/trailing whitespace
        if (header_line.size() >= 3 &&
            static_cast<unsigned char>(header_line[0]) == 0xEF &&
            static_cast<unsigned char>(header_line[1]) == 0xBB &&
            static_cast<unsigned char>(header_line[2]) == 0xBF) {
            header_line.erase(0, 3);
        }
        // Strip leading whitespace (space, tab, CR).
        const auto first = header_line.find_first_not_of(" \t\r");
        if (first != std::string::npos && first > 0) header_line.erase(0, first);
        // Strip trailing whitespace.
        const auto last = header_line.find_last_not_of(" \t\r\n");
        if (last != std::string::npos) header_line.erase(last + 1);

        // Phase 3.7: Detect CSV mode from header
        csv_mode = detect_csv_mode(header_line);
        is_physical_mode = (csv_mode == CSVMode::PHYSICAL_COORDS);

        // Validate header format based on detected mode
        auto remove_spaces = [](std::string s) {
            s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
            return s;
        };

        std::string header_no_spaces = remove_spaces(header_line);
        bool has_AH_Wm2 = false;
        bool has_hvac_profile_id = false;  // Phase 5.2: detect optional hvac_profile_id

        // Define expected headers (with and without AH_Wm2, and optional hvac_profile_id)
        const std::string expected_header_legacy_new = "i,j,bldg_id,height_m,plan_area_frac,W_road_m,W_roof_m,"
                                                      "roof_mat_id,wall_mat_id,road_mat_id,orientation_deg,ah_profile_id,AH_Wm2,is_urban";
        const std::string expected_header_legacy_old = "i,j,bldg_id,height_m,plan_area_frac,W_road_m,W_roof_m,"
                                                      "roof_mat_id,wall_mat_id,road_mat_id,orientation_deg,ah_profile_id,is_urban";
        const std::string expected_header_physical_new = "x_m,y_m,bldg_id,height_m,plan_area_frac,W_road_m,W_roof_m,"
                                                        "roof_mat_id,wall_mat_id,road_mat_id,orientation_deg,ah_profile_id,AH_Wm2,is_urban";
        const std::string expected_header_physical_old = "x_m,y_m,bldg_id,height_m,plan_area_frac,W_road_m,W_roof_m,"
                                                        "roof_mat_id,wall_mat_id,road_mat_id,orientation_deg,ah_profile_id,is_urban";
        
        // Phase 5.2: New headers with hvac_profile_id
        const std::string expected_header_legacy_new_hvac = "i,j,bldg_id,height_m,plan_area_frac,W_road_m,W_roof_m,"
                                                           "roof_mat_id,wall_mat_id,road_mat_id,orientation_deg,ah_profile_id,AH_Wm2,is_urban,hvac_profile_id";
        const std::string expected_header_legacy_old_hvac = "i,j,bldg_id,height_m,plan_area_frac,W_road_m,W_roof_m,"
                                                           "roof_mat_id,wall_mat_id,road_mat_id,orientation_deg,ah_profile_id,is_urban,hvac_profile_id";
        const std::string expected_header_physical_new_hvac = "x_m,y_m,bldg_id,height_m,plan_area_frac,W_road_m,W_roof_m,"
                                                             "roof_mat_id,wall_mat_id,road_mat_id,orientation_deg,ah_profile_id,AH_Wm2,is_urban,hvac_profile_id";
        const std::string expected_header_physical_old_hvac = "x_m,y_m,bldg_id,height_m,plan_area_frac,W_road_m,W_roof_m,"
                                                             "roof_mat_id,wall_mat_id,road_mat_id,orientation_deg,ah_profile_id,is_urban,hvac_profile_id";

        if (is_physical_mode) {
            if (header_no_spaces == remove_spaces(expected_header_physical_new_hvac)) {
                has_AH_Wm2 = true;
                has_hvac_profile_id = true;
            } else if (header_no_spaces == remove_spaces(expected_header_physical_old_hvac)) {
                has_AH_Wm2 = false;
                has_hvac_profile_id = true;
            } else if (header_no_spaces == remove_spaces(expected_header_physical_new)) {
                has_AH_Wm2 = true;
                has_hvac_profile_id = false;
            } else if (header_no_spaces == remove_spaces(expected_header_physical_old)) {
                has_AH_Wm2 = false;
                has_hvac_profile_id = false;
            } else {
                std::ostringstream oss;
                oss << "[UCM][5.2][UCMBuildingLayoutReader::read_and_broadcast] "
                    << "CSV header mismatch (physical mode).\n"
                    << "  Expected (new with hvac): " << expected_header_physical_new_hvac << "\n"
                    << "  Or (old with hvac):       " << expected_header_physical_old_hvac << "\n"
                    << "  Or (new):                 " << expected_header_physical_new << "\n"
                    << "  Or (old):                 " << expected_header_physical_old << "\n"
                    << "  Got:                      " << header_line << "\n"
                    << "  Got bytes (hex): ";
                for (unsigned char c : header_line) {
                    oss << std::hex << std::setw(2) << std::setfill('0') << int(c) << " ";
                }
                amrex::Abort(oss.str());
            }
        } else {
            // Legacy mode
            if (header_no_spaces == remove_spaces(expected_header_legacy_new_hvac)) {
                has_AH_Wm2 = true;
                has_hvac_profile_id = true;
            } else if (header_no_spaces == remove_spaces(expected_header_legacy_old_hvac)) {
                has_AH_Wm2 = false;
                has_hvac_profile_id = true;
            } else if (header_no_spaces == remove_spaces(expected_header_legacy_new)) {
                has_AH_Wm2 = true;
                has_hvac_profile_id = false;
            } else if (header_no_spaces == remove_spaces(expected_header_legacy_old)) {
                has_AH_Wm2 = false;
                has_hvac_profile_id = false;
            } else {
                std::ostringstream oss;
                oss << "[UCM][5.2][UCMBuildingLayoutReader::read_and_broadcast] "
                    << "CSV header mismatch (legacy mode).\n"
                    << "  Expected (new with hvac): " << expected_header_legacy_new_hvac << "\n"
                    << "  Or (old with hvac):       " << expected_header_legacy_old_hvac << "\n"
                    << "  Or (new):                 " << expected_header_legacy_new << "\n"
                    << "  Or (old):                 " << expected_header_legacy_old << "\n"
                    << "  Got:                      " << header_line << "\n"
                    << "  Got bytes (hex): ";
                for (unsigned char c : header_line) {
                    oss << std::hex << std::setw(2) << std::setfill('0') << int(c) << " ";
                }
                amrex::Abort(oss.str());
            }
        }

        // Read data rows
        std::string line;
        std::set<std::pair<int, int>> seen_indices;  // For legacy mode duplicate checking

        while (std::getline(csv_file, line))
        {
            // Skip empty lines
            if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
                continue;
            }

            // Phase 2.5-fix2: Task 6 — Strip UTF-8 BOM and whitespace from data rows
            if (line.size() >= 3 &&
                static_cast<unsigned char>(line[0]) == 0xEF &&
                static_cast<unsigned char>(line[1]) == 0xBB &&
                static_cast<unsigned char>(line[2]) == 0xBF) {
                line.erase(0, 3);
            }
            // Strip leading whitespace (space, tab, CR).
            const auto first = line.find_first_not_of(" \t\r");
            if (first != std::string::npos && first > 0) line.erase(0, first);
            // Strip trailing whitespace.
            const auto last = line.find_last_not_of(" \t\r\n");
            if (last != std::string::npos) line.erase(last + 1);

            UCMBuildingRow row{};
            std::stringstream ss(line);
            std::string field;

            try {
                // Parse fields based on mode
                if (is_physical_mode) {
                    // Physical mode: x_m, y_m (amrex::Real)
                    std::getline(ss, field, ','); row.x              = std::stod(field);  // x_m
                    std::getline(ss, field, ','); row.y              = std::stod(field);  // y_m
                } else {
                    // Legacy mode: i, j (int, stored as Real for compatibility)
                    std::getline(ss, field, ','); row.x              = static_cast<amrex::Real>(std::stoi(field));  // i
                    std::getline(ss, field, ','); row.y              = static_cast<amrex::Real>(std::stoi(field));  // j
                }

                // Parse remaining fields (same for both modes)
                std::getline(ss, field, ','); row.bldg_id         = std::stoi(field);
                std::getline(ss, field, ','); row.height_m        = std::stod(field);
                std::getline(ss, field, ','); row.plan_area_frac  = std::stod(field);
                std::getline(ss, field, ','); row.W_road_m        = std::stod(field);
                std::getline(ss, field, ','); row.W_roof_m        = std::stod(field);
                std::getline(ss, field, ','); row.roof_mat_id     = std::stoi(field);
                std::getline(ss, field, ','); row.wall_mat_id     = std::stoi(field);
                std::getline(ss, field, ','); row.road_mat_id     = std::stoi(field);
                std::getline(ss, field, ','); row.orientation_deg = std::stod(field);
                std::getline(ss, field, ','); row.ah_profile_id   = std::stoi(field);

                // Phase 2.9: Handle AH_Wm2 column (new format only)
                if (has_AH_Wm2) {
                    std::getline(ss, field, ','); row.AH_Wm2     = std::stod(field);
                    std::getline(ss, field, ','); row.is_urban    = std::stoi(field);
                } else {
                    // Old format: no AH_Wm2, default to 0.0
                    row.AH_Wm2 = 0.0;
                    std::getline(ss, field, ','); row.is_urban    = std::stoi(field);
                }

                // Phase 5.2: Handle hvac_profile_id column (optional trailing column, defaults to 0)
                row.hvac_profile_id = 0;  // Default
                if (has_hvac_profile_id) {
                    std::getline(ss, field, ',');
                    if (!field.empty()) {
                        try {
                            row.hvac_profile_id = std::stoi(field);
                        } catch (...) {
                            row.hvac_profile_id = 0;
                        }
                    }
                }

                // Phase 5.3: Handle optional green roof, permeable road, and soil moisture columns
                // These appear as trailing columns after hvac_profile_id (if present) or after is_urban
                row.is_green_roof = 0;  // Default: no green roof
                row.is_permeable_road = 0;  // Default: no permeable pavement
                row.soil_moisture_init_m3_per_m3 = 0.1;  // Default: reasonable soil moisture

                // Try to parse is_green_roof (optional column)
                if (std::getline(ss, field, ',')) {
                    if (!field.empty()) {
                        try {
                            row.is_green_roof = std::stoi(field);
                        } catch (...) {
                            row.is_green_roof = 0;
                        }
                    }
                }

                // Try to parse is_permeable_road (optional column)
                if (std::getline(ss, field, ',')) {
                    if (!field.empty()) {
                        try {
                            row.is_permeable_road = std::stoi(field);
                        } catch (...) {
                            row.is_permeable_road = 0;
                        }
                    }
                }

                // Try to parse soil_moisture_init_m3_per_m3 (optional column)
                if (std::getline(ss, field, ',')) {
                    if (!field.empty()) {
                        try {
                            row.soil_moisture_init_m3_per_m3 = std::stod(field);
                        } catch (...) {
                            row.soil_moisture_init_m3_per_m3 = 0.1;
                        }
                    }
                }

                // Validate is_urban
                if (row.is_urban != 0 && row.is_urban != 1) {
                    amrex::Abort("[UCM][3.7][UCMBuildingLayoutReader::read_and_broadcast] "
                                "is_urban must be 0 or 1; got " + std::to_string(row.is_urban) +
                                " at index=(" + std::to_string(static_cast<int>(row.x)) + "," +
                                std::to_string(static_cast<int>(row.y)) + ")");
                }

                // Phase 2.9: Validate AH_Wm2 >= 0
                if (row.AH_Wm2 < 0.0) {
                    amrex::Abort("[UCM][2.9][UCMBuildingLayoutReader::read_and_broadcast] "
                                "AH_Wm2 must be >= 0; got " + std::to_string(row.AH_Wm2) +
                                " at index=(" + std::to_string(static_cast<int>(row.x)) + "," +
                                std::to_string(static_cast<int>(row.y)) + ")");
                }

                // Validate mat_ids based on is_urban
                if (row.is_urban == 1) {
                    // Urban cells require all mat_ids >= 1
                    if (row.roof_mat_id < 1 || row.wall_mat_id < 1 || row.road_mat_id < 1) {
                        amrex::Abort("[UCM][3.7][UCMBuildingLayoutReader::read_and_broadcast] "
                                    "Urban cell at index=(" + std::to_string(static_cast<int>(row.x)) + "," +
                                    std::to_string(static_cast<int>(row.y)) + ") must have all mat_ids >= 1; "
                                    "got roof=" + std::to_string(row.roof_mat_id) +
                                    ", wall=" + std::to_string(row.wall_mat_id) +
                                    ", road=" + std::to_string(row.road_mat_id));
                    }
                } else {
                    // Non-urban cells: mat_ids may be 0 (sentinel) or any nonnegative value
                    if (row.roof_mat_id < 0 || row.wall_mat_id < 0 || row.road_mat_id < 0) {
                        amrex::Abort("[UCM][3.7][UCMBuildingLayoutReader::read_and_broadcast] "
                                    "Non-urban cell at index=(" + std::to_string(static_cast<int>(row.x)) + "," +
                                    std::to_string(static_cast<int>(row.y)) + ") has negative mat_id");
                    }
                }

                // Legacy mode: Check for duplicate (i,j)
                if (!is_physical_mode) {
                    int i = static_cast<int>(row.x);
                    int j = static_cast<int>(row.y);
                    std::pair<int, int> idx_pair{i, j};
                    if (seen_indices.find(idx_pair) != seen_indices.end()) {
                        has_duplicate = true;
                        duplicate_msg = "[UCM][3.7][UCMBuildingLayoutReader::read_and_broadcast] "
                                       "Duplicate (i,j) pair (" + std::to_string(i) + "," +
                                       std::to_string(j) + ") in CSV";
                        break;
                    }
                    seen_indices.insert(idx_pair);
                }

                // Track statistics
                if (n_rows == 0) {
                    min_vals = row;
                    max_vals = row;
                } else {
                    min_vals.bldg_id        = std::min(min_vals.bldg_id, row.bldg_id);
                    min_vals.height_m       = std::min(min_vals.height_m, row.height_m);
                    min_vals.plan_area_frac = std::min(min_vals.plan_area_frac, row.plan_area_frac);
                    max_vals.bldg_id        = std::max(max_vals.bldg_id, row.bldg_id);
                    max_vals.height_m       = std::max(max_vals.height_m, row.height_m);
                    max_vals.plan_area_frac = std::max(max_vals.plan_area_frac, row.plan_area_frac);
                }

                // Phase 3.7: Update bounding box for physical mode
                if (is_physical_mode) {
                    x_min = std::min(x_min, row.x);
                    x_max = std::max(x_max, row.x);
                    y_min = std::min(y_min, row.y);
                    y_max = std::max(y_max, row.y);
                }

                if (row.is_urban == 0) {
                    ++count_is_urban_zero;
                }

                // Phase 2.9: Track AH_Wm2 statistics
                if (row.AH_Wm2 > 0.0) {
                    ++count_AH_Wm2_populated;
                    min_AH_Wm2 = std::min(min_AH_Wm2, row.AH_Wm2);
                    max_AH_Wm2 = std::max(max_AH_Wm2, row.AH_Wm2);
                    sum_AH_Wm2 += row.AH_Wm2;
                }

                m_rows.push_back(row);
                ++n_rows;

            } catch (const std::exception& e) {
                amrex::Abort("[UCM][3.7][UCMBuildingLayoutReader::read_and_broadcast] "
                            "Parsing error on line:\n" + line + "\nException: " + std::string(e.what()));
            }
        }

        csv_file.close();

        // Abort if duplicate found (legacy mode only)
        if (has_duplicate) {
            amrex::Abort(duplicate_msg);
        }

        // Phase 3.7: For legacy mode, validate row count == nx_ucm * ny_ucm
        if (!is_physical_mode) {
            const int expected_rows = nx_ucm * ny_ucm;
            if (n_rows != expected_rows) {
                amrex::Abort("[UCM][3.7][UCMBuildingLayoutReader] CSV row count mismatch. "
                             "Got " + std::to_string(n_rows) + " rows, expected " +
                             std::to_string(expected_rows) + " (= nx_ucm * ny_ucm = " +
                             std::to_string(nx_ucm) + " * " + std::to_string(ny_ucm) + "). "
                             "CSV i,j MUST be UCM indices, not ATM indices.");
            }

            // Phase 3.7: Validate (i,j) ranges for all rows (legacy mode only)
            for (const auto& r : m_rows) {
                int i = static_cast<int>(r.x);
                int j = static_cast<int>(r.y);
                if (i < 0 || i >= nx_ucm || j < 0 || j >= ny_ucm) {
                    amrex::Abort("[UCM][3.7][UCMBuildingLayoutReader] Row (i=" + std::to_string(i) +
                                 ",j=" + std::to_string(j) + ") out of UCM range [0," +
                                 std::to_string(nx_ucm) + ")x[0," + std::to_string(ny_ucm) + ").");
                }
            }
        }

        // Debug trace
        if (ucm_debug) {
            amrex::Print()
                << "\n[UCM][3.7][UCMBuildingLayoutReader::read_and_broadcast]\n"
                << "  path = " << path << "\n"
                << "  mode = " << (is_physical_mode ? "physical (x_m, y_m)" : "legacy (i, j)") << "\n"
                << "  rows_parsed = " << n_rows;
            if (!is_physical_mode) {
                const int expected_rows = nx_ucm * ny_ucm;
                amrex::Print() << " (expected " << expected_rows << ")";
            }
            amrex::Print() << "\n";

            if (is_physical_mode && n_rows > 0) {
                amrex::Print()
                    << "  physical bbox: x=[" << x_min << ", " << x_max << "], "
                    << "y=[" << y_min << ", " << y_max << "]\n";
            }

            if (n_rows > 0) {
                amrex::Print()
                    << "  bldg_id: min=" << min_vals.bldg_id << ", max=" << max_vals.bldg_id << "\n"
                    << "  height_m: min=" << min_vals.height_m << ", max=" << max_vals.height_m << "\n"
                    << "  plan_area_frac: min=" << min_vals.plan_area_frac << ", max=" << max_vals.plan_area_frac << "\n"
                    << "  is_urban=0 count: " << count_is_urban_zero << "\n";
                // Phase 2.9: Log AH_Wm2 stats if populated
                if (count_AH_Wm2_populated > 0) {
                    amrex::Real mean_AH_Wm2 = sum_AH_Wm2 / count_AH_Wm2_populated;
                    amrex::Print()
                        << "  AH_Wm2: populated_count=" << count_AH_Wm2_populated 
                        << ", min=" << min_AH_Wm2 << ", max=" << max_AH_Wm2
                        << ", mean=" << mean_AH_Wm2 << " W/m^2\n";
                }
            }
        }
    }

    // =========================================================================
    // All ranks: MPI_Bcast row data + metadata
    // =========================================================================

    // Broadcast row count and mode flag
    amrex::ParallelDescriptor::Bcast(&n_rows, 1, amrex::ParallelDescriptor::IOProcessorNumber());
    int mode_int = static_cast<int>(is_physical_mode ? 1 : 0);
    amrex::ParallelDescriptor::Bcast(&mode_int, 1, amrex::ParallelDescriptor::IOProcessorNumber());
    is_physical_mode = (mode_int != 0);

    // Broadcast physical mode bounding box
    if (is_physical_mode) {
        amrex::ParallelDescriptor::Bcast(&x_min, 1, amrex::ParallelDescriptor::IOProcessorNumber());
        amrex::ParallelDescriptor::Bcast(&x_max, 1, amrex::ParallelDescriptor::IOProcessorNumber());
        amrex::ParallelDescriptor::Bcast(&y_min, 1, amrex::ParallelDescriptor::IOProcessorNumber());
        amrex::ParallelDescriptor::Bcast(&y_max, 1, amrex::ParallelDescriptor::IOProcessorNumber());
    }

    // Resize on all ranks
    if (!amrex::ParallelDescriptor::IOProcessor()) {
        m_rows.resize(n_rows);
    }

    // Broadcast row data as bytes
    if (n_rows > 0) {
        amrex::ParallelDescriptor::Bcast(
            reinterpret_cast<char*>(m_rows.dataPtr()),
            static_cast<std::size_t>(n_rows) * sizeof(UCMBuildingRow),
            amrex::ParallelDescriptor::IOProcessorNumber());

        if (amrex::ParallelDescriptor::IOProcessor() && ucm_debug) {
            amrex::Print()
                << "  MPI_Bcast: " << n_rows << " rows ("
                << (n_rows * sizeof(UCMBuildingRow)) << " bytes) to all ranks\n";
        }
    }

    // Phase 3.7: Debug output for mode and bounding box
    if (amrex::ParallelDescriptor::IOProcessor() && ucm_debug) {
        amrex::Print() << "[UCM][3.7][DEBUG][UCMBuildingLayoutReader] parsed "
                       << m_rows.size() << " rows in "
                       << (is_physical_mode ? "physical" : "legacy") << " mode\n";
    }
}
