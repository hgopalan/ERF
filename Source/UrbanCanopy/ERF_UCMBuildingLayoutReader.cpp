/**
 * @file ERF_UCMBuildingLayoutReader.cpp
 * @brief Building-layout CSV reader implementation for the ERF-SLUCM module (Phase 2.1)
 *
 * Implements rank-0-read + MPI_Bcast of building-layout data following patterns from
 * `Source/LNG/ERF_LNGSpillSchedule.cpp`.
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

// Verify POD struct is MPI_Bcast safe
static_assert(std::is_trivially_copyable_v<UCMBuildingRow>,
              "UCMBuildingRow must be trivially copyable for MPI_Bcast");

void UCMBuildingLayoutReader::read_and_broadcast(const std::string& path, 
                                                  int nx_ucm, int ny_ucm,
                                                  int lev, bool ucm_debug)
{
    // Clear any previous data
    m_rows.clear();

    int n_rows = 0;
    UCMBuildingRow min_vals{}, max_vals{};
    int count_is_urban_zero = 0;
    bool has_duplicate = false;
    std::string duplicate_msg;

    // =========================================================================
    // Rank 0: Read and parse CSV
    // =========================================================================
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        std::ifstream csv_file(path);
        if (!csv_file.is_open()) {
            amrex::Abort("[UCM][2.1][UCMBuildingLayoutReader::read_and_broadcast] "
                        "Cannot open file: " + path);
        }

        // Read and validate header
        std::string header_line;
        if (!std::getline(csv_file, header_line)) {
            amrex::Abort("[UCM][2.1][UCMBuildingLayoutReader::read_and_broadcast] "
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

        // Expected header (allow whitespace around delimiters)
        const std::string expected_header = "i,j,bldg_id,height_m,plan_area_frac,W_road_m,W_roof_m,"
                                           "roof_mat_id,wall_mat_id,road_mat_id,orientation_deg,ah_profile_id,is_urban";

        // Simple header validation: remove spaces and compare
        auto remove_spaces = [](std::string s) {
            s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
            return s;
        };
        if (remove_spaces(header_line) != remove_spaces(expected_header)) {
            std::ostringstream oss;
            oss << "[UCM][2.1] CSV header mismatch.\n"
                << "  Expected: " << expected_header << "\n"
                << "  Got:      " << header_line << "\n"
                << "  Got bytes (hex): ";
            for (unsigned char c : header_line) {
                oss << std::hex << std::setw(2) << std::setfill('0') << int(c) << " ";
            }
            amrex::Abort(oss.str());
        }

        // Read data rows
        std::string line;
        std::set<std::pair<int, int>> seen_indices;

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
                // Parse all 13 fields
                std::getline(ss, field, ','); row.i               = std::stoi(field);
                std::getline(ss, field, ','); row.j               = std::stoi(field);
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
                std::getline(ss, field, ','); row.is_urban        = std::stoi(field);

                // Validate is_urban
                if (row.is_urban != 0 && row.is_urban != 1) {
                    amrex::Abort("[UCM][2.1][UCMBuildingLayoutReader::read_and_broadcast] "
                                "is_urban must be 0 or 1; got " + std::to_string(row.is_urban) +
                                " at (i,j) = (" + std::to_string(row.i) + "," + std::to_string(row.j) + ")");
                }

                // Validate mat_ids based on is_urban
                if (row.is_urban == 1) {
                    // Urban cells require all mat_ids >= 1
                    if (row.roof_mat_id < 1 || row.wall_mat_id < 1 || row.road_mat_id < 1) {
                        amrex::Abort("[UCM][2.1][UCMBuildingLayoutReader::read_and_broadcast] "
                                    "Urban cell at (i,j) = (" + std::to_string(row.i) + "," +
                                    std::to_string(row.j) + ") must have all mat_ids >= 1; "
                                    "got roof=" + std::to_string(row.roof_mat_id) +
                                    ", wall=" + std::to_string(row.wall_mat_id) +
                                    ", road=" + std::to_string(row.road_mat_id));
                    }
                } else {
                    // Non-urban cells: mat_ids may be 0 (sentinel) or any nonnegative value;
                    // they will not be dereferenced by fill_ucm_fields_from_csv.
                    if (row.roof_mat_id < 0 || row.wall_mat_id < 0 || row.road_mat_id < 0) {
                        amrex::Abort("[UCM][2.1][UCMBuildingLayoutReader::read_and_broadcast] "
                                    "Non-urban cell at (i,j) = (" + std::to_string(row.i) + "," +
                                    std::to_string(row.j) + ") has negative mat_id");
                    }
                }

                // Check for duplicate (i,j)
                std::pair<int, int> idx_pair{row.i, row.j};
                if (seen_indices.find(idx_pair) != seen_indices.end()) {
                    has_duplicate = true;
                    duplicate_msg = "[UCM][2.1][UCMBuildingLayoutReader::read_and_broadcast] "
                                   "Duplicate (i,j) pair (" + std::to_string(row.i) + "," +
                                   std::to_string(row.j) + ") in CSV";
                    break;  // Stop processing and abort below
                }
                seen_indices.insert(idx_pair);

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

                if (row.is_urban == 0) {
                    ++count_is_urban_zero;
                }

                m_rows.push_back(row);
                ++n_rows;

            } catch (const std::exception& e) {
                amrex::Abort("[UCM][2.1][UCMBuildingLayoutReader::read_and_broadcast] "
                            "Parsing error on line:\n" + line + "\nException: " + std::string(e.what()));
            }
        }

        csv_file.close();

        // Abort if duplicate found
        if (has_duplicate) {
            amrex::Abort(duplicate_msg);
        }

        // Phase 2.3: Validate row count == nx_ucm * ny_ucm
        const int expected_rows = nx_ucm * ny_ucm;
        if (n_rows != expected_rows) {
            amrex::Abort("[UCM][2.3][UCMBuildingLayoutReader] CSV row count mismatch. "
                         "Got " + std::to_string(n_rows) + " rows, expected " +
                         std::to_string(expected_rows) + " (= nx_ucm * ny_ucm = " +
                         std::to_string(nx_ucm) + " * " + std::to_string(ny_ucm) + "). "
                         "CSV i,j MUST be UCM indices, not ATM indices.");
        }

        // Phase 2.3: Validate (i,j) ranges for all rows
        for (const auto& r : m_rows) {
            if (r.i < 0 || r.i >= nx_ucm || r.j < 0 || r.j >= ny_ucm) {
                amrex::Abort("[UCM][2.3][UCMBuildingLayoutReader] Row (i=" + std::to_string(r.i) +
                             ",j=" + std::to_string(r.j) + ") out of UCM range [0," +
                             std::to_string(nx_ucm) + ")x[0," + std::to_string(ny_ucm) + ").");
            }
        }

        // Debug trace
        if (ucm_debug) {
            amrex::Print()
                << "\n[UCM][2.1][UCMBuildingLayoutReader::read_and_broadcast]\n"
                << "  path = " << path << "\n"
                << "  rows_parsed = " << n_rows << " (expected " << expected_rows << ")\n";
            if (n_rows > 0) {
                amrex::Print()
                    << "  bldg_id: min=" << min_vals.bldg_id << ", max=" << max_vals.bldg_id << "\n"
                    << "  height_m: min=" << min_vals.height_m << ", max=" << max_vals.height_m << "\n"
                    << "  plan_area_frac: min=" << min_vals.plan_area_frac << ", max=" << max_vals.plan_area_frac << "\n"
                    << "  is_urban=0 count: " << count_is_urban_zero << "\n";
            }
        }
    }

    // =========================================================================
    // All ranks: MPI_Bcast row data (as raw bytes to avoid needing
    // amrex::ParallelDescriptor::Mpi_typemap<UCMBuildingRow> specialization).
    // =========================================================================

    // Broadcast row count
    amrex::ParallelDescriptor::Bcast(&n_rows, 1, amrex::ParallelDescriptor::IOProcessorNumber());

    // Resize on all ranks
    if (!amrex::ParallelDescriptor::IOProcessor()) {
        m_rows.resize(n_rows);
    }

    // Broadcast row data as bytes (selects the char*/byte-count overload)
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

    // Phase 2.5-fix2: Task 1 — Debug instrumentation for CSV is_urban propagation
    if (amrex::ParallelDescriptor::IOProcessor()) {
        int n_urban = 0, n_nonurban = 0;
        for (const auto& row : m_rows) {
            if (row.is_urban == 1) ++n_urban;
            else ++n_nonurban;
        }
        amrex::Print() << "[UCM][2.1][DEBUG][UCMBuildingLayoutReader] parsed "
                       << m_rows.size() << " rows: urban=" << n_urban
                       << " non-urban=" << n_nonurban << "\n";
    }
}