/**
 * @file ERF_UCMTreeLayoutReader.cpp
 * @brief Tree-layout CSV reader implementation for the ERF-SLUCM module (Phase 6.1)
 *
 * Implements rank-0-read + MPI_Bcast of tree-layout data following patterns from
 * `ERF_UCMBuildingLayoutReader.cpp` (Phase 2.1 → 3.7).
 *
 * Physical-coordinate mode only (Phase 6.1). Sparse CSV rows allowed (no row-count validation).
 */

#include <UrbanCanopy/ERF_UCMTreeLayoutReader.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>
#include <iomanip>
#include <cmath>


// Verify POD struct is MPI_Bcast safe
static_assert(std::is_trivially_copyable_v<UCMTreeRow>,
              "UCMTreeRow must be trivially copyable for MPI_Bcast");

namespace amrex {
    template <>
    struct ParallelDescriptor::Mpi_typemap<UCMTreeRow> {
        static MPI_Datatype type() { return MPI_BYTE; }
    };
}

void UCMTreeLayoutReader::read_and_broadcast(const std::string& path,
                                              int nx_ucm, int ny_ucm,
                                              amrex::Real H_tree_max,
                                              amrex::Real LAD_max,
                                              int lev, bool ucm_debug)
{
    // Clear any previous data
    m_rows.clear();

    int n_rows = 0;
    UCMTreeRow min_vals{}, max_vals{};
    int count_is_tree_one = 0;

    // =========================================================================
    // Rank 0: Read and parse CSV
    // =========================================================================
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        std::ifstream csv_file(path);
        if (!csv_file.is_open()) {
            amrex::Abort("[UCM][6.1][UCMTreeLayoutReader::read_and_broadcast] "
                        "Cannot open file: " + path);
        }

        // Read and validate header
        std::string header_line;
        if (!std::getline(csv_file, header_line)) {
            amrex::Abort("[UCM][6.1][UCMTreeLayoutReader::read_and_broadcast] "
                        "CSV file is empty: " + path);
        }

        // Phase 2.5-fix2 / 6.1: Task 6 — Strip UTF-8 BOM and leading/trailing whitespace
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

        // Expected header for physical-coordinate tree CSV
        const std::string expected_header = "x_m,y_m,tree_id,H_tree_m,H_crown_base_m,LAD_bulk,crown_area_frac,Cd_leaf,is_tree";

        auto remove_spaces = [](std::string s) {
            s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
            return s;
        };

        std::string header_no_spaces = remove_spaces(header_line);

        // Validate header format
        if (header_no_spaces != expected_header) {
            std::ostringstream oss;
            oss << "[UCM][6.1][UCMTreeLayoutReader::read_and_broadcast] "
                << "CSV header mismatch.\n"
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

        while (std::getline(csv_file, line))
        {
            // Skip empty lines
            if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
                continue;
            }

            // Skip comment lines
            if (line[0] == '#') {
                continue;
            }

            // Phase 6.1: Strip UTF-8 BOM and whitespace from data rows
            if (line.size() >= 3 &&
                static_cast<unsigned char>(line[0]) == 0xEF &&
                static_cast<unsigned char>(line[1]) == 0xBB &&
                static_cast<unsigned char>(line[2]) == 0xBF) {
                line.erase(0, 3);
            }
            // Strip leading whitespace (space, tab, CR).
            const auto line_first = line.find_first_not_of(" \t\r");
            if (line_first != std::string::npos && line_first > 0) line.erase(0, line_first);
            // Strip trailing whitespace.
            const auto line_last = line.find_last_not_of(" \t\r\n");
            if (line_last != std::string::npos) line.erase(line_last + 1);

            UCMTreeRow row{};
            std::stringstream ss(line);
            std::string field;

            try {
                // Parse fields in physical-mode order
                std::getline(ss, field, ','); row.x_m             = std::stod(field);
                std::getline(ss, field, ','); row.y_m             = std::stod(field);
                std::getline(ss, field, ','); row.tree_id         = std::stoi(field);
                std::getline(ss, field, ','); row.H_tree_m        = std::stod(field);
                std::getline(ss, field, ','); row.H_crown_base_m  = std::stod(field);
                std::getline(ss, field, ','); row.LAD_bulk        = std::stod(field);
                std::getline(ss, field, ','); row.crown_area_frac = std::stod(field);
                std::getline(ss, field, ','); row.Cd_leaf         = std::stod(field);
                std::getline(ss, field, ','); row.is_tree         = std::stoi(field);

                // Validate sanity bounds (Phase 6.1 — Contract #25)
                if (row.H_tree_m < 0.0 || row.H_tree_m > H_tree_max) {
                    amrex::Abort("[UCM][6.1][UCMTreeLayoutReader::read_and_broadcast] "
                                "H_tree_m out of bounds [0, " + std::to_string(H_tree_max) +
                                "]; got " + std::to_string(row.H_tree_m) +
                                " at (x_m=" + std::to_string(row.x_m) + ", y_m=" + std::to_string(row.y_m) + ")");
                }

                if (row.H_crown_base_m < 0.0 || row.H_crown_base_m > row.H_tree_m) {
                    amrex::Abort("[UCM][6.1][UCMTreeLayoutReader::read_and_broadcast] "
                                "H_crown_base_m must be in [0, H_tree_m]; got " + std::to_string(row.H_crown_base_m) +
                                " with H_tree_m=" + std::to_string(row.H_tree_m) +
                                " at (x_m=" + std::to_string(row.x_m) + ", y_m=" + std::to_string(row.y_m) + ")");
                }

                if (row.LAD_bulk < 0.0 || row.LAD_bulk > LAD_max) {
                    amrex::Abort("[UCM][6.1][UCMTreeLayoutReader::read_and_broadcast] "
                                "LAD_bulk out of bounds [0, " + std::to_string(LAD_max) +
                                "]; got " + std::to_string(row.LAD_bulk) +
                                " at (x_m=" + std::to_string(row.x_m) + ", y_m=" + std::to_string(row.y_m) + ")");
                }

                if (row.crown_area_frac < 0.0) {
                    amrex::Abort("[UCM][6.1][UCMTreeLayoutReader::read_and_broadcast] "
                                "crown_area_frac must be >= 0; got " + std::to_string(row.crown_area_frac) +
                                " at (x_m=" + std::to_string(row.x_m) + ", y_m=" + std::to_string(row.y_m) + ")");
                }

                if (row.Cd_leaf < 0.0 || row.Cd_leaf > 1.0) {
                    amrex::Abort("[UCM][6.1][UCMTreeLayoutReader::read_and_broadcast] "
                                "Cd_leaf out of bounds [0, 1.0]; got " + std::to_string(row.Cd_leaf) +
                                " at (x_m=" + std::to_string(row.x_m) + ", y_m=" + std::to_string(row.y_m) + ")");
                }

                if (row.is_tree != 0 && row.is_tree != 1) {
                    amrex::Abort("[UCM][6.1][UCMTreeLayoutReader::read_and_broadcast] "
                                "is_tree must be 0 or 1; got " + std::to_string(row.is_tree) +
                                " at (x_m=" + std::to_string(row.x_m) + ", y_m=" + std::to_string(row.y_m) + ")");
                }

                // Track statistics
                if (n_rows == 0) {
                    min_vals = row;
                    max_vals = row;
                } else {
                    min_vals.tree_id         = std::min(min_vals.tree_id, row.tree_id);
                    min_vals.H_tree_m        = std::min(min_vals.H_tree_m, row.H_tree_m);
                    min_vals.H_crown_base_m  = std::min(min_vals.H_crown_base_m, row.H_crown_base_m);
                    min_vals.LAD_bulk        = std::min(min_vals.LAD_bulk, row.LAD_bulk);
                    min_vals.crown_area_frac = std::min(min_vals.crown_area_frac, row.crown_area_frac);
                    min_vals.Cd_leaf         = std::min(min_vals.Cd_leaf, row.Cd_leaf);

                    max_vals.tree_id         = std::max(max_vals.tree_id, row.tree_id);
                    max_vals.H_tree_m        = std::max(max_vals.H_tree_m, row.H_tree_m);
                    max_vals.H_crown_base_m  = std::max(max_vals.H_crown_base_m, row.H_crown_base_m);
                    max_vals.LAD_bulk        = std::max(max_vals.LAD_bulk, row.LAD_bulk);
                    max_vals.crown_area_frac = std::max(max_vals.crown_area_frac, row.crown_area_frac);
                    max_vals.Cd_leaf         = std::max(max_vals.Cd_leaf, row.Cd_leaf);
                }

                if (row.is_tree == 1) {
                    ++count_is_tree_one;
                }

                m_rows.push_back(row);
                ++n_rows;

            } catch (const std::exception& e) {
                amrex::Abort("[UCM][6.1][UCMTreeLayoutReader::read_and_broadcast] "
                            "Parsing error on line:\n" + line + "\nException: " + std::string(e.what()));
            }
        }

        csv_file.close();

        // Debug trace
        if (ucm_debug) {
            amrex::Print()
                << "\n[UCM][6.1][UCMTreeLayoutReader::read_and_broadcast]\n"
                << "  path = " << path << "\n"
                << "  rows_parsed = " << n_rows << "\n";

            if (n_rows > 0) {
                amrex::Print()
                    << "  tree_id: min=" << min_vals.tree_id << ", max=" << max_vals.tree_id << "\n"
                    << "  H_tree_m: min=" << min_vals.H_tree_m << ", max=" << max_vals.H_tree_m << "\n"
                    << "  H_crown_base_m: min=" << min_vals.H_crown_base_m << ", max=" << max_vals.H_crown_base_m << "\n"
                    << "  LAD_bulk: min=" << min_vals.LAD_bulk << ", max=" << max_vals.LAD_bulk << "\n"
                    << "  crown_area_frac: min=" << min_vals.crown_area_frac << ", max=" << max_vals.crown_area_frac << "\n"
                    << "  Cd_leaf: min=" << min_vals.Cd_leaf << ", max=" << max_vals.Cd_leaf << "\n"
                    << "  is_tree=1 count = " << count_is_tree_one << "\n";
            }
        }
    }

    // =========================================================================
    // Rank 0: Broadcast row count
    // =========================================================================
    amrex::ParallelDescriptor::Bcast(&n_rows, 1, amrex::ParallelDescriptor::IOProcessorNumber());

    // =========================================================================
    // Rank 0: Broadcast row data as raw bytes
    // =========================================================================
    if (n_rows > 0) {
        m_rows.resize(n_rows);
        amrex::ParallelDescriptor::Bcast(m_rows.dataPtr(), n_rows * sizeof(UCMTreeRow),
                                         amrex::ParallelDescriptor::IOProcessorNumber());
    }
}
