/**
 * @file ERF_UCMMaterialRegistry.cpp
 * @brief Material-library registry implementation for the ERF-SLUCM module (Phase 2.1)
 *
 * Implements rank-0-read + MPI_Bcast of material properties following patterns from
 * `Source/LNG/ERF_LNGRegulatory.cpp` and `Source/Dust/ERF_DustSurfaceReader.cpp`.
 */

#include <UrbanCanopy/ERF_UCMMaterialRegistry.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <set>

// Verify POD struct is MPI_Bcast safe
static_assert(std::is_trivially_copyable_v<UCMMaterial>,
              "UCMMaterial must be trivially copyable for MPI_Bcast");

void UCMMaterialRegistry::load_and_broadcast(const std::string& path, int lev, bool ucm_debug)
{
    // Store debug flag for use in lookup()
    m_ucm_debug = ucm_debug;

    // Clear any previous data
    m_table.clear();
    m_id_to_idx.clear();
    m_lookup_trace_emitted.clear();

    int n_materials = 0;

    // =========================================================================
    // Rank 0: Read and parse CSV
    // =========================================================================
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        std::ifstream csv_file(path);
        if (!csv_file.is_open()) {
            amrex::Abort("[UCM][2.1][UCMMaterialRegistry::load_and_broadcast] "
                        "Cannot open file: " + path);
        }

        // Read and validate header
        std::string header_line;
        if (!std::getline(csv_file, header_line)) {
            amrex::Abort("[UCM][2.1][UCMMaterialRegistry::load_and_broadcast] "
                        "CSV file is empty: " + path);
        }

        // Expected header (allow whitespace around delimiters)
        const std::string expected_header = "mat_id,name,albedo,emissivity,k_therm_W_per_mK,"
                                           "rho_cp_J_per_m3K,thickness_m,description";

        // Simple header validation: remove spaces and compare
        auto remove_spaces = [](std::string s) {
            s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
            return s;
        };
        if (remove_spaces(header_line) != remove_spaces(expected_header)) {
            amrex::Abort("[UCM][2.1][UCMMaterialRegistry::load_and_broadcast] "
                        "CSV header mismatch. Expected:\n" + expected_header +
                        "\nGot:\n" + header_line);
        }

        // Read data rows
        std::string line;
        std::set<int> seen_mat_ids;

        while (std::getline(csv_file, line))
        {
            // Skip empty lines
            if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
                continue;
            }

            UCMMaterial mat{};
            std::stringstream ss(line);
            std::string field;

            try {
                // Parse fields (name and description need special handling for spaces)
                std::getline(ss, field, ','); mat.mat_id = std::stoi(field);

                // Name: read until next comma
                std::getline(ss, field, ',');
                if (field.size() < 64) {
                    std::strncpy(mat.name, field.c_str(), 63);
                    mat.name[63] = '\0';
                } else {
                    amrex::Abort("[UCM][2.1][UCMMaterialRegistry::load_and_broadcast] "
                                "Material name too long (>63 chars): " + field);
                }

                // Remaining numeric fields
                std::getline(ss, field, ','); mat.albedo            = std::stod(field);
                std::getline(ss, field, ','); mat.emissivity        = std::stod(field);
                std::getline(ss, field, ','); mat.k_therm_W_per_mK  = std::stod(field);
                std::getline(ss, field, ','); mat.rho_cp_J_per_m3K  = std::stod(field);
                std::getline(ss, field, ','); mat.thickness_m       = std::stod(field);

                // Description: read rest of line
                std::getline(ss, field, '\n');
                if (field.size() < 128) {
                    std::strncpy(mat.description, field.c_str(), 127);
                    mat.description[127] = '\0';
                } else {
                    amrex::Abort("[UCM][2.1][UCMMaterialRegistry::load_and_broadcast] "
                                "Material description too long (>127 chars)");
                }

                // Validate ranges
                if (mat.albedo < 0.0 || mat.albedo > 1.0) {
                    amrex::Abort("[UCM][2.1][UCMMaterialRegistry::load_and_broadcast] "
                                "Albedo must be in [0, 1]; got " + std::to_string(mat.albedo) +
                                " for mat_id=" + std::to_string(mat.mat_id));
                }
                if (mat.emissivity < 0.0 || mat.emissivity > 1.0) {
                    amrex::Abort("[UCM][2.1][UCMMaterialRegistry::load_and_broadcast] "
                                "Emissivity must be in [0, 1]; got " + std::to_string(mat.emissivity) +
                                " for mat_id=" + std::to_string(mat.mat_id));
                }
                if (mat.k_therm_W_per_mK <= 0.0) {
                    amrex::Abort("[UCM][2.1][UCMMaterialRegistry::load_and_broadcast] "
                                "k_therm must be > 0; got " + std::to_string(mat.k_therm_W_per_mK) +
                                " for mat_id=" + std::to_string(mat.mat_id));
                }
                if (mat.rho_cp_J_per_m3K <= 0.0) {
                    amrex::Abort("[UCM][2.1][UCMMaterialRegistry::load_and_broadcast] "
                                "rho_cp must be > 0; got " + std::to_string(mat.rho_cp_J_per_m3K) +
                                " for mat_id=" + std::to_string(mat.mat_id));
                }
                if (mat.thickness_m <= 0.0) {
                    amrex::Abort("[UCM][2.1][UCMMaterialRegistry::load_and_broadcast] "
                                "thickness_m must be > 0; got " + std::to_string(mat.thickness_m) +
                                " for mat_id=" + std::to_string(mat.mat_id));
                }

                // Check for duplicate mat_id
                if (seen_mat_ids.find(mat.mat_id) != seen_mat_ids.end()) {
                    amrex::Abort("[UCM][2.1][UCMMaterialRegistry::load_and_broadcast] "
                                "Duplicate mat_id=" + std::to_string(mat.mat_id));
                }
                seen_mat_ids.insert(mat.mat_id);

                m_table.push_back(mat);
                m_id_to_idx[mat.mat_id] = n_materials;
                ++n_materials;

            } catch (const std::exception& e) {
                amrex::Abort("[UCM][2.1][UCMMaterialRegistry::load_and_broadcast] "
                            "Parsing error on line:\n" + line + "\nException: " + std::string(e.what()));
            }
        }

        csv_file.close();

        // Debug trace
        if (ucm_debug) {
            amrex::Print()
                << "\n[UCM][2.1][UCMMaterialRegistry::load_and_broadcast]\n"
                << "  path = " << path << "\n"
                << "  n_materials = " << n_materials << "\n";
            for (int i = 0; i < n_materials; ++i) {
                const UCMMaterial& mat = m_table[i];
                amrex::Print()
                    << "  mat_id=" << mat.mat_id
                    << ": name=\"" << mat.name
                    << "\", albedo=" << std::fixed << std::setprecision(3) << mat.albedo
                    << ", emissivity=" << mat.emissivity
                    << ", k=" << mat.k_therm_W_per_mK
                    << ", rho_cp=" << std::scientific << mat.rho_cp_J_per_m3K
                    << std::fixed << ", thickness=" << mat.thickness_m << "\n";
            }
        }
    }

    // =========================================================================
    // All ranks: MPI_Bcast material data (as raw bytes to avoid needing
    // amrex::ParallelDescriptor::Mpi_typemap<UCMMaterial> specialization).
    // =========================================================================

    // Broadcast material count
    amrex::ParallelDescriptor::Bcast(&n_materials, 1,
                                     amrex::ParallelDescriptor::IOProcessorNumber());

    // Resize on all ranks
    if (!amrex::ParallelDescriptor::IOProcessor()) {
        m_table.resize(n_materials);
    }

    // Broadcast material data as bytes (selects the char*/byte-count overload)
    if (n_materials > 0) {
        amrex::ParallelDescriptor::Bcast(
            reinterpret_cast<char*>(m_table.dataPtr()),
            static_cast<std::size_t>(n_materials) * sizeof(UCMMaterial),
            amrex::ParallelDescriptor::IOProcessorNumber());

        // Reconstruct m_id_to_idx on all ranks
        m_id_to_idx.clear();
        for (int i = 0; i < n_materials; ++i) {
            m_id_to_idx[m_table[i].mat_id] = i;
        }

        if (amrex::ParallelDescriptor::IOProcessor() && ucm_debug) {
            amrex::Print()
                << "  MPI_Bcast: " << n_materials << " materials ("
                << (n_materials * sizeof(UCMMaterial)) << " bytes) to all ranks\n";
        }
    }
}

const UCMMaterial& UCMMaterialRegistry::lookup(int mat_id) const
{
    auto it = m_id_to_idx.find(mat_id);
    if (it == m_id_to_idx.end()) {
        amrex::Abort("[UCM][2.1][UCMMaterialRegistry::lookup] "
                    "Material mat_id=" + std::to_string(mat_id) + " not found in registry");
    }

    // One-time debug trace per unique mat_id
    if (m_ucm_debug && m_lookup_trace_emitted.find(mat_id) == m_lookup_trace_emitted.end()) {
        const UCMMaterial& mat = m_table[it->second];
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print()
                << "[UCM][2.1][UCMMaterialRegistry::lookup] mat_id=" << mat_id
                << " first call → name=\"" << mat.name
                << "\", albedo=" << std::fixed << std::setprecision(3) << mat.albedo << "\n";
        }
        m_lookup_trace_emitted.insert(mat_id);
    }

    return m_table[it->second];
}