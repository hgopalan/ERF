#ifndef ERF_FUEL_MAP_H
#define ERF_FUEL_MAP_H

#include <AMReX_MultiFab.H>
#include <AMReX_Gpu.H>
#include <AMReX_ParallelDescriptor.H>
#include <ERF_FuelModels.H>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstring>

/**
 * @file ERF_FuelMap.H
 * @brief Spatial fuel map readers (ASCII and binary LCP formats)
 *
 * Ported from wildfire_levelset/src/:
 *   - landscape_file.H (ASCII ESRI grid reader)
 *   - lcp_binary_reader.H (FARSITE LCP reader)
 *
 * Key design patterns:
 * - File I/O on rank 0 only, broadcast via MPI
 * - Device copy after broadcast for GPU kernel access
 * - Proper row reordering: file row 0 = northernmost (highest y)
 *   → reverse to match domain ordering
 */

/**
 * @brief Read an ESRI ASCII grid fuel map file.
 *
 * Format:
 *   ncols <N>
 *   nrows <N>
 *   xllcorner <x>
 *   yllcorner <y>
 *   cellsize <dx>
 *   nodata_value <val>
 *   <rows of integer codes, row 0 = northernmost>
 *
 * Only rank 0 reads the file. The vector is broadcast to all ranks via MPI_Bcast:
 * first the count, then the data.
 *
 * Cells with nodata_value are assigned code 0.
 *
 * Rows are reordered so that fuel_codes[j * ncols + i] corresponds to
 * domain row j at prob_lo_y (file row 0 = northernmost = highest y, so reverse).
 *
 * The map is placed on the fire grid by cell index: it must have exactly
 * fire_nx columns and fire_ny rows (the run aborts otherwise), and the header's
 * corner is not used. A cellsize that differs from fire_dx is reported.
 *
 * @param[in]  filename     Path to ASCII fuel map file
 * @param[in]  fire_nx      Fire grid resolution in x
 * @param[in]  fire_ny      Fire grid resolution in y
 * @param[out] fuel_codes   Vector of integer fuel codes (size fire_nx * fire_ny)
 * @param[out] nodata_val   The nodata_value from the header
 * @param[in]  fire_dx      Fire cell size [m] the header's cellsize is checked
 *                          against; not checked when not positive
 * @return true if file read successfully, false if file cannot be opened
 */
inline bool read_ascii_fuel_map(const std::string& filename,
                                int fire_nx, int fire_ny,
                                std::vector<int>& fuel_codes,
                                int& nodata_val,
                                double fire_dx = 0.0)
{
    int ncols = 0, nrows = 0;
    double xll = 0.0, yll = 0.0, cellsize = 0.0;
    nodata_val = -9999;
    fuel_codes.clear();

    // Every rank learns whether the file opens, so none waits in a broadcast
    // that rank 0 skipped.
    int opened = 1;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        opened = std::ifstream(filename).good() ? 1 : 0;
    }
    amrex::ParallelDescriptor::Bcast(&opened, 1);
    if (opened == 0) {
        return false;  // File cannot be opened
    }

    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ifstream file(filename);

        // Read header
        std::string line;
        std::getline(file, line);
        std::istringstream iss(line);
        std::string key;
        iss >> key >> ncols;

        std::getline(file, line);
        iss.clear();
        iss.str(line);
        iss >> key >> nrows;

        std::getline(file, line);
        iss.clear();
        iss.str(line);
        iss >> key >> xll;

        std::getline(file, line);
        iss.clear();
        iss.str(line);
        iss >> key >> yll;

        std::getline(file, line);
        iss.clear();
        iss.str(line);
        iss >> key >> cellsize;

        std::getline(file, line);
        iss.clear();
        iss.str(line);
        iss >> key >> nodata_val;

        // One code per fire cell: the map is placed by index, so its size must be
        // the fire grid's.
        if (ncols != fire_nx || nrows != fire_ny) {
            amrex::Abort("Fuel map '" + filename + "' has " + std::to_string(ncols) + " x "
                         + std::to_string(nrows) + " cells but the fire grid has "
                         + std::to_string(fire_nx) + " x " + std::to_string(fire_ny)
                         + "; the reader maps one code to each fire cell, so the two must match");
        }
        if (fire_dx > 0.0 && std::abs(cellsize - fire_dx) > 1.0e-3 * fire_dx) {
            amrex::Print() << "[FIRE] WARNING: fuel map '" << filename << "' has cellsize "
                           << cellsize << " but the fire cells are " << fire_dx
                           << " m; the map is placed by cell index, not by coordinates\n";
        }

        // The first data row is the northernmost (highest y), so fill j from the
        // top down: fuel_codes[j * ncols + i] has j = 0 at prob_lo_y.
        std::vector<int> temp_data(static_cast<size_t>(ncols) * static_cast<size_t>(nrows));
        for (int j = nrows - 1; j >= 0; --j) {
            for (int i = 0; i < ncols; ++i) {
                int val = 0;
                if (!(file >> val)) {
                    amrex::Abort("Fuel map '" + filename + "' ends before its "
                                 + std::to_string(ncols) + " x " + std::to_string(nrows) + " codes");
                }
                if (val == nodata_val) {
                    val = 0;  // Assign code 0 to nodata cells
                }
                temp_data[static_cast<size_t>(j) * static_cast<size_t>(ncols) + static_cast<size_t>(i)] = val;
            }
        }

        file.close();
        fuel_codes = temp_data;
    }

    // Broadcast count first
    int count = static_cast<int>(fuel_codes.size());
    amrex::ParallelDescriptor::Bcast(&count, 1);
    amrex::ParallelDescriptor::Bcast(&nodata_val, 1);

    // Resize on non-IO processors
    if (!amrex::ParallelDescriptor::IOProcessor()) {
        fuel_codes.resize(count);
    }

    // Broadcast data
    if (count > 0) {
        amrex::ParallelDescriptor::Bcast(fuel_codes.data(), count);
    }

    return true;
}

// Include LCP reader implementation
#include <ERF_LcpReader.H>

/**
 * @brief Fill a MultiFab with fuel model codes from a device-side array.
 *
 * GPU kernel: for each cell (i,j), set fire_fuel_model(i,j,0) to the
 * integer code at d_fuel_codes[j * fire_nx + i], converted to Real.
 *
 * @param[out] fire_fuel_model MultiFab (1 component) to fill
 * @param[in]  d_fuel_codes    Device pointer to integer fuel codes
 * @param[in]  geom            Fire grid geometry
 * @param[in]  fire_nx         Fire grid resolution in x
 */
inline void fill_fuel_model_mf(amrex::MultiFab& fire_fuel_model,
                                const int* d_fuel_codes,
                                const amrex::Geometry& /*geom*/,
                                int fire_nx)
{
    // ParallelFor over the fire grid
    for (amrex::MFIter mfi(fire_fuel_model); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto fuel_arr = fire_fuel_model.array(mfi);  // Array4 by value — required for GPU lambda capture

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            int idx = j * fire_nx + i;
            fuel_arr(i, j, k) = static_cast<amrex::Real>(d_fuel_codes[idx]);
        });
    }
}

/**
 * @brief Start each fire cell with its own fuel model's load
 *        (erf.fire.fuel_map.load_from_map).
 *
 * Non-burnable codes of the Scott-Burgan set carry no load, and unknown codes
 * follow the set's fall-through in get_fuel_params.
 *
 * @param[out] fire_fuel_load  Fuel load [kg/m2] to fill
 * @param[in]  fire_fuel_model Fuel model code of each cell (fill_fuel_model_mf)
 * @param[in]  fuel_set        Fuel model set id
 * @param[in]  sb40            Whether the Scott-Burgan codes are active
 * @param[in]  M_live          Live fuel moisture for the dynamic models
 */
inline void fill_fuel_load_from_map(amrex::MultiFab& fire_fuel_load,
                                    const amrex::MultiFab& fire_fuel_model,
                                    int fuel_set, bool sb40, amrex::Real M_live)
{
    for (amrex::MFIter mfi(fire_fuel_load); mfi.isValid(); ++mfi) {
        auto const& fuel = fire_fuel_load.array(mfi);
        auto const& code = fire_fuel_model.const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const int c = static_cast<int>(code(i, j, k));
            fuel(i, j, k) = (sb40 && sb40_nonburnable(c)) ? amrex::Real(0.0)
                          : fuel_total_load_kg_m2(get_fuel_params(c, fuel_set, M_live));
        });
    }
}

#endif  // ERF_FUEL_MAP_H
