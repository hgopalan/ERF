#ifndef ERF_LCP_READER_H
#define ERF_LCP_READER_H

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

/**
 * @file ERF_LcpReader.H
 * @brief FARSITE landscape (.lcp) fuel map reader
 *
 * The layout is FARSITE's packed, little-endian headdata struct (Farsite5.h),
 * which GDAL's LCP driver also reads and writes:
 *
 * - A 7316-byte header. The fields used here:
 *     0     int32   CrownFuels    20 = no crown fuel bands, 21 = present
 *     4     int32   GroundFuels   20 = no ground fuel bands, 21 = present
 *     8     int32   latitude
 *     12    double  loeast, hieast, lonorth, hinorth
 *     44    ten theme summaries of 412 bytes each (lo, hi, count, 100 values)
 *     4164  int32   numeast       columns
 *     4168  int32   numnorth      rows
 *     4172  double  EastUtm, WestUtm, NorthUtm, SouthUtm
 *     4204  int32   GridUnits     0 metric, 1 English
 *     4208  double  XResol, YResol
 *     4224  int16   unit codes, then ten 256-byte file names and a 512-byte
 *                   description
 * - The cells, numnorth rows from the north edge, each row numeast cells from the
 *   west, each cell its int16 bands in turn (interleaved by pixel): elevation,
 *   slope, aspect, fuel model, canopy cover; stand height, crown base height and
 *   crown bulk density when CrownFuels = 21; duff and coarse woody fuel when
 *   GroundFuels = 21.
 *
 * Only rank 0 reads; MPI_Bcast distributes the codes to all ranks.
 */

/// Size of a FARSITE landscape header [bytes]
constexpr std::size_t LCP_HEADER_BYTES = 7316;
/// Position of the fuel model among a cell's bands
constexpr int LCP_FUEL_BAND = 3;

/// The landscape header fields the fuel map reader uses
struct LcpHeader
{
    int crown_fuels  = 0;    ///< 20 = no crown fuel bands, 21 = present
    int ground_fuels = 0;    ///< 20 = no ground fuel bands, 21 = present
    int ncols        = 0;    ///< numeast
    int nrows        = 0;    ///< numnorth
    int grid_units   = 0;    ///< 0 metric, 1 English
    double xres      = 0.0;  ///< cell size east-west, in grid units
    double yres      = 0.0;  ///< cell size north-south, in grid units

    /// Number of int16 bands stored for each cell
    int bands_per_cell () const {
        return 5 + (crown_fuels == 21 ? 3 : 0) + (ground_fuels == 21 ? 2 : 0);
    }
};

namespace lcp_detail {
/// Little-endian decoding, independent of the host's byte order
inline std::int16_t get_i16 (const unsigned char* p)
{
    const auto u = static_cast<std::uint16_t>(p[0] | (p[1] << 8));
    std::int16_t v;
    std::memcpy(&v, &u, sizeof(v));
    return v;
}
inline std::int32_t get_i32 (const unsigned char* p)
{
    const std::uint32_t u = static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8)
                          | (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
    std::int32_t v;
    std::memcpy(&v, &u, sizeof(v));
    return v;
}
inline double get_f64 (const unsigned char* p)
{
    std::uint64_t u = 0;
    for (int k = 7; k >= 0; --k) { u = (u << 8) | p[k]; }
    double v;
    std::memcpy(&v, &u, sizeof(v));
    return v;
}
} // namespace lcp_detail

/**
 * @brief Parse a landscape header.
 *
 * @param[in]  header      The first LCP_HEADER_BYTES bytes of the file (the
 *                         buffer must hold that many bytes)
 * @param[in]  file_bytes  Size of the whole file
 * @param[out] h           The parsed fields
 * @return an empty string when the header describes a landscape the file holds
 *         in full, otherwise the reason it does not
 */
inline std::string parse_lcp_header (const unsigned char* header, std::size_t file_bytes, LcpHeader& h)
{
    using namespace lcp_detail;
    if (file_bytes < LCP_HEADER_BYTES) {
        return "it has " + std::to_string(file_bytes) + " bytes, fewer than the "
               + std::to_string(LCP_HEADER_BYTES) + "-byte header";
    }
    h.crown_fuels  = get_i32(header + 0);
    h.ground_fuels = get_i32(header + 4);
    if ((h.crown_fuels != 20 && h.crown_fuels != 21) || (h.ground_fuels != 20 && h.ground_fuels != 21)) {
        return "its crown and ground fuel flags are " + std::to_string(h.crown_fuels) + " and "
               + std::to_string(h.ground_fuels) + ", where a landscape header has 20 or 21";
    }
    h.ncols      = get_i32(header + 4164);
    h.nrows      = get_i32(header + 4168);
    h.grid_units = get_i32(header + 4204);
    h.xres       = get_f64(header + 4208);
    h.yres       = get_f64(header + 4216);
    if (h.ncols <= 0 || h.nrows <= 0) {
        return "its header gives " + std::to_string(h.ncols) + " x " + std::to_string(h.nrows) + " cells";
    }
    const std::size_t needed = LCP_HEADER_BYTES
        + static_cast<std::size_t>(h.ncols) * static_cast<std::size_t>(h.nrows)
        * static_cast<std::size_t>(h.bands_per_cell()) * sizeof(std::int16_t);
    if (file_bytes < needed) {
        return "it has " + std::to_string(file_bytes) + " bytes, but " + std::to_string(h.ncols) + " x "
               + std::to_string(h.nrows) + " cells of " + std::to_string(h.bands_per_cell())
               + " bands need " + std::to_string(needed);
    }
    return {};
}

/**
 * @brief Read the fuel model band of a FARSITE landscape (.lcp) file.
 *
 * fuel_codes[j * fire_nx + i] has j = 0 at prob_lo_y: the file's first row is the
 * north edge. Negative (no-data) codes become 0. The landscape must have exactly
 * fire_nx columns and fire_ny rows, since it is placed by cell index; the run
 * aborts otherwise, and when the file is not a complete landscape.
 *
 * @param[in]  filename     Path to the landscape file
 * @param[in]  fire_nx      Fire grid resolution in x
 * @param[in]  fire_ny      Fire grid resolution in y
 * @param[out] fuel_codes   Integer fuel codes (size fire_nx * fire_ny)
 * @return false if the file cannot be opened, true otherwise
 */
inline bool read_lcp_fuel_map(const std::string& filename,
                              int fire_nx, int fire_ny,
                              std::vector<int>& fuel_codes)
{
    fuel_codes.clear();

    // Every rank learns whether the file opens, so none waits in a broadcast
    // that rank 0 skipped.
    int opened = 1;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        opened = std::ifstream(filename, std::ios::binary).good() ? 1 : 0;
    }
    amrex::ParallelDescriptor::Bcast(&opened, 1);
    if (opened == 0) {
        return false;
    }

    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        const auto file_bytes = static_cast<std::size_t>(file.tellg());
        file.seekg(0, std::ios::beg);

        std::vector<unsigned char> header(LCP_HEADER_BYTES, 0);
        file.read(reinterpret_cast<char*>(header.data()),
                  static_cast<std::streamsize>(std::min(file_bytes, LCP_HEADER_BYTES)));
        LcpHeader h;
        const std::string problem = parse_lcp_header(header.data(), file_bytes, h);
        if (!problem.empty()) {
            amrex::Abort("Fuel map '" + filename + "' is not a FARSITE landscape file: " + problem);
        }
        if (h.ncols != fire_nx || h.nrows != fire_ny) {
            amrex::Abort("Fuel map '" + filename + "' has " + std::to_string(h.ncols) + " x "
                         + std::to_string(h.nrows) + " cells but the fire grid has "
                         + std::to_string(fire_nx) + " x " + std::to_string(fire_ny)
                         + "; the reader maps one code to each fire cell, so the two must match");
        }

        const auto ncols = static_cast<std::size_t>(h.ncols);
        const auto nbands = static_cast<std::size_t>(h.bands_per_cell());
        std::vector<unsigned char> row(ncols * nbands * sizeof(std::int16_t));
        fuel_codes.assign(ncols * static_cast<std::size_t>(h.nrows), 0);
        for (int r = 0; r < h.nrows; ++r) {
            // File row r counts from the north edge; fire row j from the south.
            file.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(row.size()));
            if (!file) {
                amrex::Abort("Fuel map '" + filename + "' ends in row " + std::to_string(r));
            }
            const auto j = static_cast<std::size_t>(h.nrows - 1 - r);
            for (std::size_t i = 0; i < ncols; ++i) {
                const int code = lcp_detail::get_i16(row.data() + (i * nbands + LCP_FUEL_BAND) * sizeof(std::int16_t));
                fuel_codes[j * ncols + i] = (code < 0) ? 0 : code;
            }
        }
    }

    // Broadcast count and data via MPI
    int count = static_cast<int>(fuel_codes.size());
    amrex::ParallelDescriptor::Bcast(&count, 1);

    if (!amrex::ParallelDescriptor::IOProcessor()) {
        fuel_codes.resize(count);
    }

    if (count > 0) {
        amrex::ParallelDescriptor::Bcast(fuel_codes.data(), count);
    }

    return true;
}

#endif  // ERF_LCP_READER_H
