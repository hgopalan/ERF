#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "ERF_LcpReader.H"

/**
 * @file ERF_GTestLcpReader.cpp
 * @brief The FARSITE landscape (.lcp) reader: the 7316-byte header, the fuel
 *        band of cells interleaved by pixel, and the first row at the north edge
 *        of the fire grid, for every combination of crown and ground fuel bands.
 */

namespace {
constexpr int NX = 4;
constexpr int NY = 3;
const std::string lcp_file = "erf_gtest_landscape.lcp";
// FARSITE's layout, written out here rather than taken from the reader, so a
// wrong constant in the reader fails the test
constexpr std::size_t HEADER_BYTES = 7316;
constexpr int FUEL_BAND = 3;   // elevation, slope, aspect, fuel model, canopy cover, ...

void put_i32 (std::vector<unsigned char>& b, std::size_t off, std::int32_t v)
{
    std::uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    for (int k = 0; k < 4; ++k) { b[off + k] = static_cast<unsigned char>((u >> (8 * k)) & 0xff); }
}

void put_i16 (std::vector<unsigned char>& b, std::size_t off, int v)
{
    const auto u = static_cast<std::uint16_t>(static_cast<std::int16_t>(v));
    b[off] = static_cast<unsigned char>(u & 0xff);
    b[off + 1] = static_cast<unsigned char>((u >> 8) & 0xff);
}

void put_f64 (std::vector<unsigned char>& b, std::size_t off, double v)
{
    std::uint64_t u;
    std::memcpy(&u, &v, sizeof(u));
    for (int k = 0; k < 8; ++k) { b[off + k] = static_cast<unsigned char>((u >> (8 * k)) & 0xff); }
}

// Fuel code in file row r (r = 0 is the north edge) and column i
int file_fuel (int i, int r)
{
    if (r == 0) { return (i == NX - 1) ? 101 : 183; }  // north edge: TL3, GR1 in the north-east corner
    if (r == 1) { return (i == 2) ? -9999 : 142; }     // SH2 with one no-data cell
    return 98;                                         // south edge: NB8
}

// A NX x NY landscape in FARSITE's layout. The other bands carry values unlike
// any fuel code in the map, so reading the wrong band, or the bands one after
// another instead of cell by cell, shows.
std::vector<unsigned char> landscape (int crown, int ground)
{
    const int nb = 5 + (crown == 21 ? 3 : 0) + (ground == 21 ? 2 : 0);
    std::vector<unsigned char> b(HEADER_BYTES + static_cast<std::size_t>(NX * NY * nb) * 2, 0);
    put_i32(b, 0, crown);
    put_i32(b, 4, ground);
    put_i32(b, 8, 40);
    put_i32(b, 52, -1);            // numelev: the old reader took this as a layer count
    put_i32(b, 4164, NX);
    put_i32(b, 4168, NY);
    put_f64(b, 4172, 500040.0);    // EastUtm, WestUtm, NorthUtm, SouthUtm
    put_f64(b, 4180, 500000.0);
    put_f64(b, 4188, 4000030.0);
    put_f64(b, 4196, 4000000.0);
    put_i32(b, 4204, 0);
    put_f64(b, 4208, 10.0);
    put_f64(b, 4216, 10.0);
    std::size_t off = HEADER_BYTES;
    for (int r = 0; r < NY; ++r) {
        for (int i = 0; i < NX; ++i) {
            for (int band = 0; band < nb; ++band) {
                put_i16(b, off, (band == FUEL_BAND) ? file_fuel(i, r) : 1000 + 100 * band + 10 * r + i);
                off += 2;
            }
        }
    }
    return b;
}

void write_file (const std::vector<unsigned char>& b)
{
    std::ofstream f(lcp_file, std::ios::binary);
    f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
}
}

TEST(LcpReader, FuelBandWithFirstRowNorth)
{
    const std::vector<std::pair<int, int>> flags = {{20, 20}, {21, 20}, {20, 21}, {21, 21}};
    for (const auto& [crown, ground] : flags) {
        write_file(landscape(crown, ground));
        std::vector<int> codes;
        ASSERT_TRUE(read_lcp_fuel_map(lcp_file, NX, NY, codes));
        std::remove(lcp_file.c_str());

        ASSERT_EQ(codes.size(), static_cast<std::size_t>(NX * NY));
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                const int expected = file_fuel(i, NY - 1 - j);
                EXPECT_EQ(codes[j * NX + i], (expected < 0) ? 0 : expected)
                    << "flags " << crown << "/" << ground << ", fire cell (" << i << ", " << j << ")";
            }
        }
    }
}

TEST(LcpReader, HeaderFields)
{
    const auto b = landscape(21, 20);
    LcpHeader h;
    EXPECT_EQ(parse_lcp_header(b.data(), b.size(), h), "");
    EXPECT_EQ(h.crown_fuels, 21);
    EXPECT_EQ(h.ground_fuels, 20);
    EXPECT_EQ(h.ncols, NX);
    EXPECT_EQ(h.nrows, NY);
    EXPECT_EQ(h.grid_units, 0);
    EXPECT_EQ(h.xres, 10.0);
    EXPECT_EQ(h.yres, 10.0);
    EXPECT_EQ(h.bands_per_cell(), 8);
}

TEST(LcpReader, HeaderProblemsAreReported)
{
    LcpHeader h;
    auto b = landscape(20, 20);
    // shorter than the header, and shorter than its cells
    EXPECT_NE(parse_lcp_header(b.data(), 1000, h), "");
    EXPECT_NE(parse_lcp_header(b.data(), b.size() - 2, h), "");
    // crown fuel flag outside 20/21 (0 was the ported reader's guess)
    put_i32(b, 0, 0);
    EXPECT_NE(parse_lcp_header(b.data(), b.size(), h), "");
    // no cells
    b = landscape(20, 20);
    put_i32(b, 4168, 0);
    EXPECT_NE(parse_lcp_header(b.data(), b.size(), h), "");
}

TEST(LcpReader, MissingFileIsReported)
{
    std::vector<int> codes;
    EXPECT_FALSE(read_lcp_fuel_map("erf_gtest_no_such_landscape.lcp", NX, NY, codes));
    EXPECT_TRUE(codes.empty());
}
