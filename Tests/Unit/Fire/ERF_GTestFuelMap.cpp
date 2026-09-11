#include <gtest/gtest.h>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_MultiFab.H>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "ERF_FuelMap.H"

/**
 * @file ERF_GTestFuelMap.cpp
 * @brief The ESRI ASCII fuel map reader puts the first data row at the north
 *        edge of the fire grid, and a map loaded with load_from_map carries that
 *        row's fuel load there.
 */

using namespace amrex;

namespace {
constexpr int NX = 5;
constexpr int NY = 4;
const std::string map_file = "erf_gtest_fuel_map_rows.asc";

// One Scott-Burgan code per row, written north row first as ESRI grids are. The
// east cell of the top row differs, so a flip in x shows as well, and one
// nodata cell reads as code 0.
void write_map ()
{
    std::ofstream f(map_file);
    f << "ncols 5\nnrows 4\nxllcorner 0.0\nyllcorner 0.0\ncellsize 10.0\nNODATA_value -9999\n"
      << "183 183 183 183 101\n"    // north edge: TL3, GR1 in the north-east corner
      << "142 142 142 142 142\n"    // SH2
      << "102 102 -9999 102 102\n"  // GR2 with one nodata cell
      << "98 98 98 98 98\n";        // south edge: NB8 open water
}

// The code the map above belongs at fire cell (i, j), j = 0 at the south edge
int expected_code (int i, int j)
{
    if (j == NY - 1) { return (i == NX - 1) ? 101 : 183; }
    if (j == NY - 2) { return 142; }
    if (j == 1)      { return (i == 2) ? 0 : 102; }
    return 98;
}
}

TEST(FuelMap, FirstDataRowIsNorth)
{
    write_map();
    std::vector<int> codes;
    int nodata = 0;
    ASSERT_TRUE(read_ascii_fuel_map(map_file, NX, NY, codes, nodata, 10.0));
    std::remove(map_file.c_str());

    ASSERT_EQ(codes.size(), static_cast<size_t>(NX * NY));
    EXPECT_EQ(nodata, -9999);
    for (int j = 0; j < NY; ++j) {
        for (int i = 0; i < NX; ++i) {
            EXPECT_EQ(codes[j * NX + i], expected_code(i, j)) << "fire cell (" << i << ", " << j << ")";
        }
    }
}

TEST(FuelMap, MissingFileIsReported)
{
    std::vector<int> codes;
    int nodata = 0;
    EXPECT_FALSE(read_ascii_fuel_map("erf_gtest_no_such_fuel_map.asc", NX, NY, codes, nodata));
    EXPECT_TRUE(codes.empty());
}

TEST(FuelMap, LoadFromMapAtNorthEdge)
{
    write_map();
    std::vector<int> codes;
    int nodata = 0;
    ASSERT_TRUE(read_ascii_fuel_map(map_file, NX, NY, codes, nodata));
    std::remove(map_file.c_str());

    // Several boxes, so the global index j * NX + i is used across box edges
    const Box domain(IntVect(0, 0, 0), IntVect(NX - 1, NY - 1, 0));
    BoxArray ba(domain);
    ba.maxSize(IntVect(2, 2, 1));
    const DistributionMapping dm(ba);
    const Geometry geom(domain, RealBox(0.0, 0.0, 0.0, 50.0, 40.0, 1.0),
                        CoordSys::cartesian, {false, false, false});
    MultiFab model(ba, dm, 1, 0);
    MultiFab load(ba, dm, 1, 0);
    load.setVal(-1.0);

    Gpu::DeviceVector<int> d_codes(codes.size());
    Gpu::copy(Gpu::hostToDevice, codes.begin(), codes.end(), d_codes.begin());
    fill_fuel_model_mf(model, d_codes.data(), geom, NX);
    const Real M_live = 0.6;
    fill_fuel_load_from_map(load, model, FUEL_SET_SCOTT_BURGAN40, true, M_live);

    auto load_of = [M_live] (int code) {
        return fuel_total_load_kg_m2(get_fuel_params(code, FUEL_SET_SCOTT_BURGAN40, M_live));
    };
    const Real tl3 = load_of(183);
    const Real gr1 = load_of(101);
    // The rows must carry loads the check can tell apart
    ASSERT_GT(tl3, 0.5);
    ASSERT_GT(std::abs(tl3 - gr1), 0.1);
    ASSERT_GT(std::abs(tl3 - load_of(142)), 0.1);

    const Real tol = (sizeof(Real) == 8) ? 1e-12 : 1e-5;
    int north_cells = 0;
    for (MFIter mfi(load); mfi.isValid(); ++mfi) {
        auto const& m = model.const_array(mfi);
        auto const& w = load.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                EXPECT_EQ(static_cast<int>(m(i, j, 0)), expected_code(i, j)) << "fire cell (" << i << ", " << j << ")";
                if (j == NY - 1) {
                    EXPECT_NEAR(w(i, j, 0), (i == NX - 1) ? gr1 : tl3, tol) << "north edge, i = " << i;
                    ++north_cells;
                } else if (j == 0) {
                    EXPECT_NEAR(w(i, j, 0), 0.0, tol) << "south edge (NB8), i = " << i;
                }
            }
        }
    }
    EXPECT_EQ(north_cells, NX);
}
