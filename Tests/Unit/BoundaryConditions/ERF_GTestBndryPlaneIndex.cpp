// Contract of bndry_plane_index, which picks the cell of a boundary plane (read from a precursor's
// boundary registers) that fills each ghost cell beside the face: inside the domain the cell
// itself; along a periodic direction the periodic image, as across the seam inside the domain, so
// that the ghost cells beside an inflow face at y = -1 hold the plane at y = ny - 1, not at y = 0;
// along a non-periodic direction the nearest cell of the plane.

#include <gtest/gtest.h>

#include "ERF_BndryPlaneIndex.H"

TEST(BndryPlaneIndex, InsideTheDomainACellIsItself)
{
    for (int i = 0; i < 96; ++i) {
        EXPECT_EQ(bndry_plane_index(i, 0, 95, true), i);
        EXPECT_EQ(bndry_plane_index(i, 0, 95, false), i);
    }
}

TEST(BndryPlaneIndex, APeriodicDirectionTakesThePeriodicImage)
{
    // the ghost cells either side of a periodic seam, as many as a high-order stencil reaches
    EXPECT_EQ(bndry_plane_index(-1, 0, 95, true), 95);
    EXPECT_EQ(bndry_plane_index(-3, 0, 95, true), 93);
    EXPECT_EQ(bndry_plane_index(96, 0, 95, true), 0);
    EXPECT_EQ(bndry_plane_index(98, 0, 95, true), 2);
    // a domain that does not start at zero, and a ghost further out than the domain is wide
    EXPECT_EQ(bndry_plane_index(3, 4, 7, true), 7);
    EXPECT_EQ(bndry_plane_index(-6, 4, 7, true), 6);
    EXPECT_EQ(bndry_plane_index(12, 4, 7, true), 4);
}

TEST(BndryPlaneIndex, ANonPeriodicDirectionTakesTheNearestCell)
{
    EXPECT_EQ(bndry_plane_index(-1, 0, 47, false), 0);
    EXPECT_EQ(bndry_plane_index(-3, 0, 47, false), 0);
    EXPECT_EQ(bndry_plane_index(48, 0, 47, false), 47);
    EXPECT_EQ(bndry_plane_index(50, 0, 47, false), 47);
}
