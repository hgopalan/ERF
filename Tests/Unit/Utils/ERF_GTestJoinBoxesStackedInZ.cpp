#include <algorithm>

#include <AMReX_BoxArray.H>
#include <AMReX_BoxList.H>

#include <gtest/gtest.h>

#include "ERF_Utils.H"

using namespace amrex;

namespace {

// True when two boxes of ba share a face in z
bool
has_boxes_stacked_in_z (const BoxArray& ba)
{
    for (int i = 0; i < ba.size(); ++i) {
        Box above(ba[i]);
        above.setRange(2, ba[i].bigEnd(2)+1);
        if (!ba.intersections(above).empty()) { return true; }
    }
    return false;
}

// The joined boxes must be disjoint and cover exactly the cells of the input
void
expect_same_cells (const BoxArray& ba, const BoxArray& joined)
{
    EXPECT_TRUE(joined.isDisjoint());
    EXPECT_EQ(joined.numPts(), ba.numPts());
    EXPECT_TRUE(joined.contains(ba));
    EXPECT_TRUE(ba.contains(joined));
}

bool
has_box (const BoxArray& ba, const Box& b)
{
    const auto boxes = ba.boxList().data();
    return std::find(boxes.begin(), boxes.end(), b) != boxes.end();
}

const IntVect unlimited_size(1048576);

} // namespace

// Motivation: SDM_Bubble2D_Adv_AMR1 clustered these two level-1 boxes at its first regrid.
// They share the face at k = 10, which the implicit acoustic substep took as the top of one
// column and the bottom of another, so the run stopped at the stacked-box check.
TEST(JoinBoxesStackedInZ, ClusteredSdmBubbleBoxesBecomeThreeColumns)
{
    BoxList bl;
    bl.push_back(Box(IntVect(92,0,6),  IntVect(111,7,9)));
    bl.push_back(Box(IntVect(86,0,10), IntVect(113,7,33)));
    const BoxArray ba(std::move(bl));
    ASSERT_TRUE(has_boxes_stacked_in_z(ba));

    const BoxArray joined = ERFJoinBoxesStackedInZ(ba, unlimited_size);

    expect_same_cells(ba, joined);
    EXPECT_FALSE(has_boxes_stacked_in_z(joined));
    EXPECT_EQ(joined.size(), 3);
    EXPECT_TRUE(has_box(joined, Box(IntVect(86,0,10),  IntVect(91,7,33))));
    EXPECT_TRUE(has_box(joined, Box(IntVect(92,0,6),   IntVect(111,7,33))));
    EXPECT_TRUE(has_box(joined, Box(IntVect(112,0,10), IntVect(113,7,33))));
}

// Boxes side by side, and boxes in one column with coarse cells between them, share no face
// in z; the grids must come back as they are so that decks without stacks keep their grids.
TEST(JoinBoxesStackedInZ, BoxesNotStackedAreReturnedUnchanged)
{
    BoxList bl;
    bl.push_back(Box(IntVect(0,0,0),  IntVect(7,7,3)));
    bl.push_back(Box(IntVect(8,0,0),  IntVect(15,7,9)));
    bl.push_back(Box(IntVect(0,0,5),  IntVect(7,7,9)));
    const BoxArray ba(std::move(bl));

    const BoxArray joined = ERFJoinBoxesStackedInZ(ba, unlimited_size);

    EXPECT_TRUE(joined == ba);
}

// A column chopped by amr.max_grid_size_z is joined back to its full height and then chopped
// in x only; a box in no stack is left alone.
TEST(JoinBoxesStackedInZ, MaxGridSizeAppliesInXAndYOnly)
{
    BoxList bl;
    for (int k = 0; k < 64; k += 16) {
        bl.push_back(Box(IntVect(0,0,k), IntVect(63,15,k+15)));
    }
    const Box lone(IntVect(100,0,0), IntVect(107,7,7));
    bl.push_back(lone);
    const BoxArray ba(std::move(bl));

    const BoxArray joined = ERFJoinBoxesStackedInZ(ba, IntVect(32,32,16));

    expect_same_cells(ba, joined);
    EXPECT_FALSE(has_boxes_stacked_in_z(joined));
    EXPECT_TRUE(has_box(joined, lone));
    EXPECT_TRUE(has_box(joined, Box(IntVect(0,0,0),  IntVect(31,15,63))));
    EXPECT_TRUE(has_box(joined, Box(IntVect(32,0,0), IntVect(63,15,63))));
    EXPECT_EQ(joined.size(), 3);
}

// Three boxes stacked with lateral offsets, the way nested tags cluster, form one stack.
TEST(JoinBoxesStackedInZ, OffsetThreeBoxStackIsJoined)
{
    BoxList bl;
    bl.push_back(Box(IntVect(0,0,0),  IntVect(15,7,3)));
    bl.push_back(Box(IntVect(4,0,4),  IntVect(19,7,9)));
    bl.push_back(Box(IntVect(8,2,10), IntVect(11,5,12)));
    const BoxArray ba(std::move(bl));
    ASSERT_TRUE(has_boxes_stacked_in_z(ba));

    const BoxArray joined = ERFJoinBoxesStackedInZ(ba, unlimited_size);

    expect_same_cells(ba, joined);
    EXPECT_FALSE(has_boxes_stacked_in_z(joined));
    EXPECT_TRUE(has_box(joined, Box(IntVect(8,2,0), IntVect(11,5,12))));
}
