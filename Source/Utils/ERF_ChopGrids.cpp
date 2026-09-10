#include <algorithm>
#include <array>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

#include <ERF_Utils.H>

using namespace amrex;

/**
 * Decompose the base grids to avoid creating too many grids for the number of processors.
 *
 * @param domain Box specifying the domain to decompose.
 * @param decompose_in_z Whether to decompose in the z-direction.
 * @return BoxArray of the decomposed grids.
 */
BoxArray
ERFPostProcessBaseGrids (const Box& domain, bool decompose_in_z)
{
    //
    // This is used to avoid the case where the native amrex decomposition makes
    //     too many grids for the number of processors.
    //
    // The idea is to not override the user preference if expressed by max_grid_size
    //     but instead to ensure that the default behavior is what we want.
    //
    BoxArray ba0 = amrex::decompose(domain, ParallelDescriptor::NProcs(),
                                    {true,true,decompose_in_z});
    return ba0;
}

/**
 * Iteratively decompose grids in 2D until the target number of grids is reached.
 *
 * @param[in,out] ba BoxArray to be decomposed.
 * @param domain Box specifying the domain.
 * @param target_size Target number of grids.
 */
void
ChopGrids2D (BoxArray& ba, const Box& domain, int target_size)
{
    IntVect chunk = domain.length();

    while (ba.size() < target_size)
    {
        IntVect chunk_prev = chunk;

        // We only decompose in x and y, so this array holds only those two directions;
        //     sizing it with AMREX_SPACEDIM would leave a default {0,0} entry that sorts
        //     to the front and pushes the largest direction out of the loop below
        std::array<std::pair<int,int>,2>
            chunk_dir{std::make_pair(chunk[0],int(0)),
                      std::make_pair(chunk[1],int(1))};
        std::sort(chunk_dir.begin(), chunk_dir.end());

        // Try the largest direction first, then the smaller one
        for (int idx = 1; idx >= 0; idx--) {
            int idim = chunk_dir[idx].second;
            int new_chunk_size = chunk[idim] / 2;
            if (new_chunk_size != 0)
            {
                chunk[idim] = new_chunk_size;
                ba.maxSize(chunk);
                break;
            }
        }

        if (chunk == chunk_prev) {
            break;
        }
    }
}

/**
 * Re-cut the boxes of a level so that no two of them share a face in z, covering the same cells.
 *
 * Tagging and clustering stack fine boxes in z wherever the refined region is not made of whole
 * columns of one height.  Boxes that touch in z are gathered into stacks.  Each stack is cut in x
 * and y at the x and y edges of its boxes, so that every piece holds the same runs of cells in z
 * over its whole footprint, and the runs that touch are joined, one box per run.  The pieces are
 * merged back in x and y where their z extents agree and chopped to max_grid_size in x and y.
 * Boxes in no stack are kept as they are; a BoxArray without stacks is returned unchanged.
 *
 * @param ba cell-centered BoxArray of one level.
 * @param max_grid_size Largest box size; only the x and y entries are applied.
 * @return BoxArray covering the cells of ba, with no two boxes stacked in z.
 */
BoxArray
ERFJoinBoxesStackedInZ (const BoxArray& ba, const IntVect& max_grid_size)
{
    AMREX_ALWAYS_ASSERT(ba.ixType().cellCentered());

    const int nboxes = ba.size();

    // Gather the boxes into stacks: link every box to the boxes that sit on its top face
    std::vector<int> stack_of(nboxes);
    std::iota(stack_of.begin(), stack_of.end(), 0);
    auto find_stack = [&stack_of] (int i) {
        while (stack_of[i] != i) {
            stack_of[i] = stack_of[stack_of[i]];
            i = stack_of[i];
        }
        return i;
    };

    bool any_stacked = false;
    for (int i = 0; i < nboxes; ++i) {
        Box above(ba[i]);
        above.setRange(2, ba[i].bigEnd(2)+1);
        for (const auto& isect : ba.intersections(above)) {
            stack_of[find_stack(isect.first)] = find_stack(i);
            any_stacked = true;
        }
    }
    if (!any_stacked) { return ba; }

    // std::map keeps the order, and so the new boxes, the same on every rank
    std::map<int, std::vector<Box>> stacks;
    for (int i = 0; i < nboxes; ++i) {
        stacks[find_stack(i)].push_back(ba[i]);
    }

    BoxList bl_new;
    for (const auto& stack : stacks)
    {
        const std::vector<Box>& boxes = stack.second;
        if (boxes.size() == 1) {
            bl_new.push_back(boxes[0]);
            continue;
        }

        std::array<std::vector<int>,2> cuts;
        for (const auto& b : boxes) {
            for (int idim = 0; idim < 2; ++idim) {
                cuts[idim].push_back(b.smallEnd(idim));
                cuts[idim].push_back(b.bigEnd(idim)+1);
            }
        }
        for (auto& c : cuts) {
            std::sort(c.begin(), c.end());
            c.erase(std::unique(c.begin(), c.end()), c.end());
        }
        auto cut_index = [&cuts] (int idim, int coord) {
            return static_cast<int>(std::lower_bound(cuts[idim].begin(), cuts[idim].end(), coord)
                                     - cuts[idim].begin());
        };

        // The runs of cells in z over each piece, keyed by the piece's x and y cut indices
        std::map<std::pair<int,int>, std::vector<std::pair<int,int>>> runs;
        for (const auto& b : boxes) {
            const int ix_lo = cut_index(0, b.smallEnd(0));
            const int ix_hi = cut_index(0, b.bigEnd(0)+1);
            const int iy_lo = cut_index(1, b.smallEnd(1));
            const int iy_hi = cut_index(1, b.bigEnd(1)+1);
            for (int ix = ix_lo; ix < ix_hi; ++ix) {
                for (int iy = iy_lo; iy < iy_hi; ++iy) {
                    runs[std::make_pair(ix,iy)].emplace_back(b.smallEnd(2), b.bigEnd(2));
                }
            }
        }

        BoxList bl_stack;
        for (auto& piece : runs) {
            const int ix = piece.first.first;
            const int iy = piece.first.second;
            auto& zruns = piece.second;
            std::sort(zruns.begin(), zruns.end());

            // Join the runs that touch; the boxes are disjoint, so the runs never overlap
            int klo = zruns[0].first;
            int khi = zruns[0].second;
            for (std::size_t n = 1; n <= zruns.size(); ++n) {
                if (n < zruns.size() && zruns[n].first == khi+1) {
                    khi = zruns[n].second;
                    continue;
                }
                bl_stack.push_back(Box(IntVect(cuts[0][ix],     cuts[1][iy],     klo),
                                       IntVect(cuts[0][ix+1]-1, cuts[1][iy+1]-1, khi)));
                if (n < zruns.size()) {
                    klo = zruns[n].first;
                    khi = zruns[n].second;
                }
            }
        }

        // Merging in x and y joins pieces with the same z extent, and so cannot stack them again
        bl_stack.simplify();
        bl_stack.maxSize(IntVect(max_grid_size[0], max_grid_size[1],
                                 bl_stack.minimalBox().length(2)));
        bl_new.join(bl_stack);
    }

    return BoxArray(std::move(bl_new));
}
