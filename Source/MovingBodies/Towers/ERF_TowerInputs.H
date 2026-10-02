#ifndef ERF_TOWER_INPUTS_H
#define ERF_TOWER_INPUTS_H

#include <string>

#include <AMReX_REAL.H>

namespace erf_towers {

/**
 * A lattice tower type (erf.conductors.tower_types, a block of its own per name): a square
 * lattice body tapering from base_width at the ground to top_width at the cross-arm, with an
 * optional peak above it, and a lattice cross-arm of arm_length across the line at the height
 * the conductor hangs from. The wind acts on the members' projected area, the solidity times
 * the outline of a face.
 *
 *   erf.conductors.tower_types       = lattice
 *   erf.conductors.lattice.base_width = 6.     (m, the body's width at the ground)
 *   erf.conductors.lattice.top_width  = 1.5    (m, its width at the cross-arm)
 *   erf.conductors.lattice.solidity   = 0.2    (the members' area over a face's outline, body and arm)
 *   erf.conductors.lattice.arm_length = 12.    (m, the cross-arm's length, centred on the conductor)
 *   erf.conductors.lattice.arm_depth  = 1.5    (m, the depth of the cross-arm's face; default top_width)
 *   erf.conductors.lattice.peak       = 0.     (m, the body above the cross-arm, at top_width; default 0)
 *   erf.conductors.lattice.drag_coefficient = 0. (the force coefficient on the members' area; default 0:
 *                                                 4 phi^2 - 5.9 phi + 4 from the solidity phi)
 *   erf.conductors.lattice.segments   = 10     (drag nodes along the body up to the cross-arm; default 10)
 *   erf.conductors.lattice.weight     = 9.0e4  (N, the tower's own weight on its foundation; default 0)
 *   erf.conductors.lattice.leg_spacing = 6.    (m, between the four legs' footings, a square; default base_width)
 *   erf.conductors.lattice.allowable_uplift      = 1.0e5 (N, a footing's pull-out capacity; default 0: not checked)
 *   erf.conductors.lattice.allowable_compression = 3.0e5 (N, a footing's bearing capacity; default 0: not checked)
 *   erf.conductors.lattice.frequency  = 2.0    (Hz, the first bending frequency on a rigid foundation; default 0:
 *                                               the tower stands still and the lines hang from fixed points)
 *   erf.conductors.lattice.damping_ratio = 0.02 (the structural damping of that mode; default 0.02)
 *   erf.conductors.lattice.foundation_rotational_stiffness = 1.0e9 (N m/rad, the footing's resistance to tilting;
 *                                                                   default 0: rigid)
 *   erf.conductors.lattice.foundation_lateral_stiffness    = 1.0e8 (N/m, its resistance to sliding; default 0: rigid)
 */
struct TowerType
{
    std::string name;
    amrex::Real base_width{0.0};
    amrex::Real top_width{0.0};
    amrex::Real solidity{0.0};
    amrex::Real arm_length{0.0};
    amrex::Real arm_depth{0.0};
    amrex::Real peak{0.0};
    amrex::Real drag_coefficient{0.0};
    int segments{10};
    amrex::Real weight{0.0};
    amrex::Real leg_spacing{0.0};
    amrex::Real allowable_uplift{0.0};
    amrex::Real allowable_compression{0.0};
    amrex::Real frequency{0.0};
    amrex::Real damping_ratio{0.02};
    amrex::Real foundation_rotational_stiffness{0.0};
    amrex::Real foundation_lateral_stiffness{0.0};

    /** The nodes of the cross-arm. */
    static constexpr int arm_segments = 4;

    /** The depth of the cross-arm's face: arm_depth, or top_width when not given. */
    amrex::Real arm_face () const { return arm_depth > 0.0 ? arm_depth : top_width; }
    /** The spacing of the legs' footings: leg_spacing, or base_width when not given. */
    amrex::Real legs () const { return leg_spacing > 0.0 ? leg_spacing : base_width; }
    /** Whether the tower bends under its loads (a frequency is given), or stands still. */
    bool moves () const { return frequency > 0.0; }
    /** The force coefficient on the members' area: drag_coefficient, or the lattice value of the solidity. */
    amrex::Real force_coefficient () const;
    /** The range checks: an empty string when acceptable, else the message, naming the key. */
    std::string validate () const;
};

} // namespace erf_towers

#endif
