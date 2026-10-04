#ifndef ERF_TOWER_INPUTS_H
#define ERF_TOWER_INPUTS_H

/** The inputs of a lattice tower type (erf.conductors.<type>.*) and their range checks. */

#include <string>

#include <AMReX_REAL.H>

namespace erf_towers {

/**
 * One lattice tower type: the geometry, weight, damping and foundation shared by every tower of
 * that type, read from the block erf.conductors.<type>.* for each name in erf.conductors.tower_types.
 *
 * The tower's shaft (its "body") is a square lattice tapering linearly from base_width at the
 * ground to top_width at the cross-arm, with an optional peak above the cross-arm. The cross-arm
 * is a lattice of arm_length across the line, at the height the conductor hangs from. The wind
 * acts on the members' projected area: the solidity times the outline of a face.
 *
 *   erf.conductors.tower_types       = lattice
 *   erf.conductors.lattice.base_width = 6.     (m, the shaft's width at the ground; required, > 0)
 *   erf.conductors.lattice.top_width  = 1.5    (m, the shaft's width at the cross-arm; required, in (0, base_width])
 *   erf.conductors.lattice.solidity   = 0.2    (-, the members' area over a face's outline, shaft and cross-arm;
 *                                               required, in (0, 1))
 *   erf.conductors.lattice.arm_length = 12.    (m, the cross-arm's length, centred on the shaft; required, > 0)
 *   erf.conductors.lattice.arm_depth  = 1.5    (m, the depth of the cross-arm's face; >= 0, default 0: top_width)
 *   erf.conductors.lattice.peak       = 0.     (m, the shaft above the cross-arm, at top_width; >= 0, default 0)
 *   erf.conductors.lattice.drag_coefficient = 0. (-, the force coefficient on the members' area; >= 0,
 *                                                 default 0: 4 phi^2 - 5.9 phi + 4 from the solidity phi)
 *   erf.conductors.lattice.segments   = 10     (the number of equal segments of the shaft up to the cross-arm,
 *                                               one drag node at the middle of each; >= 1, default 10)
 *   erf.conductors.lattice.weight     = 9.0e4  (N, the tower's own weight on its foundation; >= 0, default 0;
 *                                               required > 0 when frequency > 0, since it sets the mass)
 *   erf.conductors.lattice.leg_spacing = 6.    (m, between the four legs' footings, a square; >= 0,
 *                                               default 0: base_width)
 *   erf.conductors.lattice.allowable_uplift      = 1.0e5 (N, a footing's pull-out capacity; >= 0, default 0: not checked)
 *   erf.conductors.lattice.allowable_compression = 3.0e5 (N, a footing's bearing capacity; >= 0, default 0: not checked)
 *   erf.conductors.lattice.frequency  = 2.0    (Hz, the first bending frequency on a rigid foundation; >= 0,
 *                                               default 0: the tower stands still and the lines hang from
 *                                               fixed MoorDyn points; > 0 needs weight > 0)
 *   erf.conductors.lattice.damping_ratio = 0.02 (-, the fraction of critical damping of that mode; in [0, 1),
 *                                               default 0.02)
 *   erf.conductors.lattice.foundation_rotational_stiffness = 1.0e9 (N m/rad, the footing's resistance to tilting;
 *                                                                   >= 0, default 0: rigid)
 *   erf.conductors.lattice.foundation_lateral_stiffness    = 1.0e8 (N/m, the footing's resistance to sliding;
 *                                                                   >= 0, default 0: rigid)
 *
 * Every value must also be finite. The members mirror the keys, with the same units.
 */
struct TowerType
{
    std::string name;
    amrex::Real base_width{0.0};                       //!< m
    amrex::Real top_width{0.0};                        //!< m
    amrex::Real solidity{0.0};                         //!< -
    amrex::Real arm_length{0.0};                       //!< m
    amrex::Real arm_depth{0.0};                        //!< m; 0: top_width
    amrex::Real peak{0.0};                             //!< m
    amrex::Real drag_coefficient{0.0};                 //!< -; 0: the lattice curve of the solidity
    int segments{10};                                  //!< equal segments of the shaft up to the cross-arm
    amrex::Real weight{0.0};                           //!< N
    amrex::Real leg_spacing{0.0};                      //!< m; 0: base_width
    amrex::Real allowable_uplift{0.0};                 //!< N; 0: not checked
    amrex::Real allowable_compression{0.0};            //!< N; 0: not checked
    amrex::Real frequency{0.0};                        //!< Hz; 0: the tower stands still
    amrex::Real damping_ratio{0.02};                   //!< - (fraction of critical)
    amrex::Real foundation_rotational_stiffness{0.0};  //!< N m/rad; 0: rigid
    amrex::Real foundation_lateral_stiffness{0.0};     //!< N/m; 0: rigid

    /** The number of drag nodes on the cross-arm, one at the middle of each of its equal segments. */
    static constexpr int arm_segments = 4;

    /** The depth of the cross-arm's face (m): arm_depth, or top_width when arm_depth is 0. */
    amrex::Real arm_face () const { return arm_depth > 0.0 ? arm_depth : top_width; }
    /** The spacing of the legs' footings (m): leg_spacing, or base_width when leg_spacing is 0. */
    amrex::Real legs () const { return leg_spacing > 0.0 ? leg_spacing : base_width; }
    /** Whether the tower bends under its loads (a frequency is given) or stands still. */
    bool moves () const { return frequency > 0.0; }
    /** The force coefficient on the members' area (-): drag_coefficient, or the lattice value of the solidity. */
    amrex::Real force_coefficient () const;
    /** Check every range: returns an empty string when the type is acceptable, else the message, naming the key. */
    std::string validate () const;
};

} // namespace erf_towers

#endif
