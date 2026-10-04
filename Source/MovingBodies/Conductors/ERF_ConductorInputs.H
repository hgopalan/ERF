#ifndef ERF_CONDUCTOR_INPUTS_H
#define ERF_CONDUCTOR_INPUTS_H

/**
 * Inputs of the conductor module (erf.conductors.*): one LineInputs per conductor line, one
 * TransformerInputs per transformer, the lattice tower types, and the shared settings with their
 * range checks; also the still-air elastic catenary used to check MoorDyn's shape, and a helper
 * that finds non-finite values at the module's boundaries.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include <AMReX_REAL.H>

#include "ERF_TowerInputs.H"

namespace erf_conductors {

/**
 * One conductor line from the erf.conductors.<name>.* input block: one conductor (a phase or a
 * shield wire) dead-ended at end_a and end_b, stepped as one MoorDyn system. Without towers the line
 * is a single span between the two ends; with towers it is a section of several spans over
 * suspension towers, the conductor hanging at each tower from an insulator string (or clamped to
 * the tower when insulator_length is 0), so that the spans on either side of a tower balance their
 * pull along the line as on a real line. Keys marked "required" have no default.
 *
 *   erf.conductors.lines            = S1
 *   erf.conductors.S1.end_a         = 100. 500. 30.    (required; m; x, y in ERF coordinates, inside the domain;
 *                                                       z - geometry.prob_lo[2] is the height above the terrain
 *                                                       surface at x, y, so z is that height when prob_lo z is 0)
 *   erf.conductors.S1.end_b         = 1000. 500. 30.   (required; m; as end_a)
 *   erf.conductors.S1.towers        = 400. 500. 30.  700. 500. 30.   (m, the suspension points between the ends, in
 *                                                       order, each as end_a; default none: one span)
 *   erf.conductors.S1.length        = 301.5 301.5 301.5   (m, unstretched, one per span; each must exceed the
 *                                                       distance between the points the conductor hangs from where
 *                                                       they stand on the terrain; required unless stringing_tension)
 *   erf.conductors.S1.stringing_tension = 2.0e4        (N, > 0, instead of length: every span strung to this
 *                                                       horizontal tension in still air; default 0: none)
 *   erf.conductors.S1.diameter      = 0.0281           (required; m, > 0)
 *   erf.conductors.S1.mass_per_length = 1.628          (required; kg/m, more than the air the conductor displaces,
 *                                                       air_density * pi * diameter^2 / 4)
 *   erf.conductors.S1.axial_stiffness = 3.0e7          (required; N, > 0; EA, Young's modulus E times the conductor's
 *                                                       cross-section area A)
 *   erf.conductors.S1.drag_coefficient = 1.0           (>= 0, the normal drag coefficient on the diameter; default 1)
 *   erf.conductors.S1.damping_ratio = 0.5              (in (0, 1], the internal damping of the conductor and of its
 *                                                       insulator strings as a fraction of critical; default 0.5)
 *   erf.conductors.S1.segments      = 20               (>= 2, line segments per span; default 20)
 *   erf.conductors.S1.insulator_length = 2.5           (m, >= 0 and below every tower's z: the suspension strings at
 *                                                       the towers; needs towers; default 0: clamped)
 *   erf.conductors.S1.insulator_mass = 60.             (kg per string, > 0; required with insulator_length, refused
 *                                                       without it)
 *   erf.conductors.S1.insulator_diameter = 0.254       (m, > 0, the disc diameter the wind acts on; default 0.254;
 *                                                       refused without insulator_length)
 *   erf.conductors.S1.output_root   = conductors/S1    (prefix of the line's diagnostics files, one per line;
 *                                                       default <diagnostics_dir>/<name>)
 *   erf.conductors.S1.tower_type    = lattice          (one of erf.conductors.tower_types: the suspension towers are
 *                                                       lattice towers of this type, loaded by the wind; needs towers;
 *                                                       default none: points only)
 *   erf.conductors.S1.share_towers  = P2               (the line hangs from the towers of line P2, at its own points
 *                                                       on them: a circuit's phases and shield wire on one tower; default:
 *                                                       its own towers. P2 has a tower_type and shares no other line's
 *                                                       towers; this line has as many towers and no tower_type of its
 *                                                       own. Its towers' z is measured as end_a's but from the shared
 *                                                       tower's base instead of the terrain under the point, and each of
 *                                                       its tower points lies within arm_length/2 of the tower's axis
 *                                                       and between the base and the top, the cross-arm height + peak)
 *
 * Every Real input must be finite. Lines, transformers and tower types share the prefix
 * erf.conductors.<name>, so no name may be used twice.
 */
struct LineInputs
{
    std::string name;                                //!< the line's name in erf.conductors.lines
    std::array<amrex::Real,3> end_a{{0.0, 0.0, 0.0}};   //!< the first dead end (m; x, y in ERF coordinates, z as in the input block)
    std::array<amrex::Real,3> end_b{{0.0, 0.0, 0.0}};   //!< the last dead end (m; as end_a)
    std::vector<std::array<amrex::Real,3>> towers;   //!< the suspension points between the ends, in order (m; each as end_a)
    std::vector<amrex::Real> lengths;                //!< the unstretched length of each span (m), one per span
    amrex::Real stringing_tension{0.0};              //!< instead of lengths: the still-air horizontal tension of every span (N)
    amrex::Real diameter{0.0};                       //!< the conductor's diameter (m)
    amrex::Real mass_per_length{0.0};                //!< the conductor's mass per unit length (kg/m)
    amrex::Real axial_stiffness{0.0};                //!< EA (N)
    amrex::Real drag_coefficient{1.0};               //!< the normal drag coefficient on the diameter (-)
    amrex::Real damping_ratio{0.5};                  //!< the internal damping as a fraction of critical (-)
    int segments{20};                                //!< line segments per span
    amrex::Real insulator_length{0.0};               //!< the length of each insulator string (m); 0: clamped at the towers
    amrex::Real insulator_mass{0.0};                 //!< the mass of each insulator string (kg)
    amrex::Real insulator_diameter{0.254};           //!< the disc diameter of the insulator strings (m)
    bool insulator_mass_given{false};                //!< insulator_mass was given in the inputs
    bool insulator_diameter_given{false};            //!< insulator_diameter was given in the inputs
    std::string output_root;                         //!< the prefix of the line's diagnostics files
    std::string tower_type;                          //!< the type of the suspension towers; empty: none
    std::string share_towers;                        //!< the line whose towers this one hangs from; empty: its own

    /** Segments of each insulator string. */
    static constexpr int insulator_segments = 2;
    /**
     * The axial stiffness of an insulator string (N): a string of porcelain or glass discs hardly
     * stretches, and 1e7 N lets a 2.5 m string give 4 mm under a 15 kN conductor while keeping
     * MoorDyn's internal step near that of the conductor.
     */
    static constexpr amrex::Real insulator_axial_stiffness = 1.0e7;
    /** The drag coefficient of an insulator string on its disc diameter (-). */
    static constexpr amrex::Real insulator_drag_coefficient = 1.0;

    /** The number of spans: one more than the towers. */
    int num_spans () const { return static_cast<int>(towers.size()) + 1; }
    /** The attachment points in order, end_a, the towers, end_b (k = 0 .. num_spans(); m). */
    const std::array<amrex::Real,3>& point (int k) const;
    std::array<amrex::Real,3>& point (int k);
    /** Whether the conductor hangs from insulator strings at the towers. */
    bool has_insulators () const { return insulator_length > 0.0 && !towers.empty(); }
    /**
     * Where the conductor hangs at attachment point k (k = 0 .. num_spans(): 0 is end_a, num_spans()
     * is end_b): the point itself, or the bottom of the insulator string at a tower (m).
     */
    std::array<amrex::Real,3> conductor_point (int k) const;
    /** The straight distance between the attachment points of span k (k from 0) (m). */
    amrex::Real chord (int k = 0) const;
    /**
     * With stringing_tension, set every span's unstretched length so that it hangs in still air with
     * that horizontal tension H, as a line is strung. Between the points the conductor hangs from
     * (the bottoms of the insulator strings at the towers), a chord c over a horizontal distance h,
     * the parabola of horizontal tension H under the weight w per unit length is
     * l = c + w^2 h^4 / (24 H^2 c) long, the tension along the chord H c / h stretches the line by
     * that over EA, and the unstretched length is l / (1 + H c / (EA h)). All spans of a section
     * then pull alike on the towers between them and the strings hang plumb. Called once the
     * attachments stand on the terrain; w is net of the air's buoyancy (N/m). Nothing happens
     * without stringing_tension.
     */
    void lengths_from_stringing_tension (amrex::Real w);
    /**
     * The sag (m) at mid-span of span k (k from 0) for a parabola between its attachment points whose
     * length is the span's unstretched length; 0 without slack.
     */
    amrex::Real catenary_sag (int k = 0) const;
    /** The nodes of all of the line's MoorDyn lines: the spans', then the insulator strings'. */
    int num_line_nodes () const;
    /** The prefix of span k's diagnostics files (k from 0): output_root for a single span, output_root_span<k+1> in a section. */
    std::string span_root (int k) const;
    /**
     * The name under which span k's statistics are saved in a checkpoint (k from 0): name for a
     * single span, name_span<k+1> in a section.
     */
    std::string span_name (int k) const;
};

/**
 * One transformer of erf.conductors.transformers: a box standing on the terrain where lines end.
 * Every line end (end_a or end_b) whose x, y lies on the box's footprint is dead-ended on it, at the
 * end's placed height, which must clear the box's top. The transformer takes the pull of those
 * lines and is checked against an allowable horizontal force and overturning moment; every
 * conductor is watched for how close it comes to the box. Every transformer needs a line end on it.
 *
 *   erf.conductors.transformers        = T1 T2
 *   erf.conductors.T1.position         = 400. 300.     (required; m, x, y of the footprint's centre, inside the domain;
 *                                                       the box stands on the terrain there)
 *   erf.conductors.T1.size             = 8. 5. 6.      (required; m, > 0: the box's length along x, width along y and height)
 *   erf.conductors.T1.allowable_force  = 2.0e4         (N, >= 0, the horizontal pull the transformer may take; default 0: not checked)
 *   erf.conductors.T1.allowable_moment = 1.5e5         (N m, >= 0, the overturning moment about the centre of its base it
 *                                                       may take; default 0: not checked)
 */
struct TransformerInputs
{
    std::string name;                                   //!< the transformer's name in erf.conductors.transformers
    std::array<amrex::Real,2> position{{0.0, 0.0}};     //!< x, y of the footprint's centre (m)
    std::array<amrex::Real,3> size{{0.0, 0.0, 0.0}};    //!< the length along x, width along y and height (m)
    amrex::Real allowable_force{0.0};                   //!< the allowable horizontal force (N); 0: not checked
    amrex::Real allowable_moment{0.0};                  //!< the allowable overturning moment (N m); 0: not checked

    /** Whether the point x, y (m) lies on the box's footprint (its edges included). */
    bool on_footprint (amrex::Real x, amrex::Real y) const
    {
        return x >= position[0] - amrex::Real(0.5) * size[0] && x <= position[0] + amrex::Real(0.5) * size[0] &&
               y >= position[1] - amrex::Real(0.5) * size[1] && y <= position[1] + amrex::Real(0.5) * size[1];
    }
};

/** The still-air shape of an elastic catenary between two points at the same height. */
struct Catenary
{
    amrex::Real sag{0.0};                //!< mid-span sag below the chord (m)
    amrex::Real horizontal_tension{0.0}; //!< H (N)
    amrex::Real end_tension{0.0};        //!< tension at the attachments (N)
    amrex::Real stretched_length{0.0};   //!< the line's length under its own weight (m)
};

/**
 * The elastic catenary of a span of unstretched length L, net weight w per unit length and axial
 * stiffness EA hanging between two points at the same height a chord c apart: the catenary
 * parameter a = H / w solves 2 a sinh(c / 2a) = L_s, where the stretched length L_s exceeds L by
 * the strain integrated along the span, (H / EA) (c/2 + (a/2) sinh(c/a)). The unstretched length
 * this gives falls as H grows, so H is found by bisection, for a slack span (L > c) or a taut one
 * held by its stretch (L <= c) alike. Used to check MoorDyn's still-air shape.
 * Units: chord, length (m); weight_per_length (N/m, net of the air's buoyancy); axial_stiffness (N).
 * Returns: all zeros when any argument is not positive.
 */
Catenary elastic_catenary (amrex::Real chord, amrex::Real length, amrex::Real weight_per_length,
                           amrex::Real axial_stiffness);

/**
 * The first entry of v that is not finite, with v read as points of per_point components each: an
 * empty string when every entry is finite, else "<what> <p> is not finite (<its components>)", p the
 * point's 0-based index. Used at the conductor module's boundaries (the wind in, MoorDyn's outputs,
 * the forces spread into the flow).
 */
template <typename T>
std::string first_nonfinite (const std::vector<T>& v, std::size_t per_point, const std::string& what)
{
    const std::size_t n = std::max<std::size_t>(per_point, 1);
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (std::isfinite(static_cast<double>(v[i]))) { continue; }
        const std::size_t p = i / n;
        std::string values;
        for (std::size_t c = p * n; c < std::min(v.size(), (p + 1) * n); ++c) {
            values += (c > p * n ? ", " : "") + std::to_string(static_cast<double>(v[c]));
        }
        return what + " " + std::to_string(p) + " is not finite (" + values + ")";
    }
    return std::string();
}

/**
 * The erf.conductors.* inputs: the named conductor lines (one LineInputs block each), the
 * transformers and tower types, and the settings they share.
 *
 *   erf.conductors.lines            = S1 S2        (the conductor lines' names; absent or empty: the module is off)
 *   erf.conductors.diagnostics_dir  = conductors   (where the MoorDyn input files and the diagnostics go)
 *   erf.conductors.diagnostics_int  = 1            (>= 1; write a diagnostics row every this many steps)
 *   erf.conductors.anchor_level     = -1           (0 .. amr.max_level, or -1: the finest; the level the lines are
 *                                                   stepped on and the wind is sampled on)
 *   erf.conductors.air_density      = 1.225        (kg/m^3, > 0, the fluid density MoorDyn applies the drag with)
 *   erf.conductors.substeps         = 1            (>= 1; MoorDyn calls per ERF step for lines on fixed supports, the
 *                                                   wind held over the step; lines on moving towers take at least
 *                                                   this many coupling steps)
 *   erf.conductors.moordyn_dt       = 0            (s, >= 0, an upper bound on MoorDyn's internal step; 0: none)
 *   erf.conductors.moordyn_cfl      = 0.1          (in (0, 0.15], MoorDyn's Courant factor, which sets its internal step)
 *   erf.conductors.moordyn_log_level = 2           (0 debug, 1 messages, 2 warnings, 3 errors only)
 *   erf.conductors.surface_offset   = 10000.       (m; z_md = z_erf - surface_offset, so MoorDyn's free surface sits
 *                                                   at ERF z = surface_offset; it must exceed geometry.prob_hi[2], and
 *                                                   -surface_offset must lie below geometry.prob_lo[2])
 *   erf.conductors.prescribed_velocity = 0. 15. 0. (m/s, ERF frame; default none: the flow is sampled; a uniform
 *                                                   wind at every line node and tower member node instead of the flow,
 *                                                   for testing)
 *   erf.conductors.stats_start      = 0.           (s, >= 0; the running statistics of every span, insulator string
 *                                                   set, line pair, transformer and tower take the steps that end at
 *                                                   or after this time)
 *   erf.conductors.node_output_int  = 0            (>= 0; write every node of every line, insulator strings included,
 *                                                   to <output_root>_nodes.dat every this many steps; 0: never)
 *   erf.conductors.drag_on_flow     = false        (put the force the lines and the towers' members exert on the air,
 *                                                   minus the air's drag on them, into the flow as a momentum source)
 *   erf.conductors.epsilon          = 2.0          (> 0; the radius eps of the kernel exp(-r^2/eps^2) that spreads that
 *                                                   force, cut off at 3 eps, in units of the anchor level's x spacing dx)
 *   erf.conductors.flashover_distance = 1.0        (m, > 0; two lines closer than this are flagged as clashing, and a
 *                                                   conductor this close to a transformer)
 *   erf.conductors.transformers     = T1 T2        (boxes on the terrain the lines end on, each with its own block;
 *                                                   needs lines; default none)
 *   erf.conductors.tower_types      = lattice      (lattice tower types the lines' towers can be, each with its own
 *                                                   block; needs lines; default none)
 *   erf.conductors.asce74_wind      = 40.          (m/s, >= 0; the 3-second gust at 10 m over open country a design
 *                                                   check by ASCE 74 takes (ERF_ASCE74.H): every span's quasi-static
 *                                                   wind load, swing, blowout and tension under it are written to
 *                                                   <diagnostics_dir>/asce74.csv at start-up; default 0: none)
 *   erf.conductors.asce74_exposure  = C            (B or C, ASCE 74's terrain exposure of that check; default C;
 *                                                   needs asce74_wind)
 *
 * Every Real input must be finite.
 */
struct ConductorInputs
{
    bool active{false};                                 //!< erf.conductors.lines lists at least one line
    std::vector<LineInputs> lines;                      //!< the lines, in the order of erf.conductors.lines
    std::vector<TransformerInputs> transformers;        //!< the transformers, in the order of erf.conductors.transformers
    std::vector<erf_towers::TowerType> tower_types;     //!< the tower types, in the order of erf.conductors.tower_types
    std::string diagnostics_dir{"conductors"};
    int diagnostics_int{1};
    int anchor_level{-1};
    amrex::Real air_density{1.225};                     //!< (kg/m^3)
    int substeps{1};
    amrex::Real moordyn_dt{0.0};                        //!< (s)
    amrex::Real moordyn_cfl{0.1};                       //!< (-)
    int moordyn_log_level{2};
    /**
     * The largest Courant factor accepted. MoorDyn's own default, 0.5, and 0.25 make a 300 m ACSR
     * (aluminium conductor, steel reinforced) span diverge within a second in a 20 m/s wind; 0.2 runs
     * but more than doubles the still-air sag; 0.15 and 0.1 agree to the millimetre.
     */
    static constexpr amrex::Real max_moordyn_cfl = 0.15;
    amrex::Real surface_offset{10000.0};                //!< (m)
    bool has_prescribed_velocity{false};
    std::array<amrex::Real,3> prescribed_velocity{{0.0, 0.0, 0.0}};   //!< (m/s, ERF frame)
    amrex::Real stats_start{0.0};                       //!< (s)
    int node_output_int{0};
    bool drag_on_flow{false};
    amrex::Real epsilon{2.0};                           //!< (units of the anchor level's dx)
    amrex::Real flashover_distance{1.0};                //!< (m)
    amrex::Real asce74_wind{0.0};                       //!< (m/s); 0: no ASCE 74 table
    std::string asce74_exposure;                        //!< B or C; empty: C

    /** The line a line's towers belong to: the line whose towers it shares, or itself. */
    const LineInputs& tower_owner (const LineInputs& s) const;
    /** The tower type of a line's towers (those of the line it shares them with), or null when they have none. */
    const erf_towers::TowerType* tower_type (const LineInputs& s) const;
    /** Whether a line's towers move (their type has a frequency): MoorDyn then takes their cross-arms as coupled points. */
    bool towers_move (const LineInputs& s) const
    {
        const erf_towers::TowerType* t = tower_type(s);
        return t != nullptr && t->moves();
    }

    /** Read erf.conductors.*; aborts with the offending key on a bad value. */
    static ConductorInputs read ();

    /**
     * The range checks of one line (every Real finite, then each key's range): an empty string when it
     * is acceptable, else the message to abort with, naming the key. With check_slack the lengths are
     * checked against the chords as well (validate_slack); read() leaves that to after the attachment
     * points are placed on the terrain.
     */
    static std::string validate_line (const LineInputs& s, bool check_slack = true);
    /**
     * Whether every span's unstretched length exceeds the distance between the points the conductor
     * hangs from (conductor_point: the attachment points, or the bottoms of the insulator strings at
     * the towers): an empty string, else the message. A line strung to a tension is not checked: a
     * short span is taut, held by its stretch. on_terrain says the points stand at their absolute
     * heights, whose chords differ from those of the heights above the terrain wherever the ground
     * under the two ends differs; it changes only the message.
     */
    static std::string validate_slack (const LineInputs& s, bool on_terrain = false);
    /** Whether every line has an output_root of its own: an empty string, else the message naming the second line. */
    static std::string validate_output_roots (const std::vector<LineInputs>& lines);
    /** Whether a line's tower_type names one of the tower types and the line has towers: an empty string, else the message. */
    static std::string validate_tower_type (const LineInputs& s, const std::vector<erf_towers::TowerType>& types);
    /**
     * Whether every line that shares towers names a line of the run that has a tower_type, shares
     * no other line's, and has as many towers, and sets no tower_type of its own: an empty string,
     * else the message.
     */
    static std::string validate_shared_towers (const std::vector<LineInputs>& lines);
    /** The range checks of one transformer: an empty string when it is acceptable, else the message to abort with. */
    static std::string validate_transformer (const TransformerInputs& t);
    /**
     * The range checks of the shared settings (every Real finite, then each key's range) and of each
     * line's mass_per_length against the air it displaces: an empty string when acceptable, else the
     * message, naming the key.
     */
    static std::string validate_settings (const ConductorInputs& in);
    /**
     * The solver settings the conductor module supports: an empty string when they hold, else the message.
     * @param max_level amr.max_level
     * @param anchor_level the level the lines are stepped on, already resolved (not -1)
     * @param fpe_traps an amrex.fpe_trap_* input is on (MoorDyn's initial-condition solver overflows under a trap)
     */
    static std::string validate_solver (int max_level, int anchor_level, bool fpe_traps);
    /**
     * Whether MoorDyn's frame holds the whole domain: surface_offset must exceed prob_hi_z, so that
     * every node lies below MoorDyn's free surface at ERF z = surface_offset (MoorDyn applies no fluid
     * load above it), and -surface_offset must lie below prob_lo_z, so that no node reaches MoorDyn's
     * flat bottom at ERF z = -surface_offset. Units: m. Returns an empty string, else the message.
     */
    static std::string validate_frame (amrex::Real surface_offset, amrex::Real prob_lo_z, amrex::Real prob_hi_z);
    /** The level the lines are stepped on: requested, or max_level when requested is -1. */
    static int resolve_anchor_level (int requested, int max_level) { return requested < 0 ? max_level : requested; }
};

} // namespace erf_conductors

#endif
