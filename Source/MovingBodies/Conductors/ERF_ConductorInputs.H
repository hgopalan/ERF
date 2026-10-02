#ifndef ERF_CONDUCTOR_INPUTS_H
#define ERF_CONDUCTOR_INPUTS_H

#include <array>
#include <string>
#include <vector>

#include <AMReX_REAL.H>

namespace erf_conductors {

/**
 * One conductor line from the erf.conductors.<name>.* input block: a line of given unstretched
 * length per span, dead-ended at end_a and end_b. Without towers it is a single span between
 * the two; with towers it is a section of several spans over suspension towers in between, the
 * conductor hanging at each tower from an insulator string (or clamped to the tower when
 * insulator_length is 0), so that the spans on either side of a tower balance their pull along
 * the line as on a real line.
 *
 *   erf.conductors.spans            = S1
 *   erf.conductors.S1.end_a         = 100. 500. 30.    (m; z is the height above the terrain surface at x, y)
 *   erf.conductors.S1.end_b         = 1000. 500. 30.
 *   erf.conductors.S1.towers        = 400. 500. 30.  700. 500. 30.   (m, the suspension points between the ends, in order; none: one span)
 *   erf.conductors.S1.length        = 301.5 301.5 301.5   (m, unstretched, one per span; each must exceed its chord)
 *   erf.conductors.S1.diameter      = 0.0281           (m)
 *   erf.conductors.S1.mass_per_length = 1.628          (kg/m)
 *   erf.conductors.S1.axial_stiffness = 3.0e7          (N, the product EA)
 *   erf.conductors.S1.drag_coefficient = 1.0           (normal drag coefficient on the diameter; default 1)
 *   erf.conductors.S1.damping_ratio = 0.5              (internal damping as a fraction of critical; default 0.5)
 *   erf.conductors.S1.segments      = 20               (line segments per span; default 20)
 *   erf.conductors.S1.insulator_length = 2.5           (m, the suspension strings at the towers; default 0: clamped)
 *   erf.conductors.S1.insulator_mass = 60.             (kg per string; needed with insulator_length)
 *   erf.conductors.S1.insulator_diameter = 0.254       (m, the disc diameter the wind acts on; default 0.254)
 *   erf.conductors.S1.output_root   = conductors/S1    (prefix of the line's diagnostics files)
 */
struct SpanInputs
{
    std::string name;
    std::array<amrex::Real,3> end_a{{0.0, 0.0, 0.0}};
    std::array<amrex::Real,3> end_b{{0.0, 0.0, 0.0}};
    std::vector<std::array<amrex::Real,3>> towers;   //!< the suspension points between the ends, in order
    std::vector<amrex::Real> lengths;                //!< unstretched, one per span
    amrex::Real diameter{0.0};
    amrex::Real mass_per_length{0.0};
    amrex::Real axial_stiffness{0.0};
    amrex::Real drag_coefficient{1.0};
    amrex::Real damping_ratio{0.5};
    int segments{20};
    amrex::Real insulator_length{0.0};
    amrex::Real insulator_mass{0.0};
    amrex::Real insulator_diameter{0.254};
    std::string output_root;

    /** Segments of each insulator string. */
    static constexpr int insulator_segments = 2;
    /**
     * The axial stiffness of an insulator string (N): a string of porcelain or glass discs hardly
     * stretches, and 1e7 N lets a 2.5 m string give 4 mm under a 15 kN conductor while keeping
     * MoorDyn's internal step near that of the conductor.
     */
    static constexpr amrex::Real insulator_axial_stiffness = 1.0e7;
    /** The drag coefficient of an insulator string on its disc diameter. */
    static constexpr amrex::Real insulator_drag_coefficient = 1.0;

    /** The number of spans: one more than the towers. */
    int num_spans () const { return static_cast<int>(towers.size()) + 1; }
    /** The attachment points in order, end_a, the towers, end_b (k = 0 .. num_spans()). */
    const std::array<amrex::Real,3>& point (int k) const;
    std::array<amrex::Real,3>& point (int k);
    /** Whether the conductor hangs from insulator strings at the towers. */
    bool has_insulators () const { return insulator_length > 0.0 && !towers.empty(); }
    /** The straight distance between the attachment points of span k (m). */
    amrex::Real chord (int k = 0) const;
    /** The sag at mid-span of span k for a parabola whose length is the span's unstretched length (m). */
    amrex::Real catenary_sag (int k = 0) const;
    /** The nodes of all of the line's MoorDyn lines: the spans', then the insulator strings'. */
    int num_line_nodes () const;
    /** The prefix of span k's diagnostics files: output_root for a single span, output_root_span<k+1> in a section. */
    std::string span_root (int k) const;
    /** The name span k's statistics are saved under in a checkpoint. */
    std::string span_name (int k) const;
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
 * The elastic catenary of a line of unstretched length L, net weight w per unit length and axial
 * stiffness EA hanging between two points at the same height a chord c apart: the catenary
 * parameter a = H / w solves 2 a sinh(c / 2a) = L_s, where the stretched length L_s exceeds L by
 * the strain integrated along the line, (H / EA) (c/2 + (a/2) sinh(c/a)). Solved by Newton
 * iteration on a inside a fixed-point iteration on L_s. Used to check MoorDyn's still-air shape.
 */
Catenary elastic_catenary (amrex::Real chord, amrex::Real length, amrex::Real weight_per_length,
                           amrex::Real axial_stiffness);

/**
 * The erf.conductors.* inputs: named conductor lines, each with its own block, and the settings they share.
 *
 *   erf.conductors.diagnostics_dir  = conductors   (where the MoorDyn input files and the diagnostics go)
 *   erf.conductors.diagnostics_int  = 1            (write a diagnostics row every this many steps)
 *   erf.conductors.anchor_level     = -1           (the level the spans are sampled on; -1: the finest)
 *   erf.conductors.air_density      = 1.225        (kg/m^3, the fluid density MoorDyn applies the drag with)
 *   erf.conductors.substeps         = 1            (MoorDyn calls per ERF step, the wind held over the step)
 *   erf.conductors.moordyn_dt       = 0            (s, an upper bound on MoorDyn's internal step; 0: none)
 *   erf.conductors.moordyn_cfl      = 0.1          (MoorDyn's Courant factor, which sets its internal step; at most 0.15)
 *   erf.conductors.moordyn_log_level = 2           (0 debug, 1 messages, 2 warnings, 3 errors only)
 *   erf.conductors.surface_offset   = 10000.       (m; MoorDyn's free surface sits this far above ERF's z = 0)
 *   erf.conductors.prescribed_velocity = 0. 15. 0. (m/s; a uniform wind at every line node instead of the flow, for testing)
 *   erf.conductors.stats_start      = 0.           (s; the running statistics of every span start here)
 *   erf.conductors.node_output_int  = 0            (write every node of every span every this many steps; 0: never)
 *   erf.conductors.drag_on_flow     = false        (put the lines' drag back into the flow as a momentum source)
 *   erf.conductors.epsilon          = 2.0          (width of the Gaussian that spreads the drag, in cells)
 *   erf.conductors.flashover_distance = 1.0        (m; two lines closer than this are flagged as clashing)
 */
struct ConductorInputs
{
    bool active{false};
    std::vector<SpanInputs> spans;
    std::string diagnostics_dir{"conductors"};
    int diagnostics_int{1};
    int anchor_level{-1};
    amrex::Real air_density{1.225};
    int substeps{1};
    amrex::Real moordyn_dt{0.0};
    amrex::Real moordyn_cfl{0.1};
    int moordyn_log_level{2};
    /**
     * The largest Courant factor accepted. MoorDyn's own default, 0.5, and 0.25 make a 300 m ACSR span
     * diverge within a second in a 20 m/s wind; 0.2 runs but more than doubles the still-air sag; 0.15
     * and 0.1 agree to the millimetre.
     */
    static constexpr amrex::Real max_moordyn_cfl = 0.15;
    amrex::Real surface_offset{10000.0};
    bool has_prescribed_velocity{false};
    std::array<amrex::Real,3> prescribed_velocity{{0.0, 0.0, 0.0}};
    amrex::Real stats_start{0.0};
    int node_output_int{0};
    bool drag_on_flow{false};
    amrex::Real epsilon{2.0};
    amrex::Real flashover_distance{1.0};

    /** Read erf.conductors.*; aborts with the offending key on a bad value. */
    static ConductorInputs read ();

    /** The range checks of one span: an empty string when it is acceptable, else the message to abort with. */
    static std::string validate_span (const SpanInputs& s);
    /** The range checks of the shared settings: an empty string when acceptable, else the message. */
    static std::string validate_settings (const ConductorInputs& in);
    /**
     * The solver settings this version supports: an empty string when they hold, else the message.
     * @param max_level amr.max_level
     * @param anchor_level the level the spans live on, already resolved
     * @param fpe_traps an amrex.fpe_trap_* input is on (MoorDyn's initial-condition solver overflows under a trap)
     */
    static std::string validate_solver (int max_level, int anchor_level, bool fpe_traps);
    static int resolve_anchor_level (int requested, int max_level) { return requested < 0 ? max_level : requested; }
};

} // namespace erf_conductors

#endif
