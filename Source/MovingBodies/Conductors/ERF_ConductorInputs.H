#ifndef ERF_CONDUCTOR_INPUTS_H
#define ERF_CONDUCTOR_INPUTS_H

#include <array>
#include <string>
#include <vector>

#include <AMReX_REAL.H>

namespace erf_conductors {

/**
 * One conductor span from the erf.conductors.<name>.* input block: a line of given
 * unstretched length hanging between two fixed attachment points.
 *
 *   erf.conductors.spans            = S1
 *   erf.conductors.S1.end_a         = 100. 500. 30.    (m; z is the height above the terrain surface at x, y)
 *   erf.conductors.S1.end_b         = 400. 500. 30.
 *   erf.conductors.S1.length        = 301.5            (m, unstretched; must exceed the distance between the ends)
 *   erf.conductors.S1.diameter      = 0.0281           (m)
 *   erf.conductors.S1.mass_per_length = 1.628          (kg/m)
 *   erf.conductors.S1.axial_stiffness = 3.0e7          (N, the product EA)
 *   erf.conductors.S1.drag_coefficient = 1.0           (normal drag coefficient on the diameter; default 1)
 *   erf.conductors.S1.damping_ratio = 0.5              (internal damping as a fraction of critical; default 0.5)
 *   erf.conductors.S1.segments      = 20               (line segments; default 20)
 *   erf.conductors.S1.output_root   = conductors/S1    (prefix of the span's diagnostics files)
 */
struct SpanInputs
{
    std::string name;
    std::array<amrex::Real,3> end_a{{0.0, 0.0, 0.0}};
    std::array<amrex::Real,3> end_b{{0.0, 0.0, 0.0}};
    amrex::Real length{0.0};
    amrex::Real diameter{0.0};
    amrex::Real mass_per_length{0.0};
    amrex::Real axial_stiffness{0.0};
    amrex::Real drag_coefficient{1.0};
    amrex::Real damping_ratio{0.5};
    int segments{20};
    std::string output_root;

    /** The straight distance between the attachment points (m). */
    amrex::Real chord () const;
    /** The sag at mid-span of a parabola whose length is the unstretched length (m). */
    amrex::Real catenary_sag () const;
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
 * The erf.conductors.* inputs: named spans, each with its own block, and the settings they share.
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
