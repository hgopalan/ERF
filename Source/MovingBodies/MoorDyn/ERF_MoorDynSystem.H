#ifndef ERF_MOORDYN_SYSTEM_H
#define ERF_MOORDYN_SYSTEM_H

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <moordyn/MoorDyn2.h>

namespace erf_moordyn {

/** The MoorDyn-C version ERF was configured against (from its CMake package), "stub" for the bundled stand-in. */
std::string library_version ();
/** Whether the bundled stub library stands in for MoorDyn-C. */
bool is_stub ();
/** The name of a MoorDyn error code (MOORDYN_SUCCESS, MOORDYN_INVALID_INPUT_FILE, ...). */
std::string error_name (int code);
/**
 * Whether any amrex.fpe_trap_* input is on (read from ParmParse). MoorDyn-C is not trap clean: its
 * stationary initial-condition solver overflows intermediate values (amrex.fpe_trap_overflow ends
 * the run with SIGILL inside MoorDyn_Init in 2.7.1; the invalid and zero traps pass), so a body
 * that runs MoorDyn refuses to start with any of the traps on.
 */
bool fpe_traps_requested ();

/**
 * One MoorDyn-C system (one input file) behind the v2 C API, with ERF's error handling: creation
 * and initialisation report their failure in a string, so a body can name its input; every later
 * call aborts with the MoorDyn function and error code, since nothing sensible can follow it.
 *
 * Lines and points are numbered from 1, as in MoorDyn. The fluid velocity the lines feel comes
 * through MoorDyn's external wave-kinematics calls: external_kinematics_init() returns the number
 * of points MoorDyn wants the velocity at (every line node first; in MoorDyn-C 2.7.1 a fixed-ended
 * line is followed by its attachment points and one entry at MoorDyn's origin),
 * kinematics_points() their current coordinates, and set_kinematics() hands the velocity and
 * acceleration there before each step. The input file must set `WaveKin = 1` for this, and the
 * nodes must lie below z = 0 in MoorDyn's frame, which is where MoorDyn applies its fluid loads.
 */
class MoorDynSystem
{
public:
    /**
     * Create a system from a MoorDyn input file. Returns null and sets err when MoorDyn cannot read
     * it. log_level is a MOORDYN_*_LEVEL value; log_file, when not empty, receives MoorDyn's log.
     */
    static std::unique_ptr<MoorDynSystem> create (const std::string& input_file,
                                                  const std::string& log_file,
                                                  int log_level,
                                                  std::string& err);
    ~MoorDynSystem ();
    MoorDynSystem (const MoorDynSystem&) = delete;
    MoorDynSystem& operator= (const MoorDynSystem&) = delete;

    const std::string& input_file () const { return m_file; }

    /** The number of coupled degrees of freedom (3 per coupled point): 0 when every attachment is fixed. */
    unsigned num_coupled_dof () const;

    /**
     * Initialise the system with the coupled positions and velocities (num_coupled_dof() values
     * each, empty when there are none); with compute_ic MoorDyn solves for the static initial
     * shape, without it the state is expected from load(). Returns an error message, empty on success.
     */
    std::string init (const std::vector<double>& x, const std::vector<double>& xd, bool compute_ic = true);

    /** Start the external kinematics; returns the number of points, 0 with err set when the input file does not enable them. */
    unsigned external_kinematics_init (std::string& err);
    unsigned num_kinematics_points () const { return m_nkin; }
    /** 3*n coordinates of the kinematics points, in MoorDyn's frame. */
    std::vector<double> kinematics_points () const;
    /** Set the fluid velocity and acceleration (3*n each) at the kinematics points, valid at time t. */
    void set_kinematics (const std::vector<double>& U, const std::vector<double>& Ud, double t);

    /**
     * Advance the system by dt (MoorDyn sub-steps internally with its own dtM): the coupled points
     * start the step at x and move at xd over it (MoorDyn places them at x + xd s, s the time into
     * the step), and f receives the coupled forces at its end, the net force of the line ends on each
     * coupled point. t receives MoorDyn's clock, the time since it was initialised.
     */
    void step (const std::vector<double>& x, const std::vector<double>& xd, std::vector<double>& f, double& t, double dt);

    double dt () const;
    void set_dt (double dt);

    unsigned num_lines () const;
    unsigned line_num_nodes (unsigned line) const;
    double line_unstretched_length (unsigned line) const;
    std::array<double,3> line_node_position (unsigned line, unsigned node) const;
    std::array<double,3> line_node_velocity (unsigned line, unsigned node) const;
    /** The tension force vector at a node (N). */
    std::array<double,3> line_node_tension (unsigned line, unsigned node) const;
    /** The fluid's drag on a node (N): the force the air exerts on the line there. */
    std::array<double,3> line_node_drag (unsigned line, unsigned node) const;
    /**
     * The net force on a node (N): the tensions of the segments either side, the weight, the fluid's
     * drag and the rest MoorDyn adds up to move the node. At a fixed end node it is the force the
     * line puts on its support, since the support holds the node still.
     */
    std::array<double,3> line_node_force (unsigned line, unsigned node) const;
    /** The tension at the line's last node (MoorDyn's fairlead tension, N). */
    double line_end_tension (unsigned line) const;
    double line_max_tension (unsigned line) const;

    unsigned num_points () const;
    /** -1 coupled, 0 free, 1 fixed. */
    int point_type (unsigned point) const;
    std::array<double,3> point_position (unsigned point) const;
    /**
     * The net force on a point (N). MoorDyn computes it for the points it integrates (free) and
     * for the coupled ones; a fixed attachment reports zero, and its pull is the tension vector at
     * the end node of each line attached to it (line_node_tension at node 0 or at the last node).
     */
    std::array<double,3> point_force (unsigned point) const;

    /** Write the whole system state to a file (MoorDyn_Save). */
    void save (const std::string& path);
    /** Restore a state written by save() into a system created from the same input and initialised without an IC solve. */
    void load (const std::string& path);
    /** The whole system state in memory (MoorDyn_Serialize) and its restoration (MoorDyn_Deserialize), to redo a step. */
    std::vector<std::uint64_t> serialize () const;
    void deserialize (const std::vector<std::uint64_t>& data);

private:
    MoorDynSystem (MoorDyn sys, std::string file);
    void check (int rc, const std::string& what) const;
    MoorDynLine line (unsigned l) const;
    MoorDynPoint point (unsigned p) const;

    MoorDyn m_sys;
    std::string m_file;
    unsigned m_nkin{0};
};

} // namespace erf_moordyn

#endif
