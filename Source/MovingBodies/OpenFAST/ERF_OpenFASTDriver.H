#ifndef ERF_OPENFAST_DRIVER_H
#define ERF_OPENFAST_DRIVER_H

#include <array>
#include <string>
#include <vector>

#include <AMReX_REAL.H>

#include "ERF_MovingBodiesInputs.H"

// OpenFAST 5 C API, unchanged since 4.0 (the real header, or Stub/FAST_Library.h when ERF_OPENFAST_USE_STUB is set)
#include "FAST_Library.h"

namespace erf_openfast {

/**
 * Number of OpenFAST steps per ERF step. OpenFAST needs a fixed step, and the ERF step must be a
 * positive whole multiple of it. On failure returns 0 and sets err to the message to abort with.
 */
int substep_count (double dt_cfd, double dt_fast, std::string& err);

/**
 * The MPI rank that owns turbine i of a farm: the turbines are dealt round-robin over the
 * ranks, so a farm of up to nprocs turbines has one OpenFAST instance per rank and a larger
 * one spreads evenly. Every rank computes the same assignment.
 */
int owner_rank_for (int turbine_index, int nprocs);

/**
 * Whether the OpenFAST model computes its own rotor induction. When ERF puts the rotor's
 * loads into the flow, the velocities it samples already contain the induction, so AeroDyn's
 * wake model must be off (Wake_Mod = 0, or WakeMod = 0 in older AeroDyn files). Reads the
 * AeroFile named in the .fst; a model without one (CompAero = 0, or the bundled stub) passes.
 *
 * @return an empty string when the model is acceptable, otherwise the reason
 */
std::string check_induction_off (const std::string& fst_file);

/**
 * Whether the OpenFAST model applies its own tower-shadow correction to the blade inflow. When
 * ERF puts the tower's drag into the flow, the tower wake reaches the blades through the
 * sampled velocities, so AeroDyn's TwrShadow should be 0. Returns an empty string when it is
 * (or the model has no AeroDyn file), otherwise a warning naming the setting.
 */
std::string check_tower_shadow_off (const std::string& fst_file);

/**
 * One OpenFAST turbine. Sizes and node data are valid on every rank after init(); the ExtInfw
 * buffers are only allocated on the owner rank, which is the only rank that calls OpenFAST.
 */
struct TurbineState
{
    // inputs
    std::string name;
    std::string fst_file;
    std::string output_root;
    std::array<amrex::Real,3> base_pos{{0.0, 0.0, 0.0}};
    int num_force_pts_blade{0};
    int num_force_pts_tower{0};
    // ownership
    int owner_rank{0};
    int tid_local{-1};                 //!< index in the owner's OpenFAST turbine array
    // time stepping
    double dt_fast{0.0};
    int num_substeps{0};
    int time_index{0};                 //!< OpenFAST steps taken since init
    // sizes reported by OpenFAST
    int num_blades{0};
    int num_blade_elem{0};             //!< structural (velocity) nodes per blade
    int num_tower_elem{0};             //!< structural (velocity) nodes on the tower
    int num_vel_nodes{0};
    int num_force_nodes{0};
    // node data on every rank (hub first, then blades root to tip, then tower base to top)
    std::vector<amrex::Real> vel_pos;    //!< 3*num_vel_nodes velocity-node positions (m)
    std::vector<amrex::Real> force_pos;  //!< 3*num_force_nodes force-node positions (m)
    std::vector<amrex::Real> force;      //!< 3*num_force_nodes forces on the structure, as OpenFAST reports them (N)
    std::vector<amrex::Real> force_vel;  //!< 3*num_force_nodes velocities of the force nodes (the blade motion, m/s)
    std::vector<amrex::Real> chord;      //!< num_force_nodes chord at the force nodes (m; OpenFAST's, zero on the hub and tower)
    std::vector<amrex::Real> node_vel;   //!< 3*num_vel_nodes velocities the flow supplied at the velocity nodes (m/s)
    std::array<amrex::Real,3> hub_pos{{0.0, 0.0, 0.0}};
    std::array<amrex::Real,3> hub_axis{{1.0, 0.0, 0.0}}; //!< unit shaft axis (hub frame x) in ERF's frame
    amrex::Real rotor_speed{0.0};        //!< rad/s about the hub axis
    std::array<amrex::Real,3> nacelle_force{{0.0, 0.0, 0.0}}; //!< the flow's drag on the nacelle (on the structure, N), set by the caller
    // ExtInfw buffers, owner rank only; positions and forces come from OpenFAST in to_cfd and the
    // velocities the flow supplies go into from_cfd
    ExtInfw_InputType_t  to_cfd{};
    ExtInfw_OutputType_t from_cfd{};
};

/**
 * Drives one or more OpenFAST turbines through the ExtInfw C API. Each turbine is owned by one
 * MPI rank (owner_rank_for), which alone runs its OpenFAST instance; after every step the node
 * positions and forces are broadcast so every rank can use them. A farm therefore costs each
 * rank only the turbines it owns, and the result does not depend on the rank count. This version initialises, steps and reports; the flow velocity at the nodes is
 * uniform and prescribed by the caller.
 */
class OpenFASTDriver
{
public:
    explicit OpenFASTDriver (const std::vector<MovingBodyInputs>& bodies);
    ~OpenFASTDriver ();

    OpenFASTDriver (const OpenFASTDriver&) = delete;
    OpenFASTDriver& operator= (const OpenFASTDriver&) = delete;

    /**
     * Allocate and initialise every turbine and broadcast the node layout, so the caller can
     * sample the flow at the velocity nodes before the first solution. Aborts when a model does
     * not take external inflow (CompInflow must be 2) or when dt_cfd is not a whole multiple of
     * the model's dt.
     * @param dt_cfd the fixed ERF step
     * @param t_max the OpenFAST stop time
     */
    void init (double dt_cfd, double t_max);

    /**
     * Instead of init(): restore every turbine from the OpenFAST checkpoint <prefix><name>.chkp
     * written by create_checkpoint(), and take the time index from it. The first solution is
     * then already taken; the next call is step().
     */
    void restart (const std::string& prefix, double dt_cfd);

    /** Write every turbine's OpenFAST checkpoint as <prefix><name>.chkp (its owner rank writes). */
    void create_checkpoint (const std::string& prefix) const;

    /**
     * Run OpenFAST's first solution with the node velocities set so far, then broadcast the
     * positions and forces and open the diagnostics files. Must follow init() and precede step().
     */
    void solution0 ();

    /** Set the same velocity at every velocity node of every turbine. */
    void set_uniform_velocity (const std::array<amrex::Real,3>& vel);

    /** Set the velocity at each velocity node of turbine i (3*num_vel_nodes values, node order). */
    void set_node_velocities (int i, const std::vector<amrex::Real>& uvw);

    /** Record the nacelle drag of turbine i (the force on the structure, N) for the diagnostics. */
    void set_nacelle_force (int i, const std::array<amrex::Real,3>& f);

    /**
     * After restart(): give turbine i the node forces (3*num_force_nodes) and node velocities
     * (3*num_vel_nodes) ERF checkpointed at the same time, so that what depends on the loads of
     * the step before the checkpoint (the lifting-line correction) continues exactly; a library
     * that restores them itself reports the same values.
     */
    void set_restored_loads (int i, const std::vector<amrex::Real>& force, const std::vector<amrex::Real>& node_vel);

    /** Advance every turbine by one ERF step (num_substeps OpenFAST steps) and broadcast. */
    void step ();

    /**
     * Append one row per turbine to its two diagnostics files: <output_root>_erf.csv with time,
     * rotor speed, thrust (hub and blade nodes), torque and power about the hub axis, the hub
     * axis itself, the tower force (its force nodes, zero without them) and the nacelle drag
     * (zero unless set), all forces on the structure; and <output_root>_flow.csv with the
     * velocity the flow supplied at the hub node and its mean over the blade nodes.
     */
    void write_diagnostics (double time);

    const std::vector<TurbineState>& turbines () const { return m_turb; }
    bool is_owner (int i) const;
    bool initialized () const { return m_initialized; }
    bool solved0 () const { return m_solved0; }

    /** Thrust vector: sum of the hub and blade-node forces (N). */
    std::array<amrex::Real,3> thrust (const TurbineState& t) const;
    /** Sum of the tower-node forces (N); zero without tower force nodes. */
    std::array<amrex::Real,3> tower_force (const TurbineState& t) const;
    /** The velocity the flow supplied at the hub node and its mean over the blade nodes. */
    void node_velocity_means (const TurbineState& t, std::array<amrex::Real,3>& hub, std::array<amrex::Real,3>& blade) const;
    /** Aerodynamic torque about the hub axis from the blade-node forces (N m). */
    amrex::Real torque (const TurbineState& t) const;

private:
    std::vector<TurbineState> m_turb;
    bool m_initialized{false};
    bool m_solved0{false};
    int m_num_local{0};

    void fast_check (int err_stat, const char* err_msg, const std::string& where) const;
    void pull_from_fast (TurbineState& t);
    void broadcast_state (TurbineState& t);
    void allocate ();
    void finish_setup (TurbineState& t, double dt_cfd);
    void open_diagnostics (const TurbineState& t, bool truncate) const;
};

} // namespace erf_openfast

#endif
