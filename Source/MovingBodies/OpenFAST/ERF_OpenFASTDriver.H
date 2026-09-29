#ifndef ERF_OPENFAST_DRIVER_H
#define ERF_OPENFAST_DRIVER_H

#include <array>
#include <string>
#include <vector>

#include <AMReX_REAL.H>

#include "ERF_MovingBodiesInputs.H"

// OpenFAST 4 C API (the real header, or Stub/FAST_Library.h when ERF_OPENFAST_USE_STUB is set)
#include "FAST_Library.h"

namespace erf_openfast {

/**
 * Number of OpenFAST steps per ERF step. OpenFAST needs a fixed step, and the ERF step must be a
 * positive whole multiple of it. On failure returns 0 and sets err to the message to abort with.
 */
int substep_count (double dt_cfd, double dt_fast, std::string& err);

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
    std::array<amrex::Real,3> hub_pos{{0.0, 0.0, 0.0}};
    amrex::Real rotor_speed{0.0};        //!< rad/s about the hub axis
    // ExtInfw buffers, owner rank only; positions and forces come from OpenFAST in to_cfd and the
    // velocities the flow supplies go into from_cfd
    ExtInfw_InputType_t  to_cfd{};
    ExtInfw_OutputType_t from_cfd{};
};

/**
 * Drives one or more OpenFAST turbines through the ExtInfw C API. Each turbine is owned by one
 * MPI rank; after every step the node positions and forces are broadcast so every rank can use
 * them. This version initialises, steps and reports; the flow velocity at the nodes is
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
     * Allocate and initialise every turbine, run its first solution, and broadcast the node
     * layout. Aborts when a model does not take external inflow (CompInflow must be 2) or when
     * dt_cfd is not a whole multiple of the model's dt.
     * @param dt_cfd the fixed ERF step
     * @param t_max the OpenFAST stop time
     * @param initial_velocity the uniform velocity given to every node before the first solution
     */
    void init (double dt_cfd, double t_max, const std::array<amrex::Real,3>& initial_velocity);

    /** Set the same velocity at every velocity node of every turbine (owner ranks). */
    void set_uniform_velocity (const std::array<amrex::Real,3>& vel);

    /** Advance every turbine by one ERF step (num_substeps OpenFAST steps) and broadcast. */
    void step ();

    /** Append one diagnostics row per turbine (time, rotor speed, thrust, torque, power). */
    void write_diagnostics (double time);

    const std::vector<TurbineState>& turbines () const { return m_turb; }
    bool is_owner (int i) const;
    bool initialized () const { return m_initialized; }

    /** Thrust vector: sum of the blade-node forces (N). */
    std::array<amrex::Real,3> thrust (const TurbineState& t) const;
    /** Aerodynamic torque about the hub axis from the blade-node forces (N m). */
    amrex::Real torque (const TurbineState& t) const;

private:
    std::vector<TurbineState> m_turb;
    bool m_initialized{false};
    int m_num_local{0};

    void fast_check (int err_stat, const char* err_msg, const std::string& where) const;
    void pull_from_fast (TurbineState& t);
    void broadcast_state (TurbineState& t);
    void open_diagnostics (const TurbineState& t) const;
};

} // namespace erf_openfast

#endif
