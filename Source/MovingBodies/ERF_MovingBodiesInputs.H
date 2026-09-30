#ifndef ERF_MOVING_BODIES_INPUTS_H
#define ERF_MOVING_BODIES_INPUTS_H

#include <array>
#include <string>
#include <vector>

#include <AMReX_REAL.H>

/**
 * One body from the erf.moving_bodies.<name>.* input block.
 */
struct MovingBodyInputs
{
    std::string name;
    std::string type;                        //!< openfast_turbine | ct_disk
    std::array<amrex::Real,3> base_pos{{0.0, 0.0, 0.0}}; //!< tower base in ERF coordinates (m)
    amrex::Real epsilon{2.0};                //!< Gaussian spreading width in units of dx
    std::string output_root;                 //!< prefix of the per-body diagnostics files
    // openfast_turbine
    std::string fst_file;                    //!< OpenFAST primary input file
    std::string mode{"adm"};                 //!< adm | alm | none: how the OpenFAST loads reach the flow (none: they do not)
    int num_force_points_blade{50};          //!< actuator force points per blade requested from OpenFAST
    int num_force_points_tower{0};           //!< actuator force points on the tower (0: no tower forcing)
    // ct_disk
    amrex::Real rotor_radius{0.0};           //!< disk radius (m)
    amrex::Real hub_height{0.0};             //!< disk centre above the base (m)
    amrex::Real ct{0.0};                     //!< thrust coefficient on the upstream speed
    amrex::Real yaw_deg{0.0};                //!< disk normal, degrees from +x in the horizontal plane
    int num_points_r{8};                     //!< polar grid: rings
    int num_points_t{16};                    //!< points per ring (ct_disk grid, and the turbine's actuator-disk rings)
    amrex::Real sample_diameters_upstream{1.0}; //!< where the free-stream speed is sampled
    amrex::Real air_density{1.225};          //!< density in the thrust (kg/m^3)
};

/**
 * The erf.moving_bodies.* inputs: a list of named bodies, each with its own block.
 *
 *   erf.moving_bodies.bodies         = T1 T2
 *   erf.moving_bodies.T1.type        = openfast_turbine
 *   erf.moving_bodies.T1.fst_file    = T1/IEA-15-240-RWT.fst
 *   erf.moving_bodies.T1.base_pos    = 1000 1000 0
 *   erf.moving_bodies.T1.mode        = adm                (adm: disk rings; alm: a point per blade node; none)
 *   erf.moving_bodies.T1.num_force_points_blade = 50
 *   erf.moving_bodies.T1.num_force_points_tower = 0
 *   erf.moving_bodies.T1.output_root = moving_bodies/T1
 *   erf.moving_bodies.T1.epsilon     = 2.0                (spreading width, dx units)
 *   erf.moving_bodies.D1.type        = ct_disk
 *   erf.moving_bodies.D1.base_pos    = 1000 1000 0
 *   erf.moving_bodies.D1.rotor_radius = 120
 *   erf.moving_bodies.D1.hub_height  = 150
 *   erf.moving_bodies.D1.ct          = 0.75
 *   erf.moving_bodies.diagnostics_int = 1
 *   erf.moving_bodies.wake.lines_xD   = 2 4 7      (wake lines behind every rotor; see WakeInputs)
 *   erf.moving_bodies.avg_start       = 400.0      (running averages and statistics start here)
 *   erf.moving_bodies.alm_max_tip_cells = 1.0      (abort when a blade tip sweeps more cells than this per step)
 *   erf.moving_bodies.prescribed_velocity = 10.59 0 0   (testing aid: fixed inflow to every body)
 */
/** erf.moving_bodies.wake.*: sampling lines behind every rotor (off unless lines_xD is given). */
struct WakeInputs
{
    std::vector<amrex::Real> lines_xD;   //!< downstream distances in rotor diameters
    amrex::Real half_width{1.5};         //!< half-length of each line, in diameters
    int num_points{61};                  //!< points per line
    int interval{0};                     //!< sample every this many steps (0: diagnostics_int)
    bool active () const { return !lines_xD.empty(); }
};

struct MovingBodiesInputs
{
    bool active{false};                      //!< erf.moving_bodies.bodies was given and non-empty
    std::vector<MovingBodyInputs> bodies;
    int diagnostics_int{1};                  //!< write a diagnostics row every this many steps
    std::string diagnostics_dir{"moving_bodies"};
    bool has_prescribed_velocity{false};
    std::array<amrex::Real,3> prescribed_velocity{{0.0, 0.0, 0.0}};
    WakeInputs wake;
    amrex::Real avg_start{0.0};              //!< time from which the running averages and statistics accumulate
    amrex::Real alm_max_tip_cells{1.0};      //!< actuator line: largest tip travel per ERF step, in cells

    /** Read erf.moving_bodies.*; aborts on an unknown type, a duplicate name or a bad value. */
    static MovingBodiesInputs read ();

    /**
     * The solver settings this version supports. Returns an empty string when they hold, else
     * the message to abort with. A function of plain flags so a unit test can check the contract.
     * @param all_levels_anelastic every level runs the anelastic solver
     * @param fixed_dt the run uses erf.fixed_dt (OpenFAST needs a fixed step)
     * @param max_level amr.max_level (a single level in this version)
     * @param fpe_traps an amrex.fpe_trap_* input is on (OpenFAST's Fortran is not trap clean)
     */
    static std::string validate_solver (bool all_levels_anelastic, bool fixed_dt, int max_level,
                                        bool fpe_traps);
};

#endif
