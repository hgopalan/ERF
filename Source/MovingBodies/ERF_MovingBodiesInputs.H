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
    std::string type;                        //!< openfast_turbine
    std::string fst_file;                    //!< OpenFAST primary input file
    std::array<amrex::Real,3> base_pos{{0.0, 0.0, 0.0}}; //!< tower base in ERF coordinates (m)
    std::string mode{"adm"};                 //!< adm | alm: how the OpenFAST loads are spread
    int num_force_points_blade{50};          //!< actuator force points per blade requested from OpenFAST
    int num_force_points_tower{0};           //!< actuator force points on the tower (0: no tower forcing)
    std::string output_root;                 //!< prefix of the per-body diagnostics file
};

/**
 * The erf.moving_bodies.* inputs: a list of named bodies, each with its own block.
 *
 *   erf.moving_bodies.bodies         = T1 T2
 *   erf.moving_bodies.T1.type        = openfast_turbine
 *   erf.moving_bodies.T1.fst_file    = T1/IEA-15-240-RWT.fst
 *   erf.moving_bodies.T1.base_pos    = 1000 1000 0
 *   erf.moving_bodies.T1.mode        = adm
 *   erf.moving_bodies.T1.num_force_points_blade = 50
 *   erf.moving_bodies.T1.num_force_points_tower = 0
 *   erf.moving_bodies.T1.output_root = moving_bodies/T1
 *   erf.moving_bodies.diagnostics_int = 1
 *   erf.moving_bodies.prescribed_velocity = 10.59 0 0   (testing aid: fixed inflow to every body)
 */
struct MovingBodiesInputs
{
    bool active{false};                      //!< erf.moving_bodies.bodies was given and non-empty
    std::vector<MovingBodyInputs> bodies;
    int diagnostics_int{1};                  //!< write a diagnostics row every this many steps
    std::string diagnostics_dir{"moving_bodies"};
    bool has_prescribed_velocity{false};
    std::array<amrex::Real,3> prescribed_velocity{{0.0, 0.0, 0.0}};

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
