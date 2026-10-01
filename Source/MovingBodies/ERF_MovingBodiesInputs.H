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
    std::string sampling{"disk"};            //!< adm: disk (velocities at the nodes, OpenFAST induction off) | disk_corrected (disk
                                             //!< velocities, free stream recovered with the filtered-disk factor of Shapiro et al. 2019
                                             //!< and momentum theory, OpenFAST induction on) | upstream (free stream
                                             //!< sample_diameters_upstream diameters ahead of the hub, OpenFAST's own BEM induction on)
    int num_force_points_blade{50};          //!< actuator force points per blade requested from OpenFAST
    int num_force_points_tower{0};           //!< actuator force points on the tower (0: no tower forcing)
    amrex::Real nacelle_cd{0.0};             //!< nacelle drag coefficient (0: no nacelle drag)
    amrex::Real nacelle_area{0.0};           //!< nacelle frontal area (m^2), needed when nacelle_cd > 0
    bool fllc{false};                        //!< filtered lifting-line correction of the sampled blade velocities (alm only)
    amrex::Real fllc_relax{0.1};             //!< relaxation factor of the correction, in (0, 1]
    amrex::Real fllc_start_time{0.0};        //!< the correction starts at this time (s)
    amrex::Real fllc_eps_chord{0.25};        //!< optimal kernel width in chords
    amrex::Real fllc_eps_dr{1.0};            //!< fine-grid spacing of the correction as a fraction of the optimal kernel width
    // ct_disk
    amrex::Real rotor_radius{0.0};           //!< disk radius (m)
    amrex::Real hub_height{0.0};             //!< disk centre above the base (m)
    amrex::Real ct{0.0};                     //!< thrust coefficient on the upstream speed
    amrex::Real yaw_deg{0.0};                //!< disk normal, degrees from +x in the horizontal plane
    int num_points_r{8};                     //!< polar grid: rings
    int num_points_t{16};                    //!< points per ring (ct_disk grid, and the turbine's actuator-disk rings)
    amrex::Real sample_diameters_upstream{1.0}; //!< where the free-stream speed is sampled (ct_disk 1 D; turbines default to 2 D)
    amrex::Real air_density{1.225};          //!< density in the ct_disk thrust and the nacelle drag (kg/m^3)
};

/**
 * The erf.moving_bodies.* inputs: a list of named bodies, each with its own block.
 *
 *   erf.moving_bodies.bodies         = T1 T2
 *   erf.moving_bodies.T1.type        = openfast_turbine
 *   erf.moving_bodies.T1.fst_file    = T1/IEA-15-240-RWT.fst
 *   erf.moving_bodies.T1.base_pos    = 1000 1000 0
 *   erf.moving_bodies.T1.mode        = adm                (adm: disk rings; alm: a point per blade node; none)
 *   erf.moving_bodies.T1.sampling    = disk               (adm: disk | disk_corrected | upstream; upstream samples the free stream
 *                                                          sample_diameters_upstream diameters ahead and keeps OpenFAST's BEM)
 *   erf.moving_bodies.T1.num_force_points_blade = 50
 *   erf.moving_bodies.T1.num_force_points_tower = 0    (> 0: the tower's loads from OpenFAST forced too)
 *   erf.moving_bodies.T1.nacelle_cd  = 0                (> 0 with nacelle_area: a drag point at the hub)
 *   erf.moving_bodies.T1.nacelle_area = 0               (m^2)
 *   erf.moving_bodies.T1.fllc        = false              (alm: filtered lifting-line correction)
 *   erf.moving_bodies.T1.fllc_relax  = 0.1                fllc_start_time = 0, fllc_eps_chord = 0.25, fllc_eps_dr = 1
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
 *   erf.moving_bodies.density_tolerance = 0.05     (abort when ERF's density at a hub is further from the model's)
 *   erf.moving_bodies.anchor_level    = 1          (the level the bodies are sampled and forced on; default the finest)
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
    amrex::Real density_tolerance{0.05};     //!< relative tolerance between ERF's density at a hub and the OpenFAST model's
    int anchor_level{-1};                    //!< the level the bodies live on (-1: the finest level, amr.max_level)

    /** Read erf.moving_bodies.*; aborts on an unknown type, a duplicate name or a bad value. */
    static MovingBodiesInputs read ();

    /**
     * The solver settings this version supports. Returns an empty string when they hold, else
     * the message to abort with. A function of plain flags so a unit test can check the contract.
     * @param all_levels_anelastic every level runs the anelastic solver
     * @param fixed_dt the run uses erf.fixed_dt (OpenFAST needs a fixed step)
     * @param max_level amr.max_level
     * @param anchor_level the level the bodies live on, already resolved (0 <= anchor <= max_level)
     * @param fpe_traps an amrex.fpe_trap_* input is on (OpenFAST's Fortran is not trap clean)
     */
    static std::string validate_solver (bool all_levels_anelastic, bool fixed_dt, int max_level,
                                        int anchor_level, bool fpe_traps);

    /** The anchor level for the input (-1: the finest level). */
    static int resolve_anchor_level (int requested, int max_level) { return requested < 0 ? max_level : requested; }
};

#endif
