#ifndef ERF_MOVING_BODIES_INPUTS_H
#define ERF_MOVING_BODIES_INPUTS_H

#include <array>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <AMReX_REAL.H>

/** The Gaussian widths (m) a body's points are spread with, one per kind of point. */
struct KernelWidths
{
    amrex::Real rotor{0.0};     //!< the rotor's points: blade nodes and the hub node (alm), ring points (adm), a disk's points
    amrex::Real tower{0.0};     //!< an OpenFAST turbine's tower force nodes
    amrex::Real nacelle{0.0};   //!< an OpenFAST turbine's nacelle drag point
};

/**
 * One body from the erf.moving_bodies.<name>.* input block.
 */
struct MovingBodyInputs
{
    std::string name;
    std::string type;                        //!< openfast_turbine | ct_disk
    std::array<amrex::Real,3> base_pos{{0.0, 0.0, 0.0}}; //!< tower base in ERF coordinates (m)
    amrex::Real epsilon{2.0};                //!< Gaussian spreading width of the rotor's points, in cells of the anchor
                                             //!< level (its x spacing); the default of epsilon_tower and epsilon_nacelle
    amrex::Real epsilon_tower{-1.0};         //!< width of the tower force nodes, in the same cells (-1: epsilon)
    amrex::Real epsilon_nacelle{-1.0};       //!< width of the nacelle drag point, in the same cells (-1: epsilon)
    std::string output_root;                 //!< prefix of the per-body diagnostics files
    // openfast_turbine
    std::string fst_file;                    //!< OpenFAST primary input file
    std::string mode{"adm"};                 //!< adm | alm | none: how the OpenFAST loads reach the flow (none: they do not)
    std::string sampling;                    //!< adm: disk (velocities at the nodes, OpenFAST induction off) | disk_corrected (disk
                                             //!< velocities, free stream recovered with the filtered-disk factor of Shapiro et al. 2019
                                             //!< and momentum theory, OpenFAST induction on; the adm default) | upstream (free stream
                                             //!< sample_diameters_upstream diameters ahead of the hub, OpenFAST's own BEM induction on)
    int num_force_points_blade{50};          //!< actuator force points per blade requested from OpenFAST
    int num_force_points_tower{0};           //!< actuator force points on the tower (0: no tower forcing)
    amrex::Real nacelle_cd{0.0};             //!< nacelle drag coefficient (0: no nacelle drag)
    amrex::Real nacelle_area{0.0};           //!< nacelle frontal area (m^2), needed when nacelle_cd > 0
    bool fllc{false};                        //!< filtered lifting-line correction of the sampled blade velocities (alm only; on by default for alm)
    amrex::Real fllc_relax{0.1};             //!< relaxation factor of the correction, in (0, 1]
    amrex::Real fllc_start_time{0.0};        //!< the correction starts at this time (s)
    amrex::Real fllc_eps_chord{0.25};        //!< optimal kernel width in chords
    amrex::Real fllc_eps_dr{1.0};            //!< the optimal kernel width over the fine-grid spacing of the correction (spacing = eps_opt / fllc_eps_dr)
    // ct_disk
    amrex::Real rotor_radius{0.0};           //!< disk radius (m)
    amrex::Real hub_height{0.0};             //!< disk centre above the base (m)
    amrex::Real ct{0.0};                     //!< thrust coefficient on the upstream speed
    amrex::Real yaw_deg{0.0};                //!< disk normal, degrees from +x in the horizontal plane
    int num_points_r{8};                     //!< polar grid: rings
    int num_points_t{16};                    //!< points per ring (ct_disk grid, and the turbine's actuator-disk rings)
    amrex::Real sample_diameters_upstream{1.0}; //!< where the free-stream speed is sampled (ct_disk 1 D; turbines default to 2 D)
    amrex::Real correction_relax{-1.0};     //!< disk_corrected: relaxation of the free-stream update (-1: from its gain; (0,1]: fixed, 1 = none)
    amrex::Real correction_time{-1.0};      //!< with correction_relax = -1: the update's time scale, s (-1: rotor radius / free stream; 0: none)
    amrex::Real air_density{1.225};          //!< air density (kg/m^3): ct_disk thrust; turbines: nacelle drag, lifting-line correction, disk_corrected momentum theory (must equal the model's AirDens)

    /** The widths (m) on a level of x spacing dx: epsilon_tower and epsilon_nacelle fall back to epsilon when not given. */
    KernelWidths kernel_widths (amrex::Real dx) const
    {
        return {epsilon * dx, ((epsilon_tower > 0.0) ? epsilon_tower : epsilon) * dx,
                ((epsilon_nacelle > 0.0) ? epsilon_nacelle : epsilon) * dx};
    }
};

/**
 * The erf.moving_bodies.* inputs: a list of named bodies, each with its own block.
 *
 *   erf.moving_bodies.bodies         = T1 T2
 *   erf.moving_bodies.T1.type        = openfast_turbine
 *   erf.moving_bodies.T1.fst_file    = T1/IEA-15-240-RWT.fst
 *   erf.moving_bodies.T1.base_pos    = 1000 1000 0
 *   erf.moving_bodies.T1.mode        = adm                (adm: disk rings; alm: a point per blade node; none)
 *   erf.moving_bodies.T1.sampling    = disk_corrected     (adm: disk | disk_corrected | upstream; upstream samples the free stream
 *                                                          sample_diameters_upstream diameters ahead and keeps OpenFAST's BEM)
 *   erf.moving_bodies.T1.num_force_points_blade = 50
 *   erf.moving_bodies.T1.num_force_points_tower = 10   (> 0: the tower's loads from OpenFAST forced too; default 0)
 *   erf.moving_bodies.T1.nacelle_cd  = 1.0              (> 0 with nacelle_area: a drag point at the hub; default 0)
 *   erf.moving_bodies.T1.nacelle_area = 60.0            (m^2; default 0)
 *   erf.moving_bodies.T1.fllc        = true               (alm: filtered lifting-line correction; the alm default, set false to run without it)
 *   erf.moving_bodies.T1.fllc_relax  = 0.1                fllc_start_time = 0, fllc_eps_chord = 0.25, fllc_eps_dr = 1
 *   erf.moving_bodies.T1.output_root = moving_bodies/T1
 *   erf.moving_bodies.T1.epsilon     = 2.0                (spreading width of the rotor's points, dx units)
 *   erf.moving_bodies.T1.epsilon_tower   = 2.0            (tower force nodes, needs num_force_points_tower > 0; default epsilon)
 *   erf.moving_bodies.T1.epsilon_nacelle = 2.0            (nacelle drag point, needs nacelle_cd > 0; default epsilon)
 *   erf.moving_bodies.D1.type        = ct_disk
 *   erf.moving_bodies.D1.base_pos    = 1000 1000 0
 *   erf.moving_bodies.D1.rotor_radius = 120
 *   erf.moving_bodies.D1.hub_height  = 150
 *   erf.moving_bodies.D1.ct          = 0.75
 *   erf.moving_bodies.diagnostics_int = 1
 *   erf.moving_bodies.wake.lines_xD   = 2 4 7      (wake lines behind every rotor; see WakeInputs)
 *   erf.moving_bodies.avg_start       = 400.0      (running averages and statistics start here)
 *   erf.moving_bodies.alm_max_tip_cells = 1.0      (abort when a blade tip sweeps more cells than this per step)
 *   erf.moving_bodies.envelope_check_interval = 1  (check the bodies' exchange envelopes, one reduction, every this many steps)
 *   erf.moving_bodies.verify_tip_cells = 0         (1: check the cached limiting cell heights against a fresh search every step)
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
    int envelope_check_interval{1};          //!< check the exchange envelopes every this many steps (BodyExchange::set_check_interval)
    bool verify_tip_cells{false};            //!< actuator line: check the cached tip cell heights against a fresh per-region search every step (debug)
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

    /**
     * The inputs that place a body and decide how it forces the flow, as key/value pairs for the
     * checkpoint: a restart that changes one of them would continue another run's state. Only the
     * keys the body's options read are listed; the kernel widths are the resolved ones (cells).
     */
    static std::vector<std::pair<std::string,std::string>> restart_record (const MovingBodyInputs& b);

    /**
     * Find the first entry of the record `now` that differs from the checkpoint's record `chk`;
     * returns false when none does. Numbers agree to single precision (a checkpoint written by a
     * double build may be continued by a single one). A checkpoint record without epsilon_tower or
     * epsilon_nacelle spread the tower and the nacelle with its epsilon, so those entries are
     * compared with its epsilon; any other entry the checkpoint lacks is taken as unchanged.
     * @param[out] key,chk_value,now_value the entry that differs
     */
    static bool changed_record_entry (const std::map<std::string,std::string>& chk,
                                      const std::vector<std::pair<std::string,std::string>>& now,
                                      std::string& key, std::string& chk_value, std::string& now_value);
};

#endif
