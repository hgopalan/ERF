#ifndef ERF_FIRE_H
#define ERF_FIRE_H

#include <AMReX_Vector.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>

#include "ERF_FarsiteEllipse.H"
#include "ERF_FireIgnition.H"

// Rothermel parameters (from Phase 2)
struct RothermelComputed {
    amrex::Real I_R = 0.0_rt;           // Reaction intensity
    amrex::Real wind_conv = 196.85_rt;  // m/s to ft/min conversion
    amrex::Real ros_conv = 0.3048_rt;   // ft/min to m/s conversion
    amrex::Real waf = 1.0_rt;           // Wind adjustment factor
    amrex::Real U_max_ftmin = 1000.0_rt; // Maximum wind speed [ft/min]
    amrex::Real delta = 1.0_rt;         // Fuel bed depth [ft]
};

// Fire class for managing fire propagation on a refined grid
class Fire {
public:
    // Constructor and destructor
    Fire();
    ~Fire();

    // Define fire parameters from ParmParse inputs
    void Define();

    // Initialize fire state on a given AMR level
    void Init(const int& lev, const amrex::MultiFab& cons_in,
              const amrex::Geometry& geom, const amrex::Real& dt_advance);

    // Advance fire front and state through one atmospheric timestep
    void Advance(const int& lev, const amrex::Real& time, const amrex::Real& dt_advance,
                 amrex::MultiFab& cons_in, const amrex::Geometry& geom);

    // Update fire variables (e.g., heat flux) in the atmospheric state
    void Update_Fire_Vars(const int& lev, amrex::MultiFab& cons_in);

    // Compute ROS using Rothermel model
    void ComputeRothermellSpreadRate(const int& lev, const amrex::Geometry& geom);

    // Advance the fire front using elliptical expansion
    void ComputeEllipticalExpansion(const int& lev, const amrex::Geometry& geom);

    // Compute fire intensity from ROS
    void ComputeFireIntensity(const int& lev);

    // Accessors
    const amrex::MultiFab& get_phi() const { return m_phi; }
    const amrex::MultiFab& get_ros() const { return m_ros; }

private:
    // MultiFabs on fire grid (C*nx x C*ny x 1)
    amrex::MultiFab m_phi;          // level-set: <0 burned, >0 unburned
    amrex::MultiFab m_ros;          // rate of spread [m/s] from Rothermel
    amrex::MultiFab m_wind_eff;     // effective midflame wind [m/s], 2 comp
    amrex::MultiFab m_slopes;       // terrain slopes dz/dx, dz/dy, 2 comp
    amrex::MultiFab m_farsite_work; // scratch: spread vectors, 2 comp
    amrex::MultiFab m_fuel_load;    // fuel load [kg/m^2]

    // Fire grid descriptor
    amrex::BoxArray         m_ba_fire;
    amrex::DistributionMapping m_dm_fire;
    amrex::Geometry         m_geom_fire;
    int                     m_C{5};  // refinement factor from erf.fire.grid_ratio

    // Rothermel precomputed parameters (from Phase 2)
    RothermelComputed m_rc;

    // FARSITE parameters
    FarsiteParams m_fp;

    // Ignition parameters
    amrex::Real m_ignition_x{0.0_rt};
    amrex::Real m_ignition_y{0.0_rt};
    amrex::Real m_ignition_r{20.0_rt};

    // Time state
    amrex::Real m_current_time{0.0_rt};
    amrex::Real m_dt_atm{0.0_rt};

    // Fuel model ID
    int m_fuel_model_id{1};  // Default: GR1 (short grass)

    // Flag for whether Fire has been initialized
    bool m_initialized{false};
};

#endif // ERF_FIRE_H
