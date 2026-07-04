#include "ERF_Fire.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

using namespace amrex;

Fire::Fire()
{
}

Fire::~Fire()
{
}

void Fire::Define()
{
    ParmParse pp_fire("erf.fire");

    // FARSITE parameters
    pp_fire.query("farsite.phi_threshold", m_fp.phi_threshold);
    pp_fire.query("farsite.use_anderson_lw", m_fp.use_anderson_lw);
    pp_fire.query("farsite.coeff_a", m_fp.coeff_a);
    pp_fire.query("farsite.coeff_b", m_fp.coeff_b);
    pp_fire.query("farsite.coeff_c", m_fp.coeff_c);
    pp_fire.query("farsite.gaussian_sigma", m_fp.gaussian_sigma);
    pp_fire.query("farsite.cfl_fire", m_fp.cfl_fire);

    // Fire grid refinement
    pp_fire.query("grid_ratio", m_C);

    // Ignition parameters
    pp_fire.query("ignition_x", m_ignition_x);
    pp_fire.query("ignition_y", m_ignition_y);
    pp_fire.query("ignition_r", m_ignition_r);

    // Fuel model
    pp_fire.query("fuel_model_id", m_fuel_model_id);

    Print() << "[FIRE] Configuration:\n"
            << "  grid_ratio=" << m_C << "\n"
            << "  use_anderson_lw=" << m_fp.use_anderson_lw << "\n"
            << "  cfl_fire=" << m_fp.cfl_fire << "\n"
            << "  ignition=(" << m_ignition_x << ", " << m_ignition_y
            << ", r=" << m_ignition_r << ")\n";
}

void Fire::Init(const int& lev, const MultiFab& cons_in,
                const Geometry& geom, const Real& dt_advance)
{
    if (m_initialized) {
        return;
    }

    // Derive fire grid BoxArray and DistributionMapping from atmospheric level 0
    // For now, use a simple approach: create a regular grid refined by m_C
    const Box& domain_atm = geom.Domain();
    Box domain_fire = domain_atm;
    domain_fire.refine(m_C);

    // Create fire grid BoxArray
    m_ba_fire.define(domain_fire);
    m_ba_fire.maxSize(32);  // Reasonable box size for GPU work

    // Create matching DistributionMapping
    m_dm_fire.define(m_ba_fire);

    // Create fire grid Geometry
    Array<Real, AMREX_SPACEDIM> prob_lo, prob_hi;
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        prob_lo[i] = geom.ProbLo()[i];
        prob_hi[i] = geom.ProbHi()[i];
    }
    Array<int, AMREX_SPACEDIM> is_periodic;
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        is_periodic[i] = geom.isPeriodic(i);
    }
    m_geom_fire.define(domain_fire, prob_lo, prob_hi, is_periodic.data());

    // Allocate MultiFabs on fire grid
    m_phi.define(m_ba_fire, m_dm_fire, 1, IntVect(1, 1, 0));
    m_ros.define(m_ba_fire, m_dm_fire, 1, IntVect(0));
    m_wind_eff.define(m_ba_fire, m_dm_fire, 2, IntVect(0));
    m_slopes.define(m_ba_fire, m_dm_fire, 2, IntVect(0));
    m_farsite_work.define(m_ba_fire, m_dm_fire, 2, IntVect(0));
    m_fuel_load.define(m_ba_fire, m_dm_fire, 1, IntVect(0));

    // Initialize phi to +1 (unburned) everywhere
    m_phi.setVal(1.0_rt);

    // Stamp ignition circle
    initialize_ignition(m_phi, m_geom_fire, m_ignition_x, m_ignition_y, m_ignition_r);

    // Fill ghost cells
    m_phi.FillBoundary(m_geom_fire.periodicity());

    // Initialize other fields
    m_ros.setVal(0.0_rt);
    m_wind_eff.setVal(0.0_rt);
    m_slopes.setVal(0.0_rt);
    m_farsite_work.setVal(0.0_rt);
    m_fuel_load.setVal(1.0_rt);  // kg/m^2

    // Initialize Rothermel parameters
    m_rc.I_R = 10000.0_rt;        // BTU/ft^2/min (example value)
    m_rc.wind_conv = 196.85_rt;   // m/s to ft/min
    m_rc.ros_conv = 0.3048_rt;    // ft/min to m/s
    m_rc.waf = 1.0_rt;            // Wind adjustment factor
    m_rc.U_max_ftmin = 1000.0_rt;
    m_rc.delta = 1.0_rt;

    m_initialized = true;

    Print() << "[FIRE] Initialized on level " << lev << "\n"
            << "  phi domain: " << domain_fire << "\n"
            << "  dx_fire: " << m_geom_fire.CellSize()[0] << " m\n";
}

void Fire::Advance(const int& lev, const Real& time, const Real& dt_advance,
                   MultiFab& cons_in, const Geometry& geom)
{
    if (!m_initialized) {
        return;
    }

    m_current_time = time;
    m_dt_atm = dt_advance;

    ComputeRothermellSpreadRate(lev, geom);   // fills m_ros from m_wind_eff, m_slopes
    ComputeEllipticalExpansion(lev, geom);    // advances m_phi using m_ros
    ComputeFireIntensity(lev);
    Update_Fire_Vars(lev, cons_in);

    Print() << "[FIRE] t=" << time
            << "  max_ROS=" << m_ros.max(0)
            << "  mean_ROS=" << m_ros.sum(0) / m_ros.boxArray().numPts()
            << "\n";
}

void Fire::ComputeRothermellSpreadRate(const int& lev, const Geometry& geom)
{
    // Stub implementation: for now, set a constant ROS value
    // This will be replaced with full Rothermel implementation in Phase 2
    m_ros.setVal(0.05_rt);  // 0.05 m/s example value
}

void Fire::ComputeEllipticalExpansion(const int& lev, const Geometry& geom)
{
    // Subcycle the fire front through multiple substeps within one atmospheric
    // timestep. ROS (m_ros) is held constant during this function; it was
    // computed by ComputeRothermellSpreadRate before this call.
    //
    // advance_fire_subcycle() internally calls advance_farsite_one_step()
    // which uses the two-pass Huygens wavelet approach from Richards (1990).
    int n_substeps = advance_fire_subcycle(
        m_phi, m_farsite_work, m_wind_eff, m_ros,
        m_geom_fire, m_dt_atm, m_fp);

    // Fill ghost cells after propagation so gradient stencils in the next
    // ComputeRothermellSpreadRate call (next atmospheric step) are valid.
    m_phi.FillBoundary(m_geom_fire.periodicity());

    Print() << "[FIRE] t=" << m_current_time
            << "  substeps=" << n_substeps
            << "  phi_min=" << m_phi.min(0)
            << "\n";
}

void Fire::ComputeFireIntensity(const int& lev)
{
    // Stub: compute fireline intensity from ROS
    // This will use Rothermel reaction intensity formula in Phase 2
}

void Fire::Update_Fire_Vars(const int& lev, MultiFab& cons_in)
{
    // Stub: update atmospheric state with fire heat/moisture/etc.
}
