#ifndef ERF_PROB_H_
#define ERF_PROB_H_

#include <string>

#include "AMReX_REAL.H"

#include "ERF_ProbCommon.H"

struct ProbParm : ProbParmDefaults {
    amrex::Real KE_0   = 0.1;

    amrex::Real Xc_0  = 0.0;
    amrex::Real Yc_0  = 0.0;
    amrex::Real VMAX  = 15.0;   // Maximum horizontal velocity in vortex
    amrex::Real RMAX  = 100.0;  // Radius of maximum winds
    amrex::Real RZERO = 800.0;  // Radius of zero wind speed
    amrex::Real ZZERO = 2000.0; // Height of zero wind speed

    amrex::Real U_0 = 0.0;
    amrex::Real V_0 = 0.0;
    amrex::Real W_0 = 0.0;

}; // namespace ProbParm

class Problem : public ProblemBase
{
public:
    Problem();

//#include "Prob/ERF_InitConstantDensityHSE.H" // not needed if init_type=="input_sounding"
#include "Prob/ERF_InitRayleighDamping.H" // provides WRF-style damping layer, with U_0, V_0, W_0, and T_0 as ref values

    void init_custom_pert (
        const amrex::Box&  bx,
        const amrex::Box& xbx,
        const amrex::Box& ybx,
        const amrex::Box& zbx,
        amrex::Array4<amrex::Real const> const& state,
        amrex::Array4<amrex::Real      > const& state_pert,
        amrex::Array4<amrex::Real      > const& x_vel_pert,
        amrex::Array4<amrex::Real      > const& y_vel_pert,
        amrex::Array4<amrex::Real      > const& z_vel_pert,
        amrex::Array4<amrex::Real      > const& r_hse,
        amrex::Array4<amrex::Real      > const& p_hse,
        amrex::Array4<amrex::Real const> const& z_nd,
        amrex::Array4<amrex::Real const> const& z_cc,
        amrex::GeometryData const& geomdata,
        amrex::Array4<amrex::Real const> const& mf_m,
        amrex::Array4<amrex::Real const> const& mf_u,
        amrex::Array4<amrex::Real const> const& mf_v,
        const SolverChoice& sc) override;

protected:
    std::string name() override { return "Tropical Cyclone"; }

private:
    ProbParm parms;
};

#endif
