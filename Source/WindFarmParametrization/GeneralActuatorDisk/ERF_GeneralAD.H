#ifndef ERF_GENERALAD_H
#define ERF_GENERALAD_H

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include "ERF_NullWindFarm.H"

class GeneralAD : public NullWindFarm {

public:

    GeneralAD () {}

    virtual ~GeneralAD () = default;

    void advance (const amrex::Geometry& geom,
                  const amrex::Real& dt_advance,
                  amrex::MultiFab& cons_in,
                  amrex::MultiFab& mf_vars_windfarm,
                  amrex::MultiFab& U_old,
                  amrex::MultiFab& V_old,
                  amrex::MultiFab& W_old,
                  const amrex::MultiFab& mf_Nturb,
                  const amrex::MultiFab& mf_SMark,
                  const amrex::Real& time) override;

    void compute_freestream_velocity (const amrex::MultiFab& cons_in,
                                     const amrex::MultiFab& U_old,
                                     const amrex::MultiFab& V_old,
                                     const amrex::MultiFab& mf_SMark);

    void source_terms_cellcentered (const amrex::Geometry& geom,
                                    const amrex::MultiFab& cons_in,
                                    const amrex::MultiFab& mf_Smark,
                                    amrex::MultiFab& mf_vars_generalAD);

    void update (const amrex::Real& dt_advance,
                 amrex::MultiFab& cons_in,
                 amrex::MultiFab& U_old,
                 amrex::MultiFab& V_old,
                 amrex::MultiFab& W_old,
                 const amrex::MultiFab& mf_vars);

    void compute_power_output (const amrex::Real& time);

protected:
    amrex::Vector<amrex::Real> xloc, yloc;
    amrex::Real turb_disk_angle;
    amrex::Real hub_height, rotor_rad, thrust_coeff_standing, nominal_power;
    amrex::Vector<amrex::Real> wind_speed, thrust_coeff, power;
    amrex::Vector<amrex::Real> freestream_velocity, freestream_phi, disk_cell_count;
    amrex::Vector<amrex::Real> bld_rad_loc, bld_twist, bld_chord;
    amrex::Vector<amrex::Vector<amrex::Real>> bld_airfoil_aoa, bld_airfoil_Cl, bld_airfoil_Cd;
    amrex::Vector<amrex::Real> velocity, C_P, C_T, rotor_RPM, blade_pitch;
};

#endif

