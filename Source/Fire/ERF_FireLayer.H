#ifndef ERF_FIRE_LAYER_H
#define ERF_FIRE_LAYER_H

#include <AMReX_MultiFab.h>
#include <AMReX_Geometry.H>
#include <memory>
#include <ERF_FireParams.H>
#include <ERF_FireGrid.H>
#include <ERF_Rothermel.H>
#include <ERF_FarsiteEllipse.H>
#include <ERF_FireIgnition.H>
#include <ERF_FuelMoisture.H>
#include <ERF_MoistureExtinction.H>

// Forward declarations
class SurfaceLayer;
class ERF;

class FireLayer
{
public:

    FireLayer() = default;
    ~FireLayer() = default;

    void initialize(const ERF&             erf,
                    const SurfaceLayer*    surface_layer_ptr,
                    const amrex::MultiFab& z_phys_nd_atm,
                    const FireParams&      fire_params);

    void advance(amrex::Real time,
                 amrex::Real dt,
                 SurfaceLayer& surface_layer,
                 const amrex::MultiFab& xvel,
                 const amrex::MultiFab& yvel,
                 const amrex::MultiFab& z_phys_cc,
                 const amrex::MultiFab& T_atm_k0,
                 const amrex::MultiFab& RH_atm_k0);

    const amrex::MultiFab* get_heat_flux()  const { return fire_heat_flux.get(); }
    const amrex::MultiFab* get_ros()        const { return fire_ros.get(); }
    const amrex::MultiFab* get_wind_eff()   const { return fire_wind_eff.get(); }
    const amrex::MultiFab* get_levelset()   const { return fire_phi.get(); }
          amrex::MultiFab* get_levelset_mut()     { return fire_phi.get(); }
    const amrex::Geometry& get_fire_geom()  const { return m_fg.geom; }
    const amrex::MultiFab* get_wind_ref()   const { return fire_wind_ref.get(); }
    const amrex::MultiFab* get_slopes()     const { return fire_slopes.get(); }
    const amrex::MultiFab* get_fuel_mc()    const { return fire_fuel_mc.get(); }
    const FireGrid&        get_fire_grid()  const { return m_fg; }

private:

    FireGrid                                 m_fg;

    std::unique_ptr<amrex::MultiFab>         fire_phi;           // Level-set
    std::unique_ptr<amrex::MultiFab>         fire_wind_ref;      // Wind at z_ref
    std::unique_ptr<amrex::MultiFab>         fire_wind_eff;      // Effective wind
    std::unique_ptr<amrex::MultiFab>         fire_slopes;        // Terrain slopes
    std::unique_ptr<amrex::MultiFab>         fire_curvature;     // Terrain curvature
    std::unique_ptr<amrex::MultiFab>         fire_ros;           // Rate of spread
    std::unique_ptr<amrex::MultiFab>         fire_fuel_load;     // Fuel load
    std::unique_ptr<amrex::MultiFab>         fire_fuel_mc;       // Fuel moisture (3 comp)
    std::unique_ptr<amrex::MultiFab>         fire_mext;          // Moisture of extinction
    std::unique_ptr<amrex::MultiFab>         fire_heat_flux;     // Heat flux (stub)
    std::unique_ptr<amrex::MultiFab>         fire_spread_vec;    // FARSITE scratch
    std::unique_ptr<amrex::MultiFab>         fire_arrival_time;  // Fire arrival time [s]
    std::unique_ptr<amrex::MultiFab>         fire_disp_accum;    // Persistent displacement accumulator (2 comp)

    RothermelComputed                        m_rc;
    FarsiteParams                            m_fp;

    amrex::Real                              m_current_time      = 0.0;
    amrex::Real                              m_dt_atm            = 0.0;
    int                                      m_nz                = 0;
    amrex::Real                              m_fuel_bed_depth_ft = 6.5;
    amrex::Real                              m_ignition_x        = 0.0;
    amrex::Real                              m_ignition_y        = 0.0;
    amrex::Real                              m_ignition_r        = 20.0;
    FireParams                               m_params;

    void init_ignition(amrex::Real center_x, amrex::Real center_y,
                       amrex::Real radius);
    void apply_waf_to_wind();
    void advance_fuel_moisture(amrex::Real dt_s,
                               const amrex::MultiFab& T_atm_k0,
                               const amrex::MultiFab& RH_atm_k0);
};

#endif