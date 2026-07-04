/**
 * \file ERF_FireLayer.cpp
 *
 * \brief Implementation of FireLayer class.
 */

#include "ERF_FireLayer.H"
#include "ERF_FirePrerequisites.H"
#include "ERF_FireWindExtract.H"
#include "ERF_TerrainSlope.H"
#include "ERF_Rothermel.H"
#include "ERF.H"
#include <AMReX_Print.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>

void FireLayer::initialize(const ERF& erf, const FireParams& fire_params)
{
    m_fire_params = fire_params;

    // Prerequisite checks
    verify_fire_prerequisites(erf.phys_bc_type,
                            erf.m_SurfaceLayer,
                            erf.grids,
                            erf.dmap,
                            erf.Geom(),
                            fire_params,
                            0);

    // Create fire grid
    m_fire_grid = std::make_unique<FireGrid>(
        create_fire_grid(erf.grids[0], erf.dmap[0], erf.Geom(0), fire_params.grid_ratio));

    // Allocate MultiFabs on fire grid
    int C = fire_params.grid_ratio;
    int ncomp_scalar = 1;
    int ncomp_vector = 2;
    int ncomp_mc = 3;  // 1-hr, 10-hr, 100-hr moisture
    int ngrow = 1;     // Ghost cells for derivatives

    fire_phi = std::make_unique<amrex::MultiFab>(m_fire_grid->ba, m_fire_grid->dm,
                                                  ncomp_scalar, ngrow);
    fire_wind_ref = std::make_unique<amrex::MultiFab>(m_fire_grid->ba, m_fire_grid->dm,
                                                       ncomp_vector, ngrow);
    fire_wind_eff = std::make_unique<amrex::MultiFab>(m_fire_grid->ba, m_fire_grid->dm,
                                                       ncomp_vector, ngrow);
    fire_slopes = std::make_unique<amrex::MultiFab>(m_fire_grid->ba, m_fire_grid->dm,
                                                     ncomp_vector, ngrow);
    fire_curvature = std::make_unique<amrex::MultiFab>(m_fire_grid->ba, m_fire_grid->dm,
                                                        ncomp_scalar, ngrow);
    fire_ros = std::make_unique<amrex::MultiFab>(m_fire_grid->ba, m_fire_grid->dm,
                                                  ncomp_scalar, 0);
    fire_fuel_load = std::make_unique<amrex::MultiFab>(m_fire_grid->ba, m_fire_grid->dm,
                                                        ncomp_scalar, 0);
    fire_fuel_mc = std::make_unique<amrex::MultiFab>(m_fire_grid->ba, m_fire_grid->dm,
                                                      ncomp_mc, 0);
    fire_heat_flux = std::make_unique<amrex::MultiFab>(m_fire_grid->ba, m_fire_grid->dm,
                                                        ncomp_scalar, 0);
    fire_spread_vec = std::make_unique<amrex::MultiFab>(m_fire_grid->ba, m_fire_grid->dm,
                                                         ncomp_vector, 0);

    // Initialize MultiFabs
    fire_phi->setVal(1.0);           // Initially unburned (phi > 0)
    fire_wind_ref->setVal(0.0);
    fire_wind_eff->setVal(0.0);
    fire_slopes->setVal(0.0);
    fire_curvature->setVal(0.0);
    fire_ros->setVal(0.0);
    fire_fuel_load->setVal(0.0);
    fire_fuel_mc->setVal(0.0);
    fire_heat_flux->setVal(0.0);
    fire_spread_vec->setVal(0.0);

    // Compute terrain slopes and curvature (static, computed once)
    compute_terrain_slopes(*fire_slopes, *erf.z_phys_nd[0], erf.Geom(0), *m_fire_grid);
    compute_terrain_curvature(*fire_curvature, *fire_slopes, m_fire_grid->geom);

    // Set fuel load and moisture from parameters
    for (amrex::MFIter mfi(*fire_fuel_load); mfi.isValid(); ++mfi) {
        auto fl = fire_fuel_load->array(mfi);
        auto mc = fire_fuel_mc->array(mfi);

        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (const amrex::IntVect& iv) {
            // Fuel load: placeholder (typically from fuel database, here set to 1.0 kg/m²)
            fl(iv) = 1.0;
            // Moisture content
            mc(iv, 0) = fire_params.moisture_1hr;
            mc(iv, 1) = fire_params.moisture_10hr;
            mc(iv, 2) = fire_params.moisture_100hr;
        });
    }

    // Set ignition condition using physical coordinates
    amrex::Real ign_x = fire_params.ignition_x;
    amrex::Real ign_y = fire_params.ignition_y;
    amrex::Real ign_r = fire_params.ignition_r;
    amrex::Real dx_fire = m_fire_grid->geom.CellSize(0);
    amrex::Real dy_fire = m_fire_grid->geom.CellSize(1);
    amrex::Real prob_lo_x = m_fire_grid->geom.ProbLo(0);
    amrex::Real prob_lo_y = m_fire_grid->geom.ProbLo(1);

    for (amrex::MFIter mfi(*fire_phi); mfi.isValid(); ++mfi) {
        auto phi = fire_phi->array(mfi);
        const amrex::Box& box = mfi.tilebox();

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (const amrex::IntVect& iv) {
            int i = iv[0];
            int j = iv[1];

            // Convert to physical coordinates
            amrex::Real x_phys = prob_lo_x + (static_cast<amrex::Real>(i) + 0.5) * dx_fire;
            amrex::Real y_phys = prob_lo_y + (static_cast<amrex::Real>(j) + 0.5) * dy_fire;

            amrex::Real dist = std::sqrt((x_phys - ign_x) * (x_phys - ign_x) +
                                        (y_phys - ign_y) * (y_phys - ign_y));

            if (dist < ign_r) {
                phi(iv) = -1.0;  // Burned region
            } else {
                phi(iv) = dist - ign_r;  // Unburned
            }
        });
    }

    amrex::Print() << "[FIRE] Initialized fire module on " << C << "× refined grid\n"
                   << "  Fire domain: " << m_fire_grid->ba.size() << " boxes\n"
                   << "  Fuel model: " << fire_params.fuel_model_id << "\n"
                   << "  Moisture: 1hr=" << fire_params.moisture_1hr
                   << " 10hr=" << fire_params.moisture_10hr
                   << " 100hr=" << fire_params.moisture_100hr << "\n"
                   << "  Ignition: center=(" << ign_x << "," << ign_y << ") radius=" << ign_r
                   << " m\n";
}

void FireLayer::advance(amrex::Real dt, const SurfaceLayer& surface_layer)
{
    // Step 1: Extract wind from MOST layer at reference height (6.1 m)
    fill_fire_wind_from_most(*fire_wind_ref,
                            *surface_layer.get_u_star(0),
                            *surface_layer.get_z0(0),
                            *surface_layer.get_olen(0),
                            *surface_layer.get_mac_avg(0, 0),
                            *surface_layer.get_mac_avg(0, 1),
                            *m_fire_grid,
                            m_fire_params.wind_ref_ht);

    // Step 2: Copy to effective wind and apply WAF
    amrex::MultiFab::Copy(*fire_wind_eff, *fire_wind_ref, 0, 0, 2, 0);

    if (m_fire_params.use_waf) {
        // Apply Wind Adjustment Factor (currently uniform WAF, simplified)
        // Full implementation would use compute_waf_per_cell per cell
        amrex::Real waf = 0.4;  // Placeholder: typical WAF for GR1 fuel
        fire_wind_eff->mult(waf, 0, 2, 0);
    }

    // Step 3: Apply FARSITE terrain wind corrections
    if (m_fire_params.use_terrain_wind) {
        apply_farsite_terrain_wind(*fire_wind_eff,
                                  *fire_slopes,
                                  *fire_curvature,
                                  m_fire_params.k_ridge,
                                  m_fire_params.k_shelter,
                                  m_fire_params.k_valley,
                                  m_fire_params.k_deflect);
    }

    // Step 4: Compute ROS field using Rothermel model
    FuelModelParams fp = get_anderson_fuel_params(m_fire_params.fuel_model_id);
    RothermelComputed rc = compute_rothermel_params(fp,
                                                    m_fire_params.moisture_1hr,
                                                    m_fire_params.moisture_10hr,
                                                    m_fire_params.moisture_100hr);
    compute_ros_field(*fire_ros, *fire_wind_eff, *fire_slopes, rc);

    // Diagnostic output: compute max and mean ROS
    amrex::Real max_ros = fire_ros->max(0);
    
    // Compute mean ROS by summing and dividing
    amrex::Real sum_ros = 0.0;
    amrex::Real n_cells = 0.0;
    for (amrex::MFIter mfi(*fire_ros); mfi.isValid(); ++mfi) {
        auto ros = fire_ros->const_array(mfi);
        const auto& box = mfi.tilebox();
        for (int k = box.smallEnd(2); k <= box.bigEnd(2); ++k) {
            for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
                for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                    sum_ros += ros(i, j, k);
                    n_cells += 1.0;
                }
            }
        }
    }

    // MPI reduction for sum_ros and n_cells
    amrex::ParallelDescriptor::ReduceRealSum(sum_ros);
    amrex::ParallelDescriptor::ReduceRealSum(n_cells);

    amrex::Real mean_ros = (n_cells > 0.0) ? sum_ros / n_cells : 0.0;

    // Print diagnostic info
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "[FIRE] t= " << 0.0 << " max_ROS= " << max_ros
                       << " m/s  mean_ROS= " << mean_ros << " m/s\n";
    }
}

