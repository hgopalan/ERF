#include <ERF_FireLayer.H>
#include <ERF.H>
#include <ERF_SurfaceLayer.H>
#include <ERF_FirePrerequisites.H>
#include <ERF_FireGrid.H>
#include <ERF_FireWindExtract.H>
#include <ERF_TerrainSlope.H>
#include <ERF_HybridRos.H>
#include <ERF_FireTerrainReader.H>

#include <AMReX_Reduce.H>

using namespace amrex;

// Debug diagnostics over the fire grid.
//
// These run as AMReX device reductions rather than host loops: with
// the_arena_is_managed defaulting to false, a MultiFab's data is device-only
// under CUDA and cannot be dereferenced from the host at all.
//
// Namespace scope, not FireLayer members, because nvcc forbids an extended
// __device__ lambda inside a function with private or protected class access.
namespace erf_fire_diag {

struct BurningRosStats {
    amrex::Real max_ros  = 0.0;   ///< Max ROS over burning cells [m/s]
    amrex::Real mean_ros = 0.0;   ///< Mean ROS over burning cells [m/s]
    long        n_cells  = 0;     ///< Number of burning cells
};

// Masked ROS statistics over burning cells (phi < 0), reduced across ranks.
// With no burning cells anywhere, max and mean are both 0, matching the
// host-loop version this replaces.
BurningRosStats burning_ros_stats (const amrex::MultiFab& ros,
                                   const amrex::MultiFab& phi)
{
    ReduceOps<ReduceOpMax, ReduceOpSum, ReduceOpSum> reduce_op;
    ReduceData<Real, Real, unsigned long long> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (MFIter mfi(ros); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto const ros_arr = ros.const_array(mfi);
        auto const phi_arr = phi.const_array(mfi);
        reduce_op.eval(bx, reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            if (phi_arr(i,j,k,0) < 0.0_rt) {
                const Real r = ros_arr(i,j,k,0);
                return {r, r, 1ull};
            }
            return {0.0_rt, 0.0_rt, 0ull};
        });
    }

    ReduceTuple hv = reduce_data.value(reduce_op);
    Real max_ros = amrex::get<0>(hv);
    Real sum_ros = amrex::get<1>(hv);
    long n_cells = static_cast<long>(amrex::get<2>(hv));

    ParallelDescriptor::ReduceRealMax(max_ros);
    ParallelDescriptor::ReduceRealSum(sum_ros);
    ParallelDescriptor::ReduceLongSum(n_cells);

    BurningRosStats stats;
    stats.max_ros  = max_ros;
    stats.n_cells  = n_cells;
    stats.mean_ros = (n_cells > 0) ? sum_ros / Real(n_cells) : 0.0_rt;
    return stats;
}

// Number of cells with phi < 0, reduced across ranks.
long count_burning_cells (const amrex::MultiFab& phi)
{
    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<unsigned long long> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto const phi_arr = phi.const_array(mfi);
        reduce_op.eval(bx, reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            return { (phi_arr(i,j,k,0) < 0.0_rt) ? 1ull : 0ull };
        });
    }

    long n = static_cast<long>(amrex::get<0>(reduce_data.value(reduce_op)));
    ParallelDescriptor::ReduceLongSum(n);
    return n;
}

} // namespace erf_fire_diag

void FireLayer::initialize(const ERF& erf,
                            const SurfaceLayer* surface_layer_ptr,
                            const MultiFab* z_phys_nd_atm,
                            const FireParams& fire_params)
{
    m_params = fire_params;
    verify_fire_prerequisites(erf, surface_layer_ptr, fire_params);
    m_fg = create_fire_grid(erf.boxArray(0), erf.DistributionMap(0),
                            erf.Geom(0), fire_params.grid_ratio);
    m_nz = erf.Geom(0).Domain().length(2);

    fire_phi        = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 3);
    fire_wind_ref   = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 2, 0);
    fire_wind_eff   = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 2, 0);
    fire_wind_extract_z = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_slopes     = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 2, 1);
    fire_surface_z  = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_col_ground = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 4, 0);
    fire_curvature  = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_ros        = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_fuel_load  = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_fuel_mc    = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 5, 0);
    fire_mext       = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_heat_flux  = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_spread_vec = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 2, 0);
    fire_arrival_time = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_disp_accum   = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 2, 0);
    fire_surface_temp = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_surface_rh   = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);

    // Phase 5: Heat flux and diagnostics fields
    fire_fireline_intensity = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_flame_length = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);

    // Phase 9 diagnostics
    fire_flame_temp = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_crown_fraction_burned = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    if (m_params.compute_flame_tilt) {
        fire_flame_tilt = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    }
    if (m_params.crown.enable) {
        fire_crown_active = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
        fire_crown_load = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
        fire_crown_ros_active = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    }

    fire_phi->setVal(1.0);
    fire_spread_vec->setVal(0.0_rt);
    fire_wind_ref->setVal(0.0);
    fire_wind_eff->setVal(0.0);
    fire_wind_extract_z->setVal(0.0);
    fire_slopes->setVal(0.0);
    fire_curvature->setVal(0.0);
    fire_ros->setVal(0.0);
    fire_arrival_time->setVal(-1.0_rt);
    fire_disp_accum->setVal(0.0_rt);
    fire_surface_temp->setVal(0.0);
    fire_surface_rh->setVal(0.0);
    fire_fireline_intensity->setVal(0.0_rt);
    fire_flame_length->setVal(0.0_rt);
    fire_flame_temp->setVal(0.0_rt);
    fire_crown_fraction_burned->setVal(0.0_rt);
    if (fire_flame_tilt) {
        fire_flame_tilt->setVal(0.0_rt);
    }
    if (fire_crown_active) {
        fire_crown_active->setVal(0.0_rt);
    }
    if (fire_crown_load) {
        fire_crown_load->setVal(m_params.crown.canopy_bulk_den * m_params.crown.canopy_depth);
    }
    if (fire_crown_ros_active) {
        fire_crown_ros_active->setVal(0.0_rt);
    }

    // Phase 6: fire-atmosphere coupling MultiFabs
    const BoxArray& ba_atm = erf.boxArray(0);
    const DistributionMapping& dm_atm = erf.DistributionMap(0);
    BoxArray ba_atm_2d = ba_atm;
    ba_atm_2d.coarsen(IntVect(1, 1, erf.Geom(0).Domain().length(2)));
    m_Q_atm_prev = std::make_unique<MultiFab>(ba_atm_2d, dm_atm, 1, 0);
    m_Q_lat_atm_prev = std::make_unique<MultiFab>(ba_atm_2d, dm_atm, 1, 0);
    m_Q_atm_prev->setVal(0.0_rt);
    m_Q_lat_atm_prev->setVal(0.0_rt);

    fire_latent_flux = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_latent_flux->setVal(0.0_rt);

    // Phase 8: Albini spotting diagnostics
    if (m_params.spotting.enable) {
        fire_albini_data = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 4, 0);
        fire_albini_data->setVal(0.0_rt);
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Albini spotting enabled: "
                           << "I_B_min=" << m_params.spotting.I_B_min << " kW/m, "
                           << "P_base=" << m_params.spotting.P_base << "\n";
        }
    }

    // Phase 12: Allocate per-cell acceleration state for temporal model.
    // Only allocated when both enable and use_temporal are true.
    // When disabled or size-based, fire_accel_state stays nullptr.
    if (m_params.accel.enable && m_params.accel.use_temporal) {
        fire_accel_state = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 3, 0);
        fire_accel_state->setVal(0.0_rt);
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Fire acceleration enabled (temporal model): "
                           << "A_point=" << m_params.accel.A_point << " 1/min, "
                           << "A_line=" << m_params.accel.A_line << " 1/min\n";
        }
    } else if (m_params.accel.enable) {
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Fire acceleration enabled (size-based model): "
                           << "L_acc=" << m_params.accel.L_acc << " m\n";
        }
    }

    FuelModelParams fp = get_anderson_fuel_params(fire_params.fuel_model_id);
    fire_fuel_load->setVal((fp.w_d1+fp.w_d10+fp.w_d100+fp.w_lh+fp.w_lw)*4.88243);
    m_fuel_load_initial_kg_m2 = (fp.w_d1+fp.w_d10+fp.w_d100+fp.w_lh+fp.w_lw)*4.88243_rt;
    m_fuel_bed_depth_ft = fp.delta;

    fire_fuel_mc->setVal(0.0);
    for (MFIter mfi(*fire_fuel_mc); mfi.isValid(); ++mfi) {
        Array4<Real> mc = fire_fuel_mc->array(mfi);
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (const IntVect& iv) {
            mc(iv,0) = fire_params.moisture_1hr;
            mc(iv,1) = fire_params.moisture_10hr;
            mc(iv,2) = fire_params.moisture_100hr;
            mc(iv,3) = fire_params.moisture_live;  // live herbaceous (Phase 15, new component)
            mc(iv,4) = fire_params.moisture_live;  // live woody      (Phase 15, new component)
        });
    }

    Real dead_load = fp.w_d1+fp.w_d10+fp.w_d100;
    Real sigma_weighted = dead_load > 0.0_rt ? (fp.w_d1*fp.sigma_d1)/dead_load : fp.sigma_d1;
    fire_mext->setVal(compute_moisture_of_extinction(sigma_weighted));

    compute_terrain_slopes(*fire_slopes, z_phys_nd_atm, erf.Geom(0), m_fg, m_params.terrain_file_name);

    // Ground elevation of each fire cell's atmospheric column, used as the datum
    // for wind extraction. Always from the atmospheric terrain, even when a finer
    // terrain file supplies the slopes, since the wind profile being interpolated
    // belongs to that column.
    compute_fire_surface_height(*fire_surface_z, z_phys_nd_atm, erf.Geom(0), m_fg);

    // Grounds of the four columns the bilinear wind stencil blends, so each can
    // be sampled at the same height above its own terrain.
    compute_fire_column_grounds(*fire_col_ground, z_phys_nd_atm, erf.Geom(0), m_fg);
    fire_slopes->FillBoundary(m_fg.geom.periodicity());
    compute_terrain_curvature(*fire_curvature, *fire_slopes, m_fg.geom);

    m_ignition_x = fire_params.ignition_x;
    m_ignition_y = fire_params.ignition_y;
    m_ignition_r = fire_params.ignition_r;

    // The level-set solver needs a true signed distance in metres; the FARSITE
    // path keeps the normalized [-1, 1] indicator convention.
    const bool phi_normalized = (m_params.propagation_method != "levelset");
    initialize_ignition(*fire_phi, m_fg.geom, m_ignition_x, m_ignition_y, m_ignition_r,
                        phi_normalized);
    fire_phi->FillBoundary(m_fg.geom.periodicity());

    for (MFIter mfi(*fire_phi); mfi.isValid(); ++mfi) {
        auto phi_arr = fire_phi->const_array(mfi);
        auto at_arr  = fire_arrival_time->array(mfi);
        ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (const IntVect& iv) {
            if (phi_arr(iv) < 0.0_rt) at_arr(iv) = 0.0_rt;
        });
    }

    // Phase 10: Load spatial fuel map from file when specified.
    // File reading is CPU-only on rank 0; broadcast to all ranks; copy to device.
    if (!m_params.fuel_map.fuel_map_file.empty()) {
        const int fire_nx = m_fg.ba.minimalBox().length(0);
        const int fire_ny = m_fg.ba.minimalBox().length(1);
        std::vector<int> h_fuel_codes;
        int nodata_val = -9999;
        bool ok = false;
        if (m_params.fuel_map.fuel_map_format == "lcp") {
            ok = read_lcp_fuel_map(m_params.fuel_map.fuel_map_file,
                                   fire_nx, fire_ny, h_fuel_codes);
        } else {
            ok = read_ascii_fuel_map(m_params.fuel_map.fuel_map_file,
                                     fire_nx, fire_ny, h_fuel_codes, nodata_val);
        }
        if (ok) {
            m_d_fuel_codes.resize(h_fuel_codes.size());
            amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                             h_fuel_codes.begin(), h_fuel_codes.end(),
                             m_d_fuel_codes.begin());
            fire_fuel_model = std::make_unique<amrex::MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
            fill_fuel_model_mf(*fire_fuel_model, m_d_fuel_codes.data(),
                               m_fg.geom, fire_nx);
            m_has_spatial_fuel = true;
            if (m_params.fire_debug) {
                amrex::Print() << "[FIRE DEBUG] Loaded spatial fuel map '"
                               << m_params.fuel_map.fuel_map_file << "': "
                               << fire_nx << "x" << fire_ny << " cells\n";
            }
        } else {
            amrex::Print() << "[FIRE] WARNING: Cannot read fuel map '"
                           << m_params.fuel_map.fuel_map_file
                           << "'; using uniform fuel_model_id="
                           << m_params.fuel_model_id << "\n";
        }
    }

    // Phase 10: Apply firebreak barriers after ignition stamp.
    if (!m_params.firebreaks.empty() && fire_phi) {
        apply_firebreaks(*fire_phi, m_params.firebreaks, m_fg.geom);
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Applied "
                           << m_params.n_firebreaks << " firebreak(s)\n";
        }
    }

    // Phase 11: Polygon ignition (initial fire perimeter from vertex file).
    // Applied at t=0 as part of initialization, before the schedule.
    if (!m_params.ignition.polygon_file.empty()) {
        std::vector<amrex::Real> xs, ys;
        // Vertex file is read on rank 0 only; broadcast to all ranks inside
        // read_polygon_vertices() before returning.
        read_polygon_vertices(m_params.ignition.polygon_file, xs, ys);
        if (m_params.ignition.polygon_type == "polyline") {
            init_phi_from_polyline(*fire_phi, m_fg.geom, xs, ys,
                                   m_params.ignition.polyline_width);
        } else {
            init_phi_from_polygon(*fire_phi, m_fg.geom, xs, ys);
        }
        fire_phi->FillBoundary(m_fg.geom.periodicity());
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Polygon ignition applied from '"
                           << m_params.ignition.polygon_file << "' ("
                           << m_params.ignition.polygon_type << ")\n";
        }
    }

    // Phase 11: Load ignition schedule if specified.
    // File reading and broadcast are handled inside load_ignition_schedule().
    if (!m_params.ignition.ignition_schedule_file.empty()) {
        load_ignition_schedule(m_params.ignition.ignition_schedule_file,
                               m_ignition_schedule);
        m_has_schedule = true;
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Loaded ignition schedule: "
                           << m_ignition_schedule.events.size()
                           << " events\n";
        }
    }

    m_fp.phi_threshold      = fire_params.farsite_phi_threshold;
    m_fp.coeff_a            = fire_params.farsite_coeff_a;
    m_fp.coeff_b            = fire_params.farsite_coeff_b;
    m_fp.coeff_c            = fire_params.farsite_coeff_c;
    m_fp.use_anderson_lw    = fire_params.farsite_use_anderson_lw;
    m_fp.gaussian_sigma     = fire_params.farsite_gaussian_sigma;
    m_fp.cfl_fire           = fire_params.farsite_cfl_fire;

    m_rc = compute_rothermel_params(fp, fire_params.moisture_1hr,
                                    fire_params.moisture_10hr,
                                    fire_params.moisture_100hr);

    // Phase 13A: Build per-fuel wind height tables and copy to device.
    // When use_per_fuel_wind_ht = false, all entries equal wind_ref_ht (no-op).
    m_use_per_fuel_wind_ht = m_params.use_per_fuel_wind_ht;
    {
        auto h_fcwh = build_fcwh_table(m_params.wind_ref_ht, m_params.use_per_fuel_wind_ht);
        auto h_fcz0 = build_fcz0_table();
        m_d_fcwh.resize(h_fcwh.size());
        m_d_fcz0.resize(h_fcz0.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_fcwh.begin(), h_fcwh.end(), m_d_fcwh.begin());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_fcz0.begin(), h_fcz0.end(), m_d_fcz0.begin());
        if (m_params.fire_debug && m_params.use_per_fuel_wind_ht) {
            amrex::Print() << "[FIRE DEBUG] Per-fuel wind height enabled. "
                           << "FM1 fcwh=" << h_fcwh[1] << " m, FM4 fcwh=" << h_fcwh[4] << " m\n";
        }
    }

    // Phase 13B: Pre-compute alternative ROS model coefficients.
    {
        FuelModelParams fp_ros = get_anderson_fuel_params(m_params.fuel_model_id);
        if (m_params.uses_model("balbi")) {
            m_bc_default = compute_balbi_params(fp_ros, m_params.balbi,
                                                m_params.moisture_1hr);
            // Build per-fuel Balbi table when spatial fuel map is active
            if (m_has_spatial_fuel) {
                auto h_balbi = build_fuel_balbi_table(m_params.balbi,
                                                      m_params.moisture_1hr);
                m_d_balbi_table.resize(h_balbi.size());
                amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_balbi.begin(),
                                 h_balbi.end(), m_d_balbi_table.begin());
            }
            if (m_params.fire_debug) {
                if (m_params.balbi.formulation == 1) {
                    amrex::Print() << "[FIRE DEBUG] ROS model: Balbi (2020), A_rad="
                                   << m_bc_default.A_rad << ", Rb_coef="
                                   << m_bc_default.Rb_coef << " m/(s K^4), u0_coef="
                                   << m_bc_default.u0_coef << " m/s, s*r00="
                                   << m_bc_default.s_r00 << "\n";
                } else {
                    amrex::Print() << "[FIRE DEBUG] ROS model: Balbi (2009), A_coeff="
                                   << m_bc_default.A_coeff << " m/s, v_b="
                                   << m_bc_default.v_b << " m/s\n";
                }
                if (m_params.balbi.directional) {
                    amrex::Print() << "[FIRE DEBUG] Balbi: direction-dependent ROS on "
                                   << "the level-set path\n";
                }
                if (m_params.balbi.use_surface_temp) {
                    amrex::Print() << "[FIRE DEBUG] Balbi: per-cell ambient temperature "
                                   << "from fire_surface_temp\n";
                }
                if (m_params.balbi.use_cell_moisture) {
                    amrex::Print() << "[FIRE DEBUG] Balbi: per-cell fuel moisture from "
                                   << "the moisture ODE state\n";
                }
                if (m_params.balbi.use_moisture_extinction) {
                    amrex::Print() << "[FIRE DEBUG] Balbi: moisture-of-extinction cutoff "
                                   << "at M_x=" << fp_ros.Mx << "\n";
                }
                if (m_params.balbi.wind_source == 1) {
                    amrex::Print() << "[FIRE DEBUG] Balbi: reference-height wind "
                                   << "(WAF and terrain correction bypassed)\n";
                }
                if (m_params.balbi.heat_flux_coupling) {
                    amrex::Print() << "[FIRE DEBUG] Balbi: heat-flux buoyancy coupling, "
                                   << "k_upward=" << m_params.balbi.k_upward
                                   << ", H_ref=" << m_params.balbi.hf_ref_height << " m\n";
                }
            }
        }
        if (m_params.uses_model("cheney_gould")) {
            m_cgc = compute_cheney_gould_params(m_params.cheney_gould);
            if (m_params.fire_debug) {
                amrex::Print() << "[FIRE DEBUG] ROS model: Cheney-Gould (1998), "
                               << "moisture=" << m_params.cheney_gould.moisture
                               << "%, curing=" << m_params.cheney_gould.curing << "\n";
            }
        }
        if (m_params.uses_model("behave")) {
            // Phase 15: Pre-compute BEHAVE multi-class coefficients.
            FuelModelParams fp_bh = get_anderson_fuel_params(m_params.fuel_model_id);
            m_bs_default = compute_behave_state(fp_bh,
                                                m_params.moisture_1hr,
                                                m_params.moisture_10hr,
                                                m_params.moisture_100hr,
                                                m_params.moisture_live,
                                                m_params.moisture_live);
            if (m_params.fire_debug) {
                amrex::Print() << "[FIRE DEBUG] ROS model: BEHAVE multi-class Rothermel, "
                               << "R0=" << m_bs_default.r_0 * 0.00508_rt << " m/s\n";
            }
        }
        if (m_params.uses_model("macarthur")) {
            if (m_params.fire_debug) {
                amrex::Print() << "[FIRE DEBUG] ROS model: MacArthur (1966) Australian formula\n";
            }
        }
        if (m_params.uses_model("rothermel")) {
            // Rothermel coefficients were initialised above via m_rc.
            if (m_params.fire_debug) {
                amrex::Print() << "[FIRE DEBUG] ROS model: Rothermel (1972)\n";
            }
        }
    }

    // Per-cell Rothermel coefficients on a spatial fuel map (opt-in). Without
    // this the Rothermel kernel spreads with the domain fuel_model_id everywhere.
    if (m_params.rothermel_per_fuel && m_has_spatial_fuel && m_params.uses_model("rothermel")) {
        rebuild_rothermel_table(m_params.moisture_1hr,
                                m_params.moisture_10hr,
                                m_params.moisture_100hr);
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Rothermel: per-cell coefficients from the "
                           << "spatial fuel map (" << m_d_rc_table.size() << " codes)\n";
        }
    }

    // Hybrid ROS: allocate the per-cell weight and the secondary scratch field,
    // then fill the weight from the selector. The weight is static in step 1
    // (region and fuel selectors), so it is built once here.
    if (m_params.is_hybrid()) {
        fire_ros_weight  = std::make_unique<amrex::MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
        fire_ros_scratch = std::make_unique<amrex::MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
        fire_ros_scratch->setVal(0.0_rt);
        if (m_params.hybrid.selector == "structure") {
            fire_structure_height = std::make_unique<amrex::MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
            if (!read_heightmap_nearest_onto_fire_cells(*fire_structure_height, m_fg,
                                                        m_params.hybrid.structure_file)) {
                amrex::Abort("[FIRE] hybrid.selector = structure: cannot read structure file '"
                             + m_params.hybrid.structure_file + "'");
            }
            if (m_params.fire_debug) {
                amrex::Print() << "[FIRE DEBUG] Hybrid ROS: structure heightmap '"
                               << m_params.hybrid.structure_file << "' max height "
                               << fire_structure_height->max(0) << " m\n";
            }
        }
        init_ros_weight();
        if (m_params.fire_debug) { print_hybrid_weight_summary(); }
    }

    m_probe_reported.assign(m_params.probes.size() / 2, false);

    amrex::Print() << "[FIRE] FireLayer initialized: C=" << m_fg.C
                   << ", fuel_model=" << fire_params.fuel_model_id
                   << ", grid=" << m_fg.ba.size() << " boxes" << std::endl;

    if (m_params.fire_debug) {
        IntVect max_extent = m_fg.geom.Domain().size();
        RealBox prob_domain = m_fg.geom.ProbDomain();
        Real dx_fire = (prob_domain.hi(0)-prob_domain.lo(0))/max_extent[0];
        Real dy_fire = (prob_domain.hi(1)-prob_domain.lo(1))/max_extent[1];
        amrex::Print() << "[FIRE DEBUG] Fire grid: dx=" << dx_fire << " m, dy=" << dy_fire
                       << " m, extent=" << max_extent[0] << "x" << max_extent[1]
                       << ", grid_ratio=" << m_fg.C << std::endl;
        amrex::Print() << "[FIRE DEBUG] Coupling type: " << fire_params.coupling_type
                       << " (passive=" << fire_params.is_passive()
                       << ", lagged=" << fire_params.is_lagged()
                       << ", synchronous=" << fire_params.is_synchronous() << ")" << std::endl;
        amrex::Print() << "[FIRE DEBUG] Fire-atmosphere feedback multiplier: "
                       << fire_params.fire_atm_feedback << std::endl;
        amrex::Print() << "[FIRE DEBUG] Heat flux alfg (e-folding height): "
                       << fire_params.heat_flux_alfg << " m" << std::endl;
        amrex::Print() << "[FIRE DEBUG] Inject latent heat: " << fire_params.inject_latent << std::endl;
    }
}

// surface_layer is unused: the fire layer reads the atmospheric state it needs
// through the wind, temperature and humidity MultiFabs passed alongside it. The
// parameter stays for interface symmetry with initialize().
void FireLayer::advance(Real time, Real dt, SurfaceLayer& surface_layer,
                        const MultiFab& xvel, const MultiFab& yvel,
                        const MultiFab& z_phys_cc,
                        const MultiFab& T_atm_k0, const MultiFab& RH_atm_k0)
{
    amrex::ignore_unused(surface_layer);

    m_current_time = time;
    m_dt_atm       = dt;
    ++m_step;

    if (m_params.fire_debug)
        amrex::Print() << "[FIRE DEBUG] Starting fire advance step with dt=" << dt << std::endl;

    fill_fire_wind_from_interpolation(*fire_wind_ref, *fire_wind_extract_z, xvel, yvel, z_phys_cc,
                                      *fire_surface_z, *fire_col_ground,
                                      m_fg, m_params.wind_ref_ht, m_nz,
                                      m_use_per_fuel_wind_ht ? fire_fuel_model.get() : nullptr,
                                      m_use_per_fuel_wind_ht ? m_d_fcwh.data() : nullptr,
                                      m_use_per_fuel_wind_ht ? 13 : 0,
                                      m_params.wind_interp);
    if (m_params.fire_debug) {
        amrex::Print() << "[FIRE DEBUG] Wind extraction completed. Max reference wind: "
                       << fire_wind_ref->max(0) << " m/s" << std::endl;
        if (m_step > 0)
            amrex::Print() << "[FIRE DEBUG] Wind extraction height range: min="
                           << fire_wind_extract_z->min(0) << " m  max="
                           << fire_wind_extract_z->max(0) << " m" << std::endl;
    }

    MultiFab::Copy(*fire_wind_eff, *fire_wind_ref, 0, 0, 2, 0);

    if (m_params.use_waf) {
        if (m_params.fire_debug)
            amrex::Print() << "[FIRE DEBUG] Applying Wind Adjustment Factor (formula: "
                           << m_params.waf_formula << ")" << std::endl;
        apply_waf_to_wind();
    }

    if (m_params.use_terrain_wind) {
        if (m_params.fire_debug)
            amrex::Print() << "[FIRE DEBUG] Applying FARSITE terrain wind corrections" << std::endl;
        apply_farsite_terrain_wind(*fire_wind_eff, *fire_slopes, *fire_curvature,
                                   m_params.k_ridge, m_params.k_shelter,
                                   m_params.k_valley, m_params.k_deflect);
    }

    if (m_params.fire_debug)
        amrex::Print() << "[FIRE DEBUG] Effective wind computed. Max effective wind: "
                       << fire_wind_eff->max(0) << " m/s" << std::endl;

    if (m_params.fire_debug)
        amrex::Print() << "[FIRE DEBUG] Updating fuel moisture from atmospheric state" << std::endl;
    advance_fuel_moisture(dt, T_atm_k0, RH_atm_k0);
    if (m_params.fire_debug) {
        amrex::Print() << "[FIRE DEBUG] Fuel moisture update completed. Max 1-hour moisture: "
                       << fire_fuel_mc->max(0) << std::endl;
        amrex::Print() << "[FIRE DEBUG] Surface temp range: min="
                       << fire_surface_temp->min(0) << " K  max="
                       << fire_surface_temp->max(0) << " K" << std::endl;
        amrex::Print() << "[FIRE DEBUG] Surface RH range:   min="
                       << fire_surface_rh->min(0) << "    max="
                       << fire_surface_rh->max(0) << std::endl;
    }

    if (m_params.moisture_dynamic) {
        long nc = fire_fuel_mc->boxArray().numPts();
        Real avg1   = (nc>0) ? fire_fuel_mc->sum(0)/Real(nc) : m_params.moisture_1hr;
        Real avg10  = (nc>0) ? fire_fuel_mc->sum(1)/Real(nc) : m_params.moisture_10hr;
        Real avg100 = (nc>0) ? fire_fuel_mc->sum(2)/Real(nc) : m_params.moisture_100hr;
        avg1   = amrex::max(0.01_rt, amrex::min(avg1,   0.40_rt));
        avg10  = amrex::max(0.01_rt, amrex::min(avg10,  0.40_rt));
        avg100 = amrex::max(0.01_rt, amrex::min(avg100, 0.40_rt));
        FuelModelParams fp_cur = get_anderson_fuel_params(m_params.fuel_model_id);
        m_rc = compute_rothermel_params(fp_cur, avg1, avg10, avg100);
        if (!m_d_rc_table.empty()) {
            rebuild_rothermel_table(avg1, avg10, avg100);
        }
        if (m_params.fire_debug)
            amrex::Print() << "[FIRE DEBUG] Updated Rothermel coefficients with avg moisture: "
                           << "M_1hr=" << avg1 << " M_10hr=" << avg10
                           << " M_100hr=" << avg100 << " R0=" << m_rc.R0 << " m/s" << std::endl;
        
        // Phase 13B: Moisture coupling for Balbi and Cheney-Gould models
        if (m_params.moisture_dynamic && m_params.uses_model("balbi")) {
            // Recompute Balbi coefficients with updated moisture
            // A_coeff carries the moisture dependence through B*, so both the
            // default coefficients and the per-fuel table have to be rebuilt.
            FuelModelParams fp_balbi = get_anderson_fuel_params(m_params.fuel_model_id);
            m_bc_default = compute_balbi_params(fp_balbi, m_params.balbi, avg1);
            if (m_has_spatial_fuel) {
                auto h_balbi = build_fuel_balbi_table(m_params.balbi, avg1);
                m_d_balbi_table.resize(h_balbi.size());
                amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_balbi.begin(),
                                 h_balbi.end(), m_d_balbi_table.begin());
            }
        }
        if (m_params.moisture_dynamic && m_params.uses_model("cheney_gould")) {
            // Update Cheney-Gould with current 1-hr moisture converted to percent
            FireParams::CheneyGouldParams cgp_cur = m_params.cheney_gould;
            cgp_cur.moisture = avg1 * 100.0_rt;  // fraction → percent
            m_cgc = compute_cheney_gould_params(cgp_cur);
        }
        // Phase 15: Update BEHAVE state when dynamic moisture is enabled.
        if (m_params.moisture_dynamic && m_params.uses_model("behave")) {
            FuelModelParams fp_bh = get_anderson_fuel_params(m_params.fuel_model_id);
            // Domain-average live moisture from components 3 and 4
            long nc_live = fire_fuel_mc->boxArray().numPts();
            Real avg_lh  = (nc_live > 0) ? fire_fuel_mc->sum(3) / Real(nc_live) : m_params.moisture_live;
            Real avg_lw  = (nc_live > 0) ? fire_fuel_mc->sum(4) / Real(nc_live) : m_params.moisture_live;
            avg_lh = amrex::max(0.30_rt, amrex::min(avg_lh, 2.50_rt));
            avg_lw = amrex::max(0.30_rt, amrex::min(avg_lw, 2.50_rt));
            m_bs_default = compute_behave_state(fp_bh, avg1, avg10, avg100, avg_lh, avg_lw);
        }
    }

    // Phase 11: Apply any scheduled ignition events due this timestep.
    // Time window: (m_current_time - dt, m_current_time].
    if (m_has_schedule && fire_phi) {
        apply_scheduled_ignitions(*fire_phi, m_fg.geom,
                                  m_ignition_schedule,
                                  m_current_time,
                                  m_current_time - dt);
        // fill_boundary after any phi modification to propagate ghost cells
        //amrex::FillBoundary(*fire_phi, m_fg.geom);
        fire_phi->FillBoundary(m_fg.geom.periodicity());        
    }

    // Hybrid wind selector: the weight follows the effective wind, so it is
    // rebuilt every fire step; the other selectors are static.
    if (m_params.is_hybrid() && m_params.hybrid.selector == "wind") {
        update_wind_weight();
        if (m_params.fire_debug) { print_hybrid_weight_summary(); }
    }

    // Phase 13B: ROS model dispatch.
    // All models write into fire_ros [m/s].
    // Rothermel is the default (ros_model = "rothermel" or unrecognised string).
    // With ros_model = "hybrid" the primary model writes fire_ros, the
    // secondary writes fire_ros_scratch, and the two are blended per cell with
    // fire_ros_weight (0 = primary, 1 = secondary).
    // Optional per-cell fields for the Balbi kernels. Declared here because the
    // level-set path below reuses them when balbi.directional is set.
    BalbiFieldInputs balbi_in;
    if (m_params.uses_model("balbi")) {
        balbi_in.fuel_model   = m_has_spatial_fuel ? fire_fuel_model.get() : nullptr;
        balbi_in.table        = m_d_balbi_table.empty() ? nullptr : m_d_balbi_table.data();
        balbi_in.table_size   = static_cast<int>(m_d_balbi_table.size());
        balbi_in.fp           = get_anderson_fuel_params(m_params.fuel_model_id);
        balbi_in.M_f          = m_params.moisture_1hr;
        balbi_in.surface_temp = fire_surface_temp.get();
        // Per-cell moisture needs the Phase 4 ODE state; without dynamic
        // moisture that field never evolves, so the domain value is used.
        balbi_in.fuel_mc      = m_params.moisture_dynamic ? fire_fuel_mc.get() : nullptr;
        // fire_heat_flux is filled at the end of the step, so this is the
        // previous step's flux: the buoyancy feedback lags the ROS by one
        // fire step.
        balbi_in.heat_flux    = fire_heat_flux.get();

        if (m_params.moisture_dynamic && fire_fuel_mc) {
            long nc_mc = fire_fuel_mc->boxArray().numPts();
            if (nc_mc > 0) {
                Real avg_mc = fire_fuel_mc->sum(0) / Real(nc_mc);
                balbi_in.M_f = amrex::max(0.01_rt, amrex::min(avg_mc, 0.40_rt));
            }
        }

    }

    if (m_params.is_hybrid()) {
        fill_ros_for_model(m_params.hybrid.primary,   *fire_ros,         balbi_in);
        fill_ros_for_model(m_params.hybrid.secondary, *fire_ros_scratch, balbi_in);
        // Blend: R = (1 - w) R_primary + w R_secondary. With w = 0 or 1
        // everywhere this reproduces the single-model result exactly.
        blend_ros_fields(*fire_ros, *fire_ros_scratch, *fire_ros_weight);
    } else {
        fill_ros_for_model(m_params.ros_model, *fire_ros, balbi_in);
    }
    if (m_params.fire_debug) {
        // Masked ROS diagnostics (only for burning cells where phi < 0)
        const auto ros_stats = erf_fire_diag::burning_ros_stats(*fire_ros, *fire_phi);

        amrex::Print() << "[FIRE DEBUG] Rate-of-spread computed. Max: " << ros_stats.max_ros
                       << " m/s, Mean: " << ros_stats.mean_ros
                       << " m/s" << std::endl;
    }

    // Phase 10: Fuel boundary blending when spatial fuel map is active.
    if (m_has_spatial_fuel &&
        m_params.fuel_map.blending_fraction > 0.0_rt &&
        fire_fuel_model && fire_ros) {
        apply_fuel_boundary_blending(*fire_ros, *fire_fuel_model,
                                      m_params.fuel_map.blending_fraction);
    }

    fire_phi->FillBoundary(m_fg.geom.periodicity());

    // Phase 12: Apply fire acceleration scaling to ROS.
    // Reduces ROS for small fires not yet at quasi-steady-state.
    // Returns immediately when accel.enable = false (zero cost when disabled).
    if (m_params.accel.enable && fire_ros && fire_phi) {
        apply_fire_acceleration(*fire_ros, *fire_phi, m_fg.geom,
                                m_params.accel, dt,
                                fire_accel_state.get(),
                                m_params.fire_debug);
    }

    // Phase 9: crown-fire ROS enhancement.
    // Must run before propagation so the front actually advances at the
    // crowning rate; it also latches crown activation for the heat-flux pass.
    apply_crown_fire_ros();

    int n_substeps = 0;
    
    if (m_params.propagation_method == "levelset") {
        // --- Level-set path ---
        // CFL-based subcycling (same structure as FARSITE)
        amrex::Real time_remaining = dt;
        int n_ls_substeps = 0;
        while (time_remaining > 1.0e-14) {
            amrex::Real max_ros = fire_ros->max(0);
            amrex::Real dt_ls   = (max_ros > 1.0e-10)
                ? m_params.levelset_cfl * std::min(m_fg.geom.CellSize()[0],
                                                   m_fg.geom.CellSize()[1]) / max_ros
                : time_remaining;
            dt_ls = std::min(dt_ls, time_remaining);

            // With directional spread enabled the ROS is rebuilt from the front
            // normal inside every RK stage, so the front gets head, flank and
            // backing spread from the model rather than a single head-fire
            // magnitude. fire_ros still holds the isotropic head ROS and sets
            // the CFL, which stays conservative since the directional rate never
            // exceeds it.
            const bool hybrid_directional =
                m_params.is_hybrid() &&
                (m_params.directional_ros ||
                 (m_params.balbi.directional && m_params.uses_model("balbi")));
            const bool balbi_directional =
                (m_params.ros_model == "balbi") &&
                (m_params.balbi.directional || m_params.directional_ros);
            const bool generic_directional =
                m_params.directional_ros && !m_params.is_hybrid() &&
                (m_params.ros_model != "balbi");

            if (hybrid_directional) {
                // Both members are rebuilt along the front normal at every RK
                // stage and blended with the same weight the isotropic path uses.
                HybridDirectionalSpec spec;
                auto set_member = [&](const std::string& name, int& model,
                                      DirectionalRosState& state) {
                    if (name == "balbi") {
                        model = HYBRID_MODEL_BALBI;
                    } else {
                        state = make_directional_state(name);
                        model = state.model;
                    }
                };
                set_member(m_params.hybrid.primary,   spec.primary_model,   spec.primary_state);
                set_member(m_params.hybrid.secondary, spec.secondary_model, spec.secondary_state);
                spec.wind_eff   = fire_wind_eff.get();
                spec.balbi_wind = (m_params.balbi.wind_source == 1)
                                ? fire_wind_ref.get() : fire_wind_eff.get();
                spec.bc         = &m_bc_default;
                spec.bp         = &m_params.balbi;
                spec.balbi_in   = &balbi_in;
                spec.weight     = fire_ros_weight.get();
                advect_levelset_hybrid_rk3(*fire_phi, *fire_slopes, m_fg.geom, dt_ls,
                                           m_params.levelset_eps_visc, spec);
            } else if (balbi_directional) {
                advect_levelset_balbi_rk3(*fire_phi,
                                          (m_params.balbi.wind_source == 1)
                                              ? *fire_wind_ref : *fire_wind_eff,
                                          *fire_slopes,
                                          m_fg.geom, dt_ls,
                                          m_params.levelset_eps_visc,
                                          m_bc_default, m_params.balbi, balbi_in);
            } else if (generic_directional) {
                const DirectionalRosState dir_state = make_directional_state(m_params.ros_model);
                advect_levelset_directional_rk3(*fire_phi, *fire_wind_eff,
                                                *fire_slopes, m_fg.geom, dt_ls,
                                                m_params.levelset_eps_visc,
                                                dir_state);
            } else {
                fire_levelset::advect_levelset_weno5z_rk3(*fire_phi, *fire_wind_eff,
                                                *fire_ros, m_fg.geom, dt_ls,
                                                m_params.levelset_eps_visc,
                                                fire_slopes.get());
            }
            fire_phi->FillBoundary(m_fg.geom.periodicity());

            ++m_levelset_subcycle_count;
            if (m_levelset_subcycle_count % m_params.levelset_reinit_every == 0) {
                // Sussman reinitialization is stable for dtau <= dx/2; 0.5*dx sits
                // exactly on that limit and went unstable once enough iterations
                // were taken, so default to half of it.
                amrex::Real dtau = (m_params.levelset_reinit_dtau > 0.0)
                    ? m_params.levelset_reinit_dtau
                    : 0.25 * std::min(m_fg.geom.CellSize()[0], m_fg.geom.CellSize()[1]);
                fire_levelset::reinitialize_phi(*fire_phi, m_fg.geom,
                                      m_params.levelset_reinit_iters, dtau,
                                      m_params.levelset_reinit_band_m,
                                      /*normalized=*/false);
                fire_phi->FillBoundary(m_fg.geom.periodicity());
            }

            // Update arrival time for newly burned cells (phi < 0)
            {
                const amrex::Real t_now = m_current_time + (dt - time_remaining);
                for (amrex::MFIter mfi(*fire_phi); mfi.isValid(); ++mfi) {
                    auto p  = fire_phi->const_array(mfi);
                    auto at = fire_arrival_time->array(mfi);
                    amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (const amrex::IntVect& iv) noexcept {
                        if (p(iv) < 0.0_rt && at(iv) < 0.0_rt) at(iv) = t_now;
                    });
                }
            }
            time_remaining -= dt_ls;

            // Mirrors the guard the FARSITE path has had all along. Without it a
            // pathological dt_ls turns this into a silent hang, and each pass
            // costs a global fire_ros->max(0) reduction.
            if (++n_ls_substeps > 1000) {
                amrex::Abort("[FIRE] level-set subcycle: too many substeps in one "
                             "atmospheric step; check erf.fire.levelset.cfl");
            }
        }
        // Substeps taken this step, not the run-to-date total that
        // m_levelset_subcycle_count accumulates.
        n_substeps = n_ls_substeps;
    } else {
        // --- Default: FARSITE Lagrangian path (unchanged) ---
        // fire_slopes turns the ROS into a spread rate along the ground: a step
        // of ds up a slope covers ds cos(theta) in map view.
        n_substeps = advance_fire_subcycle(*fire_phi, *fire_spread_vec,
                                          *fire_disp_accum,
                                          *fire_arrival_time,
                                          *fire_wind_eff, *fire_ros,
                                          m_fg.geom, dt,
                                          m_current_time,
                                          m_fp,
                                          fire_slopes.get());
    }

    if (m_params.fire_debug) {
        amrex::Print() << "[FIRE DEBUG] Fire front propagation completed with "
                       << n_substeps << " fire subcycles" << std::endl;
        const long num_fire_cells = erf_fire_diag::count_burning_cells(*fire_phi);
        amrex::Print() << "[FIRE DEBUG] Number of active fire cells: " << num_fire_cells << std::endl;
    }

    fire_phi->FillBoundary(m_fg.geom.periodicity());

    compute_heat_flux_and_diagnostics(dt);

    // Phase 8: Albini ember spotting
    // Apply stochastic spotting at the specified interval.
    // fire_wind_eff provides the 2-D wind field for trajectory integration.
    // fire_fuel_load provides residual fuel for re-entry filtering.
    if (m_params.spotting.enable && fire_albini_data && fire_wind_eff) {
        if (m_step % m_params.spotting.spotting_interval == 0) {
            fire_albini_data->setVal(0.0_rt);
            FuelModelParams fp_sp = get_anderson_fuel_params(m_params.fuel_model_id);
            std::string fuel_sys  = m_params.spotting.fuel_system;
            compute_albini_spotting(
                *fire_phi,
                *fire_albini_data,
                *fire_wind_eff,
                *fire_ros,
                m_fg.geom,
                fp_sp,
                m_params.spotting,
                m_step,
                fire_fuel_load.get(),
                &fuel_sys,
                m_params.fuel_model_id,
                m_params.fire_debug,
                fire_fuel_load.get(),
                m_fuel_load_initial_kg_m2,
                fire_surface_z.get());
        }
    }

    if (m_params.fire_debug) {
        amrex::Real phi_min  = fire_phi->min(0, 0);   // nghost=0
        amrex::Real phi_max  = fire_phi->max(0, 0);
        
        // Masked ROS diagnostics (only for burning cells where phi < 0)
        const auto ros_stats = erf_fire_diag::burning_ros_stats(*fire_ros, *fire_phi);

        amrex::Real ros_max  = ros_stats.max_ros;
        amrex::Real ros_mean = ros_stats.mean_ros;
        
        amrex::Real Q_max    = fire_heat_flux ? fire_heat_flux->max(0) : 0.0;
        amrex::Real I_B_max  = fire_fireline_intensity ? fire_fireline_intensity->max(0) : 0.0;
        amrex::Real L_max    = fire_flame_length ? fire_flame_length->max(0) : 0.0;
        amrex::Print() << "[FIRE] t=" << m_current_time
                       << "  substeps=" << n_substeps
                       << "  phi_min=" << phi_min
                       << "  phi_max=" << phi_max
                       << "  max_ROS=" << ros_max << " m/s"
                       << "  mean_ROS=" << ros_mean << " m/s"
                       << "  Q_max=" << Q_max << " W/m2"
                       << "  I_B_max=" << I_B_max << " kW/m"
                       << "  L_max=" << L_max << " m\n";
    }

    report_probes();

    if (m_params.write_fire_stats_csv) {
        static bool csv_header_written = false;
        if (!csv_header_written) {
            write_fire_stats_header(m_params.fire_stats_csv_file);
            csv_header_written = true;
        }
        append_fire_stats(*fire_phi, *fire_arrival_time, m_fg.geom,
                         m_step, m_current_time, m_params.fire_stats_csv_file,
                         fire_ros.get(), fire_heat_flux.get(), fire_albini_data.get());
    }
}


void FireLayer::apply_waf_to_wind()
{
    Real waf = 0.4_rt;
    if      (m_params.waf_formula == "andrews")      waf = compute_waf_unsheltered(m_fuel_bed_depth_ft);
    else if (m_params.waf_formula == "behaviorplus")  waf = compute_waf_behaviorplus(m_fuel_bed_depth_ft);
    else amrex::Print() << "[FIRE WARNING] Unknown waf_formula='" << m_params.waf_formula
                        << "'. Using default WAF=" << waf << std::endl;
    fire_wind_eff->mult(waf, 0, 2, 0);
}

void FireLayer::advance_fuel_moisture(Real dt_s,
                                      const MultiFab& T_atm_k0,
                                      const MultiFab& RH_atm_k0)
{
    int C = m_fg.C;

    // Map atmospheric T and RH to fire grid and store persistently.
    // This operation runs every timestep to ensure
    // fire_surface_temp and fire_surface_rh are available for plotfile output.
    for (MFIter mfi(*fire_surface_temp, false); mfi.isValid(); ++mfi) {
        Array4<Real> T_f  = fire_surface_temp->array(mfi);
        Array4<Real> RH_f = fire_surface_rh->array(mfi);
        Array4<const Real> T_atm  = T_atm_k0.const_array(mfi);
        Array4<const Real> RH_atm = RH_atm_k0.const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (const IntVect& iv_f) {
            int ia = iv_f[0]/C, ja = iv_f[1]/C;
            T_f(iv_f[0],iv_f[1],0)  = T_atm(ia,ja,0);
            RH_f(iv_f[0],iv_f[1],0) = RH_atm(ia,ja,0);
        });
    }

    if (!m_params.moisture_dynamic) { return; }
    Real dt_hours = dt_s / 3600.0_rt;
    FuelModelParams fp = get_anderson_fuel_params(m_params.fuel_model_id);
    Real precip_mm_hr = m_params.precip_rate_mm_hr;

    for (MFIter mfi(*fire_fuel_mc); mfi.isValid(); ++mfi) {
        Array4<Real> mc   = fire_fuel_mc->array(mfi);
        Array4<Real> mext = fire_mext->array(mfi);
        Array4<const Real> T_f  = fire_surface_temp->const_array(mfi);
        Array4<const Real> RH_f = fire_surface_rh->const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (const IntVect& iv_f) {
            int i = iv_f[0], j = iv_f[1];
            Real T_C = T_f(i,j,0) - 273.15_rt;
            Real RH  = RH_f(i,j,0) * 100.0_rt;
            // Existing 3 dead fuel classes (unchanged)
            mc(i,j,0,0) = advance_fuel_moisture_one_class(mc(i,j,0,0),RH,T_C,precip_mm_hr,dt_hours,FuelMoistureConst::TAU_1HR);
            mc(i,j,0,1) = advance_fuel_moisture_one_class(mc(i,j,0,1),RH,T_C,precip_mm_hr,dt_hours,FuelMoistureConst::TAU_10HR);
            mc(i,j,0,2) = advance_fuel_moisture_one_class(mc(i,j,0,2),RH,T_C,precip_mm_hr,dt_hours,FuelMoistureConst::TAU_100HR);
            // Phase 15: live fuel moisture (components 3 and 4)
            // Live fuels respond slowly to atmospheric conditions.
            // Use TAU_100HR as a lower bound; live moisture is bounded [0.30, 2.50].
            if (mc.nComp() >= 5) {
                Real lh_new = advance_fuel_moisture_one_class(mc(i,j,0,3),RH,T_C,0.0_rt,dt_hours,FuelMoistureConst::TAU_100HR);
                Real lw_new = advance_fuel_moisture_one_class(mc(i,j,0,4),RH,T_C,0.0_rt,dt_hours,FuelMoistureConst::TAU_100HR);
                mc(i,j,0,3) = amrex::max(0.30_rt, amrex::min(lh_new, 2.50_rt));  // live herba: 30%–250%
                mc(i,j,0,4) = amrex::max(0.30_rt, amrex::min(lw_new, 2.50_rt));  // live woody: 30%–250%
            }
            Real dead_load = fp.w_d1+fp.w_d10+fp.w_d100;
            Real sw = dead_load>0.0_rt ? (fp.w_d1*fp.sigma_d1)/dead_load : fp.sigma_d1;
            mext(i,j,0) = compute_moisture_of_extinction(sw);
        });
    }
}

void FireLayer::compute_heat_flux_and_diagnostics(Real dt_fire_s)
{
    FuelModelParams fp = get_anderson_fuel_params(m_params.fuel_model_id);

    Real dead_load = fp.w_d1 + fp.w_d10 + fp.w_d100;
    Real sigma_agg = (dead_load > 1.0e-10_rt)
        ? (fp.w_d1*fp.sigma_d1 + fp.w_d10*FIRE_SIGMA_D10 + fp.w_d100*FIRE_SIGMA_D100) / dead_load
        : fp.sigma_d1;

    Real tau_sav = compute_residence_time_s(sigma_agg, fp.rho_p);
    Real tau_sav_floor = (m_params.tau_residence_s > 0.0_rt) ? m_params.tau_residence_s : tau_sav;

    if (m_params.fire_debug) {
        // Masked ROS diagnostics (only for burning cells where phi < 0)
        const auto ros_stats = erf_fire_diag::burning_ros_stats(*fire_ros, *fire_phi);

        amrex::Print() << "[FIRE DEBUG] tau_sav=" << tau_sav
                       << " s  (dx_fire=" << m_fg.geom.CellSize(0)
                       << " m, max_ROS=" << ros_stats.max_ros << " m/s)" << std::endl;
    }

    fill_fire_heat_flux(*fire_heat_flux, *fire_fuel_load,
                        *fire_phi, *fire_ros, fp,
                        m_fg.geom.CellSize(0), tau_sav_floor, dt_fire_s,
                        m_has_spatial_fuel ? fire_fuel_model.get() : nullptr);

    const Real h_kJ_per_kg = fp.heat_content * 2.326_rt;
    const Real h_fuel_Jkg = fp.heat_content * 2326.0_rt;

    // Phase 9 (part 2): crown heat release and canopy fuel depletion.
    // The ROS enhancement and the crown-activation latch happen in
    // apply_crown_fire_ros(), before the front is propagated; this pass only
    // consumes what it recorded.
    if (m_params.crown.enable && fire_crown_active && fire_crown_load && fire_crown_ros_active) {
        const Real canopy_depth  = m_params.crown.canopy_depth;
        const Real h_crown_Jkg   = m_params.crown.h_crown_BTU_lb * 2326.0_rt;

        for (MFIter mfi(*fire_ros); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto const crown_active_arr = fire_crown_active->const_array(mfi);
            auto const crown_ros_arr    = fire_crown_ros_active->const_array(mfi);
            auto crown_load_arr = fire_crown_load->array(mfi);
            auto heat_flux_arr  = fire_heat_flux->array(mfi);

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                if (crown_active_arr(i, j, k) < 0.5_rt) { return; }
                if (crown_load_arr(i, j, k) <= 0.0_rt)  { return; }

                const Real R_active  = crown_ros_arr(i, j, k);
                const Real tau_crown = amrex::max(canopy_depth / amrex::max(R_active, 1.0e-6_rt), 1.0_rt);

                heat_flux_arr(i, j, k) += crown_load_arr(i, j, k) * h_crown_Jkg / tau_crown;
                crown_load_arr(i, j, k) *= std::exp(-dt_fire_s / tau_crown);
                crown_load_arr(i, j, k) = amrex::max(crown_load_arr(i, j, k), 0.0_rt);
            });
        }
    }

    fill_fire_diagnostics(*fire_fireline_intensity, *fire_flame_length,
                          *fire_phi, *fire_ros, *fire_fuel_load,
                          m_fuel_load_initial_kg_m2, h_kJ_per_kg);

    Real M_f = m_params.moisture_1hr;
    if (m_params.moisture_dynamic && fire_fuel_mc) {
        const long nc = fire_fuel_mc->boxArray().numPts();
        const Real avg1 = (nc > 0) ? fire_fuel_mc->sum(0) / Real(nc) : m_params.moisture_1hr;
        const Real avg10 = (nc > 0) ? fire_fuel_mc->sum(1) / Real(nc) : m_params.moisture_10hr;
        const Real avg100 = (nc > 0) ? fire_fuel_mc->sum(2) / Real(nc) : m_params.moisture_100hr;
        M_f = (dead_load > 1.0e-10_rt)
            ? (fp.w_d1 * avg1 + fp.w_d10 * avg10 + fp.w_d100 * avg100) / dead_load
            : avg1;
    }

    if (fire_flame_temp) {
        fill_flame_temperature(*fire_flame_temp, *fire_fireline_intensity, *fire_phi,
                               m_params.flame_temp_method, h_fuel_Jkg, M_f,
                               m_params.flame_temp_T_amb, m_params.fire_debug);
    }

    if (fire_flame_tilt) {
        fill_flame_tilt_angle(*fire_flame_tilt, *fire_fireline_intensity, *fire_wind_eff,
                              m_params.flame_tilt_rho_air, m_params.flame_tilt_T_amb,
                              m_params.fire_debug);
    }
}

void FireLayer::apply_crown_fire_ros()
{
    if (!m_params.crown.enable || !fire_crown_active || !fire_crown_load
        || !fire_crown_ros_active) {
        return;
    }

    if (fire_crown_fraction_burned) {
        fire_crown_fraction_burned->setVal(0.0_rt);
    }

    const FuelModelParams fp = get_anderson_fuel_params(m_params.fuel_model_id);
    const Real h_kJ_per_kg   = fp.heat_content * 2.326_rt;

    // Surface-only quantities: the crown criterion is driven by the surface
    // fireline intensity, so both are evaluated from fire_ros before this
    // routine overwrites it with the crown-enhanced value.
    MultiFab surface_ros(m_fg.ba, m_fg.dm, 1, 0);
    MultiFab surface_intensity(m_fg.ba, m_fg.dm, 1, 0);
    MultiFab surface_flame_length(m_fg.ba, m_fg.dm, 1, 0);
    MultiFab::Copy(surface_ros, *fire_ros, 0, 0, 1, 0);

    fill_fire_diagnostics(surface_intensity, surface_flame_length,
                          *fire_phi, surface_ros, *fire_fuel_load,
                          m_fuel_load_initial_kg_m2, h_kJ_per_kg);

    const auto& crown = m_params.crown;
    const Real canopy_base_ht = crown.canopy_base_ht;
    const Real canopy_bulk_den = crown.canopy_bulk_den;
    const Real foliar_moisture = crown.foliar_moisture;
    const Real M_c = crown.M_c;
    const Real default_moisture_10hr = m_params.moisture_10hr;
    const Real I_B_crit = van_wagner_critical_intensity(
        canopy_base_ht, foliar_moisture, M_c);
    const Real fixed_u10_ms = (crown.wind_10m_kmh > 0.0_rt)
        ? crown.wind_10m_kmh / 3.6_rt
        : -1.0_rt;
    const int crown_model_id = (crown.ros_model == "rothermel1991") ? 1
        : (crown.ros_model == "van_wagner_proxy") ? 2 : 0;
    const bool use_dynamic_mc = (m_params.moisture_dynamic && fire_fuel_mc != nullptr);
    const bool use_passive_blend = crown.use_passive_blend;

    for (MFIter mfi(*fire_ros); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto const phi_arr = fire_phi->const_array(mfi);
        auto const wind_arr = fire_wind_eff->const_array(mfi);
        auto const surface_ros_arr = surface_ros.const_array(mfi);
        auto const surface_I_B_arr = surface_intensity.const_array(mfi);
        Array4<const Real> mc_arr;
        if (use_dynamic_mc) {
            mc_arr = fire_fuel_mc->const_array(mfi);
        }
        auto ros_arr = fire_ros->array(mfi);
        auto crown_active_arr = fire_crown_active->array(mfi);
        auto crown_ros_arr = fire_crown_ros_active->array(mfi);
        auto crown_frac_arr = fire_crown_fraction_burned->array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (phi_arr(i, j, k) >= 0.0_rt) {
                ros_arr(i, j, k) = surface_ros_arr(i, j, k);
                crown_active_arr(i, j, k) = 0.0_rt;
                crown_ros_arr(i, j, k) = 0.0_rt;
                crown_frac_arr(i, j, k) = 0.0_rt;
                return;
            }

            const Real R_surface = amrex::max(surface_ros_arr(i, j, k), 0.0_rt);
            const Real I_surface = amrex::max(surface_I_B_arr(i, j, k), 0.0_rt);
            const Real U10_ms = (fixed_u10_ms > 0.0_rt)
                ? fixed_u10_ms
                : std::sqrt(wind_arr(i, j, k, 0) * wind_arr(i, j, k, 0)
                          + wind_arr(i, j, k, 1) * wind_arr(i, j, k, 1));
            const Real moisture_10hr = use_dynamic_mc ? mc_arr(i, j, k, 1) : default_moisture_10hr;

            Real R_active = R_surface;
            if (crown_model_id == 1) {
                R_active = compute_rothermel_1991_crown_ros(R_surface);
            } else if (crown_model_id == 2) {
                R_active = compute_van_wagner_proxy_ros(canopy_bulk_den, foliar_moisture);
            } else {
                R_active = cruz_crown_ros(U10_ms, canopy_bulk_den, moisture_10hr);
            }
            R_active = amrex::max(R_active, R_surface);

            const bool crown_now_active = (crown_active_arr(i, j, k) >= 0.5_rt) || (I_surface >= I_B_crit);
            crown_active_arr(i, j, k) = crown_now_active ? 1.0_rt : 0.0_rt;
            crown_ros_arr(i, j, k) = R_active;

            Real R_total = R_surface;
            if (use_passive_blend) {
                R_total = compute_van_wagner_passive_blend(R_surface, R_active, I_surface, I_B_crit);
            } else if (crown_now_active) {
                R_total = R_active;
            }
            ros_arr(i, j, k) = amrex::max(R_total, 0.0_rt);
            crown_frac_arr(i, j, k) = compute_crown_fraction_burned(ros_arr(i, j, k), R_surface, R_active);
        });
    }
}

void FireLayer::update_atm_flux_buffer(const amrex::Geometry& geom_atm)
{
    if (!m_params.injects_flux() && !m_params.smoke_enable) {
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Skipping flux buffer update: coupling_type is passive (injects_flux=false) and smoke_enable=false" << std::endl;
        }
        return;
    }
    if (!fire_heat_flux || !m_Q_atm_prev) { return; }

    if (m_params.fire_debug) {
        amrex::Print() << "[FIRE DEBUG] Updating atmosphere flux buffer for coupling_type="
                       << m_params.coupling_type << ". Current max heat flux: "
                       << fire_heat_flux->max(0) << " W/m2" << std::endl;
    }

    coarsen_fire_flux_to_atm(*m_Q_atm_prev, *fire_heat_flux,
                             geom_atm, m_fg.geom, m_fg.C);

    if (m_params.inject_latent && m_Q_lat_atm_prev) {
        FuelModelParams fp = get_anderson_fuel_params(m_params.fuel_model_id);
        amrex::Real h_fuel_Jkg = fp.heat_content * 2326.0_rt;

        amrex::Real M_f = m_params.moisture_1hr;
        if (m_params.moisture_dynamic && fire_fuel_mc) {
            long nc = fire_fuel_mc->boxArray().numPts();
            amrex::Real avg1   = (nc > 0) ? fire_fuel_mc->sum(0) / amrex::Real(nc) : m_params.moisture_1hr;
            amrex::Real avg10  = (nc > 0) ? fire_fuel_mc->sum(1) / amrex::Real(nc) : m_params.moisture_10hr;
            amrex::Real avg100 = (nc > 0) ? fire_fuel_mc->sum(2) / amrex::Real(nc) : m_params.moisture_100hr;
            amrex::Real dead_load = fp.w_d1 + fp.w_d10 + fp.w_d100;
            M_f = (dead_load > 1e-10_rt)
                ? (fp.w_d1*avg1 + fp.w_d10*avg10 + fp.w_d100*avg100) / dead_load
                : avg1;
        }

        compute_fire_latent_flux(*fire_latent_flux, *fire_heat_flux, M_f, h_fuel_Jkg);
        coarsen_fire_flux_to_atm(*m_Q_lat_atm_prev, *fire_latent_flux,
                                 geom_atm, m_fg.geom, m_fg.C);
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Latent heat flux computed. Max latent flux: "
                           << fire_latent_flux->max(0) << " W/m2, fuel moisture: " << M_f << std::endl;
        }
    } else {
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Skipping latent heat injection (inject_latent=false or no moisture)" << std::endl;
        }
        m_Q_lat_atm_prev->setVal(0.0_rt);
    }
}

void FireLayer::apply_fire_coupling_to_cc_source(
    amrex::MultiFab& cc_source,
    const amrex::MultiFab& S_old,
    const amrex::MultiFab& z_phys_cc,
    const amrex::Geometry& geom_atm,
    bool has_moisture)
{
    if (!m_params.injects_flux()) { return; }
    if (!m_Q_atm_prev) { return; }
    if (m_params.fire_atm_feedback <= 0.0_rt) { return; }

    if (m_params.fire_debug) {
        amrex::Print() << "[FIRE DEBUG] Applying fire coupling to atmosphere (coupling_type="
                       << m_params.coupling_type << ", feedback=" << m_params.fire_atm_feedback
                       << ", max_Q_prev=" << m_Q_atm_prev->max(0) << " W/m2)" << std::endl;
    }

    const amrex::MultiFab* Q_lat_ptr = (m_params.inject_latent && has_moisture)
        ? m_Q_lat_atm_prev.get() : nullptr;

    // Pass fire_debug so apply_fire_tendency_to_cc_source prints tendency diagnostics.
    // With fire_debug=true you will see per-RK-stage output:
    //   [FIRE COUPLING] Q_atm_max=...  alfg=...
    //   [FIRE COUPLING] RhoTheta tendency sum=...  max=...  expected_surface_max=...
    apply_fire_tendency_to_cc_source(
        cc_source,
        *m_Q_atm_prev,
        Q_lat_ptr,
        z_phys_cc,
        S_old,
        geom_atm,
        m_params.heat_flux_alfg,
        m_params.fire_atm_feedback,
        has_moisture,
        m_params.fire_debug);
}

// ───────────────────────────────────────────────────────────────────────────
// Rate-of-spread model helpers
// ───────────────────────────────────────────────────────────────────────────

void FireLayer::fill_ros_for_model(const std::string& model,
                                   amrex::MultiFab& out,
                                   const BalbiFieldInputs& balbi_in)
{
    if (model == "balbi") {
        // Balbi normalises the wind by its own vertical velocity scale rather
        // than by a midflame reduction, so balbi.wind_source can bypass the
        // Wind Adjustment Factor and hand it the reference-height wind.
        const amrex::MultiFab& balbi_wind = (m_params.balbi.wind_source == 1)
                                          ? *fire_wind_ref : *fire_wind_eff;
        fill_balbi_ros(out, balbi_wind, *fire_slopes,
                       m_bc_default, m_params.balbi, balbi_in);
    } else if (model == "cheney_gould") {
        fill_cheney_gould_ros(out, *fire_wind_eff, m_cgc);
    } else if (model == "behave") {
        // Phase 15: BEHAVE multi-class Rothermel model
        FuelModelParams fp_behave = get_anderson_fuel_params(m_params.fuel_model_id);
        fill_behave_ros(out, *fire_wind_eff, *fire_slopes,
                        fp_behave,
                        m_bs_default,
                        m_params.moisture_dynamic ? fire_fuel_mc.get() : nullptr,
                        m_params.moisture_dynamic);
    } else if (model == "macarthur") {
        fill_macarthur_ros(out, *fire_wind_eff);
    } else {
        // Default: Rothermel (1972). The per-fuel table is empty unless
        // rothermel_per_fuel is set on a spatial fuel map, in which case the
        // overload falls through to the uniform kernel.
        const bool per_fuel = !m_d_rc_table.empty() && fire_fuel_model;
        compute_ros_field(out, *fire_wind_eff, *fire_slopes, m_rc,
                          per_fuel ? fire_fuel_model.get() : nullptr,
                          per_fuel ? m_d_rc_table.data() : nullptr,
                          per_fuel ? static_cast<int>(m_d_rc_table.size()) : 0);
    }
}

void FireLayer::rebuild_rothermel_table(amrex::Real m1, amrex::Real m10, amrex::Real m100)
{
    auto h_table = build_fuel_rothermel_table(m1, m10, m100);
    m_d_rc_table.resize(h_table.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_table.begin(), h_table.end(),
                     m_d_rc_table.begin());
}

void FireLayer::init_ros_weight()
{
    const auto& hy = m_params.hybrid;
    const auto prob_lo = m_fg.geom.ProbLoArray();
    const auto dx      = m_fg.geom.CellSizeArray();

    fire_ros_weight->setVal(0.0_rt);

    if (hy.selector == "region") {
        const amrex::Real x_lo = hy.region_x_lo;
        const amrex::Real y_lo = hy.region_y_lo;
        const amrex::Real x_hi = hy.region_x_hi;
        const amrex::Real y_hi = hy.region_y_hi;
        const amrex::Real bw   = hy.blend_width;
        for (amrex::MFIter mfi(*fire_ros_weight); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            auto const& w = fire_ros_weight->array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const amrex::Real x = prob_lo[0] + (i + 0.5_rt) * dx[0];
                const amrex::Real y = prob_lo[1] + (j + 0.5_rt) * dx[1];
                if (bw > 0.0_rt) {
                    // Signed distance into the rectangle (positive inside),
                    // ramped linearly over blend_width centred on the edge.
                    const amrex::Real d = amrex::min(amrex::min(x - x_lo, x_hi - x),
                                                     amrex::min(y - y_lo, y_hi - y));
                    w(i, j, k) = amrex::max(0.0_rt, amrex::min(1.0_rt, 0.5_rt + d / bw));
                } else {
                    w(i, j, k) = (x >= x_lo && x < x_hi && y >= y_lo && y < y_hi) ? 1.0_rt : 0.0_rt;
                }
            });
        }
    } else if (hy.selector == "fuel") {
        if (!fire_fuel_model) {
            amrex::Abort("erf.fire.hybrid.selector = fuel needs a spatial fuel map "
                         "(erf.fire.fuel_map.file)");
        }
        const int n_codes = static_cast<int>(hy.secondary_fuel_codes.size());
        amrex::Gpu::DeviceVector<int> d_codes(n_codes);
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                         hy.secondary_fuel_codes.begin(), hy.secondary_fuel_codes.end(),
                         d_codes.begin());
        const int* codes = d_codes.data();
        for (amrex::MFIter mfi(*fire_ros_weight); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            auto const& w    = fire_ros_weight->array(mfi);
            auto const& fuel = fire_fuel_model->const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const int code = static_cast<int>(fuel(i, j, k));
                amrex::Real wt = 0.0_rt;
                for (int n = 0; n < n_codes; ++n) {
                    if (codes[n] == code) { wt = 1.0_rt; break; }
                }
                w(i, j, k) = wt;
            });
        }
        // The kernels above read d_codes asynchronously; let them finish
        // before the device vector is freed at scope exit.
        amrex::Gpu::streamSynchronize();
    } else if (hy.selector == "structure") {
        // Cells within structure_distance of a structure cell (height above
        // structure_min_height) take the secondary model; blend_width ramps
        // the weight linearly across that distance instead of stepping.
        const amrex::Real D    = hy.structure_distance;
        const amrex::Real bw   = hy.blend_width;
        const amrex::Real hmin = hy.structure_min_height;
        const amrex::Real search = D + 0.5_rt * bw;
        const int ri = static_cast<int>(std::ceil(search / dx[0]));
        const int rj = static_cast<int>(std::ceil(search / dx[1]));
        const int ng = amrex::max(ri, rj) + 1;

        amrex::MultiFab mask(m_fg.ba, m_fg.dm, 1, ng);
        mask.setVal(0.0_rt);
        for (amrex::MFIter mfi(mask); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            auto const& m = mask.array(mfi);
            auto const& h = fire_structure_height->const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                m(i, j, k) = (h(i, j, k) > hmin) ? 1.0_rt : 0.0_rt;
            });
        }
        mask.FillBoundary(m_fg.geom.periodicity());

        const amrex::Real dxf = dx[0];
        const amrex::Real dyf = dx[1];
        for (amrex::MFIter mfi(*fire_ros_weight); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            auto const& w = fire_ros_weight->array(mfi);
            auto const& m = mask.const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                amrex::Real dmin = 1.0e30_rt;
                for (int dj = -rj; dj <= rj; ++dj) {
                    for (int di = -ri; di <= ri; ++di) {
                        if (m(i + di, j + dj, k) > 0.5_rt) {
                            const amrex::Real d = std::sqrt(di * dxf * di * dxf + dj * dyf * dj * dyf);
                            dmin = amrex::min(dmin, d);
                        }
                    }
                }
                if (bw > 0.0_rt) {
                    w(i, j, k) = amrex::max(0.0_rt, amrex::min(1.0_rt, 0.5_rt + (D - dmin) / bw));
                } else {
                    w(i, j, k) = (dmin <= D) ? 1.0_rt : 0.0_rt;
                }
            });
        }
    } else if (hy.selector == "wind") {
        // Rebuilt from the effective wind at every fire step; the wind is not
        // known yet here, so the weight starts at zero.
    } else {
        amrex::Abort("FireLayer::init_ros_weight: unsupported hybrid selector " + hy.selector);
    }
}

void FireLayer::update_wind_weight()
{
    const amrex::Real lo = m_params.hybrid.wind_lo;
    const amrex::Real hi = m_params.hybrid.wind_hi;
    for (amrex::MFIter mfi(*fire_ros_weight); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& w    = fire_ros_weight->array(mfi);
        auto const& wind = fire_wind_eff->const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const amrex::Real u = wind(i, j, k, 0);
            const amrex::Real v = wind(i, j, k, 1);
            const amrex::Real U = std::sqrt(u * u + v * v);
            w(i, j, k) = amrex::max(0.0_rt, amrex::min(1.0_rt, (U - lo) / (hi - lo)));
        });
    }
}

void FireLayer::print_hybrid_weight_summary() const
{
    const amrex::Real w_sum = fire_ros_weight->sum(0);
    // Count cells that take mostly the secondary model (weight > 0.5).
    amrex::MultiFab flag(m_fg.ba, m_fg.dm, 1, 0);
    for (amrex::MFIter mfi(flag); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& f = flag.array(mfi);
        auto const& w = fire_ros_weight->const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            f(i, j, k) = (w(i, j, k) > 0.5_rt) ? 1.0_rt : 0.0_rt;
        });
    }
    const long n_half = std::lround(flag.sum(0));
    amrex::Print() << "[FIRE DEBUG] Hybrid ROS: primary=" << m_params.hybrid.primary
                   << " secondary=" << m_params.hybrid.secondary
                   << " selector=" << m_params.hybrid.selector
                   << " weight_sum=" << w_sum
                   << " secondary_cells=" << n_half << "\n";
}

DirectionalRosState FireLayer::make_directional_state(const std::string& model) const
{
    DirectionalRosState st;
    if (model == "behave") {
        st.model = DIRECTIONAL_ROS_BEHAVE;
        st.bs    = m_bs_default;
    } else if (model == "macarthur") {
        st.model = DIRECTIONAL_ROS_MACARTHUR;
    } else if (model == "cheney_gould") {
        st.model       = DIRECTIONAL_ROS_CHENEY_GOULD;
        st.cgc         = m_cgc;
        st.cg_moisture = m_params.cheney_gould.moisture;
        st.cg_curing   = m_params.cheney_gould.curing;
    } else {
        st.model = DIRECTIONAL_ROS_ROTHERMEL;
        st.rc    = m_rc;
    }
    return st;
}

void FireLayer::report_probes()
{
    const int np = static_cast<int>(m_params.probes.size() / 2);
    if (np == 0 || !fire_arrival_time) { return; }

    const auto prob_lo = m_fg.geom.ProbLoArray();
    const auto dx      = m_fg.geom.CellSizeArray();
    const amrex::Box& domain = m_fg.geom.Domain();

    for (int n = 0; n < np; ++n) {
        if (m_probe_reported[n]) { continue; }
        const amrex::Real x = m_params.probes[2 * n];
        const amrex::Real y = m_params.probes[2 * n + 1];
        const int ip = static_cast<int>(std::floor((x - prob_lo[0]) / dx[0]));
        const int jp = static_cast<int>(std::floor((y - prob_lo[1]) / dx[1]));
        const amrex::IntVect cell(ip, jp, 0);
        if (!domain.contains(cell)) {
            amrex::Print() << "[FIRE PROBE] " << n << " x=" << x << " y=" << y
                           << " is outside the fire domain; ignored\n";
            m_probe_reported[n] = true;
            continue;
        }

        // Arrival time at the probe cell from whichever rank owns it; the
        // field is -1 until the cell burns, and the min over other ranks'
        // sentinel keeps that value.
        amrex::ReduceOps<amrex::ReduceOpMin> reduce_op;
        amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        for (amrex::MFIter mfi(*fire_arrival_time); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            if (!bx.contains(cell)) { continue; }
            auto const& at = fire_arrival_time->const_array(mfi);
            reduce_op.eval(bx, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                    return { (i == ip && j == jp) ? at(i, j, k) : amrex::Real(1.0e30) };
                });
        }
        amrex::Real val = amrex::get<0>(reduce_data.value(reduce_op));
        amrex::ParallelDescriptor::ReduceRealMin(val);

        if (val >= 0.0_rt && val < 1.0e29_rt) {
            amrex::Print() << "[FIRE PROBE] " << n << " x=" << x << " y=" << y
                           << " cell=(" << ip << "," << jp << ")"
                           << " arrival_time_s=" << val << "\n";
            m_probe_reported[n] = true;
        }
    }
}
